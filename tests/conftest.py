"""Native π0 fixtures reduce only constructor dimensions; upstream algorithms remain intact."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def native_tiny(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("openpi")
    from openpi.models import gemma
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
    from transformers import (
        GemmaConfig,
        GemmaForCausalLM,
        PaliGemmaConfig,
        PaliGemmaForConditionalGeneration,
    )

    from grounded_vla.training.pi0 import GroundedPi0

    torch.set_num_threads(1)

    def reduced_init(self, vlm_config, action_expert_config, use_adarms=None, precision="float32"):
        torch.nn.Module.__init__(self)
        cfg = gemma.get_config("dummy")
        text = dict(
            hidden_size=cfg.width,
            intermediate_size=cfg.mlp_dim,
            num_hidden_layers=cfg.depth,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            vocab_size=128,
            hidden_activation="gelu_pytorch_tanh",
            use_adarms=False,
            adarms_cond_dim=None,
        )
        vision = dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=224,
            patch_size=112,
            projection_dim=cfg.width,
        )
        hf = PaliGemmaConfig(
            text_config=text,
            vision_config=vision,
            projection_dim=cfg.width,
            hidden_size=cfg.width,
            vocab_size=128,
            image_token_index=127,
        )
        self.paligemma = PaliGemmaForConditionalGeneration(hf)
        self.gemma_expert = GemmaForCausalLM(GemmaConfig(**text))
        self.gemma_expert.model.embed_tokens = None
        self.to_bfloat16_for_selected_params(precision)

    monkeypatch.setattr(PaliGemmaWithExpertModel, "__init__", reduced_init)

    def factory(config):
        native = Pi0Config(
            dtype="float32",
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_dim=32,
            action_horizon=config.action_horizon,
            max_token_len=48,
            pi05=False,
            pytorch_compile_mode=None,
        )
        return GroundedPi0(native, config.adapter)

    import grounded_vla.training.inference as inference
    import grounded_vla.training.train as trainer

    monkeypatch.setattr(trainer, "build_model", factory)
    monkeypatch.setattr(inference, "build_model", factory)
    return factory


@pytest.fixture
def tokenizer_file(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    corpus = tmp_path / "text.txt"
    corpus.write_text("pick the clean cup and place it on the tray\ninspect the blue cup\n" * 10)
    prefix = str(tmp_path / "tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=prefix, vocab_size=40, hard_vocab_limit=False, minloglevel=2
    )
    return Path(prefix + ".model")


@pytest.fixture
def episodes(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from grounded_vla.training.data import prepare_dataset
    from grounded_vla.training.graph import FEATURES, RELATIONS, SCHEMA_ID

    generator = np.random.default_rng(12)
    meta = {
        "format_version": 1,
        "state_dim": 8,
        "action_dim": 7,
        "input_preset": "libero",
        "action_convention": "test controller targets",
        "control_period_seconds": 0.05,
        "graph_schema": {
            "id": SCHEMA_ID,
            "coordinate_frame": "test robot base",
            "features": list(FEATURES),
            "relations": list(RELATIONS),
        },
        "episodes": [],
    }
    for i, split in enumerate(("train", "train", "val")):
        n = 3 + i
        arrays = {
            "state": generator.normal(size=(n, 8)).astype(np.float32),
            "actions": generator.normal(size=(n, 7)).astype(np.float32) + i * 10,
            "timestamps": np.arange(n) * 0.05,
            "graph_timestamps": np.arange(n) * 0.05,
            "prompt": np.asarray(["pick the clean cup"] * n),
            "graph_features": generator.normal(size=(n, 2, 26)).astype(np.float32),
            "graph_edges": np.zeros((n, 2, 2), np.int64),
            "graph_valid": np.ones((n, 2), bool),
            "images/base_0_rgb": generator.integers(0, 256, (n, 224, 224, 3), dtype=np.uint8),
            "images/left_wrist_0_rgb": generator.integers(0, 256, (n, 224, 224, 3), dtype=np.uint8),
        }
        path = tmp_path / f"episode-{i}.npz"
        np.savez_compressed(path, **arrays)
        meta["episodes"].append({"id": f"episode-{i}", "path": path.name, "split": split})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(meta))
    prepared = tmp_path / "prepared.json"
    prepare_dataset(manifest, prepared)
    return manifest, prepared


@pytest.fixture
def training_config(tmp_path, native_tiny, tokenizer_file, episodes):
    from safetensors.torch import save_model

    from grounded_vla.training.config import TrainConfig

    cfg = TrainConfig(
        dataset=str(episodes[1]),
        tokenizer=str(tokenizer_file),
        base_checkpoint=str(tmp_path / "base.safetensors"),
        output_dir=str(tmp_path / "run"),
        action_horizon=3,
        precision="float32",
        device="cpu",
        batch_size=2,
        accumulation_steps=2,
        steps=4,
        warmup_steps=1,
        eval_every=2,
        eval_batches=1,
        save_every=2,
        gradient_checkpointing=True,
        graph_dropout=0.2,
    )
    model = native_tiny(cfg)
    del model.graph_adapter
    save_model(model, cfg.base_checkpoint)
    return cfg
