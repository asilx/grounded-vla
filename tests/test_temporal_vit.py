# ruff: noqa: E402
"""Native torchvision ViT blocks at reduced dimensions; no pretrained weights."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.vision_transformer import VisionTransformer

from grounded_vla.learning import temporal_vit as tv
from grounded_vla.training.critic import fit_critic, load_critic_datasets


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny(config=None):
    config = config or tv.TemporalViTConfig(
        feature_dim=16,
        temporal_dim=16,
        heads=2,
        layers=1,
        max_frames=3,
        dropout=0,
    )
    backbone = VisionTransformer(
        image_size=224,
        patch_size=112,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
    )
    backbone.heads = nn.Identity()
    return tv.TemporalViTCritic(backbone, config)


def test_masked_frames_cannot_affect_logits_and_backbone_stays_frozen():
    torch.manual_seed(4)
    model = tiny().train()
    inputs = torch.randn(2, 3, 3, 224, 224)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    original = model(inputs, valid)
    changed = inputs.clone()
    changed[0, 2] = float("nan")
    torch.testing.assert_close(original, model(changed, valid), rtol=0, atol=0)
    original.sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.projection.weight.grad.abs().sum() > 0
    assert not model.backbone.training


def test_temporal_order_matters_and_empty_windows_are_rejected():
    torch.manual_seed(7)
    model = tiny().eval()
    inputs = torch.randn(1, 3, 3, 224, 224)
    assert not torch.allclose(model(inputs), model(inputs.flip(1)))
    with pytest.raises(ValueError, match="prefix"):
        model(inputs, torch.zeros(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="prefix"):
        model(inputs, torch.tensor([[True, False, True]]))


def test_supervised_fit_updates_temporal_head_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    torch.manual_seed(12)
    model = tiny()
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 3, 224, 224), torch.arange(4)), batch_size=2
    )
    before = model.head[-1].weight.detach().clone()
    history = fit_critic(model, loader, loader, epochs=1)
    assert not torch.equal(before, model.head[-1].weight)
    assert np.isfinite(history[0]["val_loss"])
    path = tmp_path / "critic.pt"
    tv.save_critic(
        path,
        model,
        trained=True,
        training_metadata={
            "camera": "base_0_rgb",
            "control_period_seconds": 0.05,
        },
    )
    monkeypatch.setattr(tv, "build_temporal_vit", tiny)
    predictor = tv.ViTWindowPredictor(
        path,
        expected_camera="base_0_rgb",
        expected_period=0.05,
    )
    frames = np.zeros((3, 224, 224, 3), np.uint8)
    with torch.inference_mode():
        expected = model.eval()(tv.preprocess_rgb(frames)[None]).softmax(-1)[0].numpy()
    np.testing.assert_allclose(predictor(frames), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="camera"):
        tv.ViTWindowPredictor(path, expected_camera="left_wrist_0_rgb")
    with pytest.raises(ValueError, match="period"):
        tv.ViTWindowPredictor(path, expected_period=0.1)


def test_untrained_checkpoint_is_rejected_before_model_construction(tmp_path):
    path = tmp_path / "untrained.pt"
    tv.save_critic(path, tiny(), trained=False)
    with pytest.raises(ValueError, match="trained"):
        tv.ViTWindowPredictor(path)


def test_full_torchvision_backbone_factory_and_rgb_preprocessing():
    model = tv.build_temporal_vit(tv.TemporalViTConfig(max_frames=2)).eval()
    frames = np.full((1, 224, 224, 3), 255, dtype=np.uint8)
    transformed = tv.preprocess_rgb(frames)
    np.testing.assert_allclose(
        transformed[0, :, 0, 0],
        (np.ones(3) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225],
        rtol=1e-6,
    )
    with torch.inference_mode():
        logits = model(transformed[None])
    assert logits.shape == (1, 4) and torch.isfinite(logits).all()


def test_monitor_warmup_patience_latch_and_episode_reset():
    predictions = iter(
        [
            [0.1, 0.8, 0.1, 0.0],
            [0.9, 0.05, 0.05, 0.0],
            [0.1, 0.8, 0.1, 0.0],
            [0.1, 0.8, 0.1, 0.0],
        ]
    )
    calls = []

    def predictor(frames):
        calls.append(frames)
        return next(predictions)

    monitor = tv.TemporalMonitor(predictor, window_size=3, patience=2, period=0.05)
    frame = np.zeros((224, 224, 3), np.uint8)
    for i in range(5):
        assert monitor.update(frame, i * 0.05) is None
    event = monitor.update(frame, 0.25)
    assert event["reason"] == "stalled" and event["consecutive_windows"] == 2
    assert len(calls) == 4
    assert monitor.update(frame, 0.30) is None and len(calls) == 4
    monitor.reset()
    assert monitor.update(frame, 0.0) is None and len(monitor.frames) == 1


def test_monitor_rejects_duplicate_time_and_invalid_probabilities():
    frame = np.zeros((224, 224, 3), np.uint8)
    monitor = tv.TemporalMonitor(lambda _: [0.0, float("nan"), 0, 0], window_size=2)
    monitor.update(frame, 0)
    with pytest.raises(ValueError, match="ordered"):
        monitor.update(frame, 0)
    with pytest.raises(ValueError, match="distribution"):
        monitor.update(frame, 0.05)


def make_manifest(tmp_path):
    paths = []
    for index, split in enumerate(("train", "val")):
        path = tmp_path / f"{split}.npz"
        np.savez(
            path,
            **{
                "images/base_0_rgb": np.full((6, 224, 224, 3), index, np.uint8),
                "timestamps": np.arange(6) * 0.05,
                "critic_labels": np.array([0, 0, 0, 1, 2, 3], np.int64),
            },
        )
        paths.append({"path": path.name, "split": split})
    manifest = tmp_path / "critic.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "camera": "base_0_rgb",
                "control_period_seconds": 0.05,
                "episodes": paths,
            }
        )
    )
    return manifest


def test_critic_windows_use_only_past_frames_and_keep_episodes_disjoint(tmp_path):
    manifest = make_manifest(tmp_path)
    datasets, meta = load_critic_datasets(manifest, 3)
    assert len(datasets["train"]) == 4 and len(datasets["val"]) == 4
    assert datasets["train"][0][0].shape == (3, 3, 224, 224)
    assert [datasets["train"][i][1] for i in range(4)] == [0, 1, 2, 3]
    assert len({e["sha256"] for e in meta["episodes"]}) == 2
    (tmp_path / "val.npz").write_bytes((tmp_path / "train.npz").read_bytes())
    with pytest.raises(ValueError, match="Duplicate episode content"):
        load_critic_datasets(manifest, 3)
