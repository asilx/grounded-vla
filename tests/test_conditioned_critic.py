"""Context-dependent gradients, causal recording, and checkpoint compatibility."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from grounded_vla.belief import Atom, Truth  # noqa: E402
from grounded_vla.contracts import Contract, Requirement  # noqa: E402
from grounded_vla.critic_context import CONTRACT_DIM, encode_critic_context  # noqa: E402
from grounded_vla.learning import temporal_vit as tv  # noqa: E402
from grounded_vla.simulation import GraphSnapshot  # noqa: E402
from grounded_vla.training.critic import (  # noqa: E402
    collate_critic_batch,
    fit_critic,
    load_critic_datasets,
)
from grounded_vla.training.recording import EpisodeRecorder  # noqa: E402


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return tv.TemporalViTConfig(
        feature_dim=16,
        temporal_dim=16,
        heads=2,
        layers=1,
        max_frames=3,
        dropout=0,
        contract_dim=CONTRACT_DIM,
        state_dim=2,
        graph_feature_dim=3,
        graph_relation_types=2,
        graph_schema_id="test-v1",
    )


def tiny(configuration=None, **kwargs):
    backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 16))
    return tv.TemporalViTCritic(backbone, configuration or config())


def context(timestamp=0.0, *, skill="pick", nodes=2):
    contract = Contract(
        skill,
        "cargo",
        "cargo",
        effects=(
            Requirement(Atom("held" if skill == "pick" else "at_tray", "cargo"), Truth.SUPPORTED),
        ),
    )
    graph = dict(
        schema_id="test-v1",
        features=np.arange(nodes * 3, dtype=np.float32).reshape(nodes, 3),
        edges=np.eye(nodes, dtype=np.int64),
        valid=np.ones(nodes, bool),
    )
    observation = SimpleNamespace(timestamp=timestamp, state=np.zeros(2, np.float32))
    ids = ("cargo", *(f"node-{i}" for i in range(1, nodes)))
    return encode_critic_context(contract, observation, GraphSnapshot(graph, timestamp, ids))


def tensors(value):
    return {
        k: torch.from_numpy(value[k].copy())[None]
        for k in ("contract", "state", "features", "edges", "valid", "target_mask")
    }


def test_same_video_responds_to_task_and_graph_with_trainable_gradients():
    torch.manual_seed(10)
    model = tiny().eval()
    frames = torch.randn(1, 3, 3, 16, 16)
    first, second = tensors(context()), tensors(context(skill="place"))
    original = model(frames, context=first)
    assert not torch.allclose(original, model(frames, context=second))
    changed = {k: v.clone() for k, v in first.items()}
    changed["features"][:, 0] += 10
    assert not torch.allclose(original, model(frames, context=changed))
    original.square().sum().backward()
    assert model.contract_projection.weight.grad.abs().sum() > 0
    assert model.state_projection.bias.grad.abs().sum() > 0
    assert model.graph_encoder.input.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())


def test_node_reordering_preserves_context_binding_and_logits():
    torch.manual_seed(4)
    model = tiny().eval()
    frames = torch.randn(1, 3, 3, 16, 16)
    before = tensors(context(nodes=3))
    order = [2, 0, 1]
    after = {k: v.clone() for k, v in before.items()}
    for key in ("features", "valid", "target_mask"):
        after[key] = after[key][:, order]
    after["edges"] = after["edges"][:, order][:, :, order]
    torch.testing.assert_close(
        model(frames, context=before), model(frames, context=after), atol=2e-6, rtol=2e-6
    )


def test_missing_or_unbound_context_is_rejected():
    model = tiny()
    frames = torch.randn(1, 3, 3, 16, 16)
    with pytest.raises(ValueError, match="requires"):
        model(frames)
    value = tensors(context())
    value["target_mask"].zero_()
    with pytest.raises(ValueError, match="valid contract target"):
        model(frames, context=value)
    with pytest.raises(ValueError, match="all context dimensions"):
        replace(config(), state_dim=0)
    with pytest.raises(ValueError, match="Unknown critic skill"):
        context(skill="unsupported-skill")


def test_conditioned_checkpoint_roundtrip_and_schema_guard(tmp_path, monkeypatch):
    torch.manual_seed(6)
    model = tiny().eval()
    path = tmp_path / "conditioned.pt"
    tv.save_critic(
        path,
        model,
        trained=True,
        training_metadata={"camera": "base_0_rgb", "control_period_seconds": 0.05},
    )
    monkeypatch.setattr(tv, "build_temporal_vit", tiny)
    predictor = tv.ViTWindowPredictor(path, expected_camera="base_0_rgb", expected_period=0.05)
    frames = np.zeros((3, 224, 224, 3), np.uint8)
    value = context()
    with torch.inference_mode():
        expected = model(tv.preprocess_rgb(frames)[None], context=tensors(value)).softmax(-1)[0]
    np.testing.assert_allclose(predictor(frames, context=value), expected.numpy(), atol=0, rtol=0)
    assert predictor.requires_context
    value["graph_schema_id"] = "wrong"
    with pytest.raises(ValueError, match="schema"):
        predictor(frames, context=value)


def test_monitor_resets_at_task_boundaries_and_checks_causality():
    class Predictor:
        requires_context = True

        def __init__(self):
            self.calls = []

        def __call__(self, frames, *, context):
            self.calls.append(context["contract_id"])
            return [0.0, 1.0, 0.0, 0.0]

    predictor = Predictor()
    monitor = tv.TemporalMonitor(predictor, window_size=3, patience=1)
    image = np.zeros((224, 224, 3), np.uint8)
    assert monitor.update(image, 0, context=context(0)) is None
    assert monitor.update(image, 0.05, context=context(0.05)) is None
    # A new contract cannot use the previous contract's images to trigger an alarm.
    assert monitor.update(image, 0.10, context=context(0.10, skill="place")) is None
    assert monitor.update(image, 0.15, context=context(0.15, skill="place")) is None
    assert monitor.update(image, 0.20, context=context(0.20, skill="place"))["reason"] == "stalled"
    assert len(predictor.calls) == 1
    future = context(0.25)
    future["graph_timestamp"] = 0.30
    with pytest.raises(ValueError, match="causal"):
        monitor.update(image, 0.25, context=future)


def make_dataset(tmp_path):
    entries = []
    for index, split in enumerate(("train", "val")):
        recorder = EpisodeRecorder("test-v1")
        for i in range(10):
            value = context(i * 0.05, skill="pick" if i < 5 else "place", nodes=2 + index)
            recorder.append(
                state=value["state"],
                action=np.zeros(2, np.float32),
                images={"base_0_rgb": np.full((224, 224, 3), index, np.uint8)},
                prompt="task",
                graph={
                    "schema_id": "test-v1",
                    **{k: value[k] for k in ("features", "edges", "valid")},
                },
                timestamp=value["timestamp"],
                graph_timestamp=value["graph_timestamp"],
                critic_context=value,
            )
        path = tmp_path / f"{split}.npz"
        recorder.save(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        arrays["critic_labels"] = np.array([0, 0, 0, 1, 2, 0, 0, 3, 0, 1], np.int64)
        np.savez(path, **arrays)
        entries.append({"path": path.name, "split": split})
    manifest = tmp_path / "critic.json"
    manifest.write_text(
        json.dumps(
            dict(
                format_version=2,
                camera="base_0_rgb",
                control_period_seconds=0.05,
                episodes=entries,
                conditioning=dict(
                    contract_dim=CONTRACT_DIM,
                    state_dim=2,
                    graph_feature_dim=3,
                    graph_relation_types=2,
                    graph_schema_id="test-v1",
                ),
            )
        )
    )
    return manifest


def test_conditioned_dataset_never_crosses_tasks_and_pads_graphs(tmp_path):
    datasets, provenance = load_critic_datasets(make_dataset(tmp_path), 3)
    assert [end for _, end in datasets["train"].windows] == [2, 3, 4, 7, 8, 9]
    frames, labels, batch = collate_critic_batch([datasets["train"][0], datasets["val"][0]])
    assert batch["features"].shape == (2, 3, 3)
    assert batch["valid"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["target_mask"].sum(-1).tolist() == [1, 1]
    assert provenance["conditioning"]["contract_dim"] == CONTRACT_DIM


def test_conditioned_training_and_validation_use_context(tmp_path):
    datasets, _ = load_critic_datasets(make_dataset(tmp_path), 3)
    model = tiny()
    before = model.contract_projection.weight.detach().clone()
    loaders = [
        DataLoader(datasets[s], batch_size=2, collate_fn=collate_critic_batch)
        for s in ("train", "val")
    ]
    history = fit_critic(model, *loaders, epochs=1)
    assert np.isfinite(history[0]["val_loss"])
    assert not torch.equal(before, model.contract_projection.weight)


def test_recording_rejects_future_and_mismatched_context(tmp_path):
    value = context()
    recorder = EpisodeRecorder("test-v1")
    kwargs = dict(
        state=value["state"],
        images={},
        action=np.zeros(2),
        prompt="pick",
        graph={"schema_id": "test-v1", **{k: value[k] for k in ("features", "edges", "valid")}},
        timestamp=0,
        graph_timestamp=0,
        critic_context=value,
    )
    value["graph_timestamp"] = 1
    with pytest.raises(ValueError, match="causal"):
        recorder.append(**kwargs)
    value["graph_timestamp"] = 0
    kwargs["state"] = np.ones(2)
    with pytest.raises(ValueError, match="pre-action"):
        recorder.append(**kwargs)


def test_training_rejects_future_graph_context(tmp_path):
    manifest = make_dataset(tmp_path)
    path = tmp_path / "train.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["graph_timestamps"][3] = arrays["timestamps"][3] + 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="causal"):
        load_critic_datasets(manifest, 3)
