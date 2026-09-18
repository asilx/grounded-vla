# ruff: noqa: E402
import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sentencepiece")
from grounded_vla.belief import Atom, BeliefStore
from grounded_vla.training.data import (
    EpisodeDataset,
    batch_indices,
    collate,
    load_prepared,
    prepare_dataset,
)
from grounded_vla.training.graph import encode_belief_graph
from grounded_vla.training.recording import EpisodeRecorder
from grounded_vla.training.tokenizer import Pi0Tokenizer


def test_stats_are_train_only_and_chunks_do_not_cross_episodes(episodes, tokenizer_file):
    manifest, prepared = episodes
    meta = load_prepared(prepared)
    values = [
        np.load(manifest.parent / e["path"])["actions"]
        for e in meta["episodes"]
        if e["split"] == "train"
    ]
    np.testing.assert_allclose(
        meta["normalization"]["action"]["mean"], np.concatenate(values).mean(0), rtol=1e-6
    )
    dataset = EpisodeDataset(prepared, "train", 5, Pi0Tokenizer(tokenizer_file))
    last = dataset[2]
    assert last["action_mask"].sum() == 7
    assert not last["action_mask"][:, 7:].any()
    assert not last["actions"][1:].any()
    obs, graph, _, _ = collate([dataset[0], dataset[2]])
    assert not obs.image_masks["right_wrist_0_rgb"].any()
    assert obs.images["base_0_rgb"].shape == (2, 3, 224, 224)
    assert graph.features.shape == (2, 2, 26)


@pytest.mark.parametrize(
    "corruption",
    [
        "future_graph",
        "nan_action",
        "bad_relation",
        "no_camera",
        "bad_period",
        "object_prompt",
        "negative_relation",
    ],
)
def test_causal_data_rejections(episodes, corruption):
    manifest, _ = episodes
    raw_meta = json.loads(manifest.read_text())
    path = manifest.parent / raw_meta["episodes"][0]["path"]
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    if corruption == "future_graph":
        data["graph_timestamps"][0] = 1
    elif corruption == "nan_action":
        data["actions"][0, 0] = np.nan
    elif corruption == "bad_relation":
        data["graph_edges"][0, 0, 0] = 8
    elif corruption == "negative_relation":
        data["graph_edges"][0, 0, 0] = -1
    elif corruption == "no_camera":
        data = {k: v for k, v in data.items() if not k.startswith("images/")}
    elif corruption == "bad_period":
        data["timestamps"] *= 2
    else:
        data["prompt"] = np.array([object()] * 3, dtype=object)
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        prepare_dataset(manifest, manifest.parent / "invalid.json")


def test_changed_episode_and_duplicate_bytes_rejected(episodes):
    manifest, prepared = episodes
    meta = json.loads(manifest.read_text())
    first = manifest.parent / meta["episodes"][0]["path"]
    alias = manifest.parent / "alias.npz"
    alias.write_bytes(first.read_bytes())
    meta["episodes"].append({"id": "alias", "path": alias.name, "split": "val"})
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_dataset(manifest, manifest.parent / "invalid.json")
    first.write_bytes(first.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_prepared(prepared)


def test_stream_resume_is_independent_of_model_rng():
    first, cursor = batch_indices(7, 10, 0, 9)
    tail, _ = batch_indices(7, 5, cursor, 9)
    np.random.seed(321)
    whole, _ = batch_indices(7, 15, 0, 9)
    assert whole == first + tail
    assert len(set(whole[:7])) == 7


def test_graph_evidence_is_fresh_and_observed():
    store = BeliefStore()
    store.add(Atom("clean", "cup"), True, at=0, ttl=1, source="camera")
    store.add(Atom("filled", "cup"), True, at=0, source="model", kind="predicted")
    obj = [{"id": "cup", "position": [0, 0, 0], "visible": True}]
    graph = encode_belief_graph(store, obj, [], now=2)
    np.testing.assert_array_equal(graph["features"][0, 6:11], [0, 0, 1, 0, 1])
    np.testing.assert_array_equal(graph["features"][0, 11:16], [0, 0, 1, 0, 1])


def test_token_budget_is_not_silently_truncated(tokenizer_file):
    tokenizer = Pi0Tokenizer(tokenizer_file)
    with pytest.raises(ValueError, match="budget"):
        tokenizer("clean cup " * 100)


def test_recorder_snapshots_arrays_and_pads_graphs(tmp_path):
    store = BeliefStore()
    graph = encode_belief_graph(store, [{"id": "cup", "position": [1, 0, 0]}], [], now=0)
    recorder = EpisodeRecorder(graph["schema_id"])
    state = np.zeros(8, np.float32)
    kwargs = dict(
        state=state,
        images={"base_0_rgb": np.zeros((224, 224, 3), np.uint8)},
        action=np.ones(7),
        prompt="pick cup",
    )
    recorder.append(**kwargs, graph=graph, timestamp=0, graph_timestamp=0)
    state[:] = 9
    graph["features"][:] = 4
    empty = encode_belief_graph(store, [], [], now=0.05)
    recorder.append(**kwargs, graph=empty, timestamp=0.05, graph_timestamp=0.05)
    out = tmp_path / "record.npz"
    recorder.save(out)
    with np.load(out) as data:
        assert not data["state"][0].any()
        assert data["graph_features"][0, 0, 0] == 1
        assert not data["graph_valid"][1].any()
    with pytest.raises(FileExistsError):
        recorder.save(out)
