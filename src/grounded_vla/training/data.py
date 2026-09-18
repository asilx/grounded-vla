"""Causal episode data, train-only statistics, and shared train/inference transforms."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from grounded_vla.training.graph import GraphBatch
from grounded_vla.training.runtime import sha256

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_schema(meta):
    if meta.get("format_version") != 1:
        raise ValueError("Expected dataset format_version=1")
    if not 1 <= meta["state_dim"] <= 32 or not 1 <= meta["action_dim"] <= 32:
        raise ValueError("Physical state/action dimensions must be in [1,32]")
    if meta["input_preset"] not in {"canonical", "libero", "droid"}:
        raise ValueError("Unknown input preset")
    if meta["input_preset"] != "canonical" and (meta["state_dim"], meta["action_dim"]) != (8, 7):
        raise ValueError("LIBERO/DROID input presets require state_dim=8 and action_dim=7")
    if (
        not meta["action_convention"]
        or not np.isfinite(meta["control_period_seconds"])
        or meta["control_period_seconds"] <= 0
    ):
        raise ValueError("Specify action semantics and a positive control period")
    graph = meta["graph_schema"]
    if (
        not graph["id"]
        or not graph["coordinate_frame"]
        or not graph["features"]
        or not graph["relations"]
    ):
        raise ValueError("Graph schema requires an ID, frame, features, and relations")
    for key in ("features", "relations"):
        if len(set(graph[key])) != len(graph[key]):
            raise ValueError(f"Duplicate graph {key}")


def load_episode(path, meta):
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    states, actions = raw["state"], raw["actions"]
    n = len(states)
    if n == 0 or states.shape != (n, meta["state_dim"]) or actions.shape != (n, meta["action_dim"]):
        raise ValueError("Invalid episode state/action dimensions")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite episode state/actions")
    times, graph_times = raw["timestamps"], raw["graph_timestamps"]
    if (
        times.shape != (n,)
        or graph_times.shape != (n,)
        or not np.isfinite(times).all()
        or not np.isfinite(graph_times).all()
    ):
        raise ValueError("Invalid observation/graph timestamps")
    if (
        (times < 0).any()
        or (graph_times < 0).any()
        or (np.diff(times) <= 0).any()
        or (graph_times > times).any()
    ):
        raise ValueError("Graph must precede the action; timestamps must be increasing")
    if n > 1 and not np.allclose(
        np.diff(times), meta["control_period_seconds"], rtol=0.05, atol=1e-6
    ):
        raise ValueError("Episode control period differs from manifest")
    prompts = raw["prompt"]
    if (
        prompts.shape != (n,)
        or prompts.dtype.kind not in "US"
        or not all(str(x).strip() for x in prompts)
    ):
        raise ValueError("prompt must contain one nonempty Unicode string per frame")
    f, e, v = raw["graph_features"], raw["graph_edges"], raw["graph_valid"]
    if f.ndim != 3 or f.shape[0] != n or f.shape[2] != len(meta["graph_schema"]["features"]):
        raise ValueError("Invalid graph feature dimensions")
    nodes = f.shape[1]
    if (
        e.shape != (n, nodes, nodes)
        or v.shape != (n, nodes)
        or e.dtype != np.int64
        or v.dtype != np.bool_
    ):
        raise ValueError("Invalid graph edges/validity")
    if not np.isfinite(f).all() or (
        e.size and (e.min() < 0 or e.max() >= len(meta["graph_schema"]["relations"]))
    ):
        raise ValueError("Invalid graph features/relation indices")
    seen = np.zeros(n, bool)
    for camera in CAMERAS:
        key = f"images/{camera}"
        if key not in raw:
            if f"image_masks/{camera}" in raw:
                raise ValueError("Image mask supplied without image")
            continue
        if raw[key].shape != (n, 224, 224, 3) or raw[key].dtype != np.uint8:
            raise ValueError("Camera images must be uint8 [T,224,224,3] RGB")
        mask = raw.get(f"image_masks/{camera}", np.ones(n, bool))
        if mask.shape != (n,) or mask.dtype != np.bool_:
            raise ValueError("Camera masks must be boolean [T]")
        seen |= mask
    if not seen.all():
        raise ValueError("Every frame needs at least one available camera")
    return raw


def prepare_dataset(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    meta = read_json(manifest)
    validate_schema(meta)
    seen_ids, seen_paths, seen_hashes = set(), set(), set()
    stats = {
        k: [0, np.zeros(meta[f"{k}_dim"], np.float64), np.zeros(meta[f"{k}_dim"], np.float64)]
        for k in ("state", "action")
    }
    entries = []
    for entry in meta["episodes"]:
        path = (manifest.parent / entry["path"]).resolve()
        digest = sha256(path)
        if entry["id"] in seen_ids or path in seen_paths or digest in seen_hashes:
            raise ValueError("Duplicate episode ID, path, or bytes across splits")
        if entry["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        seen_ids.add(entry["id"])
        seen_paths.add(path)
        seen_hashes.add(digest)
        raw = load_episode(path, meta)
        entries.append(
            {
                **entry,
                "path": os.path.relpath(path, output.parent),
                "sha256": digest,
                "frames": len(raw["state"]),
            }
        )
        if entry["split"] == "train":
            for key in stats:
                x = raw["actions" if key == "action" else key].astype(np.float64)
                count, mean, m2 = stats[key]
                delta = x.mean(0) - mean
                total = count + len(x)
                m2 += ((x - x.mean(0)) ** 2).sum(0) + delta**2 * count * len(x) / total
                mean += delta * len(x) / total
                stats[key] = [total, mean, m2]
    if not any(e["split"] == "val" for e in entries) or not stats["state"][0]:
        raise ValueError("Separate train and validation episodes are required")
    norm = {
        key: {"mean": mean.tolist(), "std": np.sqrt(np.maximum(m2 / count, 1e-12)).tolist()}
        for key, (count, mean, m2) in stats.items()
    }
    prepared = {**meta, "episodes": entries, "normalization": norm, "normalization_split": "train"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prepared, indent=2) + "\n")
    return prepared


def load_prepared(path):
    path = Path(path).resolve()
    meta = read_json(path)
    validate_schema(meta)
    if meta.get("normalization_split") != "train":
        raise ValueError("Use grounded-vla-prepare to compute training statistics")
    for key in ("state", "action"):
        for field in ("mean", "std"):
            arr = np.asarray(meta["normalization"][key][field])
            if (
                arr.shape != (meta[f"{key}_dim"],)
                or not np.isfinite(arr).all()
                or (field == "std" and (arr <= 0).any())
            ):
                raise ValueError("Invalid normalization metadata")
    for entry in meta["episodes"]:
        if sha256(path.parent / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Episode changed after preparation: {entry['id']}")
    return meta


def normalize(x, stats):
    return (np.asarray(x, np.float32) - np.asarray(stats["mean"], np.float32)) / np.asarray(
        stats["std"], np.float32
    )


def observation_arrays(state, images, image_masks, prompt, meta, tokenizer):
    state = np.asarray(state, np.float32)
    if state.shape != (meta["state_dim"],) or not np.isfinite(state).all():
        raise ValueError("Invalid state vector")
    padded = np.zeros(32, np.float32)
    padded[: len(state)] = normalize(state, meta["normalization"]["state"])
    tokens, mask = tokenizer(prompt)
    out_images, masks = {}, {}
    for key in CAMERAS:
        if key in images:
            image = np.asarray(images[key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError("Images must be uint8 HWC RGB at 224x224")
            out_images[key] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 127.5 - 1
            masks[key] = torch.tensor(bool(image_masks.get(key, True)))
        else:
            out_images[key] = torch.zeros(3, 224, 224)
            masks[key] = torch.tensor(False)
    if not any(m.item() for m in masks.values()):
        raise ValueError("At least one camera must be present")
    return {
        "state": torch.from_numpy(padded),
        "images": out_images,
        "image_masks": masks,
        "tokenized_prompt": torch.from_numpy(tokens),
        "tokenized_prompt_mask": torch.from_numpy(mask),
    }


@dataclass
class Observation:
    state: torch.Tensor
    images: dict
    image_masks: dict
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None

    def to(self, device):
        return Observation(
            **{
                k: {n: v.to(device) for n, v in value.items()}
                if isinstance(value, dict)
                else value.to(device)
                for k, value in vars(self).items()
                if value is not None
            }
        )


class EpisodeDataset:
    def __init__(self, path, split, horizon, tokenizer, *, metadata=None):
        self.path = Path(path).resolve()
        self.meta = load_prepared(path) if metadata is None else metadata
        self.entries = [e for e in self.meta["episodes"] if e["split"] == split]
        self.frames = [(i, t) for i, e in enumerate(self.entries) for t in range(e["frames"])]
        if not self.frames:
            raise ValueError(f"Empty {split} split")
        self.horizon, self.tokenizer = horizon, tokenizer
        self._load = lru_cache(maxsize=2)(self._read)

    def _read(self, index):
        entry = self.entries[index]
        raw = load_episode(self.path.parent / entry["path"], self.meta)
        if len(raw["state"]) != entry["frames"]:
            raise ValueError("Prepared frame count mismatch")
        return raw

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        episode, t = self.frames[index]
        raw = self._load(episode)
        images = {key: raw[f"images/{key}"][t] for key in CAMERAS if f"images/{key}" in raw}
        masks = {key: raw[f"image_masks/{key}"][t] for key in images if f"image_masks/{key}" in raw}
        obs = observation_arrays(
            raw["state"][t], images, masks, raw["prompt"][t], self.meta, self.tokenizer
        )
        actions = np.zeros((self.horizon, 32), np.float32)
        valid = np.zeros_like(actions, bool)
        real = raw["actions"][t : t + self.horizon]
        dim = self.meta["action_dim"]
        actions[: len(real), :dim] = normalize(real, self.meta["normalization"]["action"])
        valid[: len(real), :dim] = True
        return {
            "observation": obs,
            "actions": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(valid),
            "features": torch.from_numpy(raw["graph_features"][t].astype(np.float32)),
            "edges": torch.from_numpy(raw["graph_edges"][t].copy()),
            "valid": torch.from_numpy(raw["graph_valid"][t].copy()),
        }


def collate(samples):
    observations = [s["observation"] for s in samples]
    obs = Observation(
        **{
            k: {key: torch.stack([o[k][key] for o in observations]) for key in observations[0][k]}
            if isinstance(observations[0][k], dict)
            else torch.stack([o[k] for o in observations])
            for k in observations[0]
        }
    )
    n = max(s["features"].shape[0] for s in samples)
    f = samples[0]["features"].shape[1]
    features, edges, valid = (
        torch.zeros(len(samples), n, f),
        torch.zeros(len(samples), n, n, dtype=torch.int64),
        torch.zeros(len(samples), n, dtype=torch.bool),
    )
    for i, sample in enumerate(samples):
        count = len(sample["valid"])
        features[i, :count] = sample["features"]
        edges[i, :count, :count] = sample["edges"]
        valid[i, :count] = sample["valid"]
    return (
        obs,
        GraphBatch(features, edges, valid),
        torch.stack([s["actions"] for s in samples]),
        torch.stack([s["action_mask"] for s in samples]),
    )


def batch_indices(length, batch_size, cursor, seed):
    """Stateless shuffled infinite stream; independent of the model RNG and resume timing."""
    indices = []
    while len(indices) < batch_size:
        epoch, offset = divmod(cursor, length)
        permutation = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(
            length
        )
        take = min(batch_size - len(indices), length - offset)
        indices.extend(permutation[offset : offset + take].tolist())
        cursor += take
    return indices, cursor
