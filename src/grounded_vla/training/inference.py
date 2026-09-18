"""Restore a trained adapter and use native π0 denoising with identical data transforms."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from grounded_vla.training.config import TrainConfig
from grounded_vla.training.data import CAMERAS, collate, observation_arrays
from grounded_vla.training.graph import GraphBatch
from grounded_vla.training.runtime import sha256, verify_runtime
from grounded_vla.training.tokenizer import Pi0Tokenizer
from grounded_vla.training.train import build_model


def canonical_inputs(payload, preset):
    if preset == "canonical":
        state = payload["state"]
        images = payload["images"]
        masks = payload.get("image_masks", {})
    elif preset == "libero":
        state = payload["observation/state"]
        images = {
            "base_0_rgb": payload["observation/image"],
            "left_wrist_0_rgb": payload["observation/wrist_image"],
        }
        masks = {}
    elif preset == "droid":
        joints = np.asarray(payload["observation/joint_position"])
        gripper = np.asarray(payload["observation/gripper_position"])
        if joints.shape != (7,) or gripper.shape != (1,):
            raise ValueError("DROID requires 7 joint positions and one gripper position")
        state = np.concatenate((joints, gripper))
        images = {
            "base_0_rgb": payload["observation/exterior_image_1_left"],
            "left_wrist_0_rgb": payload["observation/wrist_image_left"],
        }
        masks = {}
    else:
        raise ValueError("Unknown input preset")
    return state, images, masks


class TrainedPi0Policy:
    def __init__(
        self,
        checkpoint,
        *,
        base_checkpoint=None,
        tokenizer=None,
        device="cuda",
        num_steps=10,
        disable_graph=False,
    ):
        verify_runtime()
        checkpoint = Path(checkpoint)
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        config = TrainConfig(**metadata["training_config"])
        config.device = device
        if device == "cpu":
            config.precision = "float32"
        config.__post_init__()
        base_checkpoint = base_checkpoint or config.base_checkpoint
        tokenizer = tokenizer or config.tokenizer
        for key, path in (("base_checkpoint", base_checkpoint), ("tokenizer", tokenizer)):
            if sha256(path) != metadata["identity"][key]:
                raise ValueError(f"Artifact {key} identity mismatch")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.meta = metadata["dataset"]
        self.model = build_model(config)
        self.model.load_base(base_checkpoint)
        self.model.graph_adapter.load_state_dict(
            load_file(str(checkpoint / "adapter.safetensors")), strict=True
        )
        self.model.to(device).eval()
        self.tokenizer = Pi0Tokenizer(tokenizer)
        self.device, self.num_steps, self.disable_graph = device, num_steps, disable_graph
        self._lock = threading.Lock()
        self.metadata = {
            "policy": "grounded-vla-pi0",
            "graph_required": True,
            "graph_schema_id": self.meta["graph_schema"]["id"],
            "input_preset": self.meta["input_preset"],
            "action_dim": self.meta["action_dim"],
            "action_horizon": config.action_horizon,
            "action_convention": self.meta["action_convention"],
            "control_period_seconds": self.meta["control_period_seconds"],
        }

    def transform(self, payload):
        state, images, masks = canonical_inputs(payload, self.meta["input_preset"])
        observation = observation_arrays(
            state, images, masks, payload["prompt"], self.meta, self.tokenizer
        )
        raw = payload["graph"]
        if raw["schema_id"] != self.meta["graph_schema"]["id"]:
            raise ValueError("Live graph schema differs from the training artifact")
        features = torch.from_numpy(np.asarray(raw["features"], np.float32).copy())
        edges = torch.from_numpy(np.asarray(raw["edges"]).copy())
        valid = torch.from_numpy(np.asarray(raw["valid"]).copy())
        graph = GraphBatch(features[None], edges[None], valid[None])
        graph.validate(
            self.model.adapter_config.feature_dim, self.model.adapter_config.relation_types
        )
        batch = collate(
            [
                {
                    "observation": observation,
                    "features": features,
                    "edges": edges,
                    "valid": valid,
                    "actions": torch.zeros(1, 32),
                    "action_mask": torch.ones(1, 32, dtype=torch.bool),
                }
            ]
        )
        return batch[0].to(self.device), (graph.disabled() if self.disable_graph else graph).to(
            self.device
        )

    @torch.no_grad()
    def infer(self, payload, *, noise=None):
        with self._lock:
            obs, graph = self.transform(payload)
            noise = noise.to(self.device) if noise is not None else None
            result = self.model.grounded_actions(
                torch.device(self.device), obs, graph, noise=noise, num_steps=self.num_steps
            )
            actions = result[0, :, : self.meta["action_dim"]].cpu().float().numpy()
            stats = self.meta["normalization"]["action"]
            actions = actions * np.asarray(stats["std"], np.float32) + np.asarray(
                stats["mean"], np.float32
            )
            if not np.isfinite(actions).all():
                raise FloatingPointError("Non-finite policy actions")
            return {"actions": actions}

    def reset(self):
        pass  # Stateless policy: no previous episode information is retained.


def load_observation(path, preset):
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    payload = {k: v for k, v in raw.items() if k.startswith("observation/")}
    if preset == "canonical":
        payload = {
            "state": raw["state"],
            "images": {k: raw[f"images/{k}"] for k in CAMERAS if f"images/{k}" in raw},
            "image_masks": {
                k: bool(raw[f"image_masks/{k}"]) for k in CAMERAS if f"image_masks/{k}" in raw
            },
        }
    payload["prompt"] = str(raw["prompt"].item())
    payload["graph"] = {k: raw[f"graph/{k}"] for k in ("features", "edges", "valid")}
    payload["graph"]["schema_id"] = str(raw["graph/schema_id"].item())
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("observation")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--disable-graph", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    policy = TrainedPi0Policy(
        args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        tokenizer=args.tokenizer,
        device=args.device,
        num_steps=args.num_steps,
        disable_graph=args.disable_graph,
    )
    response = policy.infer(load_observation(args.observation, policy.meta["input_preset"]))
    with Path(args.out).open("xb") as stream:
        np.savez_compressed(stream, **response)
    print(f"Wrote proposed actions to {args.out}")


if __name__ == "__main__":
    main()
