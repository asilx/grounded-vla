"""Export one recorded frame for grounded-vla-predict without its action target."""

import argparse
from pathlib import Path

import numpy as np

from grounded_vla.training.data import CAMERAS, load_episode, load_prepared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset")
    p.add_argument("--episode", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    meta = load_prepared(args.dataset)
    entry = next(e for e in meta["episodes"] if e["id"] == args.episode)
    raw = load_episode(Path(args.dataset).resolve().parent / entry["path"], meta)
    t = args.frame
    if not 0 <= t < len(raw["state"]):
        raise ValueError("Frame outside episode")
    result = {
        "prompt": raw["prompt"][t],
        "graph/schema_id": np.asarray(meta["graph_schema"]["id"]),
        **{f"graph/{key}": raw[f"graph_{key}"][t] for key in ("features", "edges", "valid")},
    }
    preset = meta["input_preset"]
    if preset == "canonical":
        result["state"] = raw["state"][t]
        for camera in CAMERAS:
            if f"images/{camera}" in raw:
                result[f"images/{camera}"] = raw[f"images/{camera}"][t]
                if f"image_masks/{camera}" in raw:
                    result[f"image_masks/{camera}"] = raw[f"image_masks/{camera}"][t]
    else:
        camera_keys = (
            ("observation/image", "observation/wrist_image")
            if preset == "libero"
            else ("observation/exterior_image_1_left", "observation/wrist_image_left")
        )
        for camera, target in zip(CAMERAS[:2], camera_keys, strict=True):
            if f"image_masks/{camera}" in raw and not raw[f"image_masks/{camera}"][t]:
                raise ValueError("The live preset requires both cameras")
            result[target] = raw[f"images/{camera}"][t]
        if preset == "libero":
            result["observation/state"] = raw["state"][t]
        else:
            result["observation/joint_position"] = raw["state"][t, :7]
            result["observation/gripper_position"] = raw["state"][t, 7:]
    with Path(args.out).open("xb") as stream:
        np.savez_compressed(stream, **result)


if __name__ == "__main__":
    main()
