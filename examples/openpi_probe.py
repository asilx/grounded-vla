"""Request action proposals for a captured NPZ observation. No robot execution."""

import argparse
import os
from pathlib import Path

import numpy as np

from grounded_vla.backends.openpi import OpenPiPolicy, OpenPiTransport
from grounded_vla.contracts import Contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "observation", type=Path, help="NPZ with the documented embodiment input keys"
    )
    parser.add_argument("--uri", default="ws://localhost:8000")
    parser.add_argument("--embodiment", choices=["libero", "droid"], required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument(
        "--object", default="the blue cup", help="A visually grounded object description"
    )
    parser.add_argument("--out", type=Path, default=Path("runs/openpi/actions.npy"))
    args = parser.parse_args()
    with np.load(args.observation, allow_pickle=False) as archive:
        observation = {key: archive[key] for key in archive.files}
    with OpenPiTransport(args.uri, api_key=os.environ.get("OPENPI_API_KEY")) as transport:
        policy = OpenPiPolicy(transport, embodiment=args.embodiment, action_dim=args.action_dim)
        chunk = policy.sample(
            Contract("place", "target", args.object), {"model_inputs": observation}
        )[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, np.asarray(chunk.values))
    print(f"Saved proposed actions with shape {np.asarray(chunk.values).shape} to {args.out}")
    print(
        "No actions were executed. Apply the matching robot adapter and validation before execution."
    )


if __name__ == "__main__":
    main()
