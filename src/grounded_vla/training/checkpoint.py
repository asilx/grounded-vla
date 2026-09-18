"""Atomic adapter artifacts with exact optimizer, data cursor, and RNG resume state."""

import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def rng_state():
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], np.uint32), *np_state[2:]))


def save_checkpoint(root, model, optimizer, step, cursor, metadata):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step-{step:08d}"
    if destination.exists():
        raise FileExistsError(destination)
    temp = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root))
    try:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.graph_adapter.state_dict().items()},
            str(temp / "adapter.safetensors"),
        )
        (temp / "metadata.json").write_text(
            json.dumps({**metadata, "step": step, "cursor": cursor}, indent=2) + "\n"
        )
        torch.save(
            {"optimizer": optimizer.state_dict(), "rng": rng_state()}, temp / "training_state.pt"
        )
        os.replace(temp, destination)
        pointer = root / ".latest.tmp"
        pointer.write_text(destination.name + "\n")
        os.replace(pointer, root / "latest")
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return destination


def resume_checkpoint(path, model, optimizer, metadata):
    path = Path(path)
    previous = json.loads((path / "metadata.json").read_text())
    for key in ("identity", "training_config"):
        expected, actual = metadata[key], previous[key]
        if key == "training_config":
            ignored = {"output_dir", "base_checkpoint", "tokenizer", "dataset"}
            expected = {k: v for k, v in expected.items() if k not in ignored}
            actual = {k: v for k, v in actual.items() if k not in ignored}
        if actual != expected:
            raise ValueError(f"Resume mismatch: {key}")
    model.graph_adapter.load_state_dict(load_file(str(path / "adapter.safetensors")), strict=True)
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return previous["step"], previous["cursor"]
