"""Check the physical-loss decoder matches the independently used runtime decoder."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def test_physical_loss_runtime_decoder_parity():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/runpod/learned_corrective_v15.py"
    spec = importlib.util.spec_from_file_location("corrective_v15", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = json.loads((root / "artifacts/isaac-v18/model/metadata.json").read_text())["dataset"]
    q = np.asarray([0.012, -0.57, 0.0, -2.81, 0.0, 3.037, 0.741, 0.02, 0.02], np.float32)
    prediction = torch.linspace(-2.0, 2.0, 9)[None].requires_grad_()
    physical = module.physical_targets(prediction, torch.from_numpy(q[None]), meta)
    stats = meta["normalization"]["action"]
    logits = prediction.detach().numpy() * np.asarray(stats["std"], np.float32) + np.asarray(
        stats["mean"], np.float32
    )
    runtime = module.decode_action_chunk(logits, q, meta)
    np.testing.assert_allclose(physical.detach().numpy(), runtime, atol=1e-6)
    assert np.all(np.abs(runtime - q) <= np.asarray(meta["encoding_max_joint_step"]) + 1e-7)
    assert np.all(runtime >= np.asarray(meta["encoding_lower_limits"]))
    assert np.all(runtime <= np.asarray(meta["encoding_upper_limits"]))
    physical.sum().backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
