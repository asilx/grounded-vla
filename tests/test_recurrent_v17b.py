import importlib.util
import sys
from pathlib import Path

import torch

experiment_dir = Path(__file__).resolve().parents[1] / "experiments/runpod"
sys.path.insert(0, str(experiment_dir))
spec = importlib.util.spec_from_file_location(
    "learned_recurrent_v17b", experiment_dir / "learned_recurrent_v17b.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_causal_gru_streaming_and_episode_reset_match_full_sequence():
    torch.manual_seed(9)
    torch.set_num_threads(2)
    actor = module.Actor(189).eval()
    x = torch.randn(2, 19, 189)
    with torch.no_grad():
        expected, _, expected_hidden = actor(x)
        hidden = None
        stream = []
        for i in range(x.shape[1]):
            y, _, hidden = actor(x[:, i : i + 1], hidden)
            stream.append(y)
        torch.testing.assert_close(torch.cat(stream, 1), expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(hidden, expected_hidden, rtol=1e-5, atol=1e-6)
        future = x.clone()
        future[:, 10:] += 100.0
        altered, _, _ = actor(future)
        torch.testing.assert_close(altered[:, :10], expected[:, :10], rtol=1e-5, atol=1e-6)
        fresh, _, _ = actor(x[:, :1], None)
        torch.testing.assert_close(fresh, expected[:, :1], rtol=1e-5, atol=1e-6)


def test_padding_has_no_loss_or_gradient_influence():
    values = torch.randn(2, 5, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)
    changed = values.detach().clone()
    changed[mask == 0] = 100000.0
    torch.testing.assert_close(module.masked_mean(values, mask), module.masked_mean(changed, mask))
    module.masked_mean(values, mask).backward()
    assert torch.equal(values.grad[mask == 0], torch.zeros_like(values.grad[mask == 0]))
