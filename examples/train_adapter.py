"""Train the optional adapter on synthetic regression, with a held-out split.

Demonstrates optimization and checkpointing only; does not fine-tune π0.5.
"""

import argparse
import json
from pathlib import Path

import torch

from grounded_vla.learning.adapter import GatedKnowledgeAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path("runs/adapter"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    torch.manual_seed(7)
    torch.set_num_threads(1)
    features = torch.randn(320, 4, 6)
    edges = torch.randint(0, 3, (320, 4, 4))
    valid = torch.ones(320, 4, dtype=torch.bool)
    hidden = torch.randn(320, 3, 16) * 0.2
    # A known synthetic target, separate from all robot/action performance claims.
    signal = features[:, :, 0].mean(1) * 0.5
    target = hidden + signal[:, None, None] * torch.linspace(-1, 1, 16)
    adapter = GatedKnowledgeAdapter(6, 16, 4, 4, 3)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.01)
    losses = []

    def heldout_loss():
        adapter.eval()
        with torch.no_grad():
            predicted = adapter(hidden[256:], features[256:], edges[256:], valid[256:])
            return (predicted - target[256:]).square().mean().item()

    before = heldout_loss()
    for step in range(args.steps):
        adapter.train()
        batch = torch.randint(0, 256, (32,))
        predicted = adapter(hidden[batch], features[batch], edges[batch], valid[batch])
        loss = (predicted - target[batch]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append({"step": step, "train_mse": loss.item()})
    after = heldout_loss()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "model_config": {
                "feature_dim": 6,
                "hidden_dim": 16,
                "heads": 4,
                "context_tokens": 4,
                "relation_types": 3,
            },
        },
        args.out / "adapter.pt",
    )
    result = {
        "scope": "Synthetic regression only; no robot data or pi0.5 weights",
        "seed": 7,
        "torch": torch.__version__,
        "steps": args.steps,
        "heldout_mse_before": before,
        "heldout_mse_after": after,
        "training": losses,
    }
    (args.out / "training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Synthetic held-out MSE: {before:.6f} -> {after:.6f}")
    print(f"Checkpoint and training log: {args.out}")


if __name__ == "__main__":
    main()
