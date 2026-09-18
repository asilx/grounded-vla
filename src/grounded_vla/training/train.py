"""Single-device adapter fine-tuning through the official π0 flow-matching loss."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from grounded_vla.training.checkpoint import (
    restore_rng,
    resume_checkpoint,
    rng_state,
    save_checkpoint,
)
from grounded_vla.training.config import OPENPI_REVISION, TrainConfig
from grounded_vla.training.data import EpisodeDataset, batch_indices, collate, load_prepared
from grounded_vla.training.graph import GraphBatch
from grounded_vla.training.runtime import sha256, verify_runtime
from grounded_vla.training.tokenizer import Pi0Tokenizer


def build_model(config):
    from openpi.models.pi0_config import Pi0Config

    from grounded_vla.training.pi0 import GroundedPi0

    native = Pi0Config(
        dtype=config.precision,
        action_dim=32,
        action_horizon=config.action_horizon,
        max_token_len=48,
        pi05=False,
        pytorch_compile_mode=None,
    )
    return GroundedPi0(native, config.adapter)


def device_batch(batch, device):
    return tuple(x.to(device) for x in batch)


@torch.no_grad()
def evaluate(model, dataset, config):
    saved = rng_state()
    was_training = model.training
    model.eval()
    totals = [0.0, 0.0]
    count = 0
    try:
        torch.manual_seed(config.seed + 100000)
        for start in range(
            0, min(len(dataset), config.eval_batches * config.batch_size), config.batch_size
        ):
            obs, graph, actions, mask = device_batch(
                collate(
                    [dataset[i] for i in range(start, min(start + config.batch_size, len(dataset)))]
                ),
                config.device,
            )
            noise = model.sample_noise(actions.shape, actions.device)
            time = model.sample_time(len(actions), actions.device)
            weight = mask.sum().item()
            for i, variant in enumerate((graph, graph.disabled())):
                loss = model.flow_loss(obs, actions, variant, mask, noise=noise, time=time)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite validation loss")
                totals[i] += loss.item() * weight
            count += weight
        return {
            "val_graph_loss": totals[0] / count,
            "val_no_graph_loss": totals[1] / count,
            "val_valid_elements": count,
        }
    finally:
        model.train(was_training)
        restore_rng(saved)


def learning_rate(config, step):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def run(config: TrainConfig, *, resume=None, stop_after=None):
    verify_runtime()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This trainer is single-device; do not launch with torchrun")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; CPU validation requires device=cpu, precision=float32"
        )
    if (
        config.device.startswith("cuda")
        and config.precision == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("This GPU does not support bfloat16; select float32")
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Output is not empty; use --resume or a new output directory")
    if resume and output.exists():
        newer = [p for p in output.glob("step-*") if p.name > Path(resume).name]
        if newer:
            raise ValueError(
                "Cannot resume an older checkpoint into a directory containing newer steps"
            )
    final_step = config.steps if stop_after is None else min(config.steps, stop_after)
    if final_step <= 0:
        raise ValueError("stop-after must be positive")
    meta = load_prepared(config.dataset)
    if (
        len(meta["graph_schema"]["features"]) != config.adapter.feature_dim
        or len(meta["graph_schema"]["relations"]) != config.adapter.relation_types
    ):
        raise ValueError("Dataset graph schema and adapter configuration disagree")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    tokenizer = Pi0Tokenizer(config.tokenizer)
    train = EpisodeDataset(config.dataset, "train", config.action_horizon, tokenizer, metadata=meta)
    val = EpisodeDataset(config.dataset, "val", config.action_horizon, tokenizer, metadata=meta)
    metadata = {
        "format_version": 1,
        "openpi_revision": OPENPI_REVISION,
        "training_config": config.to_dict(),
        "dataset": {k: v for k, v in meta.items() if k != "episodes"},
        "identity": {
            "base_checkpoint": sha256(config.base_checkpoint),
            "tokenizer": sha256(config.tokenizer),
            "prepared_dataset": sha256(config.dataset),
        },
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
    }
    model = build_model(config)
    model.load_base(config.base_checkpoint)
    model.to(config.device)
    model.train()
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if {id(p) for p in parameters} != {id(p) for p in model.graph_adapter.parameters()}:
        raise RuntimeError("Only graph adapter parameters may be trainable")
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start, cursor = 0, 0
    if resume:
        start, cursor = resume_checkpoint(resume, model, optimizer, metadata)
    if start >= final_step:
        raise ValueError("Checkpoint already reached the requested stop step")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    if resume and metrics_path.exists():
        committed = [
            json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()
        ]
        metrics_path.write_text(
            "".join(json.dumps(row) + "\n" for row in committed if row["step"] <= start)
        )
    latest = None
    for step in range(start, final_step):
        batches = []
        for _ in range(config.accumulation_steps):
            indices, cursor = batch_indices(len(train), config.batch_size, cursor, config.seed)
            batches.append(collate([train[i] for i in indices]))
        total_weight = sum(batch[3].sum().item() for batch in batches)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_loss = 0.0
        for batch in batches:
            obs, graph, actions, mask = device_batch(batch, config.device)
            keep = torch.rand(len(actions), device=actions.device) >= config.graph_dropout
            graph = GraphBatch(graph.features, graph.edges, graph.valid & keep[:, None])
            loss = model.flow_loss(obs, actions, graph, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            weight = mask.sum().item() / total_weight
            (loss * weight).backward()
            total_loss += loss.item() * weight
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, config.gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        record = {
            "step": step + 1,
            "train_loss": total_loss,
            "learning_rate": lr,
            "gradient_norm": float(norm),
            "gate": float(model.graph_adapter.gate.detach().tanh()),
        }
        if (step + 1) % config.eval_every == 0 or step + 1 == final_step:
            record.update(evaluate(model, val, config))
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if (step + 1) % config.save_every == 0 or step + 1 == final_step:
            latest = save_checkpoint(output, model, optimizer, step + 1, cursor, metadata)
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--resume")
    parser.add_argument(
        "--stop-after", type=int, help="Stop early without changing the planned LR schedule"
    )
    args = parser.parse_args()
    print(run(TrainConfig.load(args.config), resume=args.resume, stop_after=args.stop_after))


if __name__ == "__main__":
    main()
