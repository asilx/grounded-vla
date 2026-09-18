"""Explicit, serializable configuration for frozen-base π0 adapter training."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"


@dataclass
class AdapterConfig:
    feature_dim: int = 26
    heads: int = 8
    context_tokens: int = 8
    relation_types: int = 8
    initial_gate: float = 0.001

    def __post_init__(self):
        if not math.isfinite(self.initial_gate):
            raise ValueError("Gate must be finite")
        if min(self.feature_dim, self.heads, self.context_tokens, self.relation_types) <= 0:
            raise ValueError("Adapter dimensions must be positive")


@dataclass
class TrainConfig:
    dataset: str
    base_checkpoint: str
    tokenizer: str
    output_dir: str
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    action_horizon: int = 50
    precision: str = "bfloat16"
    device: str = "cuda"
    batch_size: int = 2
    accumulation_steps: int = 4
    steps: int = 10000
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    graph_dropout: float = 0.1
    eval_every: int = 250
    eval_batches: int = 20
    save_every: int = 500
    seed: int = 7
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if isinstance(self.adapter, dict):
            self.adapter = AdapterConfig(**self.adapter)
        if (
            min(
                self.action_horizon,
                self.batch_size,
                self.accumulation_steps,
                self.steps,
                self.eval_every,
                self.eval_batches,
                self.save_every,
            )
            <= 0
        ):
            raise ValueError("Training sizes and intervals must be positive")
        if not all(
            math.isfinite(v)
            for v in (self.learning_rate, self.weight_decay, self.gradient_clip, self.graph_dropout)
        ):
            raise ValueError("Optimization settings must be finite")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if self.device == "cpu" and self.precision != "float32":
            raise ValueError("CPU requires float32")
        if not 0 <= self.graph_dropout < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid graph dropout or learning rate")
        if self.gradient_clip <= 0 or self.weight_decay < 0 or self.warmup_steps < 0:
            raise ValueError("Invalid optimization settings")

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path).resolve()
        raw = json.loads(path.read_text())
        for key in ("dataset", "base_checkpoint", "tokenizer", "output_dir"):
            raw[key] = str((path.parent / raw[key]).resolve())
        return cls(**raw)

    def to_dict(self):
        return asdict(self)
