"""Spatial ViT features + causal temporal attention for execution monitoring.

ImageNet initialization is not a robot critic. A trained temporal checkpoint
is required for online use; predictions remain fallible monitoring signals.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

CLASSES = ("progressing", "stalled", "failed", "completed")


@dataclass(frozen=True)
class TemporalViTConfig:
    feature_dim: int = 768
    temporal_dim: int = 128
    heads: int = 4
    layers: int = 2
    max_frames: int = 8
    dropout: float = 0.1
    freeze_backbone: bool = True

    def __post_init__(self):
        for key in ("feature_dim", "temporal_dim", "heads", "layers", "max_frames"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.temporal_dim % self.heads:
            raise ValueError("temporal_dim must be divisible by heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


class TemporalViTCritic(nn.Module):
    def __init__(self, backbone: nn.Module, config: TemporalViTConfig):
        super().__init__()
        self.config, self.backbone = config, backbone
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.projection = nn.Linear(config.feature_dim, config.temporal_dim)
        self.position = nn.Parameter(torch.zeros(1, config.max_frames, config.temporal_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            config.temporal_dim,
            config.heads,
            dim_feedforward=config.temporal_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(config.temporal_dim), nn.Linear(config.temporal_dim, 4)
        )

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, frames, valid=None):
        """Normalized float [B,T,3,H,W], right-padded boolean [B,T] -> [B,4]."""
        if frames.ndim != 5 or frames.shape[2] != 3 or not frames.is_floating_point():
            raise ValueError("frames must be normalized floating-point [B,T,3,H,W]")
        b, t = frames.shape[:2]
        if b < 1 or not 1 <= t <= self.config.max_frames:
            raise ValueError("Empty batch or window exceeds configured max_frames")
        if valid is None:
            valid = torch.ones((b, t), dtype=torch.bool, device=frames.device)
        if valid.shape != (b, t) or valid.dtype != torch.bool or valid.device != frames.device:
            raise ValueError("valid must be boolean [B,T] on the frames device")
        if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Every window needs a nonempty valid prefix and right padding")
        selected = frames[valid]
        if not torch.isfinite(selected).all():
            raise ValueError("Non-finite valid images")
        if self.config.freeze_backbone:
            with torch.no_grad():
                encoded = self.backbone(selected)
        else:
            encoded = self.backbone(selected)
        if encoded.shape != (len(selected), self.config.feature_dim):
            raise ValueError("Backbone must return one feature vector per image")
        if not torch.isfinite(encoded).all():
            raise FloatingPointError("Non-finite ViT features")
        features = encoded.new_zeros(b, t, self.config.feature_dim)
        features[valid] = encoded
        tokens = self.projection(features) + self.position[:, :t]
        causal = torch.ones((t, t), dtype=torch.bool, device=frames.device).triu(1)
        hidden = self.temporal(tokens, mask=causal, src_key_padding_mask=~valid)
        last = valid.sum(dim=1) - 1
        return self.head(hidden[torch.arange(b, device=frames.device), last])


def build_temporal_vit(config=None, *, imagenet=False):
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    config = config or TemporalViTConfig()
    if config.feature_dim != 768:
        raise ValueError("torchvision ViT-B/16 requires feature_dim=768")
    backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if imagenet else None)
    backbone.heads = nn.Identity()
    return TemporalViTCritic(backbone, config)


def preprocess_rgb(frames):
    """uint8 THWC RGB -> ImageNet ViT-B/16 normalized TCHW; no weight download."""
    from torchvision.models import ViT_B_16_Weights

    values = np.asarray(frames)
    if values.dtype != np.uint8 or values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("Expected a sequence of uint8 RGB HWC frames")
    if min(values.shape[:3]) < 1:
        raise ValueError("RGB sequence and image dimensions must be nonempty")
    tensor = torch.from_numpy(values.copy()).permute(0, 3, 1, 2)
    return ViT_B_16_Weights.IMAGENET1K_V1.transforms()(tensor)


def save_critic(path, model, *, trained, training_metadata=None):
    """Store all weights for offline restoration, including the spatial backbone."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "architecture": "torchvision_vit_b_16_temporal",
        "config": asdict(model.config),
        "classes": list(CLASSES),
        "preprocessing": "ViT_B_16_Weights.IMAGENET1K_V1",
        "trained": bool(trained),
        "training_metadata": training_metadata or {},
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)


class ViTWindowPredictor:
    def __init__(self, checkpoint, *, device="cpu", expected_camera=None, expected_period=None):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("format_version") != 1
            or payload.get("architecture") != "torchvision_vit_b_16_temporal"
            or payload.get("classes") != list(CLASSES)
            or payload.get("preprocessing") != "ViT_B_16_Weights.IMAGENET1K_V1"
            or payload.get("trained") is not True
        ):
            raise ValueError("Online monitoring requires a compatible trained critic checkpoint")
        meta = payload.get("training_metadata", {})
        if expected_camera is not None and meta.get("camera") != expected_camera:
            raise ValueError("Critic camera differs from training")
        if expected_period is not None and not math.isclose(
            float(meta.get("control_period_seconds", math.nan)),
            expected_period,
            rel_tol=1e-6,
        ):
            raise ValueError("Critic frame period differs from training")
        self.config = TemporalViTConfig(**payload["config"])
        self.model = build_temporal_vit(self.config)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.device = device
        self.model.to(device).eval()

    @torch.inference_mode()
    def __call__(self, frames):
        inputs = preprocess_rgb(frames)[None].to(self.device)
        logits = self.model(inputs)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite critic output")
        return logits.softmax(-1)[0].cpu().numpy()


class TemporalMonitor:
    """Full-window warm-up, consecutive detections, and a latched pause event."""

    def __init__(self, predictor, *, window_size=8, threshold=0.8, patience=2, period=0.05):
        if type(window_size) is not int or window_size < 2:
            raise ValueError("window_size must be at least two frames")
        if type(patience) is not int or patience < 1:
            raise ValueError("patience must be positive")
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0,1]")
        if not math.isfinite(period) or period <= 0:
            raise ValueError("period must be finite and positive")
        self.predictor, self.threshold, self.patience = predictor, threshold, patience
        self.period, self.window_size = period, window_size
        self.frames = deque(maxlen=window_size)
        self.reset()

    def reset(self):
        self.frames.clear()
        self._last_time = None
        self._hits = 0
        self._latched = False

    def update(self, image, timestamp):
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid monitor timestamp")
        if self._last_time is not None and not math.isclose(
            timestamp - self._last_time,
            self.period,
            rel_tol=0.05,
            abs_tol=1e-6,
        ):
            raise ValueError("Monitor frames must be ordered at the trained control period")
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.shape != (224, 224, 3):
            raise ValueError("Monitor needs uint8 224x224 RGB frames")
        self._last_time = timestamp
        self.frames.append(image.copy())
        if self._latched or len(self.frames) < self.window_size:
            return None
        probabilities = np.asarray(self.predictor(list(self.frames)), dtype=float)
        if (
            probabilities.shape != (4,)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError("Critic must return a finite distribution over the four classes")
        problem = float(probabilities[1] + probabilities[2])
        self._hits = self._hits + 1 if problem >= self.threshold else 0
        if self._hits < self.patience:
            return None
        self._latched = True
        return {
            "kind": "critic_pause",
            "timestamp": float(timestamp),
            "reason": CLASSES[1 + int(probabilities[2] > probabilities[1])],
            "probabilities": dict(zip(CLASSES, probabilities.tolist(), strict=True)),
            "consecutive_windows": self._hits,
        }
