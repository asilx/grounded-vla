"""Supervised temporal-ViT training on episode-disjoint, labelled RGB recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from grounded_vla.learning.temporal_vit import (
    CLASSES,
    TemporalViTConfig,
    build_temporal_vit,
    preprocess_rgb,
    save_critic,
)
from grounded_vla.observation import CAMERAS


class CriticDataset(Dataset):
    """Each window ends at its labelled frame; windows never cross episodes/splits."""

    def __init__(self, entries, *, camera, window_size):
        self.entries, self.camera, self.window_size = entries, camera, window_size
        self.windows = [
            (i, end) for i, e in enumerate(entries) for end in range(window_size - 1, e["frames"])
        ]
        if not self.windows:
            raise ValueError("No complete temporal windows in this split")
        self._read = lru_cache(maxsize=1)(self._load)

    def _load(self, index):
        with np.load(self.entries[index]["path"], allow_pickle=False) as archive:
            return archive[f"images/{self.camera}"], archive["critic_labels"]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, end = self.windows[index]
        images, labels = self._read(episode)
        return preprocess_rgb(images[end + 1 - self.window_size : end + 1]), int(labels[end])


def load_critic_datasets(manifest, window_size):
    path = Path(manifest).resolve()
    meta = json.loads(path.read_text())
    if meta.get("format_version") != 1 or meta.get("camera") not in CAMERAS:
        raise ValueError("Expected critic format_version=1 and a canonical camera")
    period = meta["control_period_seconds"]
    if not math.isfinite(period) or period <= 0:
        raise ValueError("Invalid critic frame period")
    if type(window_size) is not int or window_size < 2:
        raise ValueError("window_size must be at least two")
    paths, hashes = set(), set()
    splits = {"train": [], "val": [], "test": []}
    train_labels = set()
    for entry in meta["episodes"]:
        source = (path.parent / entry["path"]).resolve()
        if entry["split"] not in splits or source in paths:
            raise ValueError("Unknown split or duplicate episode path")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in hashes:
            raise ValueError("Duplicate episode content across splits")
        paths.add(source)
        hashes.add(digest)
        with np.load(source, allow_pickle=False) as archive:
            images = archive[f"images/{meta['camera']}"]
            times, labels = archive["timestamps"], archive["critic_labels"]
            n = len(images)
            if images.shape != (n, 224, 224, 3) or images.dtype != np.uint8 or n < window_size:
                raise ValueError("Each episode needs a full window of uint8 224x224 RGB images")
            if (
                times.shape != (n,)
                or not np.isfinite(times).all()
                or np.any(times < 0)
                or not np.allclose(np.diff(times), period, rtol=0.05, atol=1e-6)
            ):
                raise ValueError("Invalid episode timestamps or sampling period")
            if (
                labels.shape != (n,)
                or labels.dtype.kind not in "iu"
                or np.any(labels < 0)
                or np.any(labels >= len(CLASSES))
            ):
                raise ValueError("critic_labels must be integer [T] in class order 0..3")
            mask_key = f"image_masks/{meta['camera']}"
            if mask_key in archive:
                mask = archive[mask_key]
                if mask.shape != (n,) or mask.dtype != np.bool_ or not mask.all():
                    raise ValueError("Critic training requires an available camera in every frame")
            if entry["split"] == "train":
                train_labels.update(labels[window_size - 1 :].tolist())
        splits[entry["split"]].append({"path": str(source), "frames": n, "sha256": digest})
    if train_labels != set(range(len(CLASSES))):
        raise ValueError("Training windows must cover progressing, stalled, failed, and completed")
    if not splits["train"] or not splits["val"]:
        raise ValueError("Separate train and validation episodes are required")
    datasets = {
        split: CriticDataset(entries, camera=meta["camera"], window_size=window_size)
        for split, entries in splits.items()
        if entries
    }
    provenance = {
        "camera": meta["camera"],
        "control_period_seconds": period,
        "episodes": [{"split": split, **e} for split, entries in splits.items() for e in entries],
    }
    return datasets, provenance


def fit_critic(model, train_loader, val_loader, *, device="cpu", epochs=5, learning_rate=1e-4):
    if (
        type(epochs) is not int
        or epochs <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Positive epochs and learning_rate are required")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    history = []
    for epoch in range(epochs):
        row = {"epoch": epoch + 1}
        for split, loader in (("train", train_loader), ("val", val_loader)):
            model.train(split == "train")
            count, correct, total_loss = 0, 0, 0.0
            with torch.set_grad_enabled(split == "train"):
                for frames, labels in loader:
                    frames, labels = frames.to(device), labels.to(device)
                    logits = model(frames)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite critic loss")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0, error_if_nonfinite=True
                        )
                        optimizer.step()
                    count += len(labels)
                    correct += int((logits.argmax(-1) == labels).sum().item())
                    total_loss += loss.item() * len(labels)
            if count == 0:
                raise ValueError(f"Empty {split} loader")
            row[f"{split}_loss"], row[f"{split}_accuracy"] = total_loss / count, correct / count
        history.append(row)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--random-init", action="store_true", help="Skip ImageNet weights (experimental)"
    )
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    if args.batch_size <= 0 or args.seed < 0:
        raise ValueError("Invalid batch size or seed")
    torch.manual_seed(args.seed)
    datasets, provenance = load_critic_datasets(args.manifest, args.window_size)
    train = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    model = build_temporal_vit(
        TemporalViTConfig(max_frames=args.window_size),
        imagenet=not args.random_init,
    )
    history = fit_critic(
        model,
        train,
        val,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    save_critic(
        args.out,
        model,
        trained=True,
        training_metadata={**provenance, "seed": args.seed, "history": history},
    )
    print(json.dumps({"checkpoint": args.out, "history": history}, indent=2))


if __name__ == "__main__":
    main()
