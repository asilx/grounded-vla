"""Evidence-aware graph features. Stable object IDs are not learned embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from grounded_vla.belief import Atom, BeliefStore, Truth

PREDICATES = ("clean", "filled", "held", "at_tray")
FEATURES = ("x", "y", "z", "visible", "is_target", "is_container") + tuple(
    f"{p}/{v}" for p in PREDICATES for v in ("supported", "refuted", "unknown", "confidence", "age")
)
RELATIONS = ("none", "self", "on", "in", "near", "held_by", "target_of", "supports")
SCHEMA_ID = "grounded-vla-belief-v1"


@dataclass
class GraphBatch:
    features: torch.Tensor
    edges: torch.Tensor
    valid: torch.Tensor

    def validate(self, feature_dim: int, relation_types: int):
        if self.features.ndim != 3:
            raise ValueError("features must be [B,N,F]")
        b, n, f = self.features.shape
        if f != feature_dim or self.edges.shape != (b, n, n) or self.valid.shape != (b, n):
            raise ValueError("Graph dimensions disagree")
        if self.edges.dtype != torch.int64 or self.valid.dtype != torch.bool:
            raise ValueError("Graph edges must be int64 and validity boolean")
        if not torch.isfinite(self.features).all():
            raise ValueError("Non-finite graph features")
        if self.edges.numel() and (self.edges.min() < 0 or self.edges.max() >= relation_types):
            raise ValueError("Relation index outside schema")
        return self

    def to(self, device):
        return GraphBatch(self.features.to(device), self.edges.to(device), self.valid.to(device))

    def disabled(self):
        return GraphBatch(self.features, self.edges, torch.zeros_like(self.valid))


def encode_belief_graph(
    store: BeliefStore,
    objects: list[dict],
    relations: list[tuple],
    *,
    now: float,
    max_age: float = 60.0,
) -> dict:
    """objects: id, position[3], visible, is_target, is_container. Relations are directed."""
    if max_age <= 0:
        raise ValueError("max_age must be positive")
    ids = [o["id"] for o in objects]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate object IDs")
    features = np.zeros((len(objects), len(FEATURES)), np.float32)
    edges = np.zeros((len(objects), len(objects)), np.int64)
    for i, obj in enumerate(objects):
        features[i, :6] = [
            *obj["position"],
            obj.get("visible", False),
            obj.get("is_target", False),
            obj.get("is_container", False),
        ]
        edges[i, i] = 1
        for j, predicate in enumerate(PREDICATES):
            result = store.query(Atom(predicate, obj["id"]), now)
            truth = result.truth
            evidence = [store.archive[k] for k in result.evidence_ids]
            confidence = max((e.confidence for e in evidence), default=0.0)
            age = min((now - e.observed_at for e in evidence), default=max_age) / max_age
            if truth == Truth.UNKNOWN:
                confidence, age = 0.0, 1.0
            features[i, 6 + 5 * j : 11 + 5 * j] = [
                truth == Truth.SUPPORTED,
                truth == Truth.REFUTED,
                truth == Truth.UNKNOWN,
                confidence,
                min(age, 1.0),
            ]
    for source, relation, target in relations:
        edges[ids.index(source), ids.index(target)] = RELATIONS.index(relation)
    if not np.isfinite(features).all():
        raise ValueError("Object positions must be finite")
    return {
        "features": features,
        "edges": edges,
        "valid": np.ones(len(objects), bool),
        "schema_id": SCHEMA_ID,
    }
