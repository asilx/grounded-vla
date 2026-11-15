"""Causal, numeric task context shared by critic recording and online inference.

Object IDs bind the active contract to a graph node, but are never embedded.
The schema describes the existing inspect/clean/pick/place contracts only.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from grounded_vla.belief import Truth

CONTEXT_SCHEMA_ID = "grounded-vla-critic-contract-v1"
SKILLS = ("inspect", "clean", "pick", "place", "other")
PREDICATES = ("clean", "filled", "held", "at_tray")
TRUTHS = (Truth.SUPPORTED, Truth.REFUTED, Truth.UNKNOWN)
CONTRACT_FEATURES = (
    tuple(f"skill/{s}" for s in SKILLS)
    + tuple(
        f"{predicate}/{kind}"
        for predicate in PREDICATES
        for kind in (
            "pre/supported",
            "pre/refuted",
            "pre/unknown",
            "effect/supported",
            "effect/refuted",
            "effect/unknown",
            "observe",
        )
    )
    + ("tilt/enabled", "tilt/maximum_normalized")
)
CONTRACT_DIM = len(CONTRACT_FEATURES)
CONTEXT_ARRAYS = ("contract", "state", "features", "edges", "valid", "target_mask")


def encode_critic_context(contract, observation, snapshot):
    if contract.skill not in SKILLS:
        raise ValueError("Unknown critic skill; extend/version the contract schema explicitly")
    values = np.zeros(CONTRACT_DIM, np.float32)
    values[SKILLS.index(contract.skill)] = 1
    for kind, requirements in enumerate((contract.preconditions, contract.effects)):
        for requirement in requirements:
            atom = requirement.atom
            if atom.subject != contract.object_id or atom.predicate not in PREDICATES:
                raise ValueError(
                    "Critic context supports canonical predicates on the contract object"
                )
            index = len(SKILLS) + 7 * PREDICATES.index(atom.predicate) + 3 * kind
            values[index + TRUTHS.index(requirement.truth)] = 1
    for atom in contract.observes:
        if atom.subject != contract.object_id or atom.predicate not in PREDICATES:
            raise ValueError(
                "Critic observations must use canonical predicates on the contract object"
            )
        values[len(SKILLS) + 7 * PREDICATES.index(atom.predicate) + 6] = 1
    if contract.maximum_tilt_degrees is not None:
        tilt = contract.maximum_tilt_degrees
        if not math.isfinite(tilt) or not 0 <= tilt <= 90:
            raise ValueError("Invalid contract tilt")
        values[-2:] = (1, tilt / 90)
    graph = snapshot.graph
    ids = tuple(snapshot.entity_ids)
    if len(ids) != len(graph["valid"]) or len(set(ids)) != len(ids):
        raise ValueError("Conditioned critic needs unique entity_ids aligned with graph nodes")
    if contract.object_id not in ids:
        raise ValueError("Contract object is missing from the critic graph")
    target = np.asarray([entity == contract.object_id for entity in ids], dtype=bool)
    # Only semantic contract changes reset a temporal window, not refreshed evidence IDs.
    signature = {
        "object": contract.object_id,
        "features": values.tolist(),
        "rules": sorted(contract.rule_ids),
    }
    result = {
        "schema_id": CONTEXT_SCHEMA_ID,
        "graph_schema_id": graph["schema_id"],
        "contract_id": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
        "timestamp": float(observation.timestamp),
        "graph_timestamp": float(snapshot.timestamp),
        "contract": values,
        "state": np.asarray(observation.state, np.float32).copy(),
        "features": np.asarray(graph["features"], np.float32).copy(),
        "edges": np.asarray(graph["edges"]).copy(),
        "valid": np.asarray(graph["valid"]).copy(),
        "target_mask": target,
    }
    validate_critic_context(result, observation.timestamp)
    return result


def validate_critic_context(context, timestamp, *, max_graph_age_seconds=0.5):
    if context.get("schema_id") != CONTEXT_SCHEMA_ID:
        raise ValueError("Unknown critic context schema")
    if not isinstance(context.get("contract_id"), str) or not context["contract_id"]:
        raise ValueError("Missing critic contract identity")
    now, stamp = context.get("timestamp", math.nan), context.get("graph_timestamp", math.nan)
    if not all(math.isfinite(t) and t >= 0 for t in (now, stamp, timestamp)):
        raise ValueError("Invalid critic context timestamps")
    if not math.isclose(now, timestamp, abs_tol=1e-9, rel_tol=0) or stamp > now:
        raise ValueError("Critic context must be causal and match the current observation")
    if now - stamp > max_graph_age_seconds:
        raise ValueError("Stale critic graph")
    f, e, v, target = (
        np.asarray(context[k]) for k in ("features", "edges", "valid", "target_mask")
    )
    n = len(v)
    if f.ndim != 2 or f.shape[0] != n or e.shape != (n, n) or v.shape != (n,):
        raise ValueError("Invalid critic graph dimensions")
    if e.dtype != np.int64 or v.dtype != np.bool_ or target.dtype != np.bool_:
        raise ValueError("Invalid critic graph dtypes")
    if target.shape != (n,) or int(target.sum()) != 1 or np.any(target & ~v):
        raise ValueError("Critic graph needs exactly one valid contract target")
    if np.asarray(context["contract"]).shape != (CONTRACT_DIM,):
        raise ValueError("Invalid canonical contract feature dimensions")
    if np.asarray(context["state"]).ndim != 1:
        raise ValueError("Critic state must be a vector")
    if any(not np.isfinite(context[k]).all() for k in ("contract", "state", "features")):
        raise ValueError("Non-finite critic context")
    return context
