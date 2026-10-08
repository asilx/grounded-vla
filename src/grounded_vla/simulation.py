"""Closed-loop canonical openpi execution, causal recording, and critic events.

This physics runner is separate from the symbolic event-emulator showcase.
No graph facts, task success, or recovery actions are inferred from commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from grounded_vla.backends.openpi import PolicyBackendError


@dataclass(frozen=True)
class GraphSnapshot:
    graph: dict
    timestamp: float


class CanonicalSimPolicy:
    """Require a server trained for this exact joint order and absolute convention."""

    def __init__(self, client, config, *, max_horizon=100, max_graph_age_seconds=0.5):
        meta = getattr(client, "metadata", {})
        expected = {
            "input_preset": "canonical",
            "state_dim": len(config.joint_names),
            "action_dim": len(config.joint_names),
            "action_convention": config.action_convention,
            "graph_required": True,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"Policy metadata mismatch for {key}: expected {value!r}")
        period = meta.get("control_period_seconds")
        if (
            not isinstance(period, (float, int))
            or not math.isfinite(period)
            or not math.isclose(period, config.control_period_seconds, rel_tol=1e-6, abs_tol=1e-9)
        ):
            raise ValueError("Policy and simulation control periods differ")
        for key in ("graph_feature_dim", "graph_relation_types", "action_horizon"):
            if type(meta.get(key)) is not int or meta[key] <= 0:
                raise ValueError(f"Missing or invalid server metadata: {key}")
        if not isinstance(meta.get("graph_schema_id"), str) or not meta["graph_schema_id"]:
            raise ValueError("Missing server graph schema ID")
        if type(max_horizon) is not int or max_horizon <= 0:
            raise ValueError("max_horizon must be a positive integer")
        if not math.isfinite(max_graph_age_seconds) or max_graph_age_seconds <= 0:
            raise ValueError("Graph age limit must be finite and positive")
        self.client, self.meta = client, dict(meta)
        self.max_horizon = min(max_horizon, meta["action_horizon"])
        self.max_graph_age_seconds = max_graph_age_seconds

    def validate_graph(self, snapshot, now):
        if (
            not math.isfinite(snapshot.timestamp)
            or snapshot.timestamp < 0
            or snapshot.timestamp > now
            or now - snapshot.timestamp > self.max_graph_age_seconds
        ):
            raise ValueError("Graph timestamp is future, stale, or invalid")
        graph = snapshot.graph
        if graph.get("schema_id") != self.meta["graph_schema_id"]:
            raise ValueError("Graph schema differs from trained policy")
        f = np.asarray(graph["features"], dtype=np.float32)
        e, v = np.asarray(graph["edges"]), np.asarray(graph["valid"])
        if f.ndim != 2 or f.shape[1] != self.meta["graph_feature_dim"]:
            raise ValueError("Invalid graph feature dimensions")
        n = len(f)
        if e.shape != (n, n) or e.dtype != np.int64 or v.shape != (n,) or v.dtype != np.bool_:
            raise ValueError("Invalid graph edges or validity")
        if not np.isfinite(f).all() or (
            e.size and (e.min() < 0 or e.max() >= self.meta["graph_relation_types"])
        ):
            raise ValueError("Invalid graph values")
        return {
            "schema_id": graph["schema_id"],
            "features": f.copy(),
            "edges": e.copy(),
            "valid": v.copy(),
        }

    def infer(self, observation, snapshot, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide a nonempty task instruction")
        graph = self.validate_graph(snapshot, observation.timestamp)
        payload = {
            "state": observation.state.copy(),
            "images": {k: v.copy() for k, v in observation.images.items()},
            "prompt": prompt,
            "graph": graph,
        }
        response = self.client.infer(payload)
        actions = np.asarray(response["actions"], dtype=np.float32)
        if (
            actions.ndim != 2
            or actions.shape[1] != self.meta["action_dim"]
            or not 1 <= len(actions) <= self.max_horizon
            or not np.isfinite(actions).all()
        ):
            raise PolicyBackendError("Invalid canonical policy action chunk")
        return actions.copy()


@dataclass
class RolloutResult:
    steps: int = 0
    reason: str = "budget_exhausted"
    events: list[dict] = field(default_factory=list)


def run_rollout(
    environment,
    policy,
    graph_provider,
    prompt,
    *,
    max_steps=100,
    execute_steps=1,
    monitor=None,
    monitor_camera="base_0_rgb",
    recorder=None,
    on_event=None,
):
    """Execute a bounded chunk prefix, reobserve, and pause before event callbacks.

    ``graph_provider(observation)`` returns GraphSnapshot in simulation time.
    ``on_event(event, observation)`` may ask a VLM for a recovery proposal; the
    simulator is already paused and this function never executes that proposal.
    The recorder contains successfully submitted/stepped actions, not expert labels.
    """
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if type(execute_steps) is not int or not 1 <= execute_steps <= policy.max_horizon:
        raise ValueError("execute_steps must be within the policy horizon")
    result = RolloutResult()

    def inspect(obs):
        if monitor is None:
            return False
        event = monitor.update(obs.images[monitor_camera], obs.timestamp)
        if event is None:
            return False
        environment.pause()
        result.reason = "critic_pause"
        result.events.append(event)
        if on_event is not None:
            on_event(event, obs)
        return True

    try:
        if monitor is not None:
            monitor.reset()
        observation = environment.observe()
        if inspect(observation):
            return result
        snapshot = graph_provider(observation)
        while result.steps < max_steps:
            actions = policy.infer(observation, snapshot, prompt)
            for action in actions[: min(execute_steps, max_steps - result.steps)]:
                # Refresh evidence for each executed step, even inside a chunk prefix.
                graph = policy.validate_graph(snapshot, observation.timestamp)
                previous = observation
                observation = environment.step(action)
                result.steps += 1
                if recorder is not None:
                    recorder.append(
                        state=previous.state,
                        images=previous.images,
                        action=action,
                        prompt=prompt,
                        graph=graph,
                        timestamp=previous.timestamp,
                        graph_timestamp=snapshot.timestamp,
                    )
                if inspect(observation):
                    return result
                if result.steps < max_steps:
                    snapshot = graph_provider(observation)
        return result
    finally:
        # Also pause on invalid data, inference timeout, callback errors, or interrupts.
        environment.pause()
