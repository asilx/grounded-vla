"""Belief-aware, bounded execution of native VLA chunks in a synchronous environment.

The provider owns sensor ingestion; neither this executive nor a critic writes
facts from commanded or predicted effects. Declared dependencies express contract
validity, not a claim about which features the neural policy actually used.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from grounded_vla.belief import BeliefStore, Truth
from grounded_vla.contracts import Contract, Requirement
from grounded_vla.journal import Journal
from grounded_vla.simulation import GraphSnapshot


@dataclass(frozen=True)
class PositionDependency:
    entity_id: str
    tolerance: float = 0.02
    relative_to: str | None = None
    max_age_seconds: float = 0.5
    # Explicit observed phase transition (e.g. bilateral contact during a pick).
    # Re-evaluated every step; UNKNOWN never disables the geometry check.
    ignore_when: Requirement | None = None

    def __post_init__(self):
        if not self.entity_id or self.relative_to == self.entity_id:
            raise ValueError("Position dependencies need distinct, nonempty entity IDs")
        if self.ignore_when is not None and (
            not isinstance(self.ignore_when, Requirement)
            or self.ignore_when.truth != Truth.SUPPORTED
        ):
            raise ValueError("Geometry exemptions require an explicitly supported observation")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("Position tolerance must be finite and nonnegative")
        if not math.isfinite(self.max_age_seconds) or self.max_age_seconds <= 0:
            raise ValueError("Position age limit must be finite and positive")


@dataclass(frozen=True)
class RelationDependency:
    source_id: str
    target_id: str

    def __post_init__(self):
        if not self.source_id or not self.target_id:
            raise ValueError("Relation dependencies need nonempty entity IDs")


@dataclass(frozen=True)
class ExecutionPlan:
    contract: Contract
    prompt: str
    positions: tuple[PositionDependency, ...] = ()
    relations: tuple[RelationDependency, ...] = ()
    effect_timeout_steps: int = 100
    max_chunk_age_seconds: float = 0.5

    def __post_init__(self):
        if not self.prompt.strip():
            raise ValueError("A grounded prompt is required")
        if not self.contract.effects and not self.contract.observes:
            raise ValueError("A contract must have an observable completion condition")
        if type(self.effect_timeout_steps) is not int or self.effect_timeout_steps <= 0:
            raise ValueError("effect_timeout_steps must be a positive integer")
        if not math.isfinite(self.max_chunk_age_seconds) or self.max_chunk_age_seconds <= 0:
            raise ValueError("Chunk age limit must be finite and positive")


@dataclass(frozen=True)
class ExecutionView:
    snapshot: GraphSnapshot
    belief: BeliefStore


@dataclass
class ExecutionResult:
    steps: int = 0
    plans: int = 0
    replans: int = 0
    completed_contracts: int = 0
    reason: str = "step_budget_exhausted"
    events: list[dict] = field(default_factory=list)


def _entity(snapshot, entity_id, now, max_age=None):
    graph, ids = snapshot.graph, snapshot.entity_ids
    n = len(graph["valid"])
    if len(ids) != n or len(set(ids)) != n or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Dependencies require unique entity_ids aligned with graph nodes")
    if entity_id not in ids:
        raise ValueError(f"Missing entity: {entity_id}")
    index = ids.index(entity_id)
    if not graph["valid"][index]:
        raise ValueError(f"Invalid entity: {entity_id}")
    if snapshot.entity_timestamps and len(snapshot.entity_timestamps) != n:
        raise ValueError("Entity timestamps must align with graph nodes")
    stamp = snapshot.entity_timestamps[index] if snapshot.entity_timestamps else snapshot.timestamp
    if not math.isfinite(stamp) or stamp < 0 or stamp > now:
        raise ValueError(f"Invalid or future entity timestamp: {entity_id}")
    if max_age is not None and now - stamp > max_age:
        raise ValueError(f"Stale entity: {entity_id}")
    return index


def _position(snapshot, dependency, now):
    features = snapshot.graph["features"]
    if features.shape[1] < 3:
        raise ValueError("Position dependencies require xyz as the first three features")
    i = _entity(snapshot, dependency.entity_id, now, dependency.max_age_seconds)
    value = features[i, :3].astype(float).copy()
    if dependency.relative_to is not None:
        j = _entity(snapshot, dependency.relative_to, now, dependency.max_age_seconds)
        value -= features[j, :3]
    return value


def _facts(plan, view, now):
    return [r.check(view.belief, now).to_dict() for r in plan.contract.preconditions]


@dataclass(frozen=True)
class ChunkBinding:
    """Immutable numeric copies of the dependencies at proposal generation."""

    chunk_id: str
    timestamp: float
    graph_digest: str
    positions: tuple[tuple[float, ...], ...]
    relations: tuple[int, ...]
    facts: tuple[dict, ...]

    @classmethod
    def capture(cls, chunk_id, plan, view, now):
        snapshot = view.snapshot
        positions = tuple(tuple(_position(snapshot, d, now)) for d in plan.positions)
        relations = tuple(
            int(
                snapshot.graph["edges"][
                    _entity(snapshot, d.source_id, now), _entity(snapshot, d.target_id, now)
                ]
            )
            for d in plan.relations
        )
        digest = hashlib.sha256(snapshot.graph["schema_id"].encode())
        for name in ("features", "edges", "valid"):
            values = np.asarray(snapshot.graph[name])
            digest.update(str((values.shape, values.dtype)).encode())
            digest.update(values.tobytes())
        digest.update(repr(tuple(snapshot.entity_ids)).encode())
        return cls(
            chunk_id, now, digest.hexdigest(), positions, relations, tuple(_facts(plan, view, now))
        )

    def changed(self, plan, view, now):
        if now - self.timestamp > plan.max_chunk_age_seconds:
            return {"reason": "chunk_expired", "generated_at": self.timestamp}
        try:
            for dependency, previous in zip(plan.positions, self.positions, strict=True):
                if dependency.ignore_when is not None:
                    requirement = dependency.ignore_when
                    if requirement.check(view.belief, now).truth == requirement.truth:
                        continue
                current = _position(view.snapshot, dependency, now)
                displacement = float(np.linalg.norm(current - previous))
                if displacement > dependency.tolerance:
                    return {
                        "reason": "position_changed",
                        "entity": dependency.entity_id,
                        "relative_to": dependency.relative_to,
                        "before": list(previous),
                        "after": current.tolist(),
                        "displacement": displacement,
                        "tolerance": dependency.tolerance,
                    }
            for dependency, previous in zip(plan.relations, self.relations, strict=True):
                i = _entity(view.snapshot, dependency.source_id, now)
                j = _entity(view.snapshot, dependency.target_id, now)
                current = int(view.snapshot.graph["edges"][i, j])
                if current != previous:
                    return {
                        "reason": "relation_changed",
                        "source": dependency.source_id,
                        "target": dependency.target_id,
                        "before": previous,
                        "after": current,
                    }
        except ValueError as exc:
            return {"reason": "dependency_unavailable", "detail": str(exc)}
        return None


class BeliefAwareExecutor:
    """Select contracts, verify effects, and discard invalid chunk suffixes.

    view_provider(observation) -> ExecutionView ingests actual sensor evidence.
    planner(view, observation, event) -> ExecutionPlan | None runs while paused.
    A None plan means abstention (never an inferred whole-task success).
    on_event is an optional notification callback, also invoked while paused.
    Callbacks must not advance physics or submit robot commands themselves.
    """

    def __init__(self, view_provider, planner, *, journal=None):
        self.view_provider, self.planner = view_provider, planner
        self.journal = journal if journal is not None else Journal()

    def run(
        self,
        environment,
        policy,
        *,
        max_steps=100,
        execute_steps=1,
        max_replans=10,
        max_plans=50,
        monitor=None,
        monitor_camera="base_0_rgb",
        recorder=None,
        record_critic_context=False,
        critic_recorder=None,
        on_event=None,
    ):
        for name, value in (("max_steps", max_steps), ("max_plans", max_plans)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(max_replans) is not int or max_replans < 0:
            raise ValueError("max_replans must be nonnegative")
        if type(execute_steps) is not int or not 1 <= execute_steps <= policy.max_horizon:
            raise ValueError("execute_steps must be within the policy horizon")
        result = ExecutionResult()
        plan, binding, queue, event = None, None, [], None
        plan_steps = chunk_number = 0
        pending_resume = False
        last_critic_time = None

        def capture(obs):
            value = self.view_provider(obs)
            if not isinstance(value, ExecutionView) or not isinstance(value.belief, BeliefStore):
                raise TypeError("view_provider must return ExecutionView with a BeliefStore")
            s = value.snapshot
            # Copy graph arrays: an in-place provider update cannot rewrite the proposal basis.
            checked = policy.validate_graph(s, obs.timestamp)
            return ExecutionView(
                GraphSnapshot(
                    checked, s.timestamp, tuple(s.entity_ids), tuple(s.entity_timestamps)
                ),
                value.belief,
            )

        def invalidate(reason, obs, view):
            nonlocal plan, binding, queue, event, pending_resume
            environment.pause()
            pending_resume = True
            event = {
                "kind": "chunk_cancelled",
                "timestamp": obs.timestamp,
                "contract": plan.contract.key,
                "chunk_id": binding.chunk_id if binding else None,
                "discarded_actions": len(queue),
                "details": reason,
                "previous_facts": list(binding.facts) if binding else [],
                "current_facts": _facts(plan, view, obs.timestamp),
            }
            result.events.append(event)
            self.journal.append(
                "chunk_cancelled", **{k: v for k, v in event.items() if k != "kind"}
            )
            queue, binding, plan = [], None, None
            if monitor is not None:
                monitor.reset()
            if on_event is not None:
                on_event(event, obs)
            if result.replans >= max_replans:
                result.reason = "replan_budget_exhausted"
                return False
            result.replans += 1
            return True

        try:
            if monitor is not None:
                monitor.reset()
            observation = environment.observe()
            view = None
            while True:
                now = observation.timestamp
                if plan is None:
                    environment.pause()
                    pending_resume = True
                    if result.plans >= max_plans:
                        result.reason = "plan_budget_exhausted"
                        break
                    # Re-ingest after a paused recovery callback; do not fabricate a new camera frame.
                    view = capture(observation)
                    plan = self.planner(view, observation, event)
                    if plan is None:
                        result.reason = "no_plan"
                        break
                    if not isinstance(plan, ExecutionPlan):
                        raise TypeError("planner must return ExecutionPlan or None")
                    result.plans += 1
                    plan_steps, queue, binding = 0, [], None
                    self.journal.append(
                        "contract_selected",
                        timestamp=now,
                        contract=plan.contract.to_dict(),
                        prompt=plan.prompt,
                        belief_revision=view.belief.revision,
                        position_dependencies=[asdict(d) for d in plan.positions],
                        relation_dependencies=[asdict(d) for d in plan.relations],
                        trigger=event,
                    )
                    event = None
                    if monitor is not None:
                        monitor.reset()
                context = None
                if (
                    record_critic_context
                    or critic_recorder is not None
                    or (monitor is not None and getattr(monitor, "requires_context", False))
                ):
                    from grounded_vla.critic_context import encode_critic_context

                    context = encode_critic_context(plan.contract, observation, view.snapshot)
                # Include a terminal observation even when no further action is submitted.
                # Paused replanning at the same physics tick does not create duplicate frames.
                if critic_recorder is not None and (
                    last_critic_time is None or now > last_critic_time
                ):
                    critic_recorder.append(
                        state=observation.state,
                        images=observation.images,
                        graph=view.snapshot.graph,
                        timestamp=now,
                        graph_timestamp=view.snapshot.timestamp,
                        critic_context=context,
                    )
                    last_critic_time = now
                # Effects precede preconditions: a successful pick changes held=False to True.
                if plan.contract.complete(view.belief, now):
                    evidence = [r.check(view.belief, now).to_dict() for r in plan.contract.effects]
                    evidence += [
                        view.belief.query(a, now).to_dict() for a in plan.contract.observes
                    ]
                    event = {
                        "kind": "effect_verified",
                        "timestamp": now,
                        "contract": plan.contract.key,
                        "evidence": evidence,
                    }
                    self.journal.append(
                        "effect_verified",
                        timestamp=now,
                        contract=plan.contract.key,
                        evidence=evidence,
                    )
                    result.completed_contracts += 1
                    plan, binding, queue = None, None, []
                    if result.steps >= max_steps:
                        break
                    continue
                if result.steps >= max_steps:
                    break
                if not plan.contract.valid(view.belief, now):
                    if not invalidate({"reason": "precondition_invalid"}, observation, view):
                        break
                    continue
                if plan_steps >= plan.effect_timeout_steps:
                    if not invalidate(
                        {"reason": "missing_effect", "steps": plan_steps}, observation, view
                    ):
                        break
                    continue
                if binding is not None and queue:
                    for dependency in plan.positions:
                        if dependency.ignore_when is not None:
                            query = dependency.ignore_when.check(view.belief, now)
                            if query.truth == Truth.SUPPORTED:
                                self.journal.append(
                                    "position_dependency_exempted",
                                    timestamp=now,
                                    chunk_id=binding.chunk_id,
                                    entity=dependency.entity_id,
                                    evidence=query.to_dict(),
                                )
                    change = binding.changed(plan, view, now)
                    if change is not None:
                        if not invalidate(change, observation, view):
                            break
                        continue
                if monitor is not None:
                    kwargs = (
                        {"context": context} if getattr(monitor, "requires_context", False) else {}
                    )
                    signal = monitor.update(observation.images[monitor_camera], now, **kwargs)
                    if signal is not None:
                        if not invalidate(
                            {"reason": "critic_pause", "signal": signal}, observation, view
                        ):
                            break
                        continue
                if not queue:
                    chunk_number += 1
                    try:
                        binding = ChunkBinding.capture(f"c{chunk_number:06d}", plan, view, now)
                    except ValueError as exc:
                        if not invalidate(
                            {"reason": "dependency_unavailable", "detail": str(exc)},
                            observation,
                            view,
                        ):
                            break
                        continue
                    actions = policy.infer(observation, view.snapshot, plan.prompt)
                    queue = [
                        a.copy() for a in actions[: min(execute_steps, max_steps - result.steps)]
                    ]
                    self.journal.append(
                        "chunk_proposed",
                        timestamp=now,
                        chunk_id=binding.chunk_id,
                        graph_digest=binding.graph_digest,
                        contract=plan.contract.key,
                        facts=list(binding.facts),
                        actions=[a.tolist() for a in queue],
                    )
                if pending_resume:
                    environment.resume()
                    pending_resume = False
                    self.journal.append(
                        "execution_resumed", timestamp=now, chunk_id=binding.chunk_id
                    )
                action = queue.pop(0)
                previous, previous_view = observation, view
                observation = environment.step(action)
                result.steps += 1
                plan_steps += 1
                if observation.timestamp <= previous.timestamp:
                    raise ValueError("Executed actions must advance the observation timestamp")
                if recorder is not None:
                    kwargs = {"critic_context": context} if record_critic_context else {}
                    recorder.append(
                        state=previous.state,
                        images=previous.images,
                        action=action,
                        prompt=plan.prompt,
                        graph=previous_view.snapshot.graph,
                        timestamp=previous.timestamp,
                        graph_timestamp=previous_view.snapshot.timestamp,
                        **kwargs,
                    )
                self.journal.append(
                    "action_executed",
                    timestamp=observation.timestamp,
                    chunk_id=binding.chunk_id,
                    action=action.tolist(),
                )
                view = capture(observation)
            return result
        except BaseException as exc:
            result.reason = "error"
            self.journal.append("execution_error", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            environment.pause()
            self.journal.append(
                "execution_stopped",
                timestamp=(observation.timestamp if "observation" in locals() else None),
                reason=result.reason,
                steps=result.steps,
                completed_contracts=result.completed_contracts,
            )
