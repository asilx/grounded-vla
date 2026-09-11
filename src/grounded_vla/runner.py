"""End-to-end reference execution: observe, reason, validate, execute, verify."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from grounded_vla.belief import Atom, BeliefStore
from grounded_vla.contracts import Contract, Executive, TaskRules, compile_instruction
from grounded_vla.journal import Journal, explain
from grounded_vla.policy import ActionChunk, ScriptedPolicy
from grounded_vla.toy_world import ToyWorld
from grounded_vla.validation import ToyValidator, select

METHODS = ("grounded", "no_retraction", "no_semantic_guard")


@dataclass(frozen=True)
class RunConfig:
    scenario: str = "disturbance"
    seed: int = 0
    method: str = "grounded"
    candidates: int = 3
    prefix_length: int = 2
    max_ticks: int = 100

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown method: {self.method}")
        if not 1 <= self.candidates <= 3 or self.prefix_length < 1 or self.max_ticks < 1:
            raise ValueError("Use 1-3 candidates and positive prefix/tick budgets")


@dataclass
class RunResult:
    config: dict
    metrics: dict
    journal: Journal

    def to_dict(self) -> dict:
        return {"config": self.config, "metrics": self.metrics, "records": self.journal.records}


def ingest(belief: BeliefStore, observation: dict, *, retract_events: bool = True) -> None:
    now = observation["tick"]
    for event in observation["events"]:
        if event["type"] == "dirty_contact_observed" and retract_events:
            belief.retract(
                Atom("clean", event["object"]),
                at=now,
                reason="observed dirty contact invalidates prior cleanliness",
            )
    for report in observation["reports"]:
        belief.add(
            Atom(report["predicate"], report["subject"]),
            report["value"],
            at=now,
            source=report["source"],
            ttl=report["ttl"],
            replace=report.get("replace", True),
        )


def run(
    config: RunConfig | None = None, *, rules: TaskRules | None = None, knowrob_session=None
) -> RunResult:
    config, rules = config or RunConfig(), rules or TaskRules()
    world = ToyWorld(config.scenario, config.seed)
    belief, journal = BeliefStore(), Journal()
    executive = Executive(world.objects, rules)
    policy = ScriptedPolicy()
    validator = ToyValidator(semantic_guard=config.method != "no_semantic_guard")
    journal.append(
        "run_started",
        config=asdict(config),
        rules=asdict(rules),
        environment="symbolic-event-emulator",
        policy=policy.name,
        knowledge_backend="native-knowrob" if knowrob_session else "reference-evidence-ledger",
        version="0.1.0",
    )
    observation = world.observe()
    active: Contract | None = None
    queue: tuple[tuple[float, ...], ...] = ()
    selected: ActionChunk | None = None
    decision_id = ""
    decisions = recoveries = cancellations = rejected = 0
    termination = "budget_exhausted"
    while True:
        before = len(belief.events)
        ingest(belief, observation, retract_events=config.method != "no_retraction")
        now = observation["tick"]
        view = belief
        if knowrob_session is not None:
            from grounded_vla.backends.knowrob import KnowRobView

            view = KnowRobView(belief, knowrob_session, now)
        journal.append(
            "observation", tick=now, observation=observation, belief_changes=belief.events[before:]
        )
        if active is not None:
            if active.complete(view, now):
                journal.append(
                    "effect_verified", tick=now, decision_id=decision_id, contract=active.key
                )
                active, queue = None, ()
            elif not active.valid(view, now):
                cancellations += 1
                journal.append(
                    "queue_cancelled",
                    tick=now,
                    decision_id=decision_id,
                    reason="precondition changed, expired, or became uncertain",
                )
                active, queue = None, ()
            elif not queue:
                recoveries += 1
                journal.append(
                    "missing_effect",
                    tick=now,
                    decision_id=decision_id,
                    reason="chunk ended without observed postconditions; replan",
                )
                active = None
        if active is None:
            plan = executive.choose(view, now)
            if plan.contract is None:
                termination = "belief_goal_complete"
                break
            if now >= config.max_ticks:
                break
            active = plan.contract
            candidates = policy.sample(active, observation, config.candidates)
            verdicts = [validator.check(c, active, view, now) for c in candidates]
            rejected += sum(not verdict.accepted for verdict in verdicts)
            selected = select(candidates, verdicts)
            if selected is None:
                journal.append(
                    "abstained",
                    tick=now,
                    reason="no admissible candidate",
                    verdicts=[v.to_dict() for v in verdicts],
                )
                termination = "no_admissible_candidate"
                break
            decisions += 1
            decision_id = f"d{decisions:04d}"
            payload = {
                "decision_id": decision_id,
                "tick": now,
                "contract": active.to_dict(),
                "instruction": compile_instruction(active),
                "belief": belief.snapshot(now),
                "considered": plan.considered,
                "candidates": [c.to_dict() for c in candidates],
                "verdicts": [v.to_dict() for v in verdicts],
                "selected": selected.to_dict(),
            }
            payload["explanation"] = explain(payload)
            journal.append("decision", **payload)  # Must precede the first executed prefix.
            queue = selected.values
        if now >= config.max_ticks:
            break
        assert active is not None and selected is not None
        remaining = config.max_ticks - now
        length = min(config.prefix_length, len(queue), remaining)
        prefix, tail = queue[:length], queue[length:]
        # Revalidate every prefix against the most recent observation.
        prefix_chunk = ActionChunk(
            selected.id, prefix, selected.score, selected.space, selected.policy
        )
        check = validator.check(prefix_chunk, active, view, now)
        if not check.accepted:
            journal.append(
                "abstained", tick=now, reason="prefix validation failed", verdicts=[check.to_dict()]
            )
            termination = "prefix_rejected"
            break
        journal.append(
            "execution", tick=now, decision_id=decision_id, prefix=prefix, remaining_steps=len(tail)
        )
        observation = world.advance(decision_id, active, prefix, terminal=not tail)
        queue = tail
    metrics = world.score(rules)
    metrics.update(
        {
            "termination": termination,
            "decisions": decisions,
            "recoveries": recoveries,
            "cancellations": cancellations,
            "rejected_candidates": rejected,
            "backend": "scripted event emulator; not a VLA benchmark",
        }
    )
    journal.append("run_finished", metrics=metrics)
    return RunResult(asdict(config), metrics, journal)
