"""Bounded pick/place/regrasp selection driven only by measured evidence."""

from __future__ import annotations

import numpy as np

from grounded_vla.belief import Atom, Truth
from grounded_vla.contracts import Contract, Requirement
from grounded_vla.execution import BeliefAwareExecutor, ExecutionPlan, PositionDependency
from grounded_vla.grounding.pick_place import PickPlaceGrounder, SceneViewProvider


class PickPlacePlanner:
    def __init__(self, config):
        self.config = config
        self.pick_attempts = 0
        self.current = None
        self.stop_reason = None

    def __call__(self, view, observation, event):
        c, now = self.config, observation.timestamp
        held = view.belief.query(Atom("held", c.object_id), now).truth
        placed = view.belief.query(Atom("at_tray", c.object_id), now).truth
        self.current = None
        if placed == Truth.SUPPORTED and held == Truth.REFUTED:
            self.stop_reason = "observed_goal_complete"
            return None
        if held == Truth.SUPPORTED:
            # This subtask ends at observed release/loss, not at whole-task success.
            contract = Contract(
                "place",
                c.object_id,
                c.description,
                preconditions=(Requirement(Atom("held", c.object_id), Truth.SUPPORTED),),
                effects=(Requirement(Atom("held", c.object_id), Truth.REFUTED),),
            )
            prompt = f"Place {c.description} at the graph goal, release it, then withdraw."
            dependencies = (
                PositionDependency(
                    c.object_id,
                    tolerance=c.relative_tolerance,
                    relative_to="gripper",
                    max_age_seconds=c.evidence_ttl,
                ),
                PositionDependency(
                    "goal", tolerance=c.target_motion_tolerance, max_age_seconds=c.evidence_ttl
                ),
            )
        elif held == Truth.UNKNOWN or placed == Truth.UNKNOWN:
            # Real actions for inspection are measured-position holds, routed below.
            observes = (Atom("held" if held == Truth.UNKNOWN else "at_tray", c.object_id),)
            contract = Contract("inspect", c.object_id, c.description, observes=observes)
            prompt = "Hold measured joint positions while observing the object."
            dependencies = ()
        else:
            if self.pick_attempts >= c.max_pick_attempts:
                self.stop_reason = "pick_attempt_budget_exhausted"
                return None
            self.pick_attempts += 1
            contract = Contract(
                "pick",
                c.object_id,
                c.description,
                effects=(Requirement(Atom("held", c.object_id), Truth.SUPPORTED),),
            )
            prompt = f"Open the gripper, grasp {c.description}, and lift it upright."
            dependencies = (
                PositionDependency(
                    c.object_id,
                    tolerance=c.target_motion_tolerance,
                    max_age_seconds=c.evidence_ttl,
                    ignore_when=Requirement(Atom("grasp_contact", c.object_id), Truth.SUPPORTED),
                ),
            )
        self.current = contract
        return ExecutionPlan(
            contract,
            prompt,
            positions=dependencies,
            effect_timeout_steps=c.effect_timeout_steps,
            max_chunk_age_seconds=c.max_chunk_age_seconds,
        )


class ObservationHoldPolicy:
    """Route inspect to a measured hold; pick/place use the actual policy server."""

    def __init__(self, policy, planner):
        self.policy, self.planner = policy, planner
        self.max_horizon = policy.max_horizon
        self.meta = policy.meta

    def validate_graph(self, snapshot, now):
        return self.policy.validate_graph(snapshot, now)

    def infer(self, observation, snapshot, prompt):
        if self.planner.current is not None and self.planner.current.skill == "inspect":
            self.validate_graph(snapshot, observation.timestamp)
            # One-step feedback: never repeat an old measured hold across a prefix.
            return np.asarray(observation.state, np.float32)[None].copy()
        return self.policy.infer(observation, snapshot, prompt)


def build_pick_place(config, reader, policy, *, evaluator=None, journal=None):
    grounder = PickPlaceGrounder(config)
    provider = SceneViewProvider(reader, grounder, evaluator)
    planner = PickPlacePlanner(config)
    executor = BeliefAwareExecutor(provider, planner, journal=journal)
    return executor, ObservationHoldPolicy(policy, planner), grounder, planner
