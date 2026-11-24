"""Causal physical evidence for upright parallel-jaw pick/place.

A closed gripper alone never establishes held. Acquisition needs bilateral
contact, nonempty aperture, lift, and persistent relative-motion agreement.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from grounded_vla.belief import Atom, BeliefStore, Truth
from grounded_vla.execution import ExecutionView
from grounded_vla.perception.isaac_scene import (
    bilateral_contact,
    near_goal,
    released,
    validate_initial_sample,
)
from grounded_vla.simulation import GraphSnapshot
from grounded_vla.training.graph import encode_belief_graph


class PickPlaceGrounder:
    def __init__(self, config):
        self.config = config
        self.belief = BeliefStore()
        self.previous = self.initial = self.contact_start = None
        self.release_since = None
        self.held = False
        self.ever_held = False
        self.latest = None
        self.records = []
        self._view = None

    def ingest(self, sample):
        c = self.config
        if self.previous is not None:
            if sample.timestamp == self.previous.timestamp:
                if sample != self.previous:
                    raise ValueError("Conflicting measurements for the same physics tick")
                return self._view
            if sample.timestamp < self.previous.timestamp:
                raise ValueError("Scene samples must increase in time")
            if sample.source != self.previous.source:
                raise ValueError("Cannot switch measurement provenance within an episode")
        if self.initial is None:
            validate_initial_sample(sample, c)
            self.initial = sample
        dt = sample.timestamp - self.previous.timestamp if self.previous else None
        continuous = dt is not None and dt <= c.max_sample_gap + 1e-8
        object_speed = (
            float(
                np.linalg.norm(np.subtract(sample.object_position, self.previous.object_position))
                / dt
            )
            if continuous
            else None
        )
        goal_speed = (
            float(
                np.linalg.norm(np.subtract(sample.goal_position, self.previous.goal_position)) / dt
            )
            if continuous
            else None
        )
        if not continuous:
            self.contact_start = None
            self.release_since = None
            self.held = False
            self.ever_held = False
        contact = bilateral_contact(sample, c)
        if not contact:
            self.contact_start = None
            self.held = False
        else:
            if self.contact_start is None:
                self.contact_start = sample
            drift = np.linalg.norm(sample.relative_position - self.contact_start.relative_position)
            if drift > c.relative_tolerance:
                self.contact_start = sample
                self.held = False
                self.ever_held = False
            motion = np.linalg.norm(
                np.subtract(sample.gripper_position, self.contact_start.gripper_position)
            )
            lifted = sample.object_position[2] - self.initial.object_position[2] >= c.min_lift
            if (
                sample.timestamp - self.contact_start.timestamp + 1e-9 >= c.grasp_confirm_seconds
                and motion >= c.min_gripper_motion
                and lifted
            ):
                self.held = True
                self.ever_held = True

        # Missing/marginal contact is UNKNOWN unless independent separation/support
        # or an observed open gripper establishes a negative observation.
        separated = np.linalg.norm(sample.relative_position) > c.max_grasp_distance
        supported_without_grasp = sample.support_force >= c.contact_force_min and not contact
        held = (
            Truth.SUPPORTED
            if self.held
            else Truth.REFUTED
            if released(sample, c) or separated or supported_without_grasp
            else Truth.UNKNOWN
        )
        if not contact and not (released(sample, c) and near_goal(sample, c)):
            # Inconclusive contact also breaks the acquisition-to-release chain.
            # A later arrival at the goal needs a newly verified grasp.
            self.ever_held = False
        stable_release = (
            self.ever_held
            and released(sample, c)
            and near_goal(sample, c)
            and sample.support_force >= c.contact_force_min
            and object_speed is not None
            and object_speed <= c.stable_speed_max
            and goal_speed <= c.stable_speed_max
        )
        if stable_release:
            if self.release_since is None:
                self.release_since = sample.timestamp
            at_tray = (
                Truth.SUPPORTED
                if sample.timestamp - self.release_since + 1e-9 >= c.release_confirm_seconds
                else Truth.UNKNOWN
            )
        else:
            self.release_since = None
            at_tray = (
                Truth.UNKNOWN
                if self.ever_held and released(sample, c) and near_goal(sample, c)
                else Truth.REFUTED
            )

        statuses = {
            "held": held,
            "at_tray": at_tray,
            "grasp_contact": Truth.SUPPORTED if contact else Truth.REFUTED,
        }
        for predicate, truth in statuses.items():
            atom = Atom(predicate, c.object_id)
            if truth == Truth.UNKNOWN:
                self.belief.retract(atom, at=sample.timestamp, reason="measurement inconclusive")
            else:
                self.belief.add(
                    atom,
                    truth == Truth.SUPPORTED,
                    at=sample.timestamp,
                    source=f"{sample.source}:{predicate}",
                    ttl=c.evidence_ttl,
                    replace=True,
                )
        objects = [
            dict(id=c.object_id, position=sample.object_position, is_target=True),
            dict(id="goal", position=sample.goal_position, is_container=True),
            dict(id="gripper", position=sample.gripper_position),
        ]
        relations = [("goal", "target_of", c.object_id)]
        if held == Truth.SUPPORTED:
            relations.append((c.object_id, "held_by", "gripper"))
        if at_tray == Truth.SUPPORTED:
            relations.append((c.object_id, "on", "goal"))
        graph = encode_belief_graph(self.belief, objects, relations, now=sample.timestamp)
        self._view = ExecutionView(
            GraphSnapshot(
                graph, sample.timestamp, (c.object_id, "goal", "gripper"), (sample.timestamp,) * 3
            ),
            self.belief,
        )
        self.latest = sample
        self.records.append(
            {
                **asdict(sample),
                "held": held.value,
                "at_tray": at_tray.value,
                "bilateral_contact": contact,
                "object_speed": object_speed,
            }
        )
        self.previous = sample
        return self._view


class SceneViewProvider:
    """Cache paused repeat queries; never refresh evidence without a new sample."""

    def __init__(self, reader, grounder, evaluator=None):
        self.reader, self.grounder, self.evaluator = reader, grounder, evaluator
        self._stamp = None
        self._view = None

    def __call__(self, observation):
        if self._stamp == observation.timestamp:
            return self._view
        sample = self.reader.read(observation)
        if not np.isclose(sample.timestamp, observation.timestamp, atol=1e-8, rtol=0):
            raise ValueError("Scene sample does not match the current observation")
        self._view = self.grounder.ingest(sample)
        if self.evaluator is not None:
            self.evaluator.update(sample)
        self._stamp = observation.timestamp
        return self._view
