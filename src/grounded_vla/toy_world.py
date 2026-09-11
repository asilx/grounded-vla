"""A deterministic, partially observed event emulator; no contact physics or VLA.

Hidden task state is used only to generate sensor reports and score outcomes.
The executive never receives it. 'Clean' is a benchmark label, not hygiene.
"""

from __future__ import annotations

import random

from grounded_vla.contracts import Contract, TaskRules

SCENARIOS = ("clean", "unknown", "stale", "conflict", "grasp_failure", "disturbance")


class ToyWorld:
    def __init__(self, scenario: str = "disturbance", seed: int = 0) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario, self.seed, self.tick = scenario, seed, 0
        rng = random.Random(seed)
        self.objects = {"cup_a": "the blue cup", "cup_b": "the red cup"}
        self._state = {
            "cup_a": {"clean": True, "filled": False, "at_tray": False},
            "cup_b": {"clean": bool(rng.getrandbits(1)), "filled": True, "at_tray": False},
        }
        self._positions = {
            "cup_a": [0.2 + rng.random() * 0.1, 0.3, 0.25],
            "cup_b": [0.5 + rng.random() * 0.1, 0.3, 0.25],
        }
        self.robot_pose = [0.5, 0.5, 0.5, 0.0]
        self._held: str | None = None
        self._reports: list[dict] = []
        self._events: list[dict] = []
        self._progress: dict[str, int] = {}
        self._first_placement_tick: int | None = None
        self._disturbed = False
        self._failed_grasp = False
        self.violations: list[str] = []
        self._report_properties("cup_a", ttl=1 if scenario == "stale" else 40)
        if scenario == "clean":
            self._state["cup_b"]["clean"] = True
            self._report_properties("cup_b")
        if scenario == "conflict":
            self._reports.append(
                {
                    "predicate": "clean",
                    "subject": "cup_a",
                    "value": False,
                    "source": "synthetic-conflicting-sensor",
                    "ttl": 40,
                    "replace": False,
                }
            )

    def _report_properties(self, obj: str, ttl: float = 40) -> None:
        for name in ("clean", "filled"):
            self._reports.append(
                {
                    "predicate": name,
                    "subject": obj,
                    "value": self._state[obj][name],
                    "source": "toy-inspection",
                    "ttl": ttl,
                    "replace": True,
                }
            )

    def observe(self) -> dict:
        reports = self._reports
        self._reports = []
        for obj in self.objects:
            for predicate, value in (
                ("held", self._held == obj),
                ("at_tray", self._state[obj]["at_tray"]),
            ):
                reports.append(
                    {
                        "predicate": predicate,
                        "subject": obj,
                        "value": value,
                        "source": "toy-pose-sensor",
                        "ttl": 3,
                        "replace": True,
                    }
                )
        observation = {
            "tick": self.tick,
            "robot_pose": list(self.robot_pose),
            "positions": {k: list(v) for k, v in self._positions.items()},
            "reports": reports,
            "events": self._events,
        }
        self._events = []
        return observation

    def advance(
        self,
        decision_id: str,
        contract: Contract,
        prefix: tuple[tuple[float, ...], ...],
        *,
        terminal: bool,
    ) -> dict:
        for point in prefix:
            self.tick += 1
            self.robot_pose = list(point)
            self._progress[decision_id] = self._progress.get(decision_id, 0) + 1
            if self._held and self._state[self._held]["filled"] and abs(point[3]) > 12:
                event = f"spill:{self._held}"
                if event not in self.violations:
                    self.violations.append(event)
            if self._held:
                self._positions[self._held] = list(point[:3])
            self._maybe_disturb()
        if terminal:
            obj = contract.object_id
            if contract.skill == "inspect":
                self._report_properties(obj)
            elif contract.skill == "clean":
                self._state[obj]["clean"] = True
                self._state[obj]["at_tray"] = False
                self._held = None
                self._positions[obj] = [0.25, 0.8, 0.25]
                self._report_properties(obj)
            elif contract.skill == "pick":
                fail = (
                    self.scenario in {"grasp_failure", "disturbance"}
                    and obj == "cup_b"
                    and not self._failed_grasp
                )
                if fail:
                    self._failed_grasp = True
                    self._events.append({"type": "grasp_failed", "object": obj})
                else:
                    self._held = obj
            elif contract.skill == "place":
                if self._held == obj:
                    self._held = None
                    self._state[obj]["at_tray"] = True
                    self._positions[obj] = [0.8, 0.5, 0.25]
                    if not self._state[obj]["clean"]:
                        self.violations.append(f"dirty-placement:{obj}")
                    if self._first_placement_tick is None:
                        self._first_placement_tick = self.tick
        return self.observe()

    def _maybe_disturb(self) -> None:
        if self.scenario != "disturbance" or self._disturbed or self._first_placement_tick is None:
            return
        if self.tick < self._first_placement_tick + 2:
            return
        self._disturbed = True
        self._state["cup_a"]["clean"] = False
        self._state["cup_a"]["at_tray"] = False
        self._positions["cup_a"] = [0.3, 0.7, 0.25]
        if self._held == "cup_a":
            self._held = None
        self._events.append({"type": "dirty_contact_observed", "object": "cup_a"})

    def score(self, rules: TaskRules) -> dict:
        """Evaluation-only oracle. Never used as an executive input."""
        complete = all(
            state["at_tray"] and (state["clean"] or not rules.require_clean)
            for state in self._state.values()
        )
        relevant = [
            v
            for v in self.violations
            if (v.startswith("spill:") and rules.keep_filled_upright)
            or (v.startswith("dirty-placement:") and rules.require_clean)
        ]
        return {
            "goal_complete": complete,
            "constrained_success": complete and not relevant,
            "violations": relevant,
            "ticks": self.tick,
            "disturbance_occurred": self._disturbed,
        }
