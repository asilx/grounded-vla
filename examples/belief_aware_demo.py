"""Deterministic CPU control-flow demo, NOT an Isaac or learned-policy benchmark.

Run after installing the package with its simulation extra:
    python examples/belief_aware_demo.py --out runs/belief-aware-demo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grounded_vla.belief import Atom, BeliefStore, Truth
from grounded_vla.contracts import Contract, Requirement
from grounded_vla.execution import (
    BeliefAwareExecutor,
    ExecutionPlan,
    ExecutionView,
    PositionDependency,
)
from grounded_vla.simulation import CanonicalSimPolicy, GraphSnapshot


def run_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    store = BeliefStore()
    store.add(Atom("held", "cargo"), False, at=0, source="demo-observation")

    class Environment:
        timestamp = 0.0
        position = 0.0
        paused = False

        def observe(self):
            return SimpleNamespace(
                timestamp=self.timestamp,
                state=np.zeros(1, np.float32),
                images={"base_0_rgb": np.zeros((224, 224, 3), np.uint8)},
            )

        def step(self, action):
            assert not self.paused
            self.timestamp = round(self.timestamp + 0.1, 8)
            if self.timestamp == 0.1:
                self.position = 0.2  # An observed target displacement, not a commanded effect.
            if self.timestamp == 0.3:
                store.add(
                    Atom("held", "cargo"),
                    True,
                    at=self.timestamp,
                    source="demo-contact-observation",
                    replace=True,
                )
            return self.observe()

        def pause(self):
            self.paused = True

        def resume(self):
            self.paused = False

    class PolicyClient:
        metadata = dict(
            input_preset="canonical",
            state_dim=1,
            action_dim=1,
            action_convention="demo_absolute",
            graph_required=True,
            graph_feature_dim=3,
            graph_relation_types=2,
            action_horizon=3,
            graph_schema_id="demo-geometry-v1",
            control_period_seconds=0.1,
        )

        def infer(self, observation):
            return {"actions": np.zeros((3, 1), np.float32)}

    environment = Environment()
    config = SimpleNamespace(
        joint_names=("demo_joint",), action_convention="demo_absolute", control_period_seconds=0.1
    )
    policy = CanonicalSimPolicy(PolicyClient(), config)
    contract = Contract(
        "pick",
        "cargo",
        "the cargo",
        preconditions=(Requirement(Atom("held", "cargo"), Truth.REFUTED),),
        effects=(Requirement(Atom("held", "cargo"), Truth.SUPPORTED),),
    )

    def view_provider(observation):
        graph = dict(
            schema_id="demo-geometry-v1",
            features=np.array([[environment.position, 0, 0]], np.float32),
            edges=np.ones((1, 1), np.int64),
            valid=np.ones(1, bool),
        )
        return ExecutionView(GraphSnapshot(graph, observation.timestamp, ("cargo",)), store)

    def planner(view, observation, event):
        if contract.complete(view.belief, observation.timestamp):
            return None
        return ExecutionPlan(
            contract, "Pick up the cargo.", positions=(PositionDependency("cargo"),)
        )

    executive = BeliefAwareExecutor(view_provider, planner)
    result = executive.run(environment, policy, max_steps=10, execute_steps=3)
    executive.journal.save(output / "execution-trace.jsonl")
    summary = {
        "scope": "CPU deterministic control-flow demo; no learned policy or physics",
        **asdict(result),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/belief-aware-demo")
    print(json.dumps(run_demo(parser.parse_args().out), indent=2))
