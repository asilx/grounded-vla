"""Score raw scene measurements independently of policy, belief and critic outputs.

The benchmark is explicitly simulator-oracle. It does not measure perception
quality, collision safety, or generalization to real robots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from grounded_vla.grounding.pick_place import bilateral_contact, near_goal, released


class PickPlaceEvaluator:
    def __init__(self, config):
        self.config = config
        self.initial = self.previous = self.contact_start = None
        self.verified_grasp = False
        self.placement_eligible = False
        self.stable_since = None
        self.success = False
        self.samples = 0
        self.peak_lift = 0.0
        self.goal_error = None

    def update(self, sample):
        c = self.config
        if self.previous and sample.timestamp <= self.previous.timestamp:
            raise ValueError("Evaluation requires distinct increasing measurements")
        if self.initial is None:
            self.initial = sample
        dt = sample.timestamp - self.previous.timestamp if self.previous else None
        continuous = dt is not None and dt <= c.max_sample_gap + 1e-8
        if not continuous:
            self.placement_eligible = False
        lift = sample.object_position[2] - self.initial.object_position[2]
        self.peak_lift = max(self.peak_lift, lift)
        if not continuous or not bilateral_contact(sample, c):
            self.contact_start = None
        else:
            if self.contact_start is None:
                self.contact_start = sample
            if (
                np.linalg.norm(sample.relative_position - self.contact_start.relative_position)
                > c.relative_tolerance
            ):
                self.contact_start = sample
            displacement = np.linalg.norm(
                np.subtract(sample.gripper_position, self.contact_start.gripper_position)
            )
            if (
                sample.timestamp - self.contact_start.timestamp + 1e-9 >= c.grasp_confirm_seconds
                and lift >= c.min_lift
                and displacement >= c.min_gripper_motion
            ):
                self.verified_grasp = True
                self.placement_eligible = True
        if not bilateral_contact(sample, c) and not (released(sample, c) and near_goal(sample, c)):
            self.placement_eligible = False
        stable = (
            continuous
            and self.placement_eligible
            and released(sample, c)
            and near_goal(sample, c)
            and sample.support_force >= c.contact_force_min
            and np.linalg.norm(np.subtract(sample.object_position, self.previous.object_position))
            / dt
            <= c.stable_speed_max
            and np.linalg.norm(np.subtract(sample.goal_position, self.previous.goal_position)) / dt
            <= c.stable_speed_max
        )
        if stable:
            if self.stable_since is None:
                self.stable_since = sample.timestamp
        else:
            self.stable_since = None
        self.success = bool(
            self.stable_since is not None
            and sample.timestamp - self.stable_since + 1e-9 >= c.release_confirm_seconds
        )
        self.goal_error = float(
            np.linalg.norm(np.subtract(sample.object_position, sample.goal_position))
        )
        self.samples += 1
        self.previous = sample

    def result(self):
        return {
            "task_success": self.success,
            "verified_grasp": self.verified_grasp,
            "measurement_samples": self.samples,
            "peak_lift_m": self.peak_lift,
            "final_goal_error_m": self.goal_error,
            "stable_release_seconds": (
                self.previous.timestamp - self.stable_since
                if self.stable_since is not None
                else 0.0
            ),
            "measurement_source": self.initial.source if self.initial else None,
        }


def aggregate(paths):
    """Each supplied run counts, including startup errors and all retry attempts."""
    if not paths or len({Path(p).resolve() for p in paths}) != len(paths):
        raise ValueError("Provide distinct run result paths")
    runs = [json.loads(Path(p).read_text()) for p in paths]
    if any(r.get("format") != "grounded-vla-pick-place-v1" for r in runs):
        raise ValueError("Unknown pick/place result format")
    successes = sum(r.get("task_success") is True and r.get("reason") != "error" for r in runs)
    return {
        "scope": "simulator-oracle single-object pick/place; all supplied attempts",
        "started_runs": len(runs),
        "successful_runs": successes,
        "success_rate": successes / len(runs),
        "error_runs": sum(r.get("reason") == "error" for r in runs),
        "runs": [
            {
                "path": str(p),
                "seed": r.get("seed"),
                "scenario": r.get("scenario"),
                "success": r.get("task_success") is True and r.get("reason") != "error",
                "replans": r.get("replans", 0),
                "reason": r.get("reason"),
            }
            for p, r in zip(paths, runs, strict=True)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="One result.json for every started run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = aggregate(args.results)
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
