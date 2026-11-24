"""Score raw scene measurements independently of policy, belief and critic outputs.

The benchmark is explicitly simulator-oracle. It does not measure perception
quality, collision safety, or generalization to real robots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from grounded_vla.perception.isaac_scene import (
    PickPlaceConfig,
    SceneSample,
    bilateral_contact,
    near_goal,
    released,
    validate_initial_sample,
)

EVALUATION_PROTOCOL = "grounded-vla-pick-place-measurements-v2"


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
        if self.previous and sample.source != self.previous.source:
            raise ValueError("Cannot switch measurement provenance within an episode")
        if self.initial is None:
            validate_initial_sample(sample, c)
            self.initial = sample
        dt = sample.timestamp - self.previous.timestamp if self.previous else None
        continuous = dt is not None and dt <= c.max_sample_gap + 1e-8
        if not continuous:
            self.placement_eligible = False
            self.contact_start = None
        lift = sample.object_position[2] - self.initial.object_position[2]
        self.peak_lift = max(self.peak_lift, lift)
        if not bilateral_contact(sample, c):
            self.contact_start = None
        else:
            if self.contact_start is None:
                self.contact_start = sample
            if (
                np.linalg.norm(sample.relative_position - self.contact_start.relative_position)
                > c.relative_tolerance
            ):
                self.contact_start = sample
                self.placement_eligible = False
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
            "evaluation_protocol": EVALUATION_PROTOCOL,
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


def replay_measurements(path, config):
    """Re-score saved raw samples; ignore all recorded predicate/score annotations.

    Requires NumPy only, with no simulator, policy, belief store or PyTorch.
    The digest identifies the bytes replayed; it does not authenticate a sensor.
    """
    evaluator = PickPlaceEvaluator(config)
    names = {field.name for field in fields(SceneSample)}
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            digest.update(line)
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not names <= row.keys():
                    raise ValueError("Each measurement requires every raw SceneSample field")
                evaluator.update(SceneSample(**{key: row[key] for key in names}))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if evaluator.samples == 0:
        raise ValueError(f"{path}: no scene measurements")
    return {**evaluator.result(), "measurements_sha256": digest.hexdigest()}


def _counts(runs):
    successes = sum(run["success"] for run in runs)
    return {
        "started_runs": len(runs),
        "successful_runs": successes,
        "success_rate": successes / len(runs),
        "error_runs": sum(run["reason"] == "error" for run in runs),
        "verification_error_runs": sum("verification_error" in run for run in runs),
    }


def aggregate(paths, *, verify_measurements=False):
    """Each supplied run counts, including startup errors and all retry attempts."""
    paths = list(paths)
    if not paths or len({Path(p).resolve() for p in paths}) != len(paths):
        raise ValueError("Provide distinct run result paths")
    runs = [json.loads(Path(p).read_text()) for p in paths]
    if any(
        not isinstance(r, dict) or r.get("format") != "grounded-vla-pick-place-v1" for r in runs
    ):
        raise ValueError("Unknown pick/place result format")
    scored = []
    groups = {}
    for path, run in zip(paths, runs, strict=True):
        scenario = run.get("scenario", "unspecified")
        if scenario not in {"nominal", "target-shift", "unspecified"}:
            raise ValueError(f"Unknown pick/place scenario: {scenario!r}")
        eligible = scenario != "target-shift" or run.get("perturbation_applied") is True
        physical = run.get("physical_measurements", {})
        source = physical.get("measurement_source") if isinstance(physical, dict) else None
        source = source or run.get("measurement_source")
        if source is not None and not isinstance(source, str):
            raise ValueError("Measurement provenance must be a string")
        record = {
            "path": str(path),
            "seed": run.get("seed"),
            "scenario": scenario,
            "measurement_source": source,
            "reported_success": run.get("task_success") is True,
            "success": run.get("task_success") is True
            and run.get("reason") != "error"
            and eligible,
            "replans": run.get("replans", 0),
            "reason": run.get("reason"),
        }
        if verify_measurements:
            try:
                measurements = replay_measurements(
                    Path(path).parent / "scene-measurements.jsonl",
                    PickPlaceConfig(**run["task_config"]),
                )
                saved = run.get("scene_measurements")
                if saved is not None:
                    if (
                        not isinstance(saved, dict)
                        or saved.get("sha256") != measurements["measurements_sha256"]
                    ):
                        raise ValueError("Scene measurement digest mismatch")
                    if saved.get("samples") != measurements["measurement_samples"]:
                        raise ValueError("Scene measurement count mismatch")
                if isinstance(physical, dict) and "measurement_samples" in physical:
                    if physical["measurement_samples"] != measurements["measurement_samples"]:
                        raise ValueError("Physical measurement count mismatch")
            except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
                # Keep corrupt/missing recordings in the denominator, including
                # startup failures which cannot have produced any measurements.
                record.update(success=False, verification_error=str(exc))
            else:
                record.update(
                    success=bool(record["success"] and measurements["task_success"]),
                    measurement_source=measurements["measurement_source"],
                    replayed_measurements=measurements,
                )
        scored.append(record)
        key = (scenario, record["measurement_source"])
        groups.setdefault(key, []).append(record)
    return {
        "scope": "simulator-oracle single-object pick/place; all supplied attempts",
        "score_mode": "verified_raw_measurements" if verify_measurements else "reported",
        "evaluation_protocol": EVALUATION_PROTOCOL if verify_measurements else None,
        **_counts(scored),
        "groups": [
            {
                "scenario": scenario,
                "measurement_source": source,
                **_counts(group),
            }
            for (scenario, source), group in groups.items()
        ],
        "runs": scored,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="One result.json for every started run")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--verify-measurements",
        action="store_true",
        help="Re-score sibling scene-measurements.jsonl using each result's saved task_config",
    )
    args = parser.parse_args()
    result = aggregate(args.results, verify_measurements=args.verify_measurements)
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
