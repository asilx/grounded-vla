"""Run grounded pick/place in an authored Isaac scene using measured oracle evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation_config")
    parser.add_argument("task_config")
    parser.add_argument("--uri", default="ws://localhost:8000")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=("nominal", "target-shift"), default="nominal")
    parser.add_argument("--perturb-step", type=int, default=10)
    parser.add_argument("--shift-metres", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--execute-steps", type=int, default=3)
    parser.add_argument("--max-replans", type=int, default=8)
    parser.add_argument("--max-plans", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--action-guard", help="module:function(state, target); raises to veto")
    critic = parser.add_mutually_exclusive_group(required=True)
    critic.add_argument("--critic-checkpoint")
    critic.add_argument("--no-critic", action="store_true")
    parser.add_argument("--critic-device", default="cpu")
    parser.add_argument("--critic-camera", default="base_0_rgb")
    parser.add_argument("--record-critic-context", action="store_true")
    args = parser.parse_args()
    from grounded_vla.backends.isaacsim import IsaacSimConfig
    from grounded_vla.perception.isaac_scene import PickPlaceConfig

    sim_config = IsaacSimConfig.load(args.simulation_config)
    task_config = PickPlaceConfig.load(args.task_config)
    if sim_config.control_period_seconds > task_config.max_sample_gap:
        parser.error("Control period exceeds the task's maximum sample interval")
    if (
        args.seed < 0
        or args.perturb_step < 1
        or not 0 < args.shift_metres <= 0.2
        or args.max_steps < 1
        or args.execute_steps < 1
        or args.max_replans < 0
        or args.max_plans < 1
    ):
        parser.error("Invalid budgets, seed or perturbation magnitude (0, 0.2] m")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "format": "grounded-vla-pick-place-v1",
        "seed": args.seed,
        "scenario": args.scenario,
        "scope": "Isaac simulator-oracle grounding; no learned perception or safety claim",
        "recording_source": "policy_rollout_with_measured_inspection_holds_not_expert",
        "simulation_config": asdict(sim_config),
        "task_config": asdict(task_config),
        "task_success": False,
        "reason": "error",
        "perturbation_applied": False,
    }
    app = environment = client = recorder = critic_recorder = executor = None
    grounder = planner = evaluator = None
    try:
        # Kit starts before importing any Isaac extensions or the torch modules.
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": not args.gui})
        import numpy as np

        from grounded_vla.backends.isaacsim import IsaacSimEnvironment
        from grounded_vla.backends.openpi import OpenPiTransport
        from grounded_vla.benchmarks.isaac_pick_place import PickPlaceEvaluator
        from grounded_vla.isaacsim_cli import _callable
        from grounded_vla.perception.isaac_scene import IsaacSceneReader
        from grounded_vla.simulation import CanonicalSimPolicy
        from grounded_vla.tasks.pick_place import build_pick_place
        from grounded_vla.training.recording import CriticObservationRecorder, EpisodeRecorder

        attached = {}

        def setup_scene(world, robot):
            attached["reader"] = IsaacSceneReader.attach(task_config, world, sim_config.joint_names)

        environment = IsaacSimEnvironment.from_config(
            sim_config,
            scene_setup=setup_scene,
            action_guard=_callable(args.action_guard) if args.action_guard else None,
        )
        reader = attached["reader"]
        client = OpenPiTransport(args.uri, timeout=args.timeout)
        canonical = CanonicalSimPolicy(client, sim_config)
        evaluator = PickPlaceEvaluator(task_config)
        executor, policy, grounder, planner = build_pick_place(
            task_config,
            reader,
            canonical,
            evaluator=evaluator,
        )
        recorder = EpisodeRecorder(policy.meta["graph_schema_id"])
        if args.record_critic_context:
            critic_recorder = CriticObservationRecorder(policy.meta["graph_schema_id"])
        monitor = None
        if args.critic_checkpoint:
            from grounded_vla.learning.temporal_vit import TemporalMonitor, ViTWindowPredictor

            if args.critic_camera not in sim_config.camera_paths:
                raise ValueError("Monitor camera is not configured")
            predictor = ViTWindowPredictor(
                args.critic_checkpoint,
                device=args.critic_device,
                expected_camera=args.critic_camera,
                expected_period=sim_config.control_period_seconds,
            )
            monitor = TemporalMonitor(
                predictor,
                window_size=predictor.config.max_frames,
                period=sim_config.control_period_seconds,
            )
        summary["policy_metadata"] = policy.meta
        summary["critic_checkpoint"] = args.critic_checkpoint
        original_step = environment.step
        submitted = 0
        rng = np.random.default_rng(args.seed)

        def step(action):
            nonlocal submitted
            # A controlled disturbance BEFORE stepping: the resulting cameras,
            # joints and scene measurements refer to the same subsequent tick.
            if args.scenario == "target-shift" and submitted == args.perturb_step - 1:
                positions, orientations = reader.cargo.get_world_poses()
                positions = np.asarray(positions).copy()
                offset = np.array([args.shift_metres * rng.choice((-1, 1)), 0, 0])
                positions[0] += offset
                reader.cargo.set_world_poses(positions=positions, orientations=orientations)
                reader.cargo.set_velocities(np.zeros((1, 6), dtype=np.float32))
                summary["perturbation_applied"] = True
                summary["perturbation_offset_m"] = offset.tolist()
                executor.journal.append(
                    "benchmark_disturbance",
                    before_action=submitted + 1,
                    timestamp=float(environment.world.current_time),
                    offset_m=offset.tolist(),
                )
            observation = original_step(action)
            submitted += 1
            return observation

        # One action predates this exogenous disturbance. Cancellation is checked
        # at the next observation; this benchmark does not claim zero-latency detection.
        environment.step = step
        result = executor.run(
            environment,
            policy,
            max_steps=args.max_steps,
            execute_steps=args.execute_steps,
            max_replans=args.max_replans,
            max_plans=args.max_plans,
            monitor=monitor,
            monitor_camera=args.critic_camera,
            recorder=recorder,
            record_critic_context=args.record_critic_context,
            critic_recorder=critic_recorder,
        )
        summary.update(asdict(result))
        summary.update(evaluator.result())
        summary["planner_stop_reason"] = planner.stop_reason
        summary["pick_attempts"] = planner.pick_attempts
        from grounded_vla.belief import Atom, Truth

        summary["task_success"] = bool(
            evaluator.success
            and grounder.latest is not None
            and grounder.belief.query(
                Atom("at_tray", task_config.object_id), grounder.latest.timestamp
            ).truth
            == Truth.SUPPORTED
            and (args.scenario == "nominal" or summary["perturbation_applied"])
        )
    except BaseException as exc:
        summary.update(
            reason="error", task_success=False, error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        # Preserve a failure result even if a partial recording cannot be saved.
        try:
            if environment is not None:
                environment.pause()
            if grounder is not None:
                digest = hashlib.sha256()
                with (out / "scene-measurements.jsonl").open("wb") as stream:
                    for row in grounder.records:
                        line = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
                        stream.write(line)
                        digest.update(line)
                summary["scene_measurements"] = {
                    "sha256": digest.hexdigest(),
                    "samples": len(grounder.records),
                }
            if executor is not None:
                executor.journal.save(out / "execution-trace.jsonl")
            if evaluator is not None:
                summary["physical_measurements"] = evaluator.result()
            for recording, filename in (
                (recorder, "episode.npz"),
                (critic_recorder, "critic-observations.npz"),
            ):
                if recording is not None and recording.frames:
                    recording.save(out / filename)
        except BaseException as exc:
            summary.update(reason="error", task_success=False, artifact_error=str(exc))
            raise
        finally:
            try:
                (out / "result.json").write_text(
                    json.dumps(summary, indent=2, allow_nan=False) + "\n"
                )
            finally:
                try:
                    if client is not None:
                        client.close()
                finally:
                    if app is not None:
                        app.close()
    print(f"Wrote measured pick/place result to {out}")


if __name__ == "__main__":
    main()
