"""Run a configured USD scene with canonical pi0 inference and a temporal critic."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path


def _callable(spec):
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("Use module:function for a callback")
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise ValueError(f"{spec} is not callable")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--out", default="runs/isaacsim")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--capture-only", action="store_true", help="Check cameras/state without a policy"
    )
    parser.add_argument("--uri", default="ws://localhost:8000")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--prompt")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument("--action-guard", help="module:function(state, target); raise to veto")
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--graph-provider", help="module:function accepting SimObservation")
    context.add_argument(
        "--empty-graph", action="store_true", help="Explicit graph-disabled ablation"
    )
    context.add_argument(
        "--executor-factory", help="module:function(environment) returning BeliefAwareExecutor"
    )
    parser.add_argument("--max-replans", type=int, default=10)
    parser.add_argument("--max-plans", type=int, default=50)
    parser.add_argument(
        "--record-critic-context",
        action="store_true",
        help="Record causal contract/graph context for a conditioned critic",
    )
    critic = parser.add_mutually_exclusive_group()
    critic.add_argument("--critic-checkpoint")
    critic.add_argument("--no-critic", action="store_true", help="Explicit monitor ablation")
    parser.add_argument("--critic-device", default="cpu")
    parser.add_argument("--critic-camera", default="base_0_rgb")
    parser.add_argument("--critic-threshold", type=float, default=0.8)
    parser.add_argument("--critic-patience", type=int, default=2)
    parser.add_argument(
        "--on-event", help="module:function(event, observation), called after pausing"
    )
    args = parser.parse_args()
    if not args.capture_only:
        if not args.executor_factory and (
            not args.prompt or not (args.graph_provider or args.empty_graph)
        ):
            parser.error(
                "Provide --executor-factory or --prompt with --graph-provider/--empty-graph"
            )
        if not (args.critic_checkpoint or args.no_critic):
            parser.error("Provide --critic-checkpoint or explicitly select --no-critic")
        if args.record_critic_context and not args.executor_factory:
            parser.error("--record-critic-context requires --executor-factory")
    from grounded_vla.backends.isaacsim import IsaacSimConfig, IsaacSimEnvironment

    config = IsaacSimConfig.load(args.config)
    if args.critic_checkpoint and args.critic_camera not in config.camera_paths:
        parser.error("Critic camera is not configured in the scene")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    # Kit must start before importing any Isaac Sim extension (or the torch critic).
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": not args.gui})
    environment, client, recorder, executor, critic_recorder = None, None, None, None, None
    summary = {
        "simulator": "Isaac Sim",
        "config": asdict(config),
        "capture_only": args.capture_only,
    }
    try:
        import numpy as np

        from grounded_vla.backends.openpi import OpenPiTransport
        from grounded_vla.simulation import CanonicalSimPolicy, GraphSnapshot, run_rollout
        from grounded_vla.training.recording import CriticObservationRecorder, EpisodeRecorder

        environment = IsaacSimEnvironment.from_config(
            config,
            action_guard=_callable(args.action_guard) if args.action_guard else None,
        )
        if args.capture_only:
            obs = environment.observe()
            np.savez_compressed(
                out / "observation.npz",
                state=obs.state,
                timestamp=obs.timestamp,
                **{f"images/{k}": v for k, v in obs.images.items()},
            )
            summary["reason"] = "capture_only"
        else:
            client = OpenPiTransport(args.uri, timeout=args.timeout)
            policy = CanonicalSimPolicy(client, config)
            if args.empty_graph:

                def graph_provider(obs):
                    return GraphSnapshot(
                        {
                            "schema_id": policy.meta["graph_schema_id"],
                            "features": np.zeros((0, policy.meta["graph_feature_dim"]), np.float32),
                            "edges": np.zeros((0, 0), np.int64),
                            "valid": np.zeros(0, bool),
                        },
                        obs.timestamp,
                    )
            elif args.graph_provider:
                graph_provider = _callable(args.graph_provider)
            monitor = None
            if args.critic_checkpoint:
                from grounded_vla.learning.temporal_vit import TemporalMonitor, ViTWindowPredictor

                predictor = ViTWindowPredictor(
                    args.critic_checkpoint,
                    device=args.critic_device,
                    expected_camera=args.critic_camera,
                    expected_period=config.control_period_seconds,
                )
                if predictor.requires_context and not args.executor_factory:
                    raise ValueError("A conditioned critic requires --executor-factory")
                monitor = TemporalMonitor(
                    predictor,
                    window_size=predictor.config.max_frames,
                    threshold=args.critic_threshold,
                    patience=args.critic_patience,
                    period=config.control_period_seconds,
                )
            recorder = EpisodeRecorder(policy.meta["graph_schema_id"])
            if args.record_critic_context:
                critic_recorder = CriticObservationRecorder(policy.meta["graph_schema_id"])
            summary.update(
                {
                    "policy_metadata": policy.meta,
                    "graph_disabled": args.empty_graph,
                    "critic_checkpoint": args.critic_checkpoint,
                    "recording_source": "policy_rollout_not_expert_demonstration",
                    "executor_factory": args.executor_factory,
                }
            )
            options = dict(
                max_steps=args.max_steps,
                execute_steps=args.execute_steps,
                monitor=monitor,
                monitor_camera=args.critic_camera,
                recorder=recorder,
                on_event=_callable(args.on_event) if args.on_event else None,
            )
            if args.executor_factory:
                from grounded_vla.execution import BeliefAwareExecutor

                executor = _callable(args.executor_factory)(environment)
                if not isinstance(executor, BeliefAwareExecutor):
                    raise TypeError("Executor factory must return BeliefAwareExecutor")
                result = executor.run(
                    environment,
                    policy,
                    **options,
                    max_replans=args.max_replans,
                    max_plans=args.max_plans,
                    record_critic_context=args.record_critic_context,
                    critic_recorder=critic_recorder,
                )
            else:
                result = run_rollout(environment, policy, graph_provider, args.prompt, **options)
            summary.update(asdict(result))
    except BaseException as exc:
        summary.update({"reason": "error", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        try:
            if environment is not None:
                environment.pause()
            if recorder is not None and recorder.frames:
                recorder.save(out / "episode.npz")
            if executor is not None:
                executor.journal.save(out / "execution-trace.jsonl")
            if critic_recorder is not None and critic_recorder.frames:
                critic_recorder.save(out / "critic-observations.npz")
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                app.close()
    print(f"Wrote Isaac Sim capture/rollout to {out}")


if __name__ == "__main__":
    main()
