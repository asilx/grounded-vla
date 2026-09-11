"""Command-line entry points for the runnable showcase."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from grounded_vla.evaluation import evaluate
from grounded_vla.interventions import run_interventions
from grounded_vla.journal import load_verified
from grounded_vla.report import write_report
from grounded_vla.runner import METHODS, RunConfig, run
from grounded_vla.toy_world import SCENARIOS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="grounded-vla", description="Intervenable VLA execution showcase"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the offline event-emulator demo")
    demo.add_argument("--config", type=Path)
    demo.add_argument("--scenario", choices=SCENARIOS)
    demo.add_argument("--method", choices=METHODS)
    demo.add_argument("--seed", type=int)
    demo.add_argument("--max-ticks", type=int)
    demo.add_argument("--prefix-length", type=int)
    demo.add_argument("--candidates", type=int)
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))
    demo.add_argument("--knowrob-config", type=Path, help="Use native KnowRob for snapshot queries")
    demo.add_argument("--backend-timeout", type=float, default=10.0)
    benchmark = commands.add_parser("evaluate", help="Run paired synthetic ablations")
    benchmark.add_argument("--seeds", type=int, default=10)
    benchmark.add_argument("--out", type=Path, default=Path("runs/evaluation"))
    intervention = commands.add_parser("intervene", help="Test five belief/rule interventions")
    intervention.add_argument("--out", type=Path, default=Path("runs/interventions.json"))
    explanation = commands.add_parser(
        "explain", help="Verify a journal and print recorded explanations"
    )
    explanation.add_argument("trace", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            for name in ("scenario", "method", "seed", "max_ticks", "prefix_length", "candidates"):
                if getattr(args, name) is not None:
                    config[name] = getattr(args, name)
            context = nullcontext(None)
            if args.knowrob_config:
                from grounded_vla.backends.knowrob import KnowRobSession

                context = KnowRobSession(args.knowrob_config, timeout=args.backend_timeout)
            with context as session:
                result = run(RunConfig(**config), knowrob_session=session)
            args.out.mkdir(parents=True, exist_ok=True)
            result.journal.save(args.out / "trace.jsonl")
            (args.out / "run.json").write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            write_report(result, args.out / "report.html")
            print(json.dumps(result.metrics, indent=2))
            print(f"Open {args.out / 'report.html'} in your browser.")
        elif args.command == "evaluate":
            output = evaluate(args.out, args.seeds)
            print(output["scope"])
            print(f"Completed {output['started_episodes']} episodes. Results: {args.out}")
        elif args.command == "intervene":
            output = run_interventions()
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(
                f"Interventions: {output['passed']}/{output['total']} passed. Results: {args.out}"
            )
            if output["passed"] != output["total"]:
                raise SystemExit(1)
        else:
            records = load_verified(args.trace)
            for record in records:
                if record["kind"] == "decision":
                    print(record["payload"]["decision_id"], record["payload"]["explanation"])
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"grounded-vla: {exc}\n")
