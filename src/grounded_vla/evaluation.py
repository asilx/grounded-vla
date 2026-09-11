"""Paired synthetic mechanism checks. These are not policy benchmark baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from grounded_vla.runner import METHODS, RunConfig, run
from grounded_vla.toy_world import SCENARIOS


def evaluate(out: Path, seeds: int = 10) -> dict:
    if not 1 <= seeds <= 1000:
        raise ValueError("Use between 1 and 1000 seeds")
    episodes = []
    for scenario in SCENARIOS:
        for seed in range(seeds):
            for method in METHODS:
                result = run(RunConfig(scenario=scenario, seed=seed, method=method))
                episodes.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "success": result.metrics["constrained_success"],
                        "ticks": result.metrics["ticks"],
                        "violations": len(result.metrics["violations"]),
                        "recoveries": result.metrics["recoveries"],
                        "termination": result.metrics["termination"],
                    }
                )
    summary = []
    for scenario in SCENARIOS:
        for method in METHODS:
            group = [r for r in episodes if r["scenario"] == scenario and r["method"] == method]
            summary.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "episodes": len(group),
                    "success_rate": mean(r["success"] for r in group),
                    "mean_ticks": mean(r["ticks"] for r in group),
                }
            )
    output = {
        "scope": "Synthetic event-emulator ablations; not π0.5/KnowRob performance evidence",
        "seeds_per_condition": seeds,
        "started_episodes": len(episodes),
        "summary": summary,
        "episodes": episodes,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    return output
