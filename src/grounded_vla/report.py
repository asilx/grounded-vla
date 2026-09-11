"""Portable offline report. All displayed evidence comes from the run journal."""

import json
from importlib.resources import files
from pathlib import Path

from grounded_vla.runner import RunResult


def write_report(result: RunResult, path: Path) -> None:
    template = files("grounded_vla").joinpath("data/report.html").read_text(encoding="utf-8")
    # Escape script-closing sequences and HTML, including attacker-controlled labels.
    payload = json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__RUN_DATA__", payload), encoding="utf-8")
