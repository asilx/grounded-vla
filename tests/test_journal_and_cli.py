import json

import pytest

from grounded_vla.cli import main
from grounded_vla.evaluation import evaluate
from grounded_vla.journal import Journal, load_verified
from grounded_vla.report import write_report
from grounded_vla.runner import run


def test_journal_preserves_snapshot_and_detects_edit(tmp_path):
    journal, mutable = Journal(), {"truth": "supported"}
    journal.append("decision", fact=mutable)
    mutable["truth"] = "refuted"
    assert journal.records[0]["payload"]["fact"]["truth"] == "supported"
    path = tmp_path / "trace.jsonl"
    journal.save(path)
    assert len(load_verified(path)) == 1
    path.write_text(path.read_text().replace("supported", "refuted"))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified(path)


def test_report_escapes_embedded_data(tmp_path):
    result = run()
    result.config["scenario"] = '</script><script>alert("x")</script>'
    path = tmp_path / "report.html"
    write_report(result, path)
    html = path.read_text()
    assert '</script><script>alert("x")' not in html
    assert "\\u003c/script\\u003e" in html


def test_cli_demo_and_explanation_from_saved_trace(tmp_path, capsys):
    main(["demo", "--out", str(tmp_path)])
    assert (tmp_path / "report.html").is_file()
    assert load_verified(tmp_path / "trace.jsonl")[-1]["kind"] == "run_finished"
    main(["explain", str(tmp_path / "trace.jsonl")])
    assert "Selected" in capsys.readouterr().out


def test_config_and_cli_override(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"scenario": "clean", "seed": 2}))
    main(["demo", "--config", str(config), "--seed", "3", "--out", str(tmp_path / "out")])
    data = json.loads((tmp_path / "out/run.json").read_text())
    assert data["config"]["scenario"] == "clean" and data["config"]["seed"] == 3


def test_evaluation_counts_all_started_paired_episodes(tmp_path):
    results = evaluate(tmp_path, seeds=2)
    assert results["started_episodes"] == 6 * 3 * 2
    keys = {(r["scenario"], r["seed"], r["method"]) for r in results["episodes"]}
    assert len(keys) == 36
    assert any(not r["success"] for r in results["episodes"])
