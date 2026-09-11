"""Append-only decision/outcome records with a locally verifiable hash chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical(data: dict) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


class Journal:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def append(self, kind: str, **payload) -> dict:
        data = {
            "sequence": len(self.records),
            "kind": kind,
            "previous_hash": self.records[-1]["hash"] if self.records else "0" * 64,
            "payload": payload,
        }
        # Serialize now: later mutation of a payload must not rewrite past evidence.
        data = json.loads(canonical(data))
        data["hash"] = hashlib.sha256(canonical(data)).hexdigest()
        self.records.append(data)
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in self.records
            ),
            encoding="utf-8",
        )


def load_verified(path: Path) -> list[dict]:
    records, previous = [], "0" * 64
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        digest = record.pop("hash")
        if record.get("sequence") != index or record.get("previous_hash") != previous:
            raise ValueError(f"Broken journal sequence at line {index + 1}")
        if hashlib.sha256(canonical(record)).hexdigest() != digest:
            raise ValueError(f"Journal hash mismatch at line {index + 1}")
        record["hash"] = digest
        records.append(record)
        previous = digest
    if not records:
        raise ValueError("Journal is empty")
    return records


def explain(payload: dict) -> str:
    """Render only the recorded controller rationale; never invent a mental trace."""
    contract = payload["contract"]
    parts = [f"Selected {contract['skill']} for {contract['description']}."]
    for fact in contract["grounds"]:
        support = ", ".join(fact["evidence_ids"]) or "none"
        parts.append(f"{fact['fact']}: {fact['truth']} ({fact['reason']}; evidence: {support}).")
    if contract["rule_ids"]:
        parts.append("Active rules: " + ", ".join(contract["rule_ids"]) + ".")
    for verdict in payload["verdicts"]:
        if not verdict["accepted"]:
            parts.append(f"Rejected {verdict['candidate_id']}: {', '.join(verdict['reasons'])}.")
    parts.append(f"Executed candidate: {payload['selected']['id']}.")
    return " ".join(parts)
