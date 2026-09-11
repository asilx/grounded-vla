"""Explicit evidence, three-valued queries, expiry, conflicts, and retraction.

This is the reference evidence ledger, not an implementation of KnowRob or OWL.
Predictions never satisfy an observed-state precondition.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from enum import StrEnum


class Truth(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


@dataclass(frozen=True, order=True)
class Atom:
    predicate: str
    subject: str

    @property
    def key(self) -> str:
        return f"{self.predicate}({self.subject})"


@dataclass(frozen=True)
class Evidence:
    id: str
    atom: Atom
    value: bool
    observed_at: float
    source: str
    expires_at: float | None = None
    kind: str = "observed"
    confidence: float = 1.0


@dataclass(frozen=True)
class QueryResult:
    atom: Atom
    truth: Truth
    reason: str
    evidence_ids: tuple[str, ...]
    ignored_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "fact": self.atom.key,
            "truth": self.truth.value,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "ignored_ids": list(self.ignored_ids),
        }


class BeliefStore:
    def __init__(self, *, minimum_confidence: float = 0.8) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0, 1]")
        self.minimum_confidence = minimum_confidence
        self.revision = 0
        self._next_id = 1
        self._active: dict[Atom, list[Evidence]] = {}
        self.archive: dict[str, Evidence] = {}
        self.events: list[dict] = []

    def add(
        self,
        atom: Atom,
        value: bool,
        *,
        at: float,
        source: str,
        ttl: float | None = None,
        kind: str = "observed",
        confidence: float = 1.0,
        replace: bool = False,
    ) -> Evidence:
        if type(value) is not bool:
            raise TypeError("Evidence values must be explicit booleans")
        if not math.isfinite(at) or at < 0:
            raise ValueError("Evidence time must be finite and non-negative")
        if ttl is not None and (not math.isfinite(ttl) or ttl <= 0):
            raise ValueError("TTL must be finite and positive")
        if kind not in {"observed", "inferred", "predicted"}:
            raise ValueError("Unsupported evidence kind")
        if not 0 <= confidence <= 1 or not source:
            raise ValueError("A source and a confidence in [0, 1] are required")
        if replace:
            self.retract(atom, at=at, reason=f"superseded by {source}")
        evidence = Evidence(
            f"e{self._next_id:05d}",
            atom,
            value,
            at,
            source,
            None if ttl is None else at + ttl,
            kind,
            confidence,
        )
        self._next_id += 1
        self._active.setdefault(atom, []).append(evidence)
        self.archive[evidence.id] = evidence
        self.revision += 1
        self.events.append(
            {"kind": "assert", "revision": self.revision, "evidence": asdict(evidence)}
        )
        return evidence

    def retract(self, atom: Atom, *, at: float, reason: str) -> tuple[str, ...]:
        removed = tuple(e.id for e in self._active.pop(atom, []))
        if removed:
            self.revision += 1
            self.events.append(
                {
                    "kind": "retract",
                    "revision": self.revision,
                    "at": at,
                    "fact": atom.key,
                    "evidence_ids": removed,
                    "reason": reason,
                }
            )
        return removed

    def query(self, atom: Atom, now: float) -> QueryResult:
        if not math.isfinite(now) or now < 0:
            raise ValueError("Query time must be finite and non-negative")
        all_evidence = self._active.get(atom, [])
        usable, ignored = [], []
        for evidence in all_evidence:
            fresh = evidence.observed_at <= now and (
                evidence.expires_at is None or now < evidence.expires_at
            )
            reliable = evidence.confidence >= self.minimum_confidence
            if fresh and reliable and evidence.kind != "predicted":
                usable.append(evidence)
            else:
                ignored.append(evidence.id)
        polarities = {e.value for e in usable}
        if len(polarities) == 2:
            truth, reason = Truth.UNKNOWN, "conflicting evidence"
        elif polarities == {True}:
            truth, reason = Truth.SUPPORTED, "fresh supporting evidence"
        elif polarities == {False}:
            truth, reason = Truth.REFUTED, "fresh refuting evidence"
        else:
            truth = Truth.UNKNOWN
            reason = (
                "no usable evidence (stale, predicted, future, or low confidence)"
                if all_evidence
                else "no evidence"
            )
        return QueryResult(atom, truth, reason, tuple(e.id for e in usable), tuple(ignored))

    def snapshot(self, now: float) -> dict:
        return {
            "revision": self.revision,
            "at": now,
            "facts": [self.query(a, now).to_dict() for a in sorted(self._active)],
            "evidence": [asdict(e) for a in sorted(self._active) for e in self._active[a]],
        }

    def clone(self) -> BeliefStore:
        return copy.deepcopy(self)
