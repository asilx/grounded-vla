"""Hard checks precede ranking. Toy geometry never validates robot joint motion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from grounded_vla.belief import BeliefStore
from grounded_vla.contracts import Contract
from grounded_vla.policy import ActionChunk


@dataclass(frozen=True)
class Verdict:
    candidate_id: str
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class ToyValidator:
    def __init__(self, *, semantic_guard: bool = True) -> None:
        self.semantic_guard = semantic_guard

    def check(
        self, chunk: ActionChunk, contract: Contract, belief: BeliefStore, now: float
    ) -> Verdict:
        reasons = []
        if chunk.space != "toy_pose":
            return Verdict(chunk.id, False, ("unsupported-action-space",))
        if not chunk.values or not math.isfinite(chunk.score):
            return Verdict(chunk.id, False, ("empty-chunk-or-nonfinite-score",))
        for point in chunk.values:
            if len(point) != 4 or not all(math.isfinite(v) for v in point):
                return Verdict(chunk.id, False, ("malformed-or-nonfinite-action",))
            if not all(0 <= x <= 1 for x in point[:3]):
                reasons.append("workspace-bounds")
            if abs(point[3]) > 180:
                reasons.append("invalid-orientation")
            if (
                self.semantic_guard
                and contract.maximum_tilt_degrees is not None
                and (abs(point[3]) > contract.maximum_tilt_degrees)
            ):
                reasons.append("keep-filled-upright")
        if not contract.valid(belief, now):
            reasons.append("precondition-not-established")
        return Verdict(chunk.id, not reasons, tuple(sorted(set(reasons))))


def select(candidates: list[ActionChunk], verdicts: list[Verdict]) -> ActionChunk | None:
    if len({c.id for c in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    if [c.id for c in candidates] != [v.candidate_id for v in verdicts]:
        raise ValueError("Verdicts must correspond to candidates in order")
    accepted = [c for c, v in zip(candidates, verdicts, strict=True) if v.accepted]
    return max(accepted, key=lambda c: (c.score, c.id)) if accepted else None
