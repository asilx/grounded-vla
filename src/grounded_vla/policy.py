"""Action proposal interfaces and the explicitly scripted demo policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from grounded_vla.contracts import Contract


@dataclass(frozen=True)
class ActionChunk:
    id: str
    values: tuple[tuple[float, ...], ...]
    score: float
    space: str
    policy: str

    def to_dict(self) -> dict:
        return asdict(self)


class Policy(Protocol):
    def sample(self, contract: Contract, observation: dict, count: int) -> list[ActionChunk]: ...


class ScriptedPolicy:
    """Synthetic candidates in [x, y, z, tilt_degrees], NOT π0.5 inference.

    Hand-set scores deliberately favor a tilted path and an invalid workspace path
    so that the effect of hard filtering is visible. No learned quality is claimed.
    """

    name = "scripted-toy-policy"

    def sample(self, contract: Contract, observation: dict, count: int = 3) -> list[ActionChunk]:
        if not 1 <= count <= 3:
            raise ValueError("The scripted policy supports 1-3 distinct candidates")
        start = observation["robot_pose"][:3]
        target = observation["positions"][contract.object_id]
        if contract.skill == "place":
            target = [0.8, 0.5, 0.25]
        elif contract.skill == "clean":
            target = [0.25, 0.8, 0.25]
        elif contract.skill == "inspect":
            target = start
        candidates = []
        for label, tilt, score in [
            ("level", 0.0, 0.8),
            ("tilted", 35.0, 0.9),
            ("out-of-workspace", 0.0, 1.0),
        ][:count]:
            points = []
            for step in range(1, 5):
                point = [float(a + (b - a) * step / 4) for a, b in zip(start, target, strict=True)]
                if label == "out-of-workspace":
                    point[0] = 1.2
                points.append(tuple(point + [tilt]))
            candidates.append(ActionChunk(label, tuple(points), score, "toy_pose", self.name))
        return candidates
