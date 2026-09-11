"""Small explicit task rules and grounded action contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from grounded_vla.belief import Atom, BeliefStore, QueryResult, Truth


@dataclass(frozen=True)
class TaskRules:
    require_clean: bool = True
    keep_filled_upright: bool = True
    maximum_tilt_degrees: float = 12.0

    def __post_init__(self) -> None:
        if not 0 <= self.maximum_tilt_degrees <= 90:
            raise ValueError("Maximum tilt must be in [0, 90] degrees")


@dataclass(frozen=True)
class Requirement:
    atom: Atom
    truth: Truth

    def check(self, belief: BeliefStore, now: float) -> QueryResult:
        return belief.query(self.atom, now)


@dataclass(frozen=True)
class Contract:
    skill: str
    object_id: str
    description: str
    preconditions: tuple[Requirement, ...] = ()
    effects: tuple[Requirement, ...] = ()
    observes: tuple[Atom, ...] = ()
    maximum_tilt_degrees: float | None = None
    rule_ids: tuple[str, ...] = ()
    grounds: tuple[QueryResult, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.skill}:{self.object_id}"

    def complete(self, belief: BeliefStore, now: float) -> bool:
        return all(r.check(belief, now).truth == r.truth for r in self.effects) and all(
            belief.query(a, now).truth != Truth.UNKNOWN for a in self.observes
        )

    def valid(self, belief: BeliefStore, now: float) -> bool:
        return all(r.check(belief, now).truth == r.truth for r in self.preconditions)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["grounds"] = [q.to_dict() for q in self.grounds]
        return data


@dataclass(frozen=True)
class Plan:
    contract: Contract | None
    considered: tuple[dict, ...] = field(default_factory=tuple)


class Executive:
    """A deliberately small task executive. It only reads the belief ledger.

    Stable object ordering is a transparent tie-break, not a learned success score.
    Descriptions are externally grounded labels, never arbitrary IDs in a prompt.
    """

    def __init__(self, objects: dict[str, str], rules: TaskRules | None = None) -> None:
        if not objects:
            raise ValueError("At least one object is required")
        self.objects = dict(objects)
        self.rules = rules or TaskRules()

    def choose(self, belief: BeliefStore, now: float) -> Plan:
        considered: list[dict] = []
        for obj, description in sorted(self.objects.items()):
            facts = {
                name: belief.query(Atom(name, obj), now)
                for name in ("clean", "filled", "held", "at_tray")
            }
            considered.append({"object": obj, "facts": [q.to_dict() for q in facts.values()]})
            clean_ok = not self.rules.require_clean or facts["clean"].truth == Truth.SUPPORTED
            if facts["at_tray"].truth == Truth.SUPPORTED and clean_ok:
                continue
            unknown = ["at_tray", "held"]
            if self.rules.require_clean:
                unknown.append("clean")
            if self.rules.keep_filled_upright:
                unknown.append("filled")
            missing = tuple(
                Atom(name, obj) for name in unknown if facts[name].truth == Truth.UNKNOWN
            )
            if missing:
                contract = Contract(
                    "inspect",
                    obj,
                    description,
                    observes=missing,
                    rule_ids=("observe-before-acting",),
                    grounds=tuple(facts[a.predicate] for a in missing),
                )
                return Plan(contract, tuple(considered))
            tilt = (
                self.rules.maximum_tilt_degrees
                if (self.rules.keep_filled_upright and facts["filled"].truth == Truth.SUPPORTED)
                else None
            )
            rule_ids = ("keep-filled-upright",) if tilt is not None else ()
            grounds = tuple(facts[name] for name in unknown)
            fill_requirement = (
                (Requirement(Atom("filled", obj), facts["filled"].truth),)
                if self.rules.keep_filled_upright
                else ()
            )
            if not clean_ok:
                contract = Contract(
                    "clean",
                    obj,
                    description,
                    preconditions=(Requirement(Atom("clean", obj), Truth.REFUTED),)
                    + fill_requirement,
                    effects=(Requirement(Atom("clean", obj), Truth.SUPPORTED),),
                    maximum_tilt_degrees=tilt,
                    rule_ids=rule_ids + ("clean-before-serving",),
                    grounds=grounds,
                )
                return Plan(contract, tuple(considered))
            requirements = [Requirement(Atom("at_tray", obj), Truth.REFUTED)]
            requirements.extend(fill_requirement)
            if self.rules.require_clean:
                requirements.append(Requirement(Atom("clean", obj), Truth.SUPPORTED))
                rule_ids += ("clean-before-serving",)
            held = facts["held"].truth == Truth.SUPPORTED
            requirements.append(Requirement(Atom("held", obj), facts["held"].truth))
            skill = "place" if held else "pick"
            effect = Requirement(Atom("at_tray" if held else "held", obj), Truth.SUPPORTED)
            return Plan(
                Contract(
                    skill,
                    obj,
                    description,
                    tuple(requirements),
                    (effect,),
                    maximum_tilt_degrees=tilt,
                    rule_ids=rule_ids,
                    grounds=grounds,
                ),
                tuple(considered),
            )
        return Plan(None, tuple(considered))


def compile_instruction(contract: Contract, *, max_characters: int = 160) -> str:
    """Compile only grounded task language. This is not a tokenizer-budget check.

    An exact serialized token check belongs beside the selected server tokenizer.
    Never silently truncate the instruction or the robot state.
    """
    verbs = {"inspect": "Inspect", "clean": "Clean", "pick": "Pick up", "place": "Place"}
    instruction = f"{verbs[contract.skill]} {contract.description}"
    if contract.skill == "place":
        instruction += " on the serving tray"
    if contract.maximum_tilt_degrees is not None:
        instruction += "; keep it upright"
    instruction += "."
    if len(instruction) > max_characters:
        raise ValueError("Grounded instruction exceeds character budget; improve its description")
    return instruction
