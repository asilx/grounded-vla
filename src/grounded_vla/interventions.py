"""Matched interventions on belief state/rules, using the real executive code."""

from __future__ import annotations

from dataclasses import replace

from grounded_vla.belief import Atom, BeliefStore
from grounded_vla.contracts import Executive, TaskRules
from grounded_vla.policy import ScriptedPolicy
from grounded_vla.validation import ToyValidator, select


def fixture() -> BeliefStore:
    belief = BeliefStore()
    for predicate, value in {
        "clean": True,
        "filled": False,
        "held": False,
        "at_tray": False,
    }.items():
        belief.add(Atom(predicate, "cup_a"), value, at=0, source="intervention-fixture")
    return belief


def decision(belief: BeliefStore, rules: TaskRules) -> dict:
    plan = Executive({"cup_a": "the blue cup"}, rules).choose(belief, 0)
    if plan.contract is None:
        return {"contract": "done", "candidate": None}
    observation = {"robot_pose": [0.5, 0.5, 0.5, 0], "positions": {"cup_a": [0.2, 0.3, 0.25]}}
    candidates = ScriptedPolicy().sample(plan.contract, observation, 3)
    verdicts = [ToyValidator().check(c, plan.contract, belief, 0) for c in candidates]
    chosen = select(candidates, verdicts)
    return {
        "contract": plan.contract.key,
        "candidate": chosen.id if chosen else None,
        "grounds": [q.to_dict() for q in plan.contract.grounds],
    }


def run_interventions() -> dict:
    original, rules = fixture(), TaskRules()
    rows = []

    def record(name, before_belief, after_belief, before_rules, after_rules, field, expected):
        before, after = decision(before_belief, before_rules), decision(after_belief, after_rules)
        rows.append(
            {
                "intervention": name,
                "before": before,
                "after": after,
                "measured_field": field,
                "expected": expected,
                "passed": after[field] == expected,
            }
        )

    missing = original.clone()
    missing.retract(Atom("clean", "cup_a"), at=0, reason="counterfactual evidence removal")
    record("remove supporting fact", original, missing, rules, rules, "contract", "inspect:cup_a")
    dirty = original.clone()
    dirty.add(Atom("clean", "cup_a"), False, at=0, source="counterfactual", replace=True)
    record("refute cleanliness", original, dirty, rules, rules, "contract", "clean:cup_a")
    irrelevant = original.clone()
    irrelevant.add(Atom("round", "plate_c"), True, at=0, source="counterfactual")
    record("add irrelevant fact", original, irrelevant, rules, rules, "contract", "pick:cup_a")
    record(
        "remove cleanliness rule",
        dirty,
        dirty,
        rules,
        replace(rules, require_clean=False),
        "contract",
        "pick:cup_a",
    )
    filled = original.clone()
    filled.add(Atom("filled", "cup_a"), True, at=0, source="counterfactual", replace=True)
    record("change fill state", original, filled, rules, rules, "candidate", "level")
    return {
        "scope": "Five deterministic belief/rule interventions; no physical or model-level causal claim",
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "cases": rows,
    }
