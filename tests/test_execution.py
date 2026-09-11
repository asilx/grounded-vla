from dataclasses import replace

import pytest

from grounded_vla.belief import Atom, Truth
from grounded_vla.contracts import Executive
from grounded_vla.interventions import fixture, run_interventions
from grounded_vla.policy import ActionChunk
from grounded_vla.runner import RunConfig, run
from grounded_vla.toy_world import SCENARIOS
from grounded_vla.validation import ToyValidator, select


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grounded_completes_all_synthetic_conditions(scenario, seed):
    result = run(RunConfig(scenario=scenario, seed=seed))
    assert result.metrics["constrained_success"]
    assert not result.metrics["violations"]


def test_retraction_ablation_exposes_stale_world_failure():
    a = run(RunConfig(scenario="disturbance"))
    b = run(RunConfig(scenario="disturbance", method="no_retraction"))
    assert a.metrics["constrained_success"]
    assert not b.metrics["constrained_success"]
    assert "dirty-placement:cup_a" in b.metrics["violations"]


def test_guard_ablation_cannot_count_spilling_as_success():
    result = run(RunConfig(scenario="clean", method="no_semantic_guard"))
    assert result.metrics["goal_complete"]
    assert not result.metrics["constrained_success"]


def test_stale_facts_cancel_unexecuted_commands():
    result = run(RunConfig(scenario="stale"))
    assert result.metrics["cancellations"] >= 1
    cancelled = [
        r["payload"]["decision_id"]
        for r in result.journal.records
        if r["kind"] == "queue_cancelled"
    ]
    for decision_id in cancelled:
        cancellation = next(
            r["sequence"]
            for r in result.journal.records
            if r["kind"] == "queue_cancelled" and r["payload"]["decision_id"] == decision_id
        )
        assert not any(
            r["kind"] == "execution" and r["payload"]["decision_id"] == decision_id
            for r in result.journal.records[cancellation + 1 :]
        )


def test_missing_grasp_effect_is_observed_before_recovery():
    result = run(RunConfig(scenario="grasp_failure"))
    assert result.metrics["recoveries"] == 1
    events = [r["kind"] for r in result.journal.records]
    assert "missing_effect" in events
    missing = next(r for r in result.journal.records if r["kind"] == "missing_effect")
    assert any(
        r["kind"] == "observation"
        and any(e["type"] == "grasp_failed" for e in r["payload"]["observation"]["events"])
        for r in result.journal.records[: missing["sequence"]]
    )


def test_decisions_are_recorded_before_execution_with_real_evidence_ids():
    result = run()
    seen = set()
    for record in result.journal.records:
        payload = record["payload"]
        if record["kind"] == "decision":
            seen.add(payload["decision_id"])
            evidence_ids = {e["id"] for e in payload["belief"]["evidence"]}
            for fact in payload["contract"]["grounds"]:
                assert set(fact["evidence_ids"]) <= evidence_ids
        elif record["kind"] == "execution":
            assert payload["decision_id"] in seen


def test_budget_counts_partial_prefix_and_failed_episode():
    result = run(RunConfig(max_ticks=3, prefix_length=2))
    assert result.metrics["ticks"] == 3
    assert not result.metrics["constrained_success"]
    assert result.metrics["termination"] == "budget_exhausted"


def test_new_fill_information_invalidates_old_contract():
    belief = fixture()
    contract = Executive({"cup_a": "the blue cup"}).choose(belief, 0).contract
    assert contract is not None and contract.valid(belief, 0)
    belief.add(Atom("filled", "cup_a"), True, at=1, source="new sensor", replace=True)
    assert not contract.valid(belief, 1)


def test_hard_constraint_cannot_be_bought_with_a_high_score():
    belief = fixture()
    belief.add(Atom("filled", "cup_a"), True, at=0, source="sensor", replace=True)
    contract = Executive({"cup_a": "the blue cup"}).choose(belief, 0).contract
    good = ActionChunk("good", ((0.5, 0.5, 0.5, 0.0),), -1000, "toy_pose", "test")
    bad = replace(good, id="bad", score=1e9, values=((0.5, 0.5, 0.5, 40.0),))
    candidates = [good, bad]
    verdicts = [ToyValidator().check(c, contract, belief, 0) for c in candidates]
    assert select(candidates, verdicts) == good
    assert select([bad], [verdicts[1]]) is None


@pytest.mark.parametrize(
    "space,values",
    [
        ("robot_joints", ((0.0, 0.0, 0.0, 0.0),)),
        ("toy_pose", ((float("nan"), 0.0, 0.0, 0.0),)),
        ("toy_pose", ()),
    ],
)
def test_malformed_or_real_robot_actions_never_pass_toy_validator(space, values):
    belief = fixture()
    contract = Executive({"cup_a": "the blue cup"}).choose(belief, 0).contract
    verdict = ToyValidator().check(
        ActionChunk("x", values, 0.0, space, "test"), contract, belief, 0
    )
    assert not verdict.accepted


def test_interventions_use_expected_outcomes_and_irrelevant_fact_invariance():
    result = run_interventions()
    assert result["passed"] == result["total"] == 5
    irrelevant = result["cases"][2]
    assert irrelevant["before"] == irrelevant["after"]
    assert result["cases"][0]["before"]["contract"] == "pick:cup_a"
    assert result["cases"][3]["before"]["contract"] == "clean:cup_a"
    assert "clean(cup_a)" not in {fact["fact"] for fact in result["cases"][3]["after"]["grounds"]}
    assert result["cases"][4]["before"]["candidate"] == "tilted"


def test_unknown_is_observed_before_manipulation():
    belief = fixture()
    belief.retract(Atom("clean", "cup_a"), at=0, reason="unknown")
    plan = Executive({"cup_a": "the blue cup"}).choose(belief, 0)
    assert plan.contract.skill == "inspect"
    assert plan.contract.grounds[0].truth == Truth.UNKNOWN
