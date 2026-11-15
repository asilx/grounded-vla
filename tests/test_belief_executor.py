"""Closed-loop behavioral tests; no Isaac or learned robot performance claims."""

from types import SimpleNamespace

import numpy as np
import pytest

from grounded_vla.belief import Atom, BeliefStore, Truth
from grounded_vla.contracts import Contract, Requirement
from grounded_vla.execution import (
    BeliefAwareExecutor,
    ExecutionPlan,
    ExecutionView,
    PositionDependency,
    RelationDependency,
)
from grounded_vla.journal import load_verified
from grounded_vla.simulation import CanonicalSimPolicy, GraphSnapshot


class Environment:
    def __init__(self):
        self.time = 0.0
        self.paused = False
        self.commands = []
        self.on_step = lambda: None
        self.resumes = 0

    def observe(self):
        return SimpleNamespace(
            timestamp=self.time,
            state=np.zeros(1, np.float32),
            images={"base_0_rgb": np.zeros((224, 224, 3), np.uint8)},
        )

    def step(self, action):
        assert not self.paused
        self.commands.append(float(action[0]))
        self.time = round(self.time + 0.1, 8)
        self.on_step()
        return self.observe()

    def pause(self):
        self.paused = True

    def resume(self):
        assert self.paused
        self.paused = False
        self.resumes += 1


@pytest.fixture
def setup():
    env = Environment()
    store = BeliefStore()
    store.add(Atom("held", "cargo"), False, at=0, source="sensor", ttl=10)
    cfg = SimpleNamespace(
        joint_names=("q",), control_period_seconds=0.1, action_convention="absolute"
    )

    class Client:
        metadata = dict(
            input_preset="canonical",
            state_dim=1,
            action_dim=1,
            action_convention="absolute",
            graph_required=True,
            graph_feature_dim=3,
            graph_relation_types=2,
            action_horizon=3,
            graph_schema_id="test-v1",
            control_period_seconds=0.1,
        )
        calls = 0

        def infer(self, payload):
            start = self.calls * 3 + 1
            self.calls += 1
            return {"actions": np.arange(start, start + 3, dtype=np.float32)[:, None]}

    client = Client()
    policy = CanonicalSimPolicy(client, cfg)
    graph = dict(
        schema_id="test-v1",
        features=np.zeros((3, 3), np.float32),
        edges=np.eye(3, dtype=np.int64),
        valid=np.ones(3, bool),
    )
    ids = ["cargo", "gripper", "distractor"]

    def provider(obs):
        return ExecutionView(GraphSnapshot(graph, obs.timestamp, tuple(ids)), store)

    contract = Contract(
        "pick",
        "cargo",
        "the cargo",
        (Requirement(Atom("held", "cargo"), Truth.REFUTED),),
        (Requirement(Atom("held", "cargo"), Truth.SUPPORTED),),
    )
    plan = ExecutionPlan(
        contract,
        "Pick up the cargo.",
        positions=(PositionDependency("cargo"),),
        effect_timeout_steps=10,
    )
    events = []

    def planner(view, obs, event):
        assert env.paused
        if event:
            events.append(event)
        if contract.complete(view.belief, obs.timestamp):
            return None
        return plan

    return SimpleNamespace(
        env=env,
        store=store,
        policy=policy,
        client=client,
        graph=graph,
        ids=ids,
        provider=provider,
        contract=contract,
        plan=plan,
        planner=planner,
        events=events,
    )


def test_target_change_discards_old_suffix_and_explains_evidence(setup, tmp_path):
    s = setup
    s.env.on_step = lambda: s.graph["features"].__setitem__((0, 0), 0.3)
    executive = BeliefAwareExecutor(s.provider, s.planner)
    result = executive.run(s.env, s.policy, max_steps=3, execute_steps=3)
    assert s.env.commands == [1, 4, 5]
    assert result.replans == 1 and s.client.calls == 2
    event = result.events[0]
    assert event["discarded_actions"] == 2
    assert event["details"]["reason"] == "position_changed"
    assert event["previous_facts"][0]["evidence_ids"] == ["e00001"]
    path = tmp_path / "trace.jsonl"
    executive.journal.save(path)
    assert load_verified(path)[-1]["payload"]["steps"] == 3
    assert s.env.paused and s.env.resumes == 2


@pytest.mark.parametrize("change", ["irrelevant", "permutation", "same_fact_new_evidence"])
def test_irrelevant_updates_and_node_permutations_keep_chunk(setup, change):
    s = setup

    def update():
        if change == "irrelevant":
            s.graph["features"][2] += 100
        elif change == "same_fact_new_evidence":
            s.store.add(Atom("held", "cargo"), False, at=s.env.time, source="sensor", replace=True)
        elif len(s.env.commands) == 1:
            order = [2, 0, 1]
            s.graph["features"] = s.graph["features"][order]
            s.graph["edges"] = s.graph["edges"][order][:, order]
            s.graph["valid"] = s.graph["valid"][order]
            s.ids[:] = [s.ids[i] for i in order]

    s.env.on_step = update
    result = BeliefAwareExecutor(s.provider, s.planner).run(
        s.env, s.policy, max_steps=3, execute_steps=3
    )
    assert s.env.commands == [1, 2, 3] and result.replans == 0


def test_expected_relative_motion_does_not_cancel_transport(setup):
    s = setup
    plan = ExecutionPlan(
        s.contract, "Pick up cargo", positions=(PositionDependency("cargo", relative_to="gripper"),)
    )
    s.env.on_step = lambda: s.graph["features"].__setitem__(slice(0, 2), s.env.time)
    result = BeliefAwareExecutor(s.provider, lambda *args: plan).run(
        s.env, s.policy, max_steps=3, execute_steps=3
    )
    assert result.replans == 0 and s.env.commands == [1, 2, 3]


@pytest.mark.parametrize("cause", ["expiry", "conflict", "retraction"])
def test_evidence_becomes_unusable_before_next_action(setup, cause):
    s = setup
    if cause == "expiry":
        s.store.add(Atom("held", "cargo"), False, at=0, source="short", ttl=0.05, replace=True)

    def update():
        if cause == "conflict":
            s.store.add(Atom("held", "cargo"), True, at=s.env.time, source="other-sensor")
        elif cause == "retraction":
            s.store.retract(Atom("held", "cargo"), at=s.env.time, reason="lost tracking")

    s.env.on_step = update
    initial_revision = s.store.revision
    result = BeliefAwareExecutor(s.provider, s.planner).run(
        s.env, s.policy, max_steps=3, execute_steps=3, max_replans=0
    )
    assert s.env.commands == [1] and result.reason == "replan_budget_exhausted"
    assert result.events[0]["details"]["reason"] == "precondition_invalid"
    if cause == "expiry":
        assert s.store.revision == initial_revision  # Expiry needs no mutation to invalidate.


def test_relation_binding_change_cancels_before_submission(setup):
    s = setup
    plan = ExecutionPlan(
        s.contract, "Pick cargo", relations=(RelationDependency("cargo", "gripper"),)
    )
    s.env.on_step = lambda: s.graph["edges"].__setitem__((0, 1), 1)
    result = BeliefAwareExecutor(s.provider, lambda *a: plan).run(
        s.env, s.policy, max_steps=2, execute_steps=3
    )
    assert s.env.commands == [1, 4]
    assert result.events[0]["details"]["reason"] == "relation_changed"


def test_observed_effect_wins_over_changed_precondition_and_discards_suffix(setup):
    s = setup
    s.env.on_step = lambda: s.store.add(
        Atom("held", "cargo"), True, at=s.env.time, source="contact-sensor", replace=True
    )
    result = BeliefAwareExecutor(s.provider, s.planner).run(
        s.env, s.policy, max_steps=3, execute_steps=3
    )
    assert s.env.commands == [1]
    assert result.completed_contracts == 1 and result.replans == 0
    assert result.reason == "no_plan"


def test_predicted_effect_never_completes_and_recovery_is_bounded(setup):
    s = setup
    s.store.add(Atom("held", "cargo"), True, at=0, source="model", kind="predicted")
    plan = ExecutionPlan(s.contract, "Pick cargo", effect_timeout_steps=2)
    callbacks = []
    result = BeliefAwareExecutor(s.provider, lambda *a: plan).run(
        s.env,
        s.policy,
        max_steps=20,
        execute_steps=3,
        max_replans=1,
        on_event=lambda event, obs: callbacks.append(s.env.paused),
    )
    assert result.steps == 4 and result.completed_contracts == 0
    assert result.reason == "replan_budget_exhausted" and all(callbacks)
    assert s.store.query(Atom("held", "cargo"), s.env.time).truth == Truth.REFUTED


def test_chunk_age_is_bound_to_inference_not_refreshed_graph(setup):
    s = setup
    plan = ExecutionPlan(s.contract, "Pick cargo", max_chunk_age_seconds=0.05)
    result = BeliefAwareExecutor(s.provider, lambda *a: plan).run(
        s.env, s.policy, max_steps=2, execute_steps=3
    )
    assert result.events[0]["details"]["reason"] == "chunk_expired"
    assert s.env.commands == [1, 4]


def test_old_entity_geometry_cannot_be_hidden_by_new_graph_timestamp(setup):
    s = setup

    def provider(obs):
        return ExecutionView(
            GraphSnapshot(s.graph, obs.timestamp, tuple(s.ids), (0.0, 0.0, 0.0)), s.store
        )

    plan = ExecutionPlan(
        s.contract, "Pick cargo", positions=(PositionDependency("cargo", max_age_seconds=0.05),)
    )
    result = BeliefAwareExecutor(provider, lambda *a: plan).run(
        s.env, s.policy, max_steps=3, execute_steps=3, max_replans=0
    )
    assert s.env.commands == [1]
    assert result.events[0]["details"]["reason"] == "dependency_unavailable"


def test_policy_failure_pauses_and_journals_error(setup):
    s = setup

    def fail(*args):
        raise TimeoutError("policy deadline")

    s.policy.infer = fail
    executive = BeliefAwareExecutor(s.provider, s.planner)
    with pytest.raises(TimeoutError):
        executive.run(s.env, s.policy)
    assert s.env.paused and not s.env.commands
    assert executive.journal.records[-1]["payload"]["reason"] == "error"


def test_planner_cannot_loop_forever_without_executing(setup):
    s = setup
    s.store.add(Atom("held", "cargo"), True, at=0, source="contact", replace=True)
    result = BeliefAwareExecutor(s.provider, lambda *a: s.plan).run(s.env, s.policy, max_plans=2)
    assert result.reason == "plan_budget_exhausted" and result.plans == 2
    assert not s.env.commands


def test_conditioned_monitor_and_recording_share_causal_contract_context(setup, tmp_path):
    from grounded_vla.training.recording import EpisodeRecorder

    s = setup

    class Monitor:
        requires_context = True
        signalled = False

        def reset(self):
            pass

        def update(self, image, timestamp, *, context):
            assert context["timestamp"] == timestamp
            assert context["target_mask"].tolist() == [True, False, False]
            if timestamp > 0 and not self.signalled:
                self.signalled = True
                return {"reason": "stalled"}

    recorder = EpisodeRecorder("test-v1")
    result = BeliefAwareExecutor(s.provider, s.planner).run(
        s.env,
        s.policy,
        max_steps=2,
        execute_steps=3,
        monitor=Monitor(),
        recorder=recorder,
        record_critic_context=True,
    )
    assert s.env.commands == [1, 4] and result.replans == 1
    assert result.events[0]["details"]["reason"] == "critic_pause"
    assert s.store.query(Atom("held", "cargo"), s.env.time).truth == Truth.REFUTED
    path = tmp_path / "episode.npz"
    recorder.save(path)
    with np.load(path, allow_pickle=False) as data:
        assert data["critic_context/contract"].shape[0] == 2
        np.testing.assert_array_equal(data["timestamps"], [0.0, 0.1])
        assert data["critic_context/target_mask"].all(axis=0).tolist() == [True, False, False]


def test_terminal_critic_frame_is_recorded_without_fabricating_an_action(setup, tmp_path):
    from grounded_vla.training.recording import CriticObservationRecorder

    s = setup
    s.env.on_step = lambda: s.store.add(
        Atom("held", "cargo"), True, at=s.env.time, source="contact", replace=True
    )
    recorder = CriticObservationRecorder("test-v1")
    result = BeliefAwareExecutor(s.provider, s.planner).run(
        s.env, s.policy, execute_steps=3, critic_recorder=recorder
    )
    assert result.completed_contracts == 1 and s.env.commands == [1]
    path = tmp_path / "critic-observations.npz"
    recorder.save(path)
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["timestamps"], [0.0, 0.1])
        assert "actions" not in data and "critic_labels" not in data
