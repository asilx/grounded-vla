"""Raw-measurement and composed executor tests; no GPU task metrics."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from grounded_vla.belief import Atom, Truth
from grounded_vla.benchmarks.isaac_pick_place import PickPlaceEvaluator, aggregate
from grounded_vla.contracts import Requirement
from grounded_vla.execution import ChunkBinding, PositionDependency
from grounded_vla.grounding.pick_place import PickPlaceGrounder
from grounded_vla.journal import load_verified
from grounded_vla.perception.isaac_scene import IsaacSceneReader, PickPlaceConfig, SceneSample
from grounded_vla.simulation import CanonicalSimPolicy
from grounded_vla.tasks.pick_place import build_pick_place


def sample(
    t,
    *,
    cargo=(0.4, 0, 0.02),
    gripper=(0.4, 0, 0.2),
    goal=(0.6, 0, 0.02),
    width=0.08,
    forces=(0, 0),
    support=1,
):
    return SceneSample(
        t,
        cargo,
        gripper,
        goal,
        (1, 0, 0, 0),
        (width / 2,) * 2,
        forces,
        support,
        source="cpu_measurement_fixture",
    )


def trajectory():
    return [
        sample(0),
        sample(0.1, gripper=(0.4, 0, 0.03), width=0.04, forces=(2, 2)),
        sample(
            0.2, cargo=(0.4, 0, 0.06), gripper=(0.4, 0, 0.07), width=0.04, forces=(2, 2), support=0
        ),
        sample(
            0.3, cargo=(0.4, 0, 0.09), gripper=(0.4, 0, 0.10), width=0.04, forces=(2, 2), support=0
        ),
        sample(
            0.4, cargo=(0.6, 0, 0.09), gripper=(0.6, 0, 0.10), width=0.04, forces=(2, 2), support=0
        ),
        sample(0.5, cargo=(0.6, 0, 0.02), gripper=(0.6, 0, 0.15)),
        sample(0.6, cargo=(0.6, 0, 0.02), gripper=(0.6, 0, 0.15)),
        sample(0.7, cargo=(0.6, 0, 0.02), gripper=(0.6, 0, 0.15)),
        sample(0.8, cargo=(0.6, 0, 0.02), gripper=(0.6, 0, 0.15)),
    ]


@pytest.fixture
def config():
    return PickPlaceConfig(
        grasp_confirm_seconds=0.1,
        release_confirm_seconds=0.2,
        effect_timeout_steps=6,
        max_sample_gap=0.11,
    )


def truth(grounder, name):
    return grounder.belief.query(Atom(name, "cargo"), grounder.latest.timestamp).truth


def test_acquisition_release_and_stability_are_separate_observations(config):
    grounder, evaluator = PickPlaceGrounder(config), PickPlaceEvaluator(config)
    rows = trajectory()
    for i, row in enumerate(rows):
        grounder.ingest(row)
        evaluator.update(row)
        if i == 1:
            assert truth(grounder, "held") != Truth.SUPPORTED  # Contact is insufficient.
        if i == 2:
            assert truth(grounder, "held") == Truth.SUPPORTED
        if i == 5:
            assert truth(grounder, "held") == Truth.REFUTED
            assert truth(grounder, "at_tray") == Truth.UNKNOWN
            assert not evaluator.success
    assert truth(grounder, "at_tray") == Truth.SUPPORTED
    assert evaluator.success and evaluator.result()["verified_grasp"]
    assert "clean" not in {e.atom.predicate for e in grounder.belief.archive.values()}


@pytest.mark.parametrize("fault", ["closed_empty", "one_finger", "no_lift", "no_comotion"])
def test_commands_or_partial_contact_never_prove_grasp(config, fault):
    g = PickPlaceGrounder(config)
    for row in trajectory()[:4]:
        if fault == "closed_empty":
            row = replace(row, finger_positions=(0, 0), finger_forces=(0, 0))
        elif fault == "one_finger":
            row = replace(row, finger_forces=(2, 0))
        elif fault == "no_lift":
            row = replace(row, object_position=(0.4, 0, 0.02), gripper_position=(0.4, 0, 0.03))
        elif fault == "no_comotion" and row.timestamp >= 0.2:
            row = replace(row, gripper_position=(0.4, 0, 0.03))
        g.ingest(row)
    assert truth(g, "held") != Truth.SUPPORTED


def test_slip_revokes_held_evidence(config):
    g = PickPlaceGrounder(config)
    for row in trajectory()[:4]:
        g.ingest(row)
    old = g.belief.query(Atom("held", "cargo"), 0.3).evidence_ids
    g.ingest(sample(0.4, cargo=(0.5, 0, 0.02), gripper=(0.4, 0, 0.10), width=0.04, forces=(0, 0)))
    assert truth(g, "held") == Truth.REFUTED
    assert all(key in g.belief.archive for key in old)
    assert g.belief.events[-1]["kind"] == "assert"


@pytest.mark.parametrize("change", ["no_support", "closed", "far", "moving_goal"])
def test_near_goal_is_not_sufficient_for_success(config, change):
    g, e = PickPlaceGrounder(config), PickPlaceEvaluator(config)
    for row in trajectory():
        if row.timestamp >= 0.5:
            if change == "no_support":
                row = replace(row, support_force=0)
            elif change == "closed":
                row = replace(row, finger_positions=(0.02, 0.02), finger_forces=(2, 2))
            elif change == "far":
                row = replace(row, goal_position=(1, 0, 0.02))
            else:
                row = replace(
                    row, goal_position=(0.6 + 0.02 * (row.timestamp - 0.5) / 0.1, 0, 0.02)
                )
        g.ingest(row)
        e.update(row)
    assert truth(g, "at_tray") != Truth.SUPPORTED
    assert not e.success


def test_no_grasp_no_task_success_even_if_object_starts_at_goal(config):
    e, g = PickPlaceEvaluator(config), PickPlaceGrounder(config)
    for t in np.arange(10) * 0.1:
        s = sample(float(t), cargo=(0.6, 0, 0.02))
        e.update(s)
        g.ingest(s)
    assert not e.success and truth(g, "at_tray") != Truth.SUPPORTED


def test_duplicate_sample_is_idempotent_and_does_not_refresh_ttl(config):
    g = PickPlaceGrounder(config)
    s = sample(0)
    v = g.ingest(s)
    revision = g.belief.revision
    assert g.ingest(s) is v and g.belief.revision == revision
    assert len(g.records) == 1
    assert g.belief.query(Atom("held", "cargo"), config.evidence_ttl).truth == Truth.UNKNOWN
    with pytest.raises(ValueError, match="Conflicting"):
        g.ingest(replace(s, object_position=(0.5, 0, 0.02)))


def test_gaps_restart_confirmation_and_stale_samples_fail(config):
    g = PickPlaceGrounder(config)
    for s in trajectory()[:4]:
        g.ingest(s)
    g.ingest(replace(trajectory()[3], timestamp=1.0))
    assert truth(g, "held") != Truth.SUPPORTED
    with pytest.raises(ValueError, match="increase"):
        g.ingest(sample(0.4))


class ReplayEnvironment:
    def __init__(self, rows):
        self.rows, self.index, self.paused, self.commands = rows, 0, True, []

    def observe(self):
        return SimpleNamespace(
            timestamp=self.rows[self.index].timestamp,
            state=np.zeros(9, np.float32),
            images={"base_0_rgb": np.zeros((224, 224, 3), np.uint8)},
        )

    def pause(self):
        self.paused = True

    def resume(self):
        assert self.paused
        self.paused = False

    def step(self, action):
        assert not self.paused
        self.commands.append(action.copy())
        self.index += 1
        return self.observe()


class ReplayReader:
    def __init__(self, environment):
        self.environment, self.calls = environment, 0

    def read(self, observation):
        self.calls += 1
        return self.environment.rows[self.environment.index]


class Client:
    def __init__(self):
        self.metadata = dict(
            input_preset="canonical",
            state_dim=9,
            action_dim=9,
            action_convention="absolute-test",
            control_period_seconds=0.1,
            graph_required=True,
            graph_feature_dim=26,
            graph_relation_types=8,
            graph_schema_id="grounded-vla-belief-v1",
            action_horizon=3,
        )
        self.prompts = []

    def infer(self, payload):
        self.prompts.append(payload["prompt"])
        return {"actions": np.full((3, 9), len(self.prompts), np.float32)}


def components(config, rows):
    env = ReplayEnvironment(rows)
    client = Client()
    policy = CanonicalSimPolicy(
        client,
        SimpleNamespace(
            joint_names=tuple(f"q{i}" for i in range(9)),
            control_period_seconds=0.1,
            action_convention="absolute-test",
        ),
    )
    evaluator = PickPlaceEvaluator(config)
    reader = ReplayReader(env)
    executor, routed, grounder, planner = build_pick_place(
        config,
        reader,
        policy,
        evaluator=evaluator,
    )
    return SimpleNamespace(
        env=env,
        client=client,
        reader=reader,
        policy=routed,
        executor=executor,
        grounder=grounder,
        planner=planner,
        evaluator=evaluator,
    )


def test_composed_task_completes_only_after_observed_stable_release(config, tmp_path):
    s = components(config, trajectory())
    result = s.executor.run(s.env, s.policy, max_steps=8, execute_steps=3)
    assert result.completed_contracts == 3  # Pick, release, then stable observation.
    # Step budget can be reached exactly at verification; physical score remains valid.
    assert s.evaluator.success and s.env.paused
    assert s.planner.pick_attempts == 1
    assert any("Place" in p for p in s.client.prompts)
    assert not any("Hold measured" in p for p in s.client.prompts)
    assert np.array_equal(s.env.commands[-1], np.zeros(9))  # Genuine measured hold.
    assert s.reader.calls == 9  # Paused replanning did not fabricate frames.
    path = tmp_path / "trace.jsonl"
    s.executor.journal.save(path)
    assert load_verified(path)[-1]["kind"] == "execution_stopped"


def test_target_shift_cancels_suffix_before_next_command(config):
    rows = [sample(0), sample(0.1, cargo=(0.5, 0, 0.02)), sample(0.2, cargo=(0.5, 0, 0.02))]
    s = components(config, rows)
    result = s.executor.run(s.env, s.policy, max_steps=2, execute_steps=3)
    assert [a[0] for a in s.env.commands] == [1, 2]
    assert result.replans == 1
    assert result.events[0]["details"]["reason"] == "position_changed"


def test_empty_grasp_retries_are_bounded_and_never_count_as_success(config):
    config = replace(config, effect_timeout_steps=1, max_pick_attempts=2)
    s = components(config, [sample(t / 10, width=0) for t in range(5)])
    result = s.executor.run(s.env, s.policy, max_steps=4)
    assert s.planner.stop_reason == "pick_attempt_budget_exhausted"
    assert result.steps == 2 and result.completed_contracts == 0
    assert not s.evaluator.success and s.env.paused


def test_drop_selects_new_pick_and_does_not_count_release_as_task_success(config):
    rows = trajectory()[:4] + [
        sample(0.4, cargo=(0.5, 0, 0.02), gripper=(0.4, 0, 0.2), width=0.04),
        sample(0.5, cargo=(0.5, 0, 0.02)),
    ]
    s = components(config, rows)
    result = s.executor.run(s.env, s.policy, max_steps=5)
    assert s.planner.pick_attempts == 2
    assert result.completed_contracts >= 2  # Release/loss was observed, not a placed goal.
    assert not s.evaluator.success


def test_uncertain_contact_holds_instead_of_reaching_with_unverified_object(config):
    rows = trajectory()[:4] + [
        sample(
            0.4, cargo=(0.4, 0, 0.09), gripper=(0.4, 0, 0.10), width=0.04, forces=(0, 0), support=0
        ),
        sample(
            0.5, cargo=(0.4, 0, 0.09), gripper=(0.4, 0, 0.10), width=0.04, forces=(0, 0), support=0
        ),
    ]
    s = components(config, rows)
    result = s.executor.run(s.env, s.policy, max_steps=5)
    assert result.events[0]["details"]["reason"] == "precondition_invalid"
    assert s.planner.current.skill == "inspect"
    assert np.array_equal(s.env.commands[-1], np.zeros(9))


def test_provider_failure_leaves_scene_paused(config):
    s = components(config, trajectory())
    s.reader.read = lambda obs: replace(sample(0), timestamp=obs.timestamp + 1)
    with pytest.raises(ValueError, match="does not match"):
        s.executor.run(s.env, s.policy)
    assert s.env.paused and not s.env.commands


def test_geometry_exemption_requires_supported_fresh_evidence(config):
    s = components(config, trajectory())
    obs = s.env.observe()
    view = s.executor.view_provider(obs)
    plan = s.planner(view, obs, None)
    binding = ChunkBinding.capture("chunk", plan, view, 0)
    moved = replace(sample(0.1), object_position=(0.5, 0, 0.02))
    new_view = s.grounder.ingest(moved)
    assert binding.changed(plan, new_view, 0.1)["reason"] == "position_changed"
    s.grounder.belief.add(
        Atom("grasp_contact", "cargo"),
        True,
        at=0.1,
        source="prediction",
        kind="predicted",
        replace=True,
    )
    assert binding.changed(plan, new_view, 0.1)["reason"] == "position_changed"
    with pytest.raises(ValueError, match="explicitly supported"):
        PositionDependency("cargo", ignore_when=Requirement(Atom("held", "cargo"), Truth.UNKNOWN))


def test_native_reader_checks_force_shape_units_and_same_tick(config):
    cargo = SimpleNamespace(
        get_world_poses=lambda: (np.array([[0.4, 0, 0.02]]), np.array([[1, 0, 0, 0]])),
        get_contact_force_matrix=lambda **kwargs: np.array([[[1, 0, 0], [-1, 0, 0], [0, 0, 2]]]),
    )
    gripper = SimpleNamespace(get_world_pose=lambda: (np.array([0.4, 0, 0.1]), [1, 0, 0, 0]))
    goal = SimpleNamespace(get_world_pose=lambda: ([0.6, 0, 0.02], [1, 0, 0, 0]))
    world = SimpleNamespace(current_time=0.1, get_physics_dt=lambda: 1 / 60)
    reader = IsaacSceneReader(config, world, cargo, gripper, goal, config.finger_joint_names)
    obs = SimpleNamespace(timestamp=0.1, state=np.array([0.02, 0.03]))
    s = reader.read(obs)
    assert s.width == 0.05 and s.finger_forces == (1, 1) and s.support_force == 2
    assert s.gripper_position[2] == pytest.approx(0.2034)
    with pytest.raises(ValueError, match="physics tick"):
        reader.read(SimpleNamespace(timestamp=0, state=obs.state))
    cargo.get_contact_force_matrix = lambda **kw: np.zeros((1, 2, 3))
    with pytest.raises(ValueError, match="contact vectors"):
        reader.read(obs)


def test_evaluation_counts_failed_attempts_and_rejects_duplicate_paths(tmp_path):
    import json

    paths = []
    for i, result in enumerate(
        (
            {"reason": "no_plan", "task_success": True},
            {"reason": "error", "task_success": False},
            {"reason": "step_budget_exhausted", "task_success": False},
        )
    ):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps({"format": "grounded-vla-pick-place-v1", **result}))
        paths.append(path)
    result = aggregate(paths)
    assert result["started_runs"] == 3 and result["successful_runs"] == 1
    assert result["error_runs"] == 1 and result["success_rate"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="distinct"):
        aggregate([paths[0], paths[0]])


@pytest.mark.parametrize(
    "changes",
    [
        {"max_sample_gap": -1},
        {"grasp_width_max": 0.09},
        {"gripper_offset": [0, 0]},
        {"object_id": "goal"},
        {"contact_force_min": float("nan")},
    ],
)
def test_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        PickPlaceConfig(**changes)


def test_composed_recordings_keep_preaction_and_terminal_context(config, tmp_path):
    from grounded_vla.training.recording import CriticObservationRecorder, EpisodeRecorder

    s = components(config, trajectory())
    schema = s.policy.meta["graph_schema_id"]
    actions, observations = EpisodeRecorder(schema), CriticObservationRecorder(schema)
    s.executor.run(
        s.env,
        s.policy,
        max_steps=8,
        execute_steps=3,
        recorder=actions,
        record_critic_context=True,
        critic_recorder=observations,
    )
    actions.save(tmp_path / "actions.npz")
    observations.save(tmp_path / "observations.npz")
    with np.load(tmp_path / "actions.npz", allow_pickle=False) as archive:
        assert len(archive["actions"]) == 8
        assert archive["timestamps"][0] == 0
        assert archive["timestamps"][-1] == pytest.approx(0.7)
        assert archive["critic_context/contract"].shape == (8, 35)
    with np.load(tmp_path / "observations.npz", allow_pickle=False) as archive:
        assert "actions" not in archive and "critic_labels" not in archive
        assert archive["timestamps"][-1] == pytest.approx(0.8)
        assert len(archive["timestamps"]) == 9


def test_reader_uses_physics_dt_and_oracle_provenance(config):
    observed = []
    cargo = SimpleNamespace(
        get_world_poses=lambda: (np.array([[0.4, 0, 0.02]]), np.array([[1, 0, 0, 0]])),
        get_contact_force_matrix=lambda **kw: observed.append(kw["dt"]) or np.zeros((1, 3, 3)),
    )
    body = SimpleNamespace(get_world_pose=lambda: ([0.4, 0, 0.1], [1, 0, 0, 0]))
    world = SimpleNamespace(current_time=0, get_physics_dt=lambda: 1 / 120)
    reader = IsaacSceneReader(config, world, cargo, body, body, config.finger_joint_names)
    result = reader.read(SimpleNamespace(timestamp=0, state=np.array([0.04, 0.04])))
    assert observed == [1 / 120] and result.source == "isaac_ground_truth"


def test_truth_retracts_after_measured_success_leaves_goal(config):
    g, e = PickPlaceGrounder(config), PickPlaceEvaluator(config)
    for row in trajectory():
        g.ingest(row)
        e.update(row)
    assert e.success
    moved = sample(0.9, cargo=(1, 0, 0.02), gripper=(0.6, 0, 0.15))
    g.ingest(moved)
    e.update(moved)
    assert not e.success and truth(g, "at_tray") == Truth.REFUTED


def test_drop_then_slide_into_goal_does_not_reuse_old_grasp(config):
    g, e = PickPlaceGrounder(config), PickPlaceEvaluator(config)
    rows = (
        trajectory()[:4]
        + [
            sample(0.4, cargo=(0.5, 0, 0.02), gripper=(0.4, 0, 0.2), width=0.04),
        ]
        + trajectory()[5:]
    )
    for row in rows:
        g.ingest(row)
        e.update(row)
    assert e.verified_grasp  # Historical measurement remains in diagnostics.
    assert not e.success
    assert truth(g, "at_tray") != Truth.SUPPORTED
