"""Contract doubles: these tests do not run Isaac Sim physics or render a scene."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from grounded_vla.backends.isaacsim import CameraNotReady, IsaacSimConfig, IsaacSimEnvironment
from grounded_vla.backends.openpi import PolicyBackendError
from grounded_vla.simulation import CanonicalSimPolicy, GraphSnapshot, run_rollout
from grounded_vla.training.recording import EpisodeRecorder


def config():
    return IsaacSimConfig(
        "scene.usd",
        "/World/Robot",
        {"base_0_rgb": "/World/Camera"},
        ("joint_a", "joint_b"),
        (-1, -1),
        (1, 1),
        (0.1, 0.1),
        physics_dt=0.01,
        steps_per_action=5,
        warmup_steps=3,
    )


class Robot:
    dof_names = ["uncontrolled", "joint_b", "joint_a"]

    def __init__(self):
        self.q = np.array([0.7, 0.0, 0.0], np.float32)
        self.commands = []

    def get_joint_positions(self):
        return self.q

    def apply_action(self, action):
        self.commands.append(action)
        self.q[action.joint_indices] = action.joint_positions


class World:
    def __init__(self):
        self.current_time = 0.0
        self.paused = False

    def step(self, render):
        assert render and not self.paused
        self.current_time = round(self.current_time + 0.01, 8)

    def pause(self):
        self.paused = True

    def reset(self):
        self.current_time = 0.0
        self.paused = False


class Camera:
    def __init__(self, world):
        self.world = world
        self.timestamp = None
        self.rgba = np.zeros((224, 224, 4), dtype=np.uint8)
        self.rgba[..., 0] = 31
        self.rgba[..., 1] = 63
        self.rgba[..., 2] = 127
        self.rgba[..., 3] = 255

    def get_current_frame(self, clone):
        assert clone
        return {
            "rgba": self.rgba,
            "rendering_time": self.world.current_time if self.timestamp is None else self.timestamp,
        }


@pytest.fixture
def env():
    world, robot = World(), Robot()
    return IsaacSimEnvironment(
        config(),
        world,
        robot,
        {"base_0_rgb": Camera(world)},
        action_factory=SimpleNamespace,
    )


class Client:
    def __init__(self, cfg):
        self.metadata = {
            "input_preset": "canonical",
            "state_dim": 2,
            "action_dim": 2,
            "action_convention": cfg.action_convention,
            "graph_required": True,
            "graph_schema_id": "test-v1",
            "graph_feature_dim": 3,
            "graph_relation_types": 2,
            "action_horizon": 4,
            "control_period_seconds": cfg.control_period_seconds,
        }
        self.calls = []
        self.actions = np.zeros((4, 2), np.float32)

    def infer(self, payload):
        self.calls.append(payload)
        return {"actions": self.actions}


def snapshot(obs):
    return GraphSnapshot(
        {
            "schema_id": "test-v1",
            "features": np.array([[obs.timestamp, 1, 0]], np.float32),
            "edges": np.ones((1, 1), np.int64),
            "valid": np.ones(1, bool),
        },
        obs.timestamp,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"joint_names": ("joint_a", "joint_a")},
        {"lower_limits": (float("nan"), -1)},
        {"steps_per_action": 0},
        {"max_joint_step": (0, 1)},
        {"physics_dt": float("inf")},
        {"camera_paths": {"unknown": "/World/Camera"}},
    ],
)
def test_config_rejects_invalid_contract(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


def test_camera_rgb_copy_and_joint_order(env):
    env.robot.q[:] = [0.7, -0.03, 0.04]
    obs = env.observe()
    np.testing.assert_allclose(obs.state, [0.04, -0.03])
    np.testing.assert_array_equal(obs.images["base_0_rgb"][0, 0], [31, 63, 127])
    env.cameras["base_0_rgb"].rgba[:] = 0
    assert obs.images["base_0_rgb"][0, 0, 0] == 31
    after = env.step([0.0, 0.02])
    np.testing.assert_allclose(env.robot.q, [0.7, 0.02, 0.0])
    assert after.timestamp == pytest.approx(0.05)
    np.testing.assert_array_equal(env.robot.commands[0].joint_indices, [2, 1])


@pytest.mark.parametrize("action", [[0.0], [0.0, float("nan")], [1.1, 0], [0.2, 0]])
def test_invalid_action_pauses_without_submission(env, action):
    with pytest.raises(ValueError):
        env.step(action)
    assert env.world.paused and not env.robot.commands


def test_independent_guard_can_veto_before_submission(env):
    def veto(state, action):
        raise ValueError("collision")

    env.action_guard = veto
    with pytest.raises(ValueError, match="collision"):
        env.step([0, 0])
    assert not env.robot.commands and env.world.paused


def test_stale_camera_pauses_step_and_reset_clears_time_history(env):
    env.observe()
    env.cameras["base_0_rgb"].timestamp = 0.0
    with pytest.raises(CameraNotReady, match="stale"):
        env.step([0, 0])
    assert env.world.paused
    env.cameras["base_0_rgb"].timestamp = None
    obs = env.reset()
    assert obs.timestamp == pytest.approx(0.01)
    assert not env.world.paused


def test_no_blank_image_fallback(env):
    env.cameras["base_0_rgb"].rgba = None
    with pytest.raises(CameraNotReady):
        env.observe()


@pytest.mark.parametrize(
    "key,value",
    [
        ("input_preset", "droid"),
        ("state_dim", 3),
        ("action_dim", 7),
        ("action_convention", "end_effector_delta"),
        ("control_period_seconds", 0.1),
        ("control_period_seconds", float("nan")),
        ("graph_feature_dim", 0),
    ],
)
def test_policy_handshake_rejects_incompatible_checkpoint(key, value):
    client = Client(config())
    client.metadata[key] = value
    with pytest.raises(ValueError):
        CanonicalSimPolicy(client, config())


def test_canonical_wire_payload_has_camera_state_graph_and_prompt(env):
    client = Client(config())
    policy = CanonicalSimPolicy(client, config())
    obs = env.observe()
    graph = snapshot(obs)
    result = policy.infer(obs, graph, "pick the part")
    payload = client.calls[0]
    assert set(payload) == {"state", "images", "prompt", "graph"}
    assert payload["prompt"] == "pick the part"
    np.testing.assert_array_equal(payload["graph"]["features"], graph.graph["features"])
    result[:] = 0.9
    assert not client.actions.any()


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((0, 2)),
        np.zeros((5, 2)),
        np.zeros((2, 3)),
        np.full((1, 2), np.nan),
    ],
)
def test_malformed_policy_actions_are_rejected(env, actions):
    client = Client(config())
    client.actions = actions
    obs = env.observe()
    with pytest.raises(PolicyBackendError):
        CanonicalSimPolicy(client, config()).infer(obs, snapshot(obs), "pick")


@pytest.mark.parametrize("offset", [0.1, -0.6, float("nan")])
def test_future_stale_and_nonfinite_graph_prevents_policy_call(env, offset):
    client = Client(config())
    policy = CanonicalSimPolicy(client, config())
    obs = env.observe()
    raw = snapshot(obs)
    with pytest.raises(ValueError, match="timestamp"):
        policy.infer(obs, GraphSnapshot(raw.graph, obs.timestamp + offset), "pick")
    assert not client.calls


def test_rollout_replans_and_records_preaction_state(env, tmp_path):
    client = Client(config())
    client.actions[:] = [0.04, -0.03]
    recorder = EpisodeRecorder("test-v1")
    graph_times = []

    def provider(obs):
        graph_times.append(obs.timestamp)
        return snapshot(obs)

    result = run_rollout(
        env,
        CanonicalSimPolicy(client, config()),
        provider,
        "pick",
        max_steps=3,
        execute_steps=1,
        recorder=recorder,
    )
    assert result.steps == 3 and result.reason == "budget_exhausted"
    assert len(client.calls) == 3 and len(env.robot.commands) == 3 and env.world.paused
    np.testing.assert_allclose(graph_times, [0, 0.05, 0.1])
    recorder.save(tmp_path / "episode.npz")
    with np.load(tmp_path / "episode.npz") as data:
        np.testing.assert_allclose(data["timestamps"], [0.0, 0.05, 0.1])
        np.testing.assert_allclose(data["state"][0], [0, 0])
        np.testing.assert_allclose(data["state"][1], [0.04, -0.03])
        np.testing.assert_allclose(data["graph_features"][:, 0, 0], [0, 0.05, 0.1])


def test_monitor_discards_chunk_suffix_and_callback_sees_paused_world(env):
    class Monitor:
        def reset(self):
            self.count = 0

        def update(self, image, timestamp):
            self.count += 1
            return {"kind": "critic_pause"} if self.count == 2 else None

    calls = []

    def callback(event, obs):
        assert env.world.paused
        calls.append((event, obs.timestamp))

    result = run_rollout(
        env,
        CanonicalSimPolicy(Client(config()), config()),
        snapshot,
        "pick",
        max_steps=8,
        execute_steps=4,
        monitor=Monitor(),
        on_event=callback,
    )
    assert result.steps == 1 and result.reason == "critic_pause"
    assert len(env.robot.commands) == 1 and len(calls) == 1


def test_policy_timeout_pauses_without_command(env):
    client = Client(config())

    def timeout(_):
        raise TimeoutError("policy timed out")

    client.infer = timeout
    with pytest.raises(TimeoutError):
        run_rollout(env, CanonicalSimPolicy(client, config()), snapshot, "pick")
    assert env.world.paused and not env.robot.commands
