"""Exercise the scene-check CLI with a real scene reader and CPU sensor doubles."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from grounded_vla import pick_place_cli
from grounded_vla.backends.isaacsim import IsaacSimEnvironment, SimObservation
from grounded_vla.benchmarks.isaac_pick_place import aggregate
from grounded_vla.perception.isaac_scene import IsaacSceneReader

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def check_cli(tmp_path, monkeypatch):
    out = tmp_path / "check"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grounded-vla-pick-place",
            str(CONFIGS / "isaac-pick-place.example.json"),
            str(CONFIGS / "pick-place.example.json"),
            "--check-scene",
            "--out",
            str(out),
        ],
    )
    # A scene check must work before the policy and learning stack are installed.
    for module in ("grounded_vla.backends.openpi", "grounded_vla.tasks.pick_place", "torch"):
        monkeypatch.setitem(sys.modules, module, None)
    return out


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("none", None),
        ("no_support", "rest on the support"),
        ("bilateral_grasp", "without a bilateral grasp"),
        ("stale_tick", "share the physics tick"),
        ("missing_filter", "contact vectors"),
    ],
)
def test_check_first_observation(check_cli, monkeypatch, fault, error):
    app = Mock()
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=lambda options: app))
    world = SimpleNamespace(current_time=0.25, get_physics_dt=lambda: 1 / 60)
    forces = np.zeros((1, 3, 3))
    forces[0, 2, 2] = 0 if fault == "no_support" else 0.8
    if fault == "bilateral_grasp":
        forces[0, :2, 0] = (1, -1)
    if fault == "missing_filter":
        forces = forces[:, :2]
    cargo = SimpleNamespace(
        get_world_poses=lambda: ([[0.45, 0, 0.42]], [[1, 0, 0, 0]]),
        get_contact_force_matrix=Mock(return_value=forces),
    )
    gripper = SimpleNamespace(get_world_pose=lambda: ([0.45, 0, 0.32], [1, 0, 0, 0]))
    goal = SimpleNamespace(get_world_pose=lambda: ([0.45, 0.15, 0.42], [1, 0, 0, 0]))
    monkeypatch.setattr(
        IsaacSceneReader,
        "attach",
        lambda config, attached_world, names: IsaacSceneReader(
            config, attached_world, cargo, gripper, goal, names
        ),
    )
    environment = Mock(spec=IsaacSimEnvironment)
    environment.observe.return_value = SimObservation(
        0.2 if fault == "stale_tick" else world.current_time,
        np.array([0, -0.4, 0, -2.2, 0, 1.8, 0.8, 0.02, 0.02], dtype=np.float32),
        {"base_0_rgb": np.zeros((224, 224, 3), np.uint8)},
    )

    def create(config, *, scene_setup, action_guard):
        assert action_guard is None
        scene_setup(world, None)
        return environment

    monkeypatch.setattr(IsaacSimEnvironment, "from_config", create)
    if error:
        with pytest.raises(ValueError, match=error):
            pick_place_cli.main()
    else:
        pick_place_cli.main()

    path = check_cli / "scene-check.json"
    assert list(check_cli.iterdir()) == [path]
    report = json.loads(path.read_text())
    assert report["ready"] is (error is None)
    assert report["observation"]["camera_shapes"] == {"base_0_rgb": [224, 224, 3]}
    assert report["observation"]["joint_positions"]["panda_joint4"] == pytest.approx(-2.2)
    if error:
        assert error in report["error"]
    if fault in {"none", "no_support", "bilateral_grasp"}:
        measured = report["measurement"]
        assert measured["timestamp"] == report["observation"]["timestamp"]
        assert measured["support_force"] == forces[0, 2, 2]
        assert measured["gripper_position"][2] == pytest.approx(0.4234)
        assert measured["finger_forces"] == np.abs(forces[0, :2, 0]).tolist()
        assert report["gripper_width_m"] == pytest.approx(0.04)
    else:
        assert "measurement" not in report
    environment.observe.assert_called_once_with()
    environment.step.assert_not_called()
    environment.pause.assert_called_once_with()
    app.close.assert_called_once_with()
    with pytest.raises(ValueError, match="Unknown pick/place result format"):
        aggregate([path])


def test_check_records_simulator_startup_failure(check_cli, monkeypatch):
    monkeypatch.setitem(sys.modules, "isaacsim", None)
    with pytest.raises(ModuleNotFoundError):
        pick_place_cli.main()
    report = json.loads((check_cli / "scene-check.json").read_text())
    assert report["ready"] is False and report["error_type"] == "ModuleNotFoundError"
    assert "observation" not in report and not (check_cli / "result.json").exists()
