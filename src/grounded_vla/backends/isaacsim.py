"""Isaac Sim camera/joint bridge with explicit absolute joint-position semantics.

Native imports are deferred until ``from_config``. Create SimulationApp first.
The dependency-injected constructor also supports contract tests without Kit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grounded_vla.observation import CAMERAS


class SimulationBackendError(RuntimeError):
    pass


class CameraNotReady(SimulationBackendError):
    pass


@dataclass(frozen=True)
class IsaacSimConfig:
    stage_path: str
    robot_prim_path: str
    camera_paths: dict[str, str]
    joint_names: tuple[str, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    max_joint_step: tuple[float, ...]
    physics_dt: float = 1 / 120
    steps_per_action: int = 6
    warmup_steps: int = 60
    max_camera_age_seconds: float = 0.05

    def __post_init__(self):
        for key in ("joint_names", "lower_limits", "upper_limits", "max_joint_step"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        n = len(self.joint_names)
        if not 1 <= n <= 32 or len(set(self.joint_names)) != n:
            raise ValueError("Provide 1..32 unique joint names in model state/action order")
        if not all(isinstance(s, str) and s.strip() for s in self.joint_names):
            raise ValueError("Joint names must be nonempty strings")
        if not self.stage_path or not self.robot_prim_path.startswith("/"):
            raise ValueError("Provide a stage and absolute robot prim path")
        if not self.camera_paths or not set(self.camera_paths) <= set(CAMERAS):
            raise ValueError("Use canonical base/wrist camera names")
        if not all(isinstance(p, str) and p.startswith("/") for p in self.camera_paths.values()):
            raise ValueError("Camera prim paths must be absolute")
        if len(set(self.camera_paths.values())) != len(self.camera_paths):
            raise ValueError("Each camera must refer to a distinct prim")
        for key in ("lower_limits", "upper_limits", "max_joint_step"):
            a = np.asarray(getattr(self, key), dtype=float)
            if a.shape != (n,) or not np.isfinite(a).all():
                raise ValueError(f"{key} must contain one finite value per joint")
        if np.any(np.asarray(self.lower_limits) >= self.upper_limits):
            raise ValueError("Joint lower limits must be below upper limits")
        if np.any(np.asarray(self.max_joint_step) <= 0):
            raise ValueError("max_joint_step must be positive")
        for key in ("physics_dt", "max_camera_age_seconds"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("steps_per_action", "warmup_steps"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")

    @property
    def control_period_seconds(self):
        return self.physics_dt * self.steps_per_action

    @property
    def action_convention(self):
        # Include order in the wire contract: equal dimensions do not imply equal semantics.
        return "absolute_joint_positions[" + ",".join(self.joint_names) + "]"

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        raw = json.loads(path.read_text())
        if "://" not in raw["stage_path"]:
            raw["stage_path"] = str((path.parent / raw["stage_path"]).resolve())
        return cls(**raw)


@dataclass(frozen=True)
class SimObservation:
    timestamp: float
    state: np.ndarray
    images: dict[str, np.ndarray]


class IsaacSimEnvironment:
    """Synchronous simulation only; no physics advances during policy inference.

    ``action_guard`` is an optional independent robot/task validator. Joint
    limits alone are not a collision checker or a hardware safety controller.
    """

    def __init__(self, config, world, robot, cameras, *, action_factory, action_guard=None):
        self.config, self.world, self.robot = config, world, robot
        self.cameras, self.action_factory = cameras, action_factory
        self.action_guard = action_guard
        if set(cameras) != set(config.camera_paths):
            raise ValueError("Configured and attached cameras differ")
        dof_names = list(robot.dof_names)
        if any(name not in dof_names for name in config.joint_names):
            raise ValueError("Configured joint not present in the initialized articulation")
        self.indices = np.array([dof_names.index(name) for name in config.joint_names], np.int32)
        self._last_camera_times = {}
        self._paused = False

    @classmethod
    def from_config(cls, config, *, action_guard=None):
        """Attach an existing, authored USD scene after SimulationApp has started."""
        try:
            from isaacsim.core.api import World
            from isaacsim.core.prims import SingleArticulation
            from isaacsim.core.utils.prims import is_prim_path_valid
            from isaacsim.core.utils.stage import open_stage
            from isaacsim.core.utils.types import ArticulationAction
            from isaacsim.sensors.camera import Camera
        except ImportError as exc:
            raise SimulationBackendError(
                "Run with Isaac Sim's Python and create SimulationApp before native imports"
            ) from exc
        if not open_stage(config.stage_path):
            raise SimulationBackendError(f"Could not open USD stage: {config.stage_path}")
        for path in (config.robot_prim_path, *config.camera_paths.values()):
            if not is_prim_path_valid(path):
                raise SimulationBackendError(f"Missing prim in scene: {path}")
        world = World(physics_dt=config.physics_dt, rendering_dt=config.physics_dt)
        try:
            robot = world.scene.add(
                SingleArticulation(prim_path=config.robot_prim_path, name="grounded_vla_robot")
            )
            cameras = {
                name: Camera(prim_path=path, name=name, resolution=(224, 224))
                for name, path in config.camera_paths.items()
            }
            world.reset()
            for camera in cameras.values():
                camera.initialize()
            env = cls(
                config,
                world,
                robot,
                cameras,
                action_factory=ArticulationAction,
                action_guard=action_guard,
            )
            # Warm-up is bounded and never substitutes black images for missing sensors.
            for _ in range(config.warmup_steps):
                world.step(render=True)
                try:
                    env.observe()
                    env._last_camera_times.clear()
                    return env
                except CameraNotReady:
                    continue
            raise CameraNotReady("Cameras did not produce valid frames during warm-up")
        except BaseException:
            world.pause()
            raise

    def _positions(self):
        state = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)[self.indices]
        if state.shape != (len(self.indices),) or not np.isfinite(state).all():
            raise SimulationBackendError("Invalid measured articulation state")
        return state.copy()

    def observe(self):
        now = float(self.world.current_time)
        if not math.isfinite(now) or now < 0:
            raise SimulationBackendError("Invalid simulation clock")
        images, camera_times = {}, {}
        for name, camera in self.cameras.items():
            frame = camera.get_current_frame(clone=True)
            rgba = np.asarray(frame.get("rgba"))
            stamp = frame.get("rendering_time")
            if rgba.dtype != np.uint8 or rgba.shape != (224, 224, 4) or stamp is None:
                raise CameraNotReady(f"{name}: expected a rendered uint8 RGBA frame at 224x224")
            stamp = float(stamp)
            if (
                not math.isfinite(stamp)
                or stamp < 0
                or stamp > now + 1e-6
                or now - stamp > self.config.max_camera_age_seconds
                or stamp <= self._last_camera_times.get(name, -math.inf)
            ):
                raise CameraNotReady(f"{name}: stale, duplicate, or invalid camera timestamp")
            images[name] = np.ascontiguousarray(rgba[:, :, :3])
            camera_times[name] = stamp
        state = self._positions()
        self._last_camera_times = camera_times
        return SimObservation(now, state, images)

    def step(self, action):
        if self._paused:
            raise SimulationBackendError("Episode is paused; reset before executing another action")
        try:
            target = np.asarray(action, dtype=np.float32)
            if target.shape != (len(self.indices),) or not np.isfinite(target).all():
                raise ValueError("Action must be a finite absolute position per configured joint")
            if np.any(target < self.config.lower_limits) or np.any(
                target > self.config.upper_limits
            ):
                raise ValueError("Action exceeds joint limits")
            state = self._positions()
            if np.any(np.abs(target - state) > self.config.max_joint_step):
                raise ValueError("Action exceeds maximum step from measured joint positions")
            if self.action_guard is not None:
                self.action_guard(state.copy(), target.copy())
            self.robot.apply_action(
                self.action_factory(
                    joint_positions=target.copy(), joint_indices=self.indices.copy()
                )
            )
            for _ in range(self.config.steps_per_action):
                self.world.step(render=True)
            return self.observe()
        except BaseException:
            self.pause()
            raise

    def pause(self):
        self._paused = True
        self.world.pause()

    def reset(self):
        self.world.reset()
        self._last_camera_times.clear()
        self._paused = False
        try:
            for _ in range(self.config.warmup_steps):
                self.world.step(render=True)
                try:
                    return self.observe()
                except CameraNotReady:
                    continue
            raise CameraNotReady("Cameras did not recover after reset")
        except BaseException:
            self.pause()
            raise
