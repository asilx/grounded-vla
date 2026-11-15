"""Explicit simulator-oracle measurements for an upright, single-object task.

No RGB detection or commanded-state inference. Forces are filtered object/finger
and object/support forces in N. Positions and gripper width are in metres.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have {size} finite components")
    return tuple(float(x) for x in result)


def rotation(quaternion):
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class SceneSample:
    timestamp: float
    object_position: tuple[float, float, float]
    gripper_position: tuple[float, float, float]
    goal_position: tuple[float, float, float]
    gripper_orientation: tuple[float, float, float, float]
    finger_positions: tuple[float, float]
    finger_forces: tuple[float, float]
    support_force: float
    source: str = "isaac_ground_truth"

    def __post_init__(self):
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("Invalid scene timestamp")
        for name, size in (
            ("object_position", 3),
            ("gripper_position", 3),
            ("goal_position", 3),
            ("gripper_orientation", 4),
            ("finger_positions", 2),
            ("finger_forces", 2),
        ):
            object.__setattr__(self, name, vector(getattr(self, name), size, name))
        if not np.isclose(np.linalg.norm(self.gripper_orientation), 1, atol=1e-4):
            raise ValueError("Expected normalized wxyz gripper quaternion")
        if (
            min(self.finger_positions) < 0
            or min(self.finger_forces) < 0
            or not math.isfinite(self.support_force)
            or self.support_force < 0
        ):
            raise ValueError("Invalid measured finger positions or contact forces")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Measurement provenance is required")

    @property
    def width(self):
        return sum(self.finger_positions)

    @property
    def relative_position(self):
        return rotation(self.gripper_orientation).T @ (
            np.asarray(self.object_position) - self.gripper_position
        )


@dataclass(frozen=True)
class PickPlaceConfig:
    object_prim_path: str = "/World/Cargo"
    gripper_prim_path: str = "/World/Robot/panda_hand"
    goal_prim_path: str = "/World/Goal"
    support_prim_path: str = "/World/Table"
    finger_prim_paths: tuple[str, str] = (
        "/World/Robot/panda_leftfinger",
        "/World/Robot/panda_rightfinger",
    )
    finger_joint_names: tuple[str, str] = ("panda_finger_joint1", "panda_finger_joint2")
    gripper_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034)
    object_id: str = "cargo"
    description: str = "the red cube"
    contact_force_min: float = 0.1
    grasp_width_min: float = 0.005
    grasp_width_max: float = 0.065
    release_width_min: float = 0.07
    max_grasp_distance: float = 0.08
    min_lift: float = 0.03
    min_gripper_motion: float = 0.01
    relative_tolerance: float = 0.008
    grasp_confirm_seconds: float = 0.1
    release_confirm_seconds: float = 0.5
    goal_xy_tolerance: float = 0.035
    goal_z_tolerance: float = 0.02
    stable_speed_max: float = 0.02
    target_motion_tolerance: float = 0.02
    evidence_ttl: float = 0.25
    max_sample_gap: float = 0.1
    effect_timeout_steps: int = 600
    max_chunk_age_seconds: float = 0.5
    max_pick_attempts: int = 4

    def __post_init__(self):
        for name in ("finger_prim_paths", "finger_joint_names"):
            values = tuple(getattr(self, name))
            if len(values) != 2 or len(set(values)) != 2:
                raise ValueError(f"{name} requires two distinct entries")
            object.__setattr__(self, name, values)
        paths = (
            self.object_prim_path,
            self.gripper_prim_path,
            self.goal_prim_path,
            self.support_prim_path,
            *self.finger_prim_paths,
        )
        if len(set(paths)) != len(paths) or any(
            not isinstance(p, str) or not p.startswith("/") or any(c in p for c in "*[]")
            for p in paths
        ):
            raise ValueError("Configure distinct exact absolute scene paths")
        if (
            not self.object_id
            or self.object_id in {"goal", "gripper"}
            or not self.description.strip()
        ):
            raise ValueError("Provide an unambiguous object identity and description")
        object.__setattr__(self, "gripper_offset", vector(self.gripper_offset, 3, "gripper_offset"))
        for name in (
            "contact_force_min",
            "grasp_width_min",
            "grasp_width_max",
            "release_width_min",
            "max_grasp_distance",
            "min_lift",
            "min_gripper_motion",
            "relative_tolerance",
            "grasp_confirm_seconds",
            "release_confirm_seconds",
            "goal_xy_tolerance",
            "goal_z_tolerance",
            "stable_speed_max",
            "target_motion_tolerance",
            "evidence_ttl",
            "max_sample_gap",
            "max_chunk_age_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.grasp_width_min < self.grasp_width_max < self.release_width_min:
            raise ValueError("Grasp and release width thresholds must be ordered")
        if self.evidence_ttl < self.max_sample_gap:
            raise ValueError("Evidence TTL must cover the maximum sample interval")
        for name in ("effect_timeout_steps", "max_pick_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class IsaacSceneReader:
    """Attach before World.reset; read without stepping physics or editing poses."""

    def __init__(self, config, world, cargo, gripper, goal, joint_names):
        self.config, self.world = config, world
        self.cargo, self.gripper, self.goal = cargo, gripper, goal
        self.finger_indices = tuple(joint_names.index(n) for n in config.finger_joint_names)

    @classmethod
    def attach(cls, config, world, joint_names):
        # Isaac 5.1 core API (also present as deprecated core API in Isaac 6).
        from isaacsim.core.prims import RigidPrim, SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom, UsdPhysics

        stage = get_current_stage()
        if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise ValueError("Pick/place scene requires metres as stage units")
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise ValueError("Pick/place scene requires Z-up")
        for path in (
            config.object_prim_path,
            config.gripper_prim_path,
            config.goal_prim_path,
            config.support_prim_path,
            *config.finger_prim_paths,
        ):
            if not stage.GetPrimAtPath(path).IsValid():
                raise ValueError(f"Missing configured scene prim: {path}")
        if not stage.GetPrimAtPath(config.object_prim_path).HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Cargo must be a rigid body")
        cargo = world.scene.add(
            RigidPrim(
                prim_paths_expr=config.object_prim_path,
                name="grounded_pick_place_contacts",
                reset_xform_properties=False,
                track_contact_forces=True,
                prepare_contact_sensors=True,
                disable_stablization=False,
                contact_filter_prim_paths_expr=[
                    *config.finger_prim_paths,
                    config.support_prim_path,
                ],
                max_contact_count=64,
            )
        )
        gripper = world.scene.add(
            SingleRigidPrim(
                config.gripper_prim_path,
                name="grounded_pick_place_gripper",
                reset_xform_properties=False,
            )
        )
        goal = world.scene.add(
            SingleXFormPrim(
                config.goal_prim_path,
                name="grounded_pick_place_goal",
                reset_xform_properties=False,
            )
        )
        return cls(config, world, cargo, gripper, goal, list(joint_names))

    def read(self, observation):
        stamp = float(self.world.current_time)
        if not math.isclose(stamp, observation.timestamp, abs_tol=1e-8, rel_tol=0):
            raise ValueError("Scene and camera/joint observation must share the physics tick")
        positions, _ = self.cargo.get_world_poses()
        position, orientation = self.gripper.get_world_pose()
        goal, _ = self.goal.get_world_pose()
        forces = np.asarray(self.cargo.get_contact_force_matrix(dt=self.world.get_physics_dt()))
        if forces.shape != (1, 3, 3) or not np.isfinite(forces).all():
            raise ValueError("Expected measured cargo contact vectors for two fingers and support")
        position = np.asarray(position) + rotation(orientation) @ self.config.gripper_offset
        return SceneSample(
            stamp,
            np.asarray(positions)[0],
            position,
            goal,
            orientation,
            np.asarray(observation.state)[list(self.finger_indices)],
            np.linalg.norm(forces[0, :2], axis=-1),
            float(np.linalg.norm(forces[0, 2])),
        )
