"""Author the example metre/Z-up scene using a user-supplied Franka USD asset.

Run with Isaac Sim's Python. This authors a scene; it does not run a policy.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-usd", required=True, help="Local absolute path or asset URL")
    parser.add_argument("--out", default="assets/grounded-pick-place.usda")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    robot = args.robot_usd
    if "://" not in robot:
        robot = str(Path(robot).resolve(strict=True))
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from pxr import UsdGeom

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
        stage = get_current_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world.scene.add_default_ground_plane()
        world.scene.add(
            FixedCuboid(
                "/World/Table",
                name="table",
                position=np.array([0.4, 0, 0.2]),
                scale=np.array([1.2, 0.9, 0.4]),
                size=1.0,
            )
        )
        add_reference_to_stage(robot, "/World/Robot")
        arm = world.scene.add(
            SingleArticulation(
                "/World/Robot",
                name="franka",
                position=np.array([0, 0, 0.4]),
            )
        )
        arm.set_joints_default_state(
            positions=np.array([0, -0.4, 0, -2.2, 0, 1.8, 0.8, 0.04, 0.04])
        )
        world.scene.add(
            DynamicCuboid(
                "/World/Cargo",
                name="cargo",
                position=np.array([0.45, -0.15, 0.425]),
                scale=np.array([0.04, 0.04, 0.04]),
                size=1.0,
                mass=0.08,
                color=np.array([0.8, 0.05, 0.05]),
            )
        )
        goal = UsdGeom.Xform.Define(stage, "/World/Goal")
        goal.AddTranslateOp().Set((0.45, 0.15, 0.42))
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.AddTranslateOp().Set((0.4, 0.0, 1.5))  # USD camera looks along -Z.
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateClippingRangeAttr((0.01, 10.0))
        # Persist the initial robot joint state into USD (not just World defaults).
        world.reset()
        for _ in range(2):
            world.step(render=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(out))
        print(out)
    finally:
        app.close()


if __name__ == "__main__":
    main()
