"""
UR10e — extension-mode URDF load & movement demo.

Imports a UR10e from a local URDF file, starts the simulation, and moves the
arm to a target pose.

Run inside a live Isaac Sim session via the VS Code extension (not standalone).
The stage should be empty before running.
"""

import asyncio
import pathlib

import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdLux

_ROOT = pathlib.Path(r'D:\Telekinesis\Code\isaacsim-examples\assets').parent
_MODEL_DIR = _ROOT / "assets"

UR10E_URDF = (
    _MODEL_DIR
    / "example-robot-data"
    / "robots"
    / "universal_robots"
    / "urdf"
    / "ur10e.urdf"
)

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
ARM_TARGET_RAD = np.deg2rad([0.0, -90.0, 90.0, 0.0, 90.0, 0.0])
WAIT_STEPS = 120


async def main():
    app = omni.kit.app.get_app()
    stage = omni.usd.get_context().get_stage()

    # !! Very important: timeline has to be stopped when importing
    omni.timeline.get_timeline_interface().stop()
    if not stage.GetPrimAtPath("/groundPlane").IsValid():
        PhysicsSchemaTools.addGroundPlane(
            stage, "/groundPlane", "Z", 15, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.7)
        )
    if not stage.GetPrimAtPath("/DistantLight").IsValid():
        distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        distant_light.CreateIntensityAttr(1000)

    if not UR10E_URDF.is_file():
        raise FileNotFoundError(f"Cannot find UR10e URDF: {UR10E_URDF}")

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.distance_scale = 1.0

    _, ur10e_prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(UR10E_URDF),
        import_config=import_config,
        get_articulation_root=True,
    )
    print(f"Imported UR10e at: {ur10e_prim_path}")

    await app.next_update_async()
    await app.next_update_async()

    omni.timeline.get_timeline_interface().play()

    robot = None
    for _ in range(60):
        try:
            art = SingleArticulation(prim_path=ur10e_prim_path, name="ur10e")
            art.initialize()
            if art.num_dof and art.num_dof > 0:
                robot = art
                break
        except Exception:
            pass
        await app.next_update_async()

    if robot is None:
        raise RuntimeError("UR10e articulation did not become valid.")

    print(f"num_dof: {robot.num_dof}")
    print(f"dof_names: {robot.dof_names}")

    arm_indices = [robot.dof_names.index(n) for n in ARM_JOINT_NAMES]

    robot.apply_action(
        ArticulationAction(
            joint_positions=ARM_TARGET_RAD.tolist(),
            joint_indices=arm_indices,
        )
    )
    print(f"Commanded arm (rad): {ARM_TARGET_RAD}")

    for _ in range(WAIT_STEPS):
        await app.next_update_async()

    print("\nDone. UR10e imported and arm moved.")


asyncio.ensure_future(main())
