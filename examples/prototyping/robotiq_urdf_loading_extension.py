"""
!!!!!!!!!!!!!!!!!!!
This does not work for robotiq urdf. Not sure why but something is wrong. Use the .usd instead of urdf to load!
Here we added loading of usd from isaacsims storage. This is a good usd that works


This should be done to the loaded urdf model
https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/rig_closed_loop_structures.html

!!!!!!!!!!!!!!!!!!!


Robotiq gripper — extension-mode URDF load & movement demo.

Imports a Robotiq gripper from a local URDF file, starts the simulation, and
closes the gripper by driving its finger joint.

Run inside a live Isaac Sim session via the VS Code extension (not standalone).
The stage should be empty before running.

Adjust GRIPPER_URDF to point to your local URDF file.
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
from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdLux, UsdPhysics
import isaacsim.core.utils.stage as stage_utils
from isaacsim.storage.native import get_assets_root_path

_ROOT = pathlib.Path(r'D:\Telekinesis\Code\isaacsim-examples\assets').parent
_MODEL_DIR = _ROOT / "assets"

# Adjust this path to your Robotiq URDF location.

GRIPPER_URDF = _MODEL_DIR / r"example-robot-data\end_effectors\robotiq\urdf\robotiq_2f_85_gripper.urdf"
assets_root_path = get_assets_root_path()
GRIPPER_USD = (
    assets_root_path + "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
)
ADD_PRIM_PATH = "/World/robot_2f_85"

GRIPPER_CLOSE_RAD = 0.5  # TODO this is dummy value its not for closed gripper
WAIT_STEPS = 60


def find_driver_joint(stage, prim_path, dof_names):
    """Return the gripper's actuated driver joint name from ``dof_names``.

    A single-input gripper has one actuated joint; the rest are mimic joints that
    follow it. The driver is the non-mimic joint, preferring one with a
    UsdPhysics.DriveAPI. Falls back to the first DOF if detection is inconclusive.
    """
    root = stage.GetPrimAtPath(prim_path)
    if root.IsA(UsdPhysics.Joint):  # importer may return the root joint, not the container
        root = root.GetParent()
    joints = {p.GetName(): p for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)}

    fallback = None
    for name in dof_names:
        prim = joints.get(name)
        if prim is None or any("MimicJoint" in s for s in prim.GetAppliedSchemas()):
            continue
        if prim.HasAPI(UsdPhysics.DriveAPI):
            return name
        fallback = fallback or name
    return fallback or dof_names[0]


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

    # if not GRIPPER_URDF.is_file():
    #     raise FileNotFoundError(f"Cannot find gripper URDF: {GRIPPER_URDF}")

    # _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    # import_config.merge_fixed_joints = False
    # import_config.convex_decomp = False
    # import_config.import_inertia_tensor = True
    # import_config.fix_base = True
    # import_config.distance_scale = 1.0
    # import_config.parse_mimic = True

    # _, gripper_prim_path = omni.kit.commands.execute(
    #     "URDFParseAndImportFile",
    #     urdf_path=str(GRIPPER_URDF),
    #     import_config=import_config,
    #     get_articulation_root=True,
    # )
    # print(f"Imported gripper at: {gripper_prim_path}")

    stage_utils.add_reference_to_stage(str(GRIPPER_USD), ADD_PRIM_PATH)

    # gripper_prim_path = '/World/Robotiq_2F_85_edit'
    await app.next_update_async()
    await app.next_update_async()

    omni.timeline.get_timeline_interface().play()

    gripper = None
    for _ in range(60):
        try:
            art = SingleArticulation(prim_path=ADD_PRIM_PATH, name="gripper")
            art.initialize()
            if art.num_dof and art.num_dof > 0:
                gripper = art
                break
        except Exception:
            pass
        await app.next_update_async()

    if gripper is None:
        raise RuntimeError("Gripper articulation did not become valid.")

    print(f"num_dof: {gripper.num_dof}")
    print(f"dof_names: {gripper.dof_names}")

    driver_joint = find_driver_joint(stage, ADD_PRIM_PATH, gripper.dof_names)
    finger_idx = gripper.dof_names.index(driver_joint)

    gripper.apply_action(
        ArticulationAction(
            joint_positions=[GRIPPER_CLOSE_RAD],
            joint_indices=[finger_idx],
        )
    )
    print(f"Commanded {driver_joint} (rad): {GRIPPER_CLOSE_RAD}")

    tolerance = 1e-3
    while True:
        await app.next_update_async()
        current = gripper.get_joint_positions()[finger_idx]
        if abs(current - GRIPPER_CLOSE_RAD) < tolerance:
            break

    print("\nDone. Gripper imported and closed.")


asyncio.ensure_future(main())
