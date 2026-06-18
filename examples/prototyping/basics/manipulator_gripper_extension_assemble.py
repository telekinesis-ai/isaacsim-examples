"""
References:
-https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_simulation/articulation_controller.html
-https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/assemble_robots.html
-https://github.com/isaac-sim/IsaacSim/blob/40316786340b3f034a229d9e12650df1ac0b68ab/source/extensions/isaacsim.robot_setup.assembler/isaacsim/robot_setup/assembler/ui/ui_builder.py#L64
-https://github.com/isaac-sim/IsaacSim/blob/40316786340b3f034a229d9e12650df1ac0b68ab/source/extensions/isaacsim.robot_setup.assembler/isaacsim/robot_setup/assembler/tests/test_robot_assembler.py#L44

Manipulator + gripper — extension-mode ASSEMBLE demo (manufacturer-agnostic).

This script runs *inside* a live Isaac Sim session via the VS Code extension. It
connects to an arm articulation and a *separate* gripper articulation already in
the open stage, **assembles** the gripper onto the arm's flange with
``RobotAssembler``, then moves the arm, waits, and closes the gripper. After
assembly the gripper is fixed to the arm, so the whole rig becomes a SINGLE
articulation rooted at the arm — one ``SingleArticulation`` handle drives both
the arm joints and the gripper's driver joint.

Nothing is hardcoded to a specific robot: arm joint names are read from the arm
articulation, and the gripper's driver joint is detected (non-mimic, prefers a
DriveAPI). Configured below for **KUKA kr120 + OnRobot RG6**; swap robots by
editing the config block only.

How to use:
- Add the arm to the scene (e.g. import KUKA kr120 via the URDF Importer).
- Add the gripper to the scene (e.g. import the RG6).
- Adjust the config block (prim paths + mount links) to match your stage.
- Run this with the VS Code extension.

Notes:
- Attachment points must be RigidBodyAPI links (e.g. KUKA ``link_6`` / UR
  ``wrist_3_link``), NOT empty frames like ``tool0`` / ``flange``.
- Joint targets are in **degrees**, converted to radians via ``np.deg2rad``.
"""
from isaacsim.robot_setup.assembler import RobotAssembler
import asyncio
import numpy as np

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import omni.timeline
import omni.usd
import omni.kit.app
import omni.kit.commands
from pxr import Gf, Usd, UsdPhysics

# Robot Assembler is packaged as an extension. Enable it before importing so the
# import doesn't fail if the extension isn't loaded yet.
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.robot_setup.assembler", True)

# ---------------------------------------------------------------------------
# Config — edit these to match your stage / robots (configured for KUKA + RG6).
# ---------------------------------------------------------------------------
ARM_PRIM = "/World/manipulator"                       # existing arm articulation root
GRIPPER_PRIM = "/World/rg6"                    # existing gripper articulation root
ARM_MOUNT_LINK = "link_6"                      # rigid-body flange link (NOT tool0)
GRIPPER_MOUNT_LINK = "onrobot_rg6_base_link"   # gripper base link

ASSEMBLY_NAMESPACE = "Gripper"
VARIANT_NAME = "arm_with_gripper"

TARGET_DEG = [0,0,0,0,0,0]         # applied to every arm joint
GRIPPER_CLOSE_DEG = 30.0   # gripper driver-joint close target
WAIT_STEPS = 120           # sim steps to settle the arm before closing

# Custom mount offset applied to the gripper, in its mount-local frame.
ATTACH_OFFSET_TRANSLATION_M = (0.0, 0.0, 0.0)  # meters (x, y, z)
ATTACH_OFFSET_ROTATION_DEG = (0.0, 0.0, 0.0)   # XYZ Euler degrees


def find_prim_by_name(stage, root_path, name):
    """Full path of the prim named ``name`` under ``root_path`` (handles nesting)."""
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if p.GetName() == name:
            return p.GetPath().pathString
    raise RuntimeError(f"Prim named {name!r} not found under {root_path!r}.")



def find_driver_joint(stage, root_path):
    """The gripper's actuated driver joint name: non-mimic, preferring a DriveAPI.

    The other gripper joints are mimic joints that follow it, so driving this one
    opens/closes the whole linkage.
    """
    fallback = None
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not (p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)):
            continue
        if any("MimicJoint" in s for s in p.GetAppliedSchemas()):
            continue
        if p.HasAPI(UsdPhysics.DriveAPI):
            return p.GetName()
        fallback = fallback or p.GetName()
    if fallback is None:
        raise RuntimeError(f"No drivable joint found under {root_path!r}.")
    return fallback


def apply_attach_offset(prim_path, translation_m, rotation_deg):
    """Nudge the attach prim by a custom offset in its own (mount-local) frame.

    Applied between begin_assembly() and assemble() so the fixed joint is baked
    at the adjusted relative pose. No-op when both offsets are zero.
    """
    if not any(translation_m) and not any(rotation_deg):
        return

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    old_mat = omni.usd.get_local_transform_matrix(prim)

    rot = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_deg[0])
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation_deg[1])
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_deg[2])
    )
    offset = Gf.Matrix4d().SetRotate(rot)
    offset.SetTranslateOnly(Gf.Vec3d(*translation_m))

    # offset * old_mat -> offset expressed in the prim's local frame.
    # Swap to old_mat * offset for world/parent-axis offsets instead.
    new_mat = offset * old_mat
    omni.kit.commands.execute(
        "TransformPrimCommand",
        path=prim.GetPath(),
        new_transform_matrix=new_mat,
        old_transform_matrix=old_mat,
    )


async def assemble_gripper_onto_arm():
    """Attach the gripper base link to the arm flange via RobotAssembler.

    ``ARM_PRIM`` / ``GRIPPER_PRIM`` are the robot *base* prims. The mount frames
    are resolved by link name (``ARM_MOUNT_LINK`` / ``GRIPPER_MOUNT_LINK``) — the
    arm mount must be a rigid-body link, not an empty frame like ``tool0``.
    """
    # Assembly edits USD, so do it while the timeline is stopped.
    omni.timeline.get_timeline_interface().stop()

    stage = omni.usd.get_context().get_stage()

    if not stage.GetPrimAtPath(ARM_PRIM).IsValid():
        raise RuntimeError(f"Arm prim {ARM_PRIM!r} not found in the open stage.")
    if not stage.GetPrimAtPath(GRIPPER_PRIM).IsValid():
        raise RuntimeError(f"Gripper prim {GRIPPER_PRIM!r} not found in the open stage.")

    arm_mount = find_prim_by_name(stage, ARM_PRIM, ARM_MOUNT_LINK)
    gripper_mount = find_prim_by_name(stage, GRIPPER_PRIM, GRIPPER_MOUNT_LINK)

    assembler = RobotAssembler()
    # begin_assembly() positions the gripper at the mount; assemble() creates the
    # fixed joint and removes the gripper's own articulation root (so the merged
    # tree has ONE root at the arm). finish_assemble() is called after a few
    # simulated frames (see main()) — matching the upstream test/UI ordering.
    assembler.begin_assembly(
        stage,
        ARM_PRIM,
        arm_mount,
        GRIPPER_PRIM,
        gripper_mount,
        ASSEMBLY_NAMESPACE,
        VARIANT_NAME,
    )

    # Adjust the gripper pose relative to the mount with a custom offset.
    apply_attach_offset(
        GRIPPER_PRIM,
        ATTACH_OFFSET_TRANSLATION_M,
        ATTACH_OFFSET_ROTATION_DEG,
    )

    assembler.assemble()
    assembler.finish_assemble()

    app = omni.kit.app.get_app()
    await app.next_update_async()
    await app.next_update_async()


async def main():
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()

    # Capture joint identities by NAME before assembly (paths are stable here;
    # names survive the topology change, raw indices/paths may not).
    arm_names = [
        p.GetName()
        for p in Usd.PrimRange(stage.GetPrimAtPath(ARM_PRIM))
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)
    ]

    driver = find_driver_joint(stage, GRIPPER_PRIM)

    # assemble_gripper_onto_arm() stops the timeline, assembles, and finalizes.
    await assemble_gripper_onto_arm()

    timeline.play()

    # After assembly, arm + gripper are a single articulation rooted at the arm.
    # Build the handle with a small retry guard in case physics needs a few more
    # frames to stabilize the merged topology.
    app = omni.kit.app.get_app()
    robot = None
    for _ in range(60):
        try:
            robot = SingleArticulation(prim_path=ARM_PRIM, name=VARIANT_NAME)
            robot.initialize()
            if robot.num_dof and robot.num_dof > 0:
                break
        except Exception:
            robot = None
        await app.next_update_async()

    if robot is None or not robot.num_dof:
        raise RuntimeError(
            f"Articulation at {ARM_PRIM!r} did not become valid after assembly."
        )

    print(f"num_dof: {robot.num_dof}")
    print(f"dof_names: {robot.dof_names}")

    # Resolve indices by name (robust to articulation ordering). Gripper mimic
    # joints follow the driver joint automatically.
    dof_names = robot.dof_names
    arm_indices = [dof_names.index(n) for n in arm_names if n in dof_names]
    finger_idx = dof_names.index(driver)

    # 1) Move the arm (every arm joint to TARGET_DEG).
    arm_target = np.deg2rad(TARGET_DEG)
    robot.apply_action(
        ArticulationAction(
            joint_positions=arm_target.tolist(),
            joint_indices=arm_indices,
        )
    )
    print(f"Commanded arm joints {arm_names} -> {TARGET_DEG} deg")

    # 2) Wait a fixed number of steps so the arm can settle.
    for _ in range(WAIT_STEPS):
        await app.next_update_async()

    # 3) Close the gripper.
    finger_target = np.deg2rad([GRIPPER_CLOSE_DEG])
    robot.apply_action(
        ArticulationAction(
            joint_positions=finger_target.tolist(),
            joint_indices=[finger_idx],
        )
    )
    print(f"Commanded {driver} (deg): {GRIPPER_CLOSE_DEG}")

    # Poll until the gripper reaches the target.
    tolerance = 1e-3
    reached = False
    while not reached:
        await app.next_update_async()
        current = robot.get_joint_positions()[finger_idx]
        reached = abs(current - finger_target[0]) < tolerance

    print("\nDone. Assembled, arm moved, and gripper closed.")


asyncio.ensure_future(main())
