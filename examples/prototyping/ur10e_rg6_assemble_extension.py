"""
References:
-https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_simulation/articulation_controller.html
-https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/assemble_robots.html
-https://github.com/isaac-sim/IsaacSim/blob/40316786340b3f034a229d9e12650df1ac0b68ab/source/extensions/isaacsim.robot_setup.assembler/isaacsim/robot_setup/assembler/ui/ui_builder.py#L64
-https://github.com/isaac-sim/IsaacSim/blob/40316786340b3f034a229d9e12650df1ac0b68ab/source/extensions/isaacsim.robot_setup.assembler/isaacsim/robot_setup/assembler/tests/test_robot_assembler.py#L44

How to use:
- Add ur10e to scene
- Add rg6 to scene
- run this with extension

Notes:
- Attachin points have to be RigidBodyAPI liek wrist 3 and not simple frames like flange or tool0 

UR10e + OnRobot RG6 — extension-mode ASSEMBLE demo, native Isaac Sim APIs.

This script runs *inside* a live Isaac Sim session via the VS Code extension. It
connects to a UR10e articulation and a *separate* RG6 articulation already in the
open stage, **assembles** the RG6 onto the UR10e ``tool0`` flange with
``RobotAssembler``, then moves the arm, waits, and closes the gripper.

After assembly the RG6 is fixed to the arm, so the whole rig becomes a SINGLE
articulation rooted at the UR10e — one ``SingleArticulation`` handle drives both
the 6 arm joints and the gripper's ``finger_joint``.

(For the variant that drives two independent articulations without assembly, see
``ur10e_rg6_extension_connect.py``.)

Setup before running:

* Open Isaac Sim with a stage containing a UR10e articulation root at
  ``UR10E_PRIM`` and an RG6 articulation root at ``RG6_PRIM`` (adjust to match
  your stage).
* Assembly edits USD, so it is done before the timeline is played; the script
  then plays the timeline and ticks Kit so the physics view exists.
* Joint targets below are in **degrees**, converted to radians via ``np.deg2rad``.
"""
import asyncio
import numpy as np

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import omni.timeline
import omni.usd
import omni.kit.app
import omni.kit.commands
from pxr import Gf

# Robot Assembler is packaged as an extension. Enable it before importing so the
# import doesn't fail if the extension isn't loaded yet.
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.robot_setup.assembler", True)
from isaacsim.robot_setup.assembler import RobotAssembler

# Existing articulation-root prims in the open stage.
UR10E_PRIM = "/World/ur10e"
RG6_PRIM = "/World/rg6"

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Arm target pose (deg) and gripper closed target (deg, within ~[-36, 36]).
ARM_TARGET_DEG = [0.0, -90.0, 90.0, 0.0, 90.0, 0.0]
GRIPPER_CLOSE_DEG = 30.0

# Sim steps to wait after the arm command before closing the gripper.
WAIT_STEPS = 120

# Custom mount offset applied to the gripper, in its mount-local frame.
ATTACH_OFFSET_TRANSLATION_M = (0.0, 0.0, 0.1)      # meters (x, y, z)
ATTACH_OFFSET_ROTATION_DEG = (0.0, 0.0, 0.0)       # XYZ Euler degrees


def _apply_attach_offset(prim_path, translation_m, rotation_deg):
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
    """Attach the RG6 base link to the UR10e tool0 flange via RobotAssembler.

    ``UR10E_PRIM`` / ``RG6_PRIM`` are the robot *base* prims (the xforms whose
    children are the links). The mount frames are direct children:
    ``<arm>/tool0`` and ``<gripper>/onrobot_rg6_base_link``. Do NOT use the
    parent path here — using the parent (e.g. ``/World``) makes the mount frame
    invalid, and if it collides with the stage DefaultPrim the assembler drops
    into a broken "direct edit" export path.
    """
    # Assembly edits USD, so do it while the timeline is stopped.
    omni.timeline.get_timeline_interface().stop()

    stage = omni.usd.get_context().get_stage()

    # Base + mount path of the UR10e.
    ur10e_base = UR10E_PRIM
    ur10e_base_mount = "/World/ur10e/wrist_3_link" #TODO find dinamically
    # Base + mount path of the RG6 (the attach robot).
    onrobot_rg6_attach = RG6_PRIM
    onrobot_rg6_attach_mount = f"/World/rg6/onrobot_rg6_base_link"  #TODO find dinamically

    # Fail early with a clear message if the stage layout doesn't match.
    for path in (ur10e_base, ur10e_base_mount, onrobot_rg6_attach, onrobot_rg6_attach_mount):
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(
                f"Prim {path!r} not found in the open stage. Adjust UR10E_PRIM / "
                f"RG6_PRIM (and the mount-frame child names) to match your stage."
            )

    assembler = RobotAssembler()
    # begin_assembly() positions the gripper at the mount; assemble() creates the
    # fixed joint and removes the RG6's own articulation root (so the merged tree
    # has ONE root at the UR10e). finish_assemble() is deliberately NOT called
    # here: the upstream test/UI simulate the assembly for several frames between
    # assemble() and finish_assemble(), so main() drives that timing.
    assembler.begin_assembly(
        stage,
        ur10e_base,
        ur10e_base_mount,
        onrobot_rg6_attach,
        onrobot_rg6_attach_mount,
        "Gripper",          # assembly namespace
        "ur10e_with_rg6",   # variant name
    )


    # Adjust the gripper pose relative to the mount with a custom offset.
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(onrobot_rg6_attach)
    old_mat = omni.usd.get_local_transform_matrix(prim)

    rot = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), ATTACH_OFFSET_ROTATION_DEG[0])
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), ATTACH_OFFSET_ROTATION_DEG[1])
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), ATTACH_OFFSET_ROTATION_DEG[2])
    )
    offset = Gf.Matrix4d().SetRotate(rot)
    offset.SetTranslateOnly(Gf.Vec3d(*ATTACH_OFFSET_TRANSLATION_M))

    # offset * old_mat -> offset expressed in the prim's local frame.
    # Swap to old_mat * offset for world/parent-axis offsets instead.
    new_mat = offset * old_mat
    omni.kit.commands.execute(
        "TransformPrimCommand",
        path=prim.GetPath(),
        new_transform_matrix=new_mat,
        old_transform_matrix=old_mat,
    )



    assembler.assemble()
    assembler.finish_assemble()

    app = omni.kit.app.get_app()
    await app.next_update_async()
    await app.next_update_async()


async def main():
    timeline = omni.timeline.get_timeline_interface()

    # assemble_gripper_onto_arm() stops the timeline, assembles, and finalizes.
    await assemble_gripper_onto_arm()
    
    timeline.play()

    # After assembly, UR10e + RG6 are a single articulation rooted at the arm.
    # Build the handle with a small retry guard in case physics needs a few more
    # frames to stabilize the merged topology.
    app = omni.kit.app.get_app()
    robot = None
    for _ in range(60):
        try:
            robot = SingleArticulation(prim_path=UR10E_PRIM, name="ur10e_with_rg6")
            robot.initialize()
            if robot.num_dof and robot.num_dof > 0:
                break
        except Exception:
            robot = None
        await app.next_update_async()

    if robot is None or not robot.num_dof:
        raise RuntimeError(
            f"Articulation at {UR10E_PRIM!r} did not become valid after assembly."
        )

    print(f"num_dof: {robot.num_dof}")
    print(f"dof_names: {robot.dof_names}")

    # Resolve joint indices by name (robust to articulation ordering). The other
    # 5 RG6 joints are URDF mimic joints that follow finger_joint automatically.
    arm_indices = [robot.dof_names.index(n) for n in ARM_JOINT_NAMES]
    finger_idx = robot.dof_names.index("finger_joint")

    # 1) Move the arm.
    arm_target = np.deg2rad(ARM_TARGET_DEG)
    robot.apply_action(
        ArticulationAction(
            joint_positions=arm_target.tolist(),
            joint_indices=arm_indices,
        )
    )
    print(f"Commanded arm (deg): {ARM_TARGET_DEG}")

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
    print(f"Commanded finger_joint (deg): {GRIPPER_CLOSE_DEG}")

    # Poll until the gripper reaches the target.
    tolerance = 1e-3
    reached = False
    while not reached:
        await app.next_update_async()
        current = robot.get_joint_positions()[finger_idx]
        reached = abs(current - finger_target[0]) < tolerance

    print("\nDone. Assembled, arm moved, and gripper closed.")


asyncio.ensure_future(main())
