"""
References:
-https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_simulation/articulation_controller.html
-https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/assemble_robots.html

Manipulator + gripper — extension-mode CONNECT demo (manufacturer-agnostic).

This script runs *inside* a live Isaac Sim session via the VS Code extension. It
connects to an already-assembled manipulator+gripper articulation in the open
stage, moves the arm, waits, and closes the gripper.

Arm joint names and the gripper driver joint are discovered at runtime from the
USD prim tree — no robot-specific names are hardcoded. Swap robots by editing
the config block only.

Setup before running:
* Open Isaac Sim with a stage containing an assembled manipulator+gripper
  articulation at ``MANIPULATOR_PRIM`` (e.g. produced by
  ``manipulator_gripper_extension_assemble.py``).
* Joint targets are in **degrees**, converted to radians via ``np.deg2rad``.
"""
import asyncio
import numpy as np

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import omni.timeline
import omni.usd
import omni.kit.app
from pxr import Usd, UsdPhysics

# ---------------------------------------------------------------------------
# Config — edit these to match your stage / robots.
# ---------------------------------------------------------------------------
MANIPULATOR_PRIM = "/World/manipulator"   # assembled articulation root
GRIPPER_PRIM = "/World/rg6"              # gripper subtree root (for joint discovery)

TARGET_DEG = [90,0,0,0,0,0]        # applied to every arm joint
GRIPPER_CLOSE_DEG = 30.0   # gripper driver-joint close target
WAIT_STEPS = 120           # sim steps to settle the arm before closing


def _arm_joint_names(stage, arm_prim_path):
    """Discover all revolute/prismatic joint names under the arm prim."""
    names = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(arm_prim_path)):
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
            names.append(p.GetName())
    return names


def _find_driver_joint(stage, root_path):
    """The gripper's actuated driver joint: non-mimic, preferring a DriveAPI."""
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


def _build_articulation(prim_path, name):
    """Construct + initialize a SingleArticulation, or None if not ready yet."""
    try:
        art = SingleArticulation(prim_path=prim_path, name=name)
        art.initialize()
        if art.num_dof and art.num_dof > 0:
            return art
    except Exception:
        pass
    return None


async def main():
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    app = omni.kit.app.get_app()
    stage = omni.usd.get_context().get_stage()

    if not stage.GetPrimAtPath(MANIPULATOR_PRIM).IsValid():
        raise RuntimeError(f"Manipulator prim {MANIPULATOR_PRIM!r} not found in the open stage.")

    # Discover joint names before physics initializes (prim tree is stable here).
    arm_names = _arm_joint_names(stage, MANIPULATOR_PRIM)
    driver = _find_driver_joint(stage, GRIPPER_PRIM)
    print(f"Discovered arm joints: {arm_names}")
    print(f"Discovered gripper driver joint: {driver!r}")

    robot = None
    for _ in range(60):
        if robot is None:
            robot = _build_articulation(MANIPULATOR_PRIM, "manipulator")
        if robot is not None:
            break
        await app.next_update_async()

    if robot is None:
        raise RuntimeError(f"Articulation at {MANIPULATOR_PRIM!r} did not become valid.")

    print(f"num_dof: {robot.num_dof}")
    print(f"dof_names: {robot.dof_names}")

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
    print(f"Commanded arm joints -> {TARGET_DEG} deg")

    # 2) Wait for the arm to settle.
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
    print(f"Commanded {driver!r} (deg): {GRIPPER_CLOSE_DEG}")

    # Poll until the gripper reaches the target.
    tolerance = 1e-3
    reached = False
    while not reached:
        await app.next_update_async()
        current = robot.get_joint_positions()[finger_idx]
        reached = abs(current - finger_target[0]) < tolerance

    print("\nDone. Arm moved and gripper closed.")


asyncio.ensure_future(main())
