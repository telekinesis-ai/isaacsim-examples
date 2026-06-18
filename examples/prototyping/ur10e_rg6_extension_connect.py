"""
References:
-https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_simulation/articulation_controller.html
-https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/assemble_robots.html

UR10e + OnRobot RG6 — extension-mode CONNECT demo, native Isaac Sim APIs.

This script runs *inside* a live Isaac Sim session via the VS Code extension. It
does NOT assemble anything: it simply **connects** to a UR10e articulation and a
*separate* RG6 articulation that already exist in the open stage, then moves the
arm, waits, and closes the gripper. The two are independent articulations, so
each gets its own ``SingleArticulation`` handle.

(For the variant that attaches the gripper to the arm first, see
``ur10e_rg6_extension_assemble.py``.)

Setup before running:

* Open Isaac Sim with a stage containing a UR10e at ``UR10E_PATH`` and an RG6 at
  ``RG6_PATH`` (adjust the constants to match your stage).
* The script plays the timeline itself and ticks Kit a couple of times so the
  PhysX / tensor simulation view exists before ``initialize()``.
* Joint targets below are in **degrees**, converted to radians via ``np.deg2rad``.
"""
import asyncio
import numpy as np

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import omni.timeline
import omni.kit.app

# Existing articulation prims in the open stage.
UR10E_PATH = "/World/ur10e"
RG6_PATH = "/World/rg6"

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Arm target pose (deg) and gripper closed target (deg, within ~[-36, 36]).
ARM_TARGET_DEG = [90.0, -90.0, 90.0, 0.0, 90.0, 0.0]
GRIPPER_CLOSE_DEG = 30.0

# Sim steps to wait after the arm command before closing the gripper.
WAIT_STEPS = 120


async def main():
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Let Isaac Sim tick so PhysX / tensor simulation view exists.
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    # Two independent articulations -> two handles.
    robot = SingleArticulation(prim_path=UR10E_PATH, name="ur10e")
    robot.initialize()
    # gripper = SingleArticulation(prim_path=RG6_PATH, name="rg6")
    # gripper.initialize()
    return 
    print(f"arm dof_names: {robot.dof_names}")
    # print(f"gripper dof_names: {gripper.dof_names}")

    # Resolve joint indices by name (robust to articulation ordering).
    arm_indices = [robot.dof_names.index(n) for n in ARM_JOINT_NAMES]
    finger_idx = 7

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
        await omni.kit.app.get_app().next_update_async()

    # # 3) Close the gripper.
    # finger_target = np.deg2rad([GRIPPER_CLOSE_DEG])
    # gripper.apply_action(
    #     ArticulationAction(
    #         joint_positions=finger_target.tolist(),
    #         joint_indices=[finger_idx],
    #     )
    # )
    # print(f"Commanded finger_joint (deg): {GRIPPER_CLOSE_DEG}")

    # # Poll until the gripper reaches the target.
    # tolerance = 1e-3
    # reached = False
    # while not reached:
    #     await omni.kit.app.get_app().next_update_async()
    #     current = gripper.get_joint_positions()[finger_idx]
    #     reached = abs(current - finger_target[0]) < tolerance

    print("\nDone. Arm moved and gripper closed.")


asyncio.ensure_future(main())
