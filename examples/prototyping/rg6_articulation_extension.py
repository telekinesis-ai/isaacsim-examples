"""
References:
-https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_simulation/articulation_controller.html
-https://docs.isaacsim.omniverse.nvidia.com/6.0.0/core_api_tutorials/tutorial_core_adding_manipulator.html

OnRobot RG6 gripper — extension-mode demo, native Isaac Sim APIs.

Unlike ``rg6_articulation_standalone.py`` (which launches its own
``SimulationApp`` and imports the RG6 from URDF), this script runs *inside* a
live Isaac Sim session via the VS Code extension and simply **connects** to an
RG6 articulation that already exists in the open stage, using the native
``SingleArticulation`` API.

Setup before running:

* Open Isaac Sim with a stage that contains an RG6 articulation at ``PRIM_PATH``
  below (its base should be fixed so it doesn't fall).
* The script plays the timeline itself and ticks Kit a couple of times so the
  PhysX / tensor simulation view exists before ``initialize()``.
* Joint targets below are in **degrees** and converted to radians (what the
  native articulation API expects) via ``np.deg2rad``. finger_joint limits are
  ~[-36, 36] deg (~[-0.628, 0.628] rad).
"""
import asyncio
import numpy as np

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import omni.timeline

# Existing RG6 articulation prim in the open stage.
PRIM_PATH = "/onrobot_rg6_model"

# finger_joint target (deg): 0 ~ open, 30 ~ closed (within ~[-36, 36]).
CLOSE_DEG = 34


async def main():
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Let Isaac Sim tick at least once so PhysX / tensor simulation view exists.
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()
    # Bind to the articulation already present in the stage. The timeline must be
    # playing for the physics view to be valid, hence initialize() here.
    gripper = SingleArticulation(prim_path=PRIM_PATH, name="rg6")
    gripper.initialize()

    print(f"num_dof: {gripper.num_dof}")
    print(f"dof_names: {gripper.dof_names}")
    print(f"current joints (rad): {gripper.get_joint_positions()}")

    # `finger_joint` is the single actuated joint; the other 5 knuckle/finger
    # joints are URDF mimic joints and should follow it automatically if the
    # importer created PhysX mimic-joint constraints. If they don't, command all
    # DOFs with the ±1 multipliers or use ParallelGripper(use_mimic_joints=True).
    finger_idx = gripper.dof_names.index("finger_joint")

    # One-shot close command. Fire-and-forget: this just sets the PD target; the
    # fingers drive toward it as the timeline keeps ticking.
    target = np.deg2rad([CLOSE_DEG])
    gripper.apply_action(
        ArticulationAction(
            joint_positions=target.tolist(),
            joint_indices=[finger_idx],
        )
    )

    tolerance = 1e-4
    reached = False
    while not reached:
        await omni.kit.app.get_app().next_update_async()
        current = gripper.get_joint_positions()[finger_idx]
        reached = abs(current - target[0]) < tolerance

    print(f"Commanded finger_joint (deg): {CLOSE_DEG}")
    print("\nDone. Watch the viewport — the gripper closes now.")

asyncio.ensure_future(main())
