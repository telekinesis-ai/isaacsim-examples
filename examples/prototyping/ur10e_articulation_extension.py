"""
References:
-https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_simulation/articulation_controller.html
-https://docs.isaacsim.omniverse.nvidia.com/6.0.0/core_api_tutorials/tutorial_core_adding_manipulator.html

UR10e (no gripper) — extension-mode example, native Isaac Sim APIs.

Unlike ``ur10e_with_rg6 copy.py`` (which launches its own ``SimulationApp`` and
imports the UR10e from URDF), this script runs *inside* a live Isaac Sim session
via the VS Code extension and simply **connects** to a UR10e articulation that
already exists in the open stage, using the native ``SingleArticulation`` API.

Setup before running:

* Open Isaac Sim with a stage that contains a UR10e articulation at
  ``PRIM_PATH`` below.
* **Press Play in the Isaac Sim GUI first** and keep it playing. The extension
  runs this script on Kit's main event loop, which is blocked for the whole run,
  so no physics steps happen while the script executes — the arm only moves
  *after* the script returns and the timeline resumes ticking.
* Joint targets below are in **radians** (what the native articulation API
  expects).
"""
import asyncio
import numpy as np

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import omni.timeline

# Existing UR10e articulation prim in the open stage.
PRIM_PATH = "/World/ur10e"


async def main():
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Let Isaac Sim tick at least once so PhysX / tensor simulation view exists.
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()
    # Bind to the articulation already present in the stage. The timeline must be
    # playing for the physics view to be valid, hence initialize() here.
    robot = SingleArticulation(prim_path=PRIM_PATH, name="ur10e_no_gripper")
    robot.initialize()

    print(f"num_dof: {robot.num_dof}")
    print(f"dof_names: {robot.dof_names}")
    print(f"current joints (rad): {robot.get_joint_positions()}")

    # One-shot pose command (radians). Fire-and-forget: this just sets the PD
    # target; the arm drives toward it after the script returns and the timeline
    # keeps ticking.
    target = np.deg2rad([0.0, -90.0, 0.0, 0.0, 90.0, 0.0])
    robot.apply_action(
        ArticulationAction(
            joint_positions=target,
            joint_indices=np.arange(robot.num_dof).tolist(),
        )
    )

    tolerance = 1e-4
    reached = False
    while not reached:
        await omni.kit.app.get_app().next_update_async()
        current = robot.get_joint_positions()
        reached = np.max(np.abs(current - target)) < tolerance

    print(f"Commanded joints (rad): {target}")
    print("\nDone. Watch the viewport — the arm drives to the target now.")

asyncio.ensure_future(main())
