"""
Example: drive a Kuka inside Isaac Sim using the same Synapse robot class as for
real hardware.

Add the robot to your open USD stage and set SIM_ROBOT_PRIM_PATH to its prim path
(right-click the prim in the Stage tree -> "Copy Prim Path"). The robot must already
exist in the stage; if the prim is absent, connect_async() raises an error (add it via
Isaac Sim's URDF Importer or by referencing its USD first).

This script targets the **Isaac Sim VS Code extension "Run" button** (or the Kit
script editor), which executes your code inside an already-running Kit asyncio loop.
The interface is therefore driven through its async API: we wrap the work in
`async def run()` and schedule it with `asyncio.ensure_future(run())` — NOT
`asyncio.run()`, which would try to start a second event loop. Every wait yields to
the running loop via `await` instead of synchronously pumping `app.update()` (which
would re-enter the loop and flood "Cannot enter into task ..." errors).

You manage the timeline: run() plays it before connecting. The interface never
stops/plays/pauses the timeline, so connecting never restarts the simulation.
"""

import numpy as np
import omni.timeline

from telekinesis.synapse.robots.manipulators import kuka
from isaacsim.core.utils.types import ArticulationAction

SIM_ROBOT_PRIM_PATH = "/World/robot_assembly/kuka_kr120r2500pro"
# SIM_ROBOT_PRIM_PATH = "/World/RobotAssembly/kuka_kr120r2500pro"


def main():
    # You own the timeline — play it before connecting.
    omni.timeline.get_timeline_interface().play()

    # Create robot and connect to sim (async: loop-safe inside the extension).
    robot = kuka.KukaKR120R2500PRO()
    robot.connect(simulation_prim_path=SIM_ROBOT_PRIM_PATH)

    # print("Joint positions (deg):", robot.state.joint_positions)
    # print("TCP pose (m, deg):    ", robot.state.tcp_pose)

    actuator = robot._communication_interface._active_actuator()
    joint_indices = robot._communication_interface._arm_joint_indices(actuator)

    # actuator.apply_action(
    #         ArticulationAction(
    #             joint_positions=np.deg2rad([0, -70, 100, 0, 60, 90]).tolist(),
    #             joint_indices=joint_indices,
    #         )
    #     )
    
    # render?

    # Move to default joint configuration before commanding Cartesian pose,
    # to avoid large joint movements.
    print(f"Setting joints {robot.default_joint_configuration}")
    robot.set_joint_positions([0, -70, 100, 0, 60, 90], asynchronous=False)
    print("Joint positions after move:", robot.state.joint_positions)


    # Cartesian round-trip. Inside the VS Code extension only the last command
    # of a single Run takes effect, so run this on its own Run to watch it move.
    pose = [0.3, -1.4, 1, -180, 0, -180]
    print(f"\nCommanding pose (m, deg): {pose}")

    # Move the robot.
    robot.set_cartesian_pose(pose, speed=100)
    print("Joint positions after move:", robot.state.joint_positions)

    # Cartesian round-trip. Inside the VS Code extension only the last command
    # of a single Run takes effect, so run this on its own Run to watch it move.
    pose = [0.3, -1.5, 0.5, -180, 0, -180]
    print(f"\nCommanding pose (m, deg): {pose}")

    # Move the robot.
    robot.set_cartesian_pose(pose, speed=100)
    print("Joint positions after move:", robot.state.joint_positions)

    robot.disconnect()
    # omni.timeline.get_timeline_interface().stop()

main()

