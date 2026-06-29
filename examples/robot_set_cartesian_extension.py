"""
MoveL prototype — linear Cartesian motion via cuMotion TrajectoryGenerator.

Generates a straight-line end-effector path (moveL) from the robot's current
pose to a user-defined target pose, then executes the resulting joint trajectory.

Setup before running:
* Open Isaac Sim with a stage containing a UR10e articulation at PRIM_PATH.
* Timeline must be playing (the script starts it automatically).

References:
- https://docs.isaacsim.omniverse.nvidia.com/6.0.1/motion_generation/trajectory_planning.html
"""

import asyncio

import cumotion
import numpy as np
import omni.kit.app
import omni.timeline

from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.cumotiTrajectoryGeneratoron import TrajectoryGenerator, load_cumotion_supported_robot
from isaacsim.robot_motion.cumotion.impl.utils import isaac_sim_to_cumotion_pose

# ── configuration ──────────────────────────────────────────────────────────────

PRIM_PATH = "/World/ur10e"
ROBOT_NAME = "ur10e"    # cuMotion-supported robot name
TOOL_FRAME = "tool0"    # end-effector frame defined in the robot URDF

# Target end-effector pose in world space.
# Orientation as quaternion [w, x, y, z].
TARGET_POSITION = np.array([0.4, 0.2, 0.5], dtype=np.float64)
TARGET_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

# ── main ───────────────────────────────────────────────────────────────────────


async def main():
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    robot = Articulation(PRIM_PATH)

    # Wait until physics tensors are ready.
    while not robot.is_physics_tensor_entity_valid():
        await omni.kit.app.get_app().next_update_async()

    print(f"dof_names: {robot.dof_names}")

    # ── build cuMotion generator ──────────────────────────────────────────────

    robot_config = load_cumotion_supported_robot(ROBOT_NAME)
    generator = TrajectoryGenerator(
        cumotion_robot=robot_config,
        robot_joint_space=robot.dof_names,
    )

    # ── compute start EE pose from current joint positions via FK ─────────────

    current_joints = robot.get_dof_positions().numpy().flatten()
    start_pose = robot_config.kinematics.pose(current_joints, TOOL_FRAME)

    # ── build linear task-space path spec ─────────────────────────────────────

    robot_positions, robot_orientations = robot.get_world_poses()

    target_pose = isaac_sim_to_cumotion_pose(
        position_world_to_target=TARGET_POSITION,
        orientation_world_to_target=TARGET_ORIENTATION,
        position_world_to_base=robot_positions,
        orientation_world_to_base=robot_orientations,
    )

    path_spec = cumotion.create_task_space_path_spec(start_pose)
    path_spec.add_linear_path(target_pose)

    trajectory = generator.generate_trajectory_from_path_specification(
        path_specification=path_spec,
        tool_frame_name=TOOL_FRAME,
    )

    if trajectory is None:
        print("Failed to generate trajectory — target pose may be unreachable.")
        return

    print(f"Trajectory generated. Duration: {trajectory.duration:.2f}s")

    # ── follow trajectory ─────────────────────────────────────────────────────

    dt = SimulationManager.get_physics_dt()
    trajectory_time = 0.0

    while trajectory_time <= trajectory.duration:
        desired_state = trajectory.get_target_state(trajectory_time)
        if desired_state is not None:
            robot.set_dof_positions(
                positions=desired_state.joints.positions,
                dof_indices=desired_state.joints.position_indices,
            )
        trajectory_time += dt
        await omni.kit.app.get_app().next_update_async()

    print("MoveL complete.")


asyncio.ensure_future(main())
