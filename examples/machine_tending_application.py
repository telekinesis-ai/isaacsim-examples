"""First stage of the UR10e machine-tending application in Isaac Sim.

The UR10e and Robotiq 2F-85 must already exist in the open Isaac Sim stage.
This script reads the world pose of the scene's mount marker, places the
existing robot on that marker, connects both articulations through Synapse,
and attaches the gripper to the robot.

Before running:
    1. Open the machine-tending USD in Isaac Sim.
    2. Enable the local Telekinesis Isaac Sim bridge extension.
    3. Import the UR10e and Robotiq 2F-85 into the stage at the prim paths below.
    4. Start from an unassembled scene.
    5. Run this file from the Synapse Conda environment:
       python examples/machine_tending_application.py

This example is an external application. It intentionally imports neither
``omni`` nor ``isaacsim``; all scene operations cross the local bridge.
"""

from __future__ import annotations

import math
import time

import requests

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import robotiq


BASE_URL = "http://127.0.0.1:8766"
REQUEST_TIMEOUT_SECONDS = 120.0

# Existing scene marker whose world pose defines the robot-base placement.
MOUNT_PRIM_PATH = "/World/sortbot_housing/Mount_point"

# Add one entry here for each robot used by the machine-tending application.
ACTIVE_ROBOT = "ur10e"
ROBOT_CONFIGS = {
    "ur10e": {
        "class": universal_robots.UniversalRobotsUR10E,
        "display_name": "UR10e",
        "instance_name": "machine_tending_ur10e",
        "prim_path": "/World/ur10e_robot",
        "mount_offset_z": -0.005,
        "yaw_offset_radians": math.radians(180.0),
        "home_joint_positions": [-90.0, -90.0, -90.0, 0.0, 90.0, 0.0],
        "drive_stiffness": 1.0e5,
        "drive_damping": 1.0e4,
    },
}
ROBOT_CONFIG = ROBOT_CONFIGS[ACTIVE_ROBOT]
ROBOT_PRIM_PATH = ROBOT_CONFIG["prim_path"]

# Prim path of the gripper already present in the open stage.
GRIPPER_PRIM_PATH = "/World/Robotiq_2F_85_edit"
GRIPPER_TCP_NAME = "gripper_tcp"
GRIPPER_TCP_OFFSET = [0.0, 0.0, 0.175, 0.0, 0.0, 0.0]
ASSEMBLY_SETTLE_SECONDS = 1.0

# A motion call returns once the joints are close enough or have stopped moving,
# which can leave the drive a fraction of a degree from the target. This gives it
# time to close that before the pose is read back.
MOTION_SETTLE_SECONDS = 1.0

# CNC door configuration from the current machine-tending scene.
CNC_DOOR_PRIM_PATH = "/World/model_cnc_machine_tool/E_body_1/door"
CNC_DOOR_OPEN_X = -0.68654
CNC_DOOR_CLOSED_X = -0.20938
CNC_DOOR_WORLD_X_PER_LOCAL_X = 2.54
CNC_DOOR_MOVE_SECONDS = 2.0
CNC_DOOR_MOVE_STEPS = 60
CNC_PROCESS_SECONDS = 3.0


class CNCMachine:
    """Represent the CNC machine used by the application."""

    def __init__(self, door_prim_path: str, open_x: float, closed_x: float) -> None:
        self.door_prim_path = door_prim_path
        self.open_x = open_x
        self.closed_x = closed_x
        self.part_completed = False

    def get_door_pose(self) -> list[float]:
        """Read the door pose relative to its parent prim."""
        result = bridge_request(
            "GET",
            "/prims/poses",
            params={
                "prim_path": self.door_prim_path,
                "coordinate_system": "local",
                "rotation_type": "cartesian",
            },
        )
        return [float(value) for value in result["pose"]]

    def move_door(self, target_x: float) -> None:
        """Move smoothly to a parent-local X position."""
        start_x = self.get_door_pose()[0]
        local_distance = target_x - start_x
        world_distance_x = local_distance * CNC_DOOR_WORLD_X_PER_LOCAL_X
        previous_progress = 0.0

        for step in range(1, CNC_DOOR_MOVE_STEPS + 1):
            progress = step / CNC_DOOR_MOVE_STEPS
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            step_world_x = world_distance_x * (
                smooth_progress - previous_progress
            )

            bridge_request(
                "POST",
                "/prims/poses/relative",
                body={
                    "prim_path": self.door_prim_path,
                    "relative_pose": {
                        "pose": [step_world_x, 0.0, 0.0, 0.0, 0.0, 0.0]
                    },
                    "object_first": False,
                },
            )
            previous_progress = smooth_progress
            time.sleep(CNC_DOOR_MOVE_SECONDS / CNC_DOOR_MOVE_STEPS)

    def open_door(self) -> None:
        """Move the door smoothly to its fully open position."""
        self.move_door(self.open_x)

    def close_door(self) -> None:
        """Move the door smoothly to its fully closed position."""
        self.move_door(self.closed_x)

    def process_part(self) -> None:
        """Simulate one CNC machining cycle."""
        self.part_completed = False
        print("CNC machining started...")
        time.sleep(CNC_PROCESS_SECONDS)
        self.part_completed = True
        print("CNC machining completed.")

    def is_part_complete(self) -> bool:
        """Return whether the current machining cycle has completed."""
        return self.part_completed

def bridge_request(method, path, *, params=None, body=None):
    """Send one request to the local Isaac Sim bridge and decode its JSON."""
    response = requests.request(
        method,
        BASE_URL + path,
        params=params,
        json=body,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json() if response.content else None


def stop_timeline() -> None:
    """Stop physics before importing or changing a top-level world pose."""
    bridge_request("PATCH", "/stage/simulation/timeline/stop")


def get_world_pose(prim_path: str) -> list[float]:
    """Return a prim world pose as metres plus rotation-vector radians."""
    result = bridge_request(
        "GET",
        "/prims/poses",
        params={
            "prim_path": prim_path,
            "coordinate_system": "world",
            "rotation_type": "cartesian",
        },
    )
    return [float(value) for value in result["pose"]]


def prim_exists(prim_path: str) -> bool:
    """Return whether a prim exists without modifying the open stage."""
    try:
        get_world_pose(prim_path)
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == 404:
            return False
        raise
    return True


def require_prim(prim_path: str) -> None:
    """Raise a clear error when an expected stage prim is missing."""
    if not prim_exists(prim_path):
        raise RuntimeError(
            f"Required prim not found: {prim_path}. "
            "Import it into the Isaac Sim stage before running this script."
        )


def set_world_pose(prim_path: str, pose: list[float]) -> None:
    """Set a top-level world pose using metres and rotation-vector radians."""
    bridge_request(
        "PUT",
        "/prims/poses",
        body={"prim_path": prim_path, "input_pose": {"pose": pose}},
    )


def register_articulation(prim_path: str) -> dict:
    """Register an existing articulation and return its bridge information."""
    return bridge_request(
        "PUT",
        "/articulations",
        body={"prim_path": prim_path},
    )


def apply_drive_gains(articulation_id: str) -> dict:
    """Retune the arm's position drives, which a URDF import leaves too soft.

    This applies to the running simulation rather than to the stage, so it is
    redone on every run and nothing is written back into the USD.
    """
    return bridge_request(
        "POST",
        f"/articulations/{articulation_id}/dof_gains",
        body={
            "stiffness": ROBOT_CONFIG["drive_stiffness"],
            "damping": ROBOT_CONFIG["drive_damping"],
        },
    )


def robot_pose_from_mount(mount_pose: list[float]) -> list[float]:
    """Apply the scene-specific height and yaw offsets to the mount pose."""
    robot_pose = mount_pose.copy()
    robot_pose[2] += ROBOT_CONFIG["mount_offset_z"]

    # This scene's mount has a pure Z-axis rotation vector. Normalize the
    # resulting yaw to [-pi, pi] so pi + pi is represented cleanly as zero.
    yaw = robot_pose[5] + ROBOT_CONFIG["yaw_offset_radians"]
    robot_pose[5] = math.atan2(math.sin(yaw), math.cos(yaw))
    return robot_pose


def main() -> None:
    """Place, connect, and assemble the existing UR10e and Robotiq 2F-85."""
    bridge_request("GET", "/status")

    cnc = CNCMachine(
        door_prim_path=CNC_DOOR_PRIM_PATH,
        open_x=CNC_DOOR_OPEN_X,
        closed_x=CNC_DOOR_CLOSED_X,
    )

    require_prim(MOUNT_PRIM_PATH)
    require_prim(ROBOT_PRIM_PATH)
    require_prim(GRIPPER_PRIM_PATH)
    require_prim(cnc.door_prim_path)
    print(f"CNC door local pose: {cnc.get_door_pose()}")

    mount_pose = get_world_pose(MOUNT_PRIM_PATH)
    robot_world_pose = robot_pose_from_mount(mount_pose)
    print(f"Mount world pose: {mount_pose}")

    # Place the robot while physics is stopped. Synapse connect starts playback.
    print(f"Placing the {ROBOT_CONFIG['display_name']} on the mount...")
    stop_timeline()
    set_world_pose(ROBOT_PRIM_PATH, robot_world_pose)
    print(
        f"{ROBOT_CONFIG['display_name']} world pose: "
        f"{get_world_pose(ROBOT_PRIM_PATH)}"
    )

    robot = ROBOT_CONFIG["class"](name=ROBOT_CONFIG["instance_name"])
    gripper = robotiq.Robotiq2F85()

    # Register the fingertip frame while the robot is still offline. This
    # avoids reading simulation state during the post-assembly rebind.
    robot.add_tcp(GRIPPER_TCP_NAME, GRIPPER_TCP_OFFSET)
    print(f"Active TCP: {robot.active_tcp}")

    try:
        print("Connecting through Synapse...")
        robot.connect(simulation_prim_path=ROBOT_PRIM_PATH)

        # Report the gripper's imported joints and detected driver.
        gripper_articulation = register_articulation(GRIPPER_PRIM_PATH)
        gripper_driver = bridge_request(
            "GET",
            f"/articulations/{gripper_articulation['articulation_id']}"
            "/driver_joint",
        )
        print(f"Robotiq imported DOFs: {gripper_articulation['dof_names']}")
        print(f"Robotiq detected driver: {gripper_driver}")

        gripper.connect(simulation_prim_path=GRIPPER_PRIM_PATH)
        try:
            print(
                f"Attaching the Robotiq 2F-85 to the "
                f"{ROBOT_CONFIG['display_name']}..."
            )
            robot.attach_tool(gripper)
            time.sleep(ASSEMBLY_SETTLE_SECONDS)

            # After assembly, because the gains apply to the running simulation
            # and assembly restarts it, which reloads the drives from the USD.
            robot_articulation = register_articulation(ROBOT_PRIM_PATH)
            drive_properties = apply_drive_gains(robot_articulation["articulation_id"])
            print(
                f"Drive gains set (stiffness={ROBOT_CONFIG['drive_stiffness']:g}, "
                f"damping={ROBOT_CONFIG['drive_damping']:g})."
            )
            print(f"Resulting drive properties: {drive_properties}")

            print(f"Moving the {ROBOT_CONFIG['display_name']} to its home position...")
            robot.set_joint_positions(
                ROBOT_CONFIG["home_joint_positions"],
                speed=20.0,
                acceleration=30.0,
            )

            time.sleep(MOTION_SETTLE_SECONDS)

            joint_positions = robot.get_joint_positions()
            gripper_position = gripper.get_current_position()
            print(f"{ROBOT_CONFIG['display_name']} joints (degrees): {joint_positions}")
            print(f"Gripper position (mm): {gripper_position}")
            print("Initial machine-tending setup completed.")
        finally:
            gripper.disconnect()
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
