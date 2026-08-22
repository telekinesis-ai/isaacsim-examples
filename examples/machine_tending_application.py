"""UR10e machine tending in Isaac Sim, for every part it finds in the tray.

    tray -> CNC -> [door, machine, door] -> CNC -> container

Each reach is two moves: an L pose (folded, base turned to face the work,
commanded in JOINT space) then Cartesian from there in, so the descent is a
straight vertical line. The arm refolds between operations so it crosses the
cell tucked up. Static targets come from the cell's TF tree in the robot-base
frame. The expected object and downward-facing grasp targets are named static
frames; live frames that track the physical cylinders are deliberately omitted.

Before running:
    1. Open the machine-tending USD in Isaac Sim.
    2. Enable the local Telekinesis Isaac Sim bridge extension.
    3. Import the UR10e and OnRobot RG2 at the prim paths below. Import the RG2
       with the dialog's Natural Frequency set to 0, or its fingers sag.
    4. Run from the examples:
       python examples/machine_tending_application.py

This example is an external application. It imports neither ``omni`` nor
``isaacsim``; all scene operations cross the local bridge.
"""

from __future__ import annotations

import math
import time

import requests

from telekinesis.tf import tftree, tfutils
from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot

from cnc_machine_tending_static_frames_visualization import (
    CNC_GRASP_FRAME,
    CNC_PRE_FRAME,
    CNC_PRIM_PATH,
    PICK_GRASP_FRAME_PREFIX,
    PICK_PRE_FRAME_PREFIX,
    PLACE_GRASP_FRAME_PREFIX,
    PLACE_PRE_FRAME_PREFIX,
    TABLE_FRAME_PRIM_PATH,
    build_static_frame_tree,
)


BASE_URL = "http://127.0.0.1:8766"
REQUEST_TIMEOUT_SECONDS = 120.0

ROBOT_PRIM_PATH = "/World/ur10e_robot"
GRIPPER_PRIM_PATH = "/World/onrobot_rg2_model"
PART_COUNT = 16

# A URDF import leaves the drives soft: the arm oscillates, the fingers swing shut.
ARM_DRIVE_STIFFNESS = 1.0e5
ARM_DRIVE_DAMPING = 1.0e4
GRIPPER_DRIVE_STIFFNESS = 5.0e3
GRIPPER_DRIVE_DAMPING = 5.0e2

GRIPPER_TCP_NAME = "gripper_tcp"
GRIPPER_TCP_OFFSET = [0.0, 0.0, 0.21677, 0.0, 0.0, 0.0]

# Widths in mm, on a 75 mm part: commanded inside it so the compliant
# fingers stall under load. STOPPED_INNER_OBJECT is a grasp, AT_DEST is a miss.
GRIPPER_OPEN_MM = 105.0
GRIPPER_GRASP_MM = 58.0
GRIPPER_CLOSE_SECONDS = 2.0

# The folded rest pose -- a shape, not a direction: only the base angle changes
# per target. The elbow is open of the natural -90 to lift the wrist clear of the
# machine and wrist 1 gives that back, keeping the tool's tilt (joints 2+3+4)
# unchanged. Raise it further in that same pairing.
HOME_L_JOINTS = [-90.0, -90.0, -60.0, -120.0, 90.0, 0.0]

GRID_RELEASE_CLEARANCE = 0.003
GRID_PLACE_SPEED = 0.05

# How far the wrist sits to one side of the arm's plane.
SHOULDER_OFFSET = 0.17415

# Comfortably above the furthest flange distance that has actually succeeded
# (1.328 m, measured) and below the UR10e's own spec (1.3 m to the wrist, more
# with a tool), so this only skips targets that were never going to solve.
MAX_PHYSICAL_REACH_METERS = 1.40

TRANSFER_SPEED = 0.3
JOINT_SPEED = 45.0
JOINT_ACCELERATION = 60.0
SETTLE_SECONDS = 0.3

CNC_DOOR_PRIM_PATH = "/World/model_cnc_machine_tool/E_body_1/door"
CNC_DOOR_OPEN_X = -0.68654
CNC_DOOR_CLOSED_X = -0.20938
CNC_DOOR_WORLD_X_PER_LOCAL_X = 2.54
CNC_DOOR_MOVE_SECONDS = 2.0
CNC_DOOR_MOVE_STEPS = 60
CNC_PROCESS_SECONDS = 3.0


# ---------------------------------------------------------------------------
# Talking to the simulation
# ---------------------------------------------------------------------------


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


def get_pose_in_world(prim_path: str) -> list[float]:
    """A prim's world pose, as metres plus rotation-vector radians."""
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


def place_robot_on_mount(tree: tftree.TransformTree) -> list[float]:
    """Place the simulated robot at the TF tree's robot-base frame.

    Args:
        tree (tftree.TransformTree): Static machine-tending frame tree.

    Returns:
        list[float]: Robot-base world pose with XYZ in metres and rotation
            vector in radians.

    Raises:
        ValueError: If the robot-base transform cannot be converted to a pose.
        requests.RequestException: If a bridge request fails.
    """
    bridge_request("PATCH", "/stage/simulation/timeline/stop")

    world_T_robot_base = tree.lookup_transform("world", "robot_base")
    robot_base_pose_in_world = [
        float(value)
        for value in tfutils.transformation_matrix_to_pose(
            world_T_robot_base,
            rot_type="rotvec",
        )
    ]

    bridge_request(
        "PUT",
        "/prims/poses",
        body={
            "prim_path": ROBOT_PRIM_PATH,
            "input_pose": {"pose": robot_base_pose_in_world},
        },
    )
    return robot_base_pose_in_world


def register_articulation(prim_path: str) -> str:
    """Register an articulation with the bridge and return its id."""
    info = bridge_request("PUT", "/articulations", body={"prim_path": prim_path})
    return info["articulation_id"]


def set_drive_gains(articulation_id: str, stiffness: float, damping: float) -> None:
    """Retune a device's drives on the running simulation, not on the stage."""
    bridge_request(
        "POST",
        f"/articulations/{articulation_id}/dof_gains",
        body={"stiffness": stiffness, "damping": damping},
    )


class CNCMachine:
    """A door that slides, and a machining cycle that waits."""

    def __init__(self, door_prim_path: str, open_x: float, closed_x: float) -> None:
        self.door_prim_path = door_prim_path
        self.open_x = open_x
        self.closed_x = closed_x

    def move_door(self, target: float) -> None:
        """Slide the door, easing in and out rather than jumping."""
        result = bridge_request(
            "GET",
            "/prims/poses",
            params={
                "prim_path": self.door_prim_path,
                "coordinate_system": "local",
                "rotation_type": "cartesian",
            },
        )
        distance = (target - float(result["pose"][0])) * CNC_DOOR_WORLD_X_PER_LOCAL_X
        previous = 0.0

        for step in range(1, CNC_DOOR_MOVE_STEPS + 1):
            progress = step / CNC_DOOR_MOVE_STEPS
            eased = progress * progress * (3.0 - 2.0 * progress)
            bridge_request(
                "POST",
                "/prims/poses/relative",
                body={
                    "prim_path": self.door_prim_path,
                    "relative_pose": {
                        "pose": [distance * (eased - previous), 0.0, 0.0, 0.0, 0.0, 0.0]
                    },
                    "object_first": False,
                },
            )
            previous = eased
            time.sleep(CNC_DOOR_MOVE_SECONDS / CNC_DOOR_MOVE_STEPS)


# ---------------------------------------------------------------------------
# Where to go
# ---------------------------------------------------------------------------


def get_motion_target(
    tree: tftree.TransformTree,
    grasp_frame: str,
    hover_frame: str,
    label: str,
) -> dict | None:
    """Build the robot motion target for one named TF grasp frame.

    Args:
        tree (tftree.TransformTree): Static machine-tending frame tree.
        grasp_frame (str): Named downward-facing grasp frame in the tree.
        hover_frame (str): Named pre-contact frame above the grasp frame.
        label (str): Human-readable target name used in the startup output.

    Returns:
        dict | None: Robot-base grasp and hover poses plus the robot-specific
            folded joint pose, or ``None`` when the approximate physical-reach
            check rejects the target.

    Raises:
        ValueError: If a frame is missing or a transform is invalid.
    """
    grasp_pose = tree.lookup_transform("robot_base", grasp_frame, rot_type="deg")
    hover_pose = tree.lookup_transform("robot_base", hover_frame, rot_type="deg")

    # The complete Cartesian grasp orientation comes from TF. Only the folded
    # clearance pose below is robot-specific: its first joint turns the folded
    # UR toward this target before the Cartesian reach begins.
    xyz = grasp_pose[:3]
    heading = f"  {label:8s} {[round(value, 4) for value in xyz]}"
    reach = math.dist([0.0, 0.0, 0.0], xyz)
    radial = math.degrees(math.atan2(xyz[1], xyz[0]))

    # The flange trails the TCP by a tool length, so it is what actually has to
    # reach, not the TCP. Checking the physical limit first gives a clearer
    # failure than asking IK to solve a target outside the arm's workspace.
    flange = math.dist(
        [0.0, 0.0, 0.0],
        [hover_pose[0], hover_pose[1], hover_pose[2] + GRIPPER_TCP_OFFSET[2]],
    )
    if flange > MAX_PHYSICAL_REACH_METERS:
        print(
            f"{heading}  OUT OF REACH at {reach:.3f} m -- {flange:.3f} m past the flange's limit"
        )
        return None

    # The folded shape, with only the base turned to the target. The few degrees
    # past it are the wrist's sideways offset from the arm's plane.
    folded = list(HOME_L_JOINTS)
    folded[0] = radial + math.degrees(
        math.asin(min(1.0, SHOULDER_OFFSET / math.hypot(xyz[0], xyz[1])))
    )

    rpy = [round(value, 2) for value in grasp_pose[3:]]
    print(f"{heading}  RPY {rpy}  base turns to {folded[0]:7.2f}  reach {reach:.3f} m")
    return {
        "grasp_pose": grasp_pose,
        "hover_pose": hover_pose,
        "folded": folded,
    }


# ---------------------------------------------------------------------------
# Moving the arm
# ---------------------------------------------------------------------------


def shortest_turn(
    target_joints: list[float],
    current_joints,
    joint_limits,
) -> list[float]:
    """The same configuration, with no joint taking the long way round.

    -350 and +10 degrees are the same place; commanded literally, the arm unwinds
    a whole turn to reach a pose it already stands in.
    """
    nearest = []
    for target, current, limits in zip(target_joints, current_joints, joint_limits):
        lower, upper = (float(limits[0]), float(limits[1]))
        equivalents = [
            target + 360.0 * turns
            for turns in range(-2, 3)
            if lower <= target + 360.0 * turns <= upper
        ]
        nearest.append(
            min(equivalents, key=lambda value: abs(value - float(current)))
            if equivalents
            else target
        )
    return nearest


def move_joints(robot, joints: list[float]) -> None:
    """Drive to a joint configuration the short way, and let it settle."""
    robot.set_joint_positions(
        shortest_turn(joints, robot.get_joint_positions(), robot.joint_limits),
        speed=JOINT_SPEED,
        acceleration=JOINT_ACCELERATION,
    )
    time.sleep(SETTLE_SECONDS)


def move_to(
    robot,
    target_pose: list[float],
    *,
    speed: float = TRANSFER_SPEED,
) -> None:
    """Drive the TCP to a pose and let it settle. Always blocking -- guessing how
    long a descent takes is how the gripper once closed in mid-air."""
    robot.set_cartesian_pose(target_pose, speed=speed)
    time.sleep(SETTLE_SECONDS)


def grip(gripper, width_mm: float, *, blocking: bool = True) -> str:
    """Move the fingers and allow either arrival or a fixed contact wait."""
    status = gripper.move(width_mm, asynchronous=not blocking)
    time.sleep(SETTLE_SECONDS if blocking else GRIPPER_CLOSE_SECONDS)
    return status


def pick_at(robot, gripper, target: dict) -> None:
    """Turn to face the work, reach down, take the part, and back out."""
    grip(gripper, GRIPPER_OPEN_MM)
    move_joints(robot, target["folded"])

    move_to(robot, target["hover_pose"])
    move_to(robot, target["grasp_pose"])

    grip(gripper, GRIPPER_GRASP_MM, blocking=False)
    print(
        f"  grip command held for {GRIPPER_CLOSE_SECONDS:.1f} s; "
        f"reported width {gripper.get_current_position():.1f} mm"
    )

    move_to(robot, target["hover_pose"])
    move_joints(robot, target["folded"])


def place_at(
    robot,
    gripper,
    target: dict,
    *,
    release_clearance: float = 0.0,
    descent_speed: float = TRANSFER_SPEED,
) -> None:
    """Reach down, release above the seated target, and back out."""
    move_joints(robot, target["folded"])

    move_to(robot, target["hover_pose"])
    release_pose = list(target["grasp_pose"])
    release_pose[2] += release_clearance
    move_to(robot, release_pose, speed=descent_speed)

    status = grip(gripper, GRIPPER_OPEN_MM)
    print(f"  released at {gripper.get_current_position():.1f} mm, status {status}")

    move_to(robot, target["hover_pose"])
    move_joints(robot, target["folded"])


# ---------------------------------------------------------------------------


def main() -> None:
    """Run the machine-tending cycle once for every part in the tray."""
    bridge_request("GET", "/status")
    cnc = CNCMachine(CNC_DOOR_PRIM_PATH, CNC_DOOR_OPEN_X, CNC_DOOR_CLOSED_X)

    cnc_machine_pose_in_world = get_pose_in_world(CNC_PRIM_PATH)
    table_pose_in_world = get_pose_in_world(TABLE_FRAME_PRIM_PATH)
    tree, pick_grid, place_grid = build_static_frame_tree(
        cnc_machine_pose_in_world,
        table_pose_in_world,
    )

    print("Placing the robot on the mount...")
    place_robot_on_mount(tree)

    pick_capacity = pick_grid.numx * pick_grid.numy
    place_capacity = place_grid.numx * place_grid.numy
    if PART_COUNT > min(pick_capacity, place_capacity):
        raise RuntimeError(
            f"configured for {PART_COUNT} parts, but the pick and place grids "
            f"have capacities {pick_capacity} and {place_capacity}."
        )
    print(f"Using {PART_COUNT} fixed pick and place slots.")

    robot = universal_robots.UniversalRobotsUR10E()
    gripper = onrobot.OnRobotRG2()

    # Added while offline: this is model surgery, not a simulation command.
    robot.add_tcp(GRIPPER_TCP_NAME, GRIPPER_TCP_OFFSET)

    print("Connecting...")
    robot.connect(simulation_prim_path=ROBOT_PRIM_PATH)

    try:
        # Registered before the attach: doing it afterwards rebinds the merged
        # arm-and-gripper articulation and jolts the fingers.
        arm_id = register_articulation(ROBOT_PRIM_PATH)
        gripper_id = register_articulation(GRIPPER_PRIM_PATH)

        gripper.connect(simulation_prim_path=GRIPPER_PRIM_PATH)
        try:
            print("Attaching the gripper...")
            robot.attach_tool(gripper)
            time.sleep(1.0)

            # Gains go on after the attach: they apply to the running simulation,
            # and the attach restarts it, reloading the drives from the USD.
            set_drive_gains(arm_id, ARM_DRIVE_STIFFNESS, ARM_DRIVE_DAMPING)
            set_drive_gains(gripper_id, GRIPPER_DRIVE_STIFFNESS, GRIPPER_DRIVE_DAMPING)

            # Convert all named TF frames into the poses used by the cycle.
            # set_cartesian_pose() performs the trajectory IK when each move runs.
            print()
            print("Preparing targets:")
            machine = get_motion_target(
                tree,
                CNC_GRASP_FRAME,
                CNC_PRE_FRAME,
                "machine",
            )

            picks = []
            drops = []
            for index in range(PART_COUNT):
                pick_x_index, pick_y_index = divmod(index, pick_grid.numy)
                picks.append(
                    get_motion_target(
                        tree,
                        f"{PICK_GRASP_FRAME_PREFIX}_{pick_x_index}_{pick_y_index}",
                        f"{PICK_PRE_FRAME_PREFIX}_{pick_x_index}_{pick_y_index}",
                        f"part {index + 1}",
                    )
                )

                place_x_index, place_y_index = divmod(index, place_grid.numy)
                drops.append(
                    get_motion_target(
                        tree,
                        f"{PLACE_GRASP_FRAME_PREFIX}_{place_x_index}_{place_y_index}",
                        f"{PLACE_PRE_FRAME_PREFIX}_{place_x_index}_{place_y_index}",
                        f"slot {index + 1}",
                    )
                )

            missing = sum(target is None for target in [machine, *picks, *drops])
            if missing:
                raise RuntimeError(
                    f"{missing} target(s) out of reach -- see the list above. Move the "
                    "scene, not the code: the arm cannot get there from where it is."
                )

            started = time.monotonic()

            def step(label: str) -> None:
                """Announce a phase with the time since the cycle began."""
                print(f"[{time.monotonic() - started:5.1f}s] {label}")

            move_joints(robot, HOME_L_JOINTS)

            for index in range(PART_COUNT):
                print()
                step(f"--- part {index + 1} of {PART_COUNT} ---")

                step("Picking it off the table...")
                pick_at(robot, gripper, picks[index])

                step("Loading it into the machine...")
                place_at(robot, gripper, machine)

                step("Closing the door...")
                cnc.move_door(cnc.closed_x)
                print("  machining...")
                time.sleep(CNC_PROCESS_SECONDS)
                step("Opening the door...")
                cnc.move_door(cnc.open_x)

                step("Taking the finished part out...")
                pick_at(robot, gripper, machine)

                step(f"Standing it in slot {index + 1}...")
                place_at(
                    robot,
                    gripper,
                    drops[index],
                    release_clearance=GRID_RELEASE_CLEARANCE,
                    descent_speed=GRID_PLACE_SPEED,
                )
                # No return to HOME_L_JOINTS here: place_at already ends folded
                # at this slot's angle, and the next pick_at folds straight to
                # its own angle, so the arm just turns from one to the other.

            print()
            print(f"{PART_COUNT} part(s) in {time.monotonic() - started:.1f} s.")
        finally:
            gripper.disconnect()
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
