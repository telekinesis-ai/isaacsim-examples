"""UR10e machine tending in Isaac Sim, for every part it finds in the tray.

    tray -> CNC -> [door, machine, door] -> CNC -> container

Each reach is two moves: an L pose (folded, base turned to face the work,
commanded in JOINT space) then Cartesian from there in, so the descent is a
straight vertical line. The arm refolds between operations so it crosses the cell
tucked up. Positions are held in WORLD coordinates and converted at startup, so
moving the robot needs no edit here.

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

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot


BASE_URL = "http://127.0.0.1:8766"
REQUEST_TIMEOUT_SECONDS = 120.0

MOUNT_PRIM_PATH = (
    "/World/table/VentionAssembly/VentionAssembly"
    "/tn__0_26_/tn__MOLM0510006__2_vK59"
)
ROBOT_PRIM_PATH = "/World/ur10e_robot"
GRIPPER_PRIM_PATH = "/World/onrobot_rg2_model"
PART_PRIM_PATTERN = "/World/part_{:02d}"
MAX_PARTS = 20

# Placement on the mount marker: 5 mm below it, turned 180 degrees.
MOUNT_OFFSET_Z = -0.005
MOUNT_YAW_RADIANS = math.radians(180.0)

# A URDF import leaves the drives soft: the arm oscillates, the fingers swing shut.
ARM_DRIVE_STIFFNESS = 1.0e5
ARM_DRIVE_DAMPING = 1.0e4
GRIPPER_DRIVE_STIFFNESS = 5.0e3
GRIPPER_DRIVE_DAMPING = 5.0e2

# 0.21677 m to the RG2's fingertips; pulled in so the pads meet the part.
GRIPPER_TCP_NAME = "gripper_tcp"
GRIPPER_TCP_OFFSET = [0.0, 0.0, 0.195, 0.0, 0.0, 0.0]

# Widths in mm, on an 80 mm part: commanded well inside it so the compliant
# fingers stall under load. STOPPED_INNER_OBJECT is a grasp, AT_DEST is a miss.
GRIPPER_OPEN_MM = 105.0
GRIPPER_GRASP_MM = 68.0

# The cylinder, and where the fingers close: 47.5 mm above centre is 20 mm below
# its top -- clear of the grid plate, still plenty of part held.
PART_LENGTH = 0.135
GRIP_ABOVE_CENTRE = 0.0475

# The folded rest pose -- a shape, not a direction: only the base angle changes
# per target. The elbow is open of the natural -90 to lift the wrist clear of the
# machine and wrist 1 gives that back, keeping the tool's tilt (joints 2+3+4)
# unchanged. Raise it further in that same pairing.
HOME_L_JOINTS = [-90.0, -90.0, -60.0, -120.0, 90.0, 0.0]

# The CNC fixture. Nothing to drop into, so a part stands on the surface.
CNC_PLACE_WORLD_XY = [-2.73674, -2.64291]
CNC_SURFACE_WORLD_Z = 1.44384

# Every container slot's seated position, in world coordinates -- measured, not
# extrapolated. A 4-corner parallelogram check (anchor, far row end, far column
# end, opposite corner) confirmed the grid is regular enough to fill the middle
# by interpolation once the corner parts were allowed to settle under gravity
# rather than read from wherever they were dragged.
SLOT_POSITIONS_WORLD = [
    [-2.57978, -1.08219, 1.30291],  # slot 1
    [-2.39840, -1.08211, 1.30291],  # slot 2
    [-2.21702, -1.08202, 1.30291],  # slot 3
    [-2.03564, -1.08194, 1.30291],  # slot 4
    [-2.57822, -0.92066, 1.30291],  # slot 5
    [-2.39684, -0.92058, 1.30291],  # slot 6
    [-2.21546, -0.92050, 1.30291],  # slot 7
    [-2.03408, -0.92041, 1.30291],  # slot 8
    [-2.57667, -0.75914, 1.30291],  # slot 9
    [-2.39529, -0.75905, 1.30291],  # slot 10
    [-2.21391, -0.75897, 1.30291],  # slot 11
    [-2.03253, -0.75889, 1.30291],  # slot 12
    [-2.57511, -0.59761, 1.30291],  # slot 13
    [-2.39373, -0.59753, 1.30291],  # slot 14
    [-2.21235, -0.59744, 1.30291],  # slot 15
    [-2.03097, -0.59736, 1.30291],  # slot 16
]

# Fingers straight down for every grasp; the angle about vertical is per target.
TOOL_ROLL = 180.0
TOOL_PITCH = 0.0

# Hover height. The machine gets less: furthest reach in the cycle, and lift costs reach.
APPROACH_HEIGHT = 0.20
CNC_APPROACH_HEIGHT = 0.06

# How far the wrist sits to one side of the arm's plane.
SHOULDER_OFFSET = 0.17415

# Comfortably above the furthest flange distance that has actually succeeded
# (1.328 m, measured) and below the UR10e's own spec (1.3 m to the wrist, more
# with a tool), so this only skips targets that were never going to solve.
MAX_PHYSICAL_REACH_METERS = 1.40

# A UR joint can wind past a full turn, which is what makes unwinding possible.
JOINT_LIMIT_DEGREES = 360.0

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
        method, BASE_URL + path, params=params, json=body, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json() if response.content else None


def world_pose(prim_path: str) -> list[float]:
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


def place_robot_on_mount() -> list[float]:
    """Stand the robot on the mount; the pose returned is everything else's origin."""
    bridge_request("PATCH", "/stage/simulation/timeline/stop")

    mount = world_pose(MOUNT_PRIM_PATH)
    mount[2] += MOUNT_OFFSET_Z
    yaw = mount[5] + MOUNT_YAW_RADIANS
    mount[5] = math.atan2(math.sin(yaw), math.cos(yaw))

    bridge_request(
        "PUT", "/prims/poses", body={"prim_path": ROBOT_PRIM_PATH, "input_pose": {"pose": mount}}
    )
    return mount


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

    def door_position(self) -> float:
        """Where the door sits along its own travel."""
        result = bridge_request(
            "GET",
            "/prims/poses",
            params={
                "prim_path": self.door_prim_path,
                "coordinate_system": "local",
                "rotation_type": "cartesian",
            },
        )
        return float(result["pose"][0])

    def move_door(self, target: float) -> None:
        """Slide the door, easing in and out rather than jumping."""
        distance = (target - self.door_position()) * CNC_DOOR_WORLD_X_PER_LOCAL_X
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

    def open_door(self) -> None:
        self.move_door(self.open_x)

    def close_door(self) -> None:
        self.move_door(self.closed_x)

    def machine(self) -> None:
        print("  machining...")
        time.sleep(CNC_PROCESS_SECONDS)


# ---------------------------------------------------------------------------
# Where to go
# ---------------------------------------------------------------------------


def base_frame(robot_base_world: list[float], world_xyz: list[float]) -> list[float]:
    """A world position in the robot's own frame. Mounted square, so: a subtraction."""
    return [world_xyz[index] - robot_base_world[index] for index in range(3)]


def slot_position(robot_base_world: list[float], index: int) -> list[float]:
    """Grasp point for one container slot, from its measured position."""
    x, y, z = SLOT_POSITIONS_WORLD[index]
    return base_frame(robot_base_world, [x, y, z + GRIP_ABOVE_CENTRE])


def find_parts(robot_base_world: list[float]) -> list[dict]:
    """Every cylinder in the tray, as a grasp point in the robot's base frame.

    Probed by name (the bridge cannot walk the stage) and read once (it reports
    the authored transform, which physics then stops matching).
    """
    parts = []
    for index in range(1, MAX_PARTS + 1):
        prim_path = PART_PRIM_PATTERN.format(index)
        try:
            world = world_pose(prim_path)
        except requests.HTTPError:
            break
        # A cylinder's origin is its own centre; the fingers close above that.
        parts.append(
            {
                "prim_path": prim_path,
                "xyz": base_frame(
                    robot_base_world, [world[0], world[1], world[2] + GRIP_ABOVE_CENTRE]
                ),
            }
        )
    return parts


def pose(xyz: list[float], yaw: float, *, dz: float = 0.0) -> list[float]:
    """A TCP pose at ``xyz``, optionally raised, fingers pointing down."""
    return [xyz[0], xyz[1], xyz[2] + dz, TOOL_ROLL, TOOL_PITCH, yaw]


def plan(robot, xyz: list[float], approach: float, label: str) -> dict | None:
    """Work out how to reach one target. None if the arm cannot, so a run lists them all.

    Two angles come out of this. The WRIST angle starts pointing straight out
    from the base and fans out until IK solves at both the hover and the target;
    one angle for both keeps the wrist still on the way down. The BASE angle is
    arithmetic rather than IK, because a UR can also reach a target backwards
    over its own shoulder and the solver returns whichever it finds first -- the
    backwards one folds the arm on the far side of the cell, so the move out of
    it crosses the robot.
    """
    heading = f"  {label:8s} {[round(value, 4) for value in xyz]}"
    reach = math.dist([0.0, 0.0, 0.0], xyz)
    radial = math.degrees(math.atan2(xyz[1], xyz[0]))

    # The flange trails the TCP by a tool length, so it is what actually has to
    # reach, not the TCP. Past MAX_PHYSICAL_REACH_METERS nothing solves at any
    # wrist angle -- checking that first skips the 25-angle search (roughly
    # 0.5s per angle) on targets that were never going to work, without
    # touching genuinely borderline ones near the arm's real limit.
    flange = math.dist(
        [0.0, 0.0, 0.0], [xyz[0], xyz[1], xyz[2] + approach + GRIPPER_TCP_OFFSET[2]]
    )
    if flange > MAX_PHYSICAL_REACH_METERS:
        print(f"{heading}  OUT OF REACH at {reach:.3f} m -- {flange:.3f} m past the flange's limit")
        return None

    yaw, reason = None, "nothing tried"
    for offset in [0] + [sign * step for step in range(15, 181, 15) for sign in (1, -1)]:
        candidate = (radial + offset + 180.0) % 360.0 - 180.0
        try:
            for dz in (0.0, approach):
                # multi_start, because the UR default seeds from wherever the arm
                # is standing and fails on reachable targets from an awkward seed.
                robot.inverse_kinematics(
                    pose(xyz, candidate, dz=dz), solver="multi_start_clik"
                )
        except Exception as error:
            reason = f"{type(error).__name__}: {str(error).splitlines()[0]}"
            continue
        yaw = candidate
        break

    if yaw is None:
        print(f"{heading}  OUT OF REACH at {reach:.3f} m -- {reason}")
        return None

    # The folded shape, with only the base turned to the target. The few degrees
    # past it are the wrist's sideways offset from the arm's plane.
    folded = list(HOME_L_JOINTS)
    folded[0] = radial + math.degrees(
        math.asin(min(1.0, SHOULDER_OFFSET / math.hypot(xyz[0], xyz[1])))
    )

    print(f"{heading}  yaw {yaw:7.2f}  base turns to {folded[0]:7.2f}  reach {reach:.3f} m")
    return {"xyz": xyz, "yaw": yaw, "folded": folded, "approach": approach}


# ---------------------------------------------------------------------------
# Moving the arm
# ---------------------------------------------------------------------------


def shortest_turn(target_joints: list[float], current_joints) -> list[float]:
    """The same configuration, with no joint taking the long way round.

    -350 and +10 degrees are the same place; commanded literally, the arm unwinds
    a whole turn to reach a pose it already stands in.
    """
    unwound = []
    for target, current in zip(target_joints, current_joints):
        shifted = target + 360.0 * round((float(current) - target) / 360.0)
        unwound.append(shifted if abs(shifted) < JOINT_LIMIT_DEGREES else target)
    return unwound


def move_joints(robot, joints: list[float]) -> None:
    """Drive to a joint configuration the short way, and let it settle."""
    robot.set_joint_positions(
        shortest_turn(joints, robot.get_joint_positions()),
        speed=JOINT_SPEED,
        acceleration=JOINT_ACCELERATION,
    )
    time.sleep(SETTLE_SECONDS)


def move_to(robot, target_pose: list[float]) -> None:
    """Drive the TCP to a pose and let it settle. Always blocking -- guessing how
    long a descent takes is how the gripper once closed in mid-air."""
    robot.set_cartesian_pose(target_pose, speed=TRANSFER_SPEED)
    time.sleep(SETTLE_SECONDS)


def grip(gripper, width_mm: float) -> str:
    """Move the fingers to a width and wait. The status says what they hit."""
    status = gripper.move(width_mm, asynchronous=False)
    time.sleep(SETTLE_SECONDS)
    return status


def pick_at(robot, gripper, target: dict) -> None:
    """Turn to face the work, reach down, take the part, and back out."""
    grip(gripper, GRIPPER_OPEN_MM)
    move_joints(robot, target["folded"])

    move_to(robot, pose(target["xyz"], target["yaw"], dz=target["approach"]))
    move_to(robot, pose(target["xyz"], target["yaw"]))

    status = grip(gripper, GRIPPER_GRASP_MM)
    print(f"  gripped at {gripper.get_current_position():.1f} mm, status {status}")

    move_to(robot, pose(target["xyz"], target["yaw"], dz=target["approach"]))
    move_joints(robot, target["folded"])


def place_at(robot, gripper, target: dict) -> None:
    """Turn to face the work, reach down, let the part go, and back out."""
    move_joints(robot, target["folded"])

    move_to(robot, pose(target["xyz"], target["yaw"], dz=target["approach"]))
    move_to(robot, pose(target["xyz"], target["yaw"]))

    status = grip(gripper, GRIPPER_OPEN_MM)
    print(f"  released at {gripper.get_current_position():.1f} mm, status {status}")

    move_to(robot, pose(target["xyz"], target["yaw"], dz=target["approach"]))
    move_joints(robot, target["folded"])


# ---------------------------------------------------------------------------


def main() -> None:
    """Run the machine-tending cycle once for every part in the tray."""
    bridge_request("GET", "/status")
    cnc = CNCMachine(CNC_DOOR_PRIM_PATH, CNC_DOOR_OPEN_X, CNC_DOOR_CLOSED_X)

    print("Placing the robot on the mount...")
    robot_base_world = place_robot_on_mount()

    parts = find_parts(robot_base_world)
    if not parts:
        raise RuntimeError(
            f"no parts found. Expected at least {PART_PRIM_PATTERN.format(1)} in the stage."
        )
    print(f"Found {len(parts)} part(s): {[part['prim_path'] for part in parts]}")

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

            # Every target worked out before anything moves, so an unreachable
            # one is a printed list rather than the arm finding out mid-cycle.
            print()
            print("Planning:")
            machine = plan(
                robot,
                base_frame(
                    robot_base_world,
                    [
                        *CNC_PLACE_WORLD_XY,
                        CNC_SURFACE_WORLD_Z + PART_LENGTH / 2 + GRIP_ABOVE_CENTRE,
                    ],
                ),
                CNC_APPROACH_HEIGHT,
                "machine",
            )
            picks = [
                plan(robot, part["xyz"], APPROACH_HEIGHT, f"part {index + 1}")
                for index, part in enumerate(parts)
            ]
            drops = [
                plan(
                    robot,
                    slot_position(robot_base_world, index),
                    APPROACH_HEIGHT,
                    f"slot {index + 1}",
                )
                for index in range(len(parts))
            ]

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

            for index, part in enumerate(parts):
                print()
                step(f"--- part {index + 1} of {len(parts)}: {part['prim_path']} ---")

                step("Picking it off the table...")
                pick_at(robot, gripper, picks[index])
                move_joints(robot, HOME_L_JOINTS)

                step("Loading it into the machine...")
                place_at(robot, gripper, machine)
                move_joints(robot, HOME_L_JOINTS)

                step("Closing the door...")
                cnc.close_door()
                cnc.machine()
                step("Opening the door...")
                cnc.open_door()

                step("Taking the finished part out...")
                pick_at(robot, gripper, machine)
                move_joints(robot, HOME_L_JOINTS)

                step(f"Standing it in slot {index + 1}...")
                place_at(robot, gripper, drops[index])
                # No return to HOME_L_JOINTS here: place_at already ends folded
                # at this slot's angle, and the next pick_at folds straight to
                # its own angle, so the arm just turns from one to the other.

            print()
            print(f"{len(parts)} part(s) in {time.monotonic() - started:.1f} s.")
        finally:
            gripper.disconnect()
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
