"""Place roofs on cars stopped at the automotive lightbeam.

The application places the selected robot on the calibrated TF ``robot_mount``
frame, attaches the modeled suction gripper, runs the three car conveyors until
the lightbeam detects a sledge, and stops them together. The roof is picked from
the rack and placed using the matching calibrated TF car frame. Two car/roof
paths are recycled from inactive templates for longer production runs.

Before running:
    1. Open the automotive-assembly scene in Isaac Sim.
    2. Enable the Telekinesis Isaac Sim bridge extension.
    3. Import the robot and gripper at the configured prim paths.
    4. Run ``python automotive_assembly_application_multi_robot.py --kuka``.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

from telekinesis.synapse.robots.manipulators import abstract_robot, kuka
from telekinesis.synapse.tools.suction_grippers import custom
from telekinesis.tf import tftree, tfutils

from automotive_assembly_isaacsim_backend import AutomotiveIsaacSimBackend
from automotive_assembly_static_frames_visualization import (
    PLACE_PRE_GRASP_OFFSET,
    build_static_frame_tree,
)


BASE_URL = "http://127.0.0.1:8766"
REQUEST_TIMEOUT_SECONDS = 30.0
MOTION_SETTLE_SECONDS = 0.5
SUCTION_SETTLE_SECONDS = 1.0
SUCTION_GRIP_TIMEOUT_SECONDS = 5.0
SUCTION_STATUS_POLL_SECONDS = 0.25

KUKA_PRIM_PATH = "/World/kuka_kr210"
SUCTION_GRIPPER_PRIM_PATH = "/World/suction_gripper_modelled"
ROOF_PRIM_PATH = "/World/roof"
SECOND_ROOF_PRIM_PATH = "/World/roof_02"
LIGHTBEAM_PRIM_PATH = "/World/LightBeam_Sensor"
SLEDGE_PRIM_PATHS = ["/World/sledge", "/World/sledge_01"]
ROOF_PRIM_PATHS = [ROOF_PRIM_PATH, SECOND_ROOF_PRIM_PATH]
SPAWN_TEMPLATE_ROOT = "/World/_automotive_spawn_templates"
ASSEMBLY_CYCLE_COUNT = 8
CONVEYOR_RUN_VELOCITIES = {
    "/World/ConveyorBelt_A05": [0.6, 0.0, 0.0],
    "/World/ConveyorBelt_A05_01": [-0.6, 0.0, 0.0],
    "/World/ConveyorBelt_A05_02": [0.6, 0.0, 0.0],
    "/World/ConveyorBelt_A05_03/Rollers": [0.6, 0.0, 0.0],
}
# The gripper asset root is its physical mounting point. The translation below
# is the KUKA URDF's fixed ``link_6`` to ``tool0`` offset; the rotation turns
# the modeled gripper horizontal.
KUKA_LINK6_T_SUCTION_GRIPPER = [
    0.0375,
    0.0,
    -0.00023924,
    90.0,
    0.0,
    90.0,
]

# KUKA tool0 to the centre of the modeled suction contact area. This combines
# the gripper mounting orientation with its rebased ``contact_frame``.
KUKA_TOOL0_T_SUCTION_TCP = [
    0.453148259,
    0.050733766,
    -0.008042106,
    90.0,
    0.0,
    90.0,
]


@dataclass(frozen=True)
class RobotConfig:
    """Robot-specific setup values used by the shared application.

    Attributes:
        robot_class: Synapse class used to create the robot client.
        prim_path: Default robot prim path in Isaac Sim.
        mount_T_robot_base: Robot-base pose relative to ``robot_mount``, using
            XYZ metres and Euler XYZ degrees.
        tool_mount_frame: Rigid robot link used for simulated tool assembly.
        tool_mount_transform: Mount-link-to-gripper-root pose, using XYZ metres
            and Euler XYZ degrees.
        suction_tcp_offset: Suction contact-centre pose relative to the robot's
            default TCP, using XYZ metres and Euler XYZ degrees.
        home_joints: Clear home pose in joint degrees.
        joint_speed: Joint-motion speed in degrees per second.
        joint_acceleration: Joint-motion acceleration in degrees per second
            squared.
        cartesian_speed: TCP travel speed in metres per second.
        cartesian_acceleration: TCP acceleration in metres per second squared.
        contact_speed: Slower TCP speed used for the final roof approach.
        drive_stiffness: Simulation position-drive stiffness.
        drive_damping: Simulation position-drive damping.
    """

    robot_class: type
    prim_path: str
    mount_T_robot_base: list[float]
    tool_mount_frame: str
    tool_mount_transform: list[float]
    suction_tcp_offset: list[float]
    home_joints: list[float]
    joint_speed: float
    joint_acceleration: float
    cartesian_speed: float
    cartesian_acceleration: float
    contact_speed: float
    drive_stiffness: float
    drive_damping: float


ROBOT_CONFIGS = {
    "kuka": RobotConfig(
        robot_class=kuka.KukaKR210L150,
        prim_path=KUKA_PRIM_PATH,
        mount_T_robot_base=[0.0, 0.0, 0.0, 0.0, 0.0, 180.0],
        tool_mount_frame="link_6",
        tool_mount_transform=KUKA_LINK6_T_SUCTION_GRIPPER,
        suction_tcp_offset=KUKA_TOOL0_T_SUCTION_TCP,
        home_joints=[0.0, 0.0, 0.0, 0.0, 90.0, 0.0],
        joint_speed=20.0,
        joint_acceleration=30.0,
        cartesian_speed=0.25,
        cartesian_acceleration=0.35,
        contact_speed=0.05,
        drive_stiffness=1.0e6,
        drive_damping=1.0e5,
    ),
}


def bridge_request(
    method: str,
    path: str,
    *,
    body: dict | None = None,
) -> dict | None:
    """Send a request to the local Isaac Sim bridge.

    Args:
        method: HTTP request method.
        path: Bridge endpoint path.
        body: Optional JSON request body.

    Returns:
        Decoded JSON response, or ``None`` for an empty response.

    Raises:
        requests.RequestException: If the bridge request or response fails.
    """
    response = requests.request(
        method,
        BASE_URL + path,
        json=body,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json() if response.content else None


def place_robot_on_mount(
    tree: tftree.TransformTree,
    robot_prim_path: str,
) -> None:
    """Place the robot prim at the TF tree's ``robot_base`` frame.

    Args:
        tree: Automotive-assembly transform tree.
        robot_prim_path: USD path of the robot prim to move.

    Returns:
        None.

    Raises:
        requests.RequestException: If an Isaac Sim bridge request fails.
        ValueError: If the TF pose cannot be converted.
    """
    bridge_request("PATCH", "/stage/simulation/timeline/stop")
    robot_pose_in_world = [
        float(value)
        for value in tfutils.transformation_matrix_to_pose(
            tree.lookup_transform("world", "robot_base"),
            rot_type="rotvec",
        )
    ]
    bridge_request(
        "PUT",
        "/prims/poses",
        body={
            "prim_path": robot_prim_path,
            "input_pose": {"pose": robot_pose_in_world},
        },
    )
    print(f"Robot pose in world: {[round(value, 6) for value in robot_pose_in_world]}")


def apply_drive_gains(
    robot_prim_path: str,
    stiffness: float,
    damping: float,
) -> None:
    """Apply position-drive gains to the simulated robot articulation.

    Args:
        robot_prim_path: USD path of the robot articulation.
        stiffness: Position-drive stiffness.
        damping: Position-drive damping.

    Returns:
        None.

    Raises:
        RuntimeError: If the bridge returns no articulation information.
        requests.RequestException: If an Isaac Sim bridge request fails.
    """
    articulation = bridge_request(
        "PUT",
        "/articulations",
        body={"prim_path": robot_prim_path},
    )
    if articulation is None:
        raise RuntimeError("No robot articulation information returned")
    bridge_request(
        "POST",
        f"/articulations/{articulation['articulation_id']}/dof_gains",
        body={"stiffness": stiffness, "damping": damping},
    )


def move_tcp_to_frame(
    robot: abstract_robot.AbstractManipulator,
    tree: tftree.TransformTree,
    frame_name: str,
    speed: float,
    acceleration: float,
    vertical_offset: float = 0.0,
) -> None:
    """Move the active TCP to a named TF frame.

    Args:
        robot: Connected Synapse robot.
        tree: Automotive-assembly transform tree.
        frame_name: TF target frame.
        speed: Cartesian speed in metres per second.
        acceleration: Cartesian acceleration in metres per second squared.
        vertical_offset: Additional robot-base Z height in metres.

    Returns:
        None.

    Raises:
        RuntimeError: If Synapse cannot execute the Cartesian move.
        ValueError: If the TF frame is invalid.
    """
    target_pose = list(
        tree.lookup_transform("robot_base", frame_name, rot_type="deg")
    )
    target_pose[2] += vertical_offset
    robot.set_cartesian_pose(
        target_pose,
        speed=speed,
        acceleration=acceleration,
    )
    time.sleep(MOTION_SETTLE_SECONDS)


def main() -> None:
    """Recycle two car slots and place one roof during every cycle.

    Returns:
        None.

    Raises:
        RuntimeError: If the robot, gripper, or articulation cannot connect.
        requests.RequestException: If an Isaac Sim bridge request fails.
        ValueError: If a configured transform is invalid.
    """
    parser = argparse.ArgumentParser(
        description="Set up a supported robot for automotive roof assembly.",
    )
    parser.add_argument(
        "--kuka",
        dest="robot_name",
        action="store_const",
        const="kuka",
        help="Use the KUKA KR210 L150 (default).",
    )
    parser.set_defaults(robot_name="kuka")
    args = parser.parse_args()

    config = ROBOT_CONFIGS[args.robot_name]
    scene_backend = AutomotiveIsaacSimBackend(
        conveyor_velocities=CONVEYOR_RUN_VELOCITIES,
        lightbeam_path=LIGHTBEAM_PRIM_PATH,
        sledge_paths=SLEDGE_PRIM_PATHS,
        roof_paths=ROOF_PRIM_PATHS,
        sledge_template_source_path=SLEDGE_PRIM_PATHS[1],
        template_root=SPAWN_TEMPLATE_ROOT,
    )
    bridge_request("GET", "/status")

    tree = build_static_frame_tree()
    tree.add(
        "robot_mount",
        "robot_base",
        config.mount_T_robot_base,
        rot_type="deg",
    )

    print(f"Robot: {args.robot_name} ({config.prim_path})")
    print("Placing the robot on the TF robot_mount frame...")
    place_robot_on_mount(tree, config.prim_path)

    robot = config.robot_class(name=args.robot_name)
    gripper = custom.CustomSuctionGripper()

    scene_backend.hold_conveyors_during_setup()
    bridge_request("PATCH", "/stage/simulation/timeline/play")
    time.sleep(0.5)

    print("Connecting the robot...")
    robot.connect(simulation_prim_path=config.prim_path)
    try:
        print(f"Connecting the suction gripper at {SUCTION_GRIPPER_PRIM_PATH}...")
        gripper.connect(simulation_prim_path=SUCTION_GRIPPER_PRIM_PATH)
        try:
            print(f"Attaching the suction gripper to {config.tool_mount_frame}...")
            robot.attach_tool(
                gripper,
                mount_frame=config.tool_mount_frame,
                transform=config.tool_mount_transform,
            )
            robot.add_tcp(
                name="suction_tcp",
                transform=config.suction_tcp_offset,
                set_active=True,
            )
            print(
                "Active TCP: suction_tcp at the modeled suction contact centre"
            )
            apply_drive_gains(
                config.prim_path,
                config.drive_stiffness,
                config.drive_damping,
            )
            print(
                "Applied KUKA drive gains: "
                f"stiffness={config.drive_stiffness:g}, "
                f"damping={config.drive_damping:g}"
            )
            print(f"Moving to the KUKA home pose: {config.home_joints}")
            robot.set_joint_positions(
                config.home_joints,
                speed=config.joint_speed,
                acceleration=config.joint_acceleration,
            )
            time.sleep(MOTION_SETTLE_SECONDS)

            scene_backend.prepare_spawn_templates()

            with ThreadPoolExecutor(max_workers=1) as conveyor_executor:
                next_car_future = None

                for cycle_index in range(1, ASSEMBLY_CYCLE_COUNT + 1):
                    pool_slot_index = (cycle_index - 1) % len(SLEDGE_PRIM_PATHS)
                    detected_sledge = SLEDGE_PRIM_PATHS[pool_slot_index]
                    place_index = cycle_index

                    if cycle_index == 1:
                        world_T_car_pose = scene_backend.wait_for_car(
                            wait_for_clear=False,
                            sledge_path=detected_sledge,
                        )
                        conveyor_T_car = tree.lookup_transform(
                            "conveyor",
                            "world",
                        ) @ tfutils.pose_to_transformation_matrix(
                            world_T_car_pose,
                            rot_type="deg",
                        )
                        tree.update("car", conveyor_T_car, rot_type="mat")
                        print(
                            f"Stopped {detected_sledge.rsplit('/', 1)[-1]}; "
                            "updated the dynamic car/place frames."
                        )
                    else:
                        print(
                            f"Car {cycle_index} is advancing while the robot "
                            f"picks roof {cycle_index}."
                        )

                    print("Moving the suction TCP to roof_pre_pick...")
                    move_tcp_to_frame(
                        robot,
                        tree,
                        "roof_pre_pick",
                        speed=config.cartesian_speed,
                        acceleration=config.cartesian_acceleration,
                    )

                    print("Descending to roof_pick...")
                    move_tcp_to_frame(
                        robot,
                        tree,
                        "roof_pick",
                        speed=config.contact_speed,
                        acceleration=config.cartesian_acceleration,
                    )

                    print("Turning suction on...")
                    gripper.grasp()
                    grip_deadline = (
                        time.monotonic() + SUCTION_GRIP_TIMEOUT_SECONDS
                    )
                    while not gripper.get_part_present():
                        if time.monotonic() >= grip_deadline:
                            raise RuntimeError(
                                "Suction did not detect the roof after waiting "
                                "for its configured retry; the robot was not "
                                "lifted."
                            )
                        time.sleep(SUCTION_STATUS_POLL_SECONDS)

                    print("Roof attached. Lifting back to roof_pre_pick...")
                    move_tcp_to_frame(
                        robot,
                        tree,
                        "roof_pre_pick",
                        speed=config.cartesian_speed,
                        acceleration=config.cartesian_acceleration,
                    )

                    if next_car_future is not None:
                        world_T_car_pose = next_car_future.result()
                        conveyor_T_car = tree.lookup_transform(
                            "conveyor",
                            "world",
                        ) @ tfutils.pose_to_transformation_matrix(
                            world_T_car_pose,
                            rot_type="deg",
                        )
                        tree.update("car", conveyor_T_car, rot_type="mat")
                        print(
                            f"Stopped {detected_sledge.rsplit('/', 1)[-1]}; "
                            "updated the dynamic car/place frames."
                        )

                    print(f"Moving above car {place_index}...")
                    move_tcp_to_frame(
                        robot,
                        tree,
                        "place",
                        speed=config.cartesian_speed,
                        acceleration=config.cartesian_acceleration,
                        vertical_offset=PLACE_PRE_GRASP_OFFSET,
                    )
                    print(f"Lowering the roof onto car {place_index}...")
                    move_tcp_to_frame(
                        robot,
                        tree,
                        "place",
                        speed=config.contact_speed,
                        acceleration=config.cartesian_acceleration,
                    )

                    print("Turning suction off...")
                    gripper.release()
                    time.sleep(SUCTION_SETTLE_SECONDS)
                    if gripper.get_part_present():
                        raise RuntimeError(
                            "Suction still reports the roof after release."
                        )
                    scene_backend.attach_placed_roof(
                        roof_path=ROOF_PRIM_PATHS[pool_slot_index],
                        sledge_path=detected_sledge,
                    )

                    print("Retracting above the car...")
                    move_tcp_to_frame(
                        robot,
                        tree,
                        "place",
                        speed=config.cartesian_speed,
                        acceleration=config.cartesian_acceleration,
                        vertical_offset=PLACE_PRE_GRASP_OFFSET,
                    )

                    if cycle_index < ASSEMBLY_CYCLE_COUNT:
                        next_pool_slot_index = cycle_index % len(SLEDGE_PRIM_PATHS)
                        next_sledge_path = SLEDGE_PRIM_PATHS[next_pool_slot_index]
                        next_roof_path = ROOF_PRIM_PATHS[next_pool_slot_index]

                        if cycle_index == 1:
                            scene_backend.activate_roof(
                                SECOND_ROOF_PRIM_PATH,
                            )
                        else:
                            # This pool slot was processed two cycles ago. Its
                            # old car is removed before it reaches the conveyor
                            # edge, then the car and roof are respawned at their
                            # original calibrated start poses.
                            scene_backend.spawn_cycle_assets(
                                next_sledge_path,
                                next_roof_path,
                            )

                        next_car_future = conveyor_executor.submit(
                            scene_backend.wait_for_car,
                            wait_for_clear=True,
                            sledge_path=next_sledge_path,
                        )
                        print(
                            f"Roof {cycle_index} placed. Car "
                            f"{cycle_index + 1} is now advancing in parallel."
                        )

            input(
                f"All {ASSEMBLY_CYCLE_COUNT} roofs were placed. "
                "Press Enter to disconnect..."
            )
        finally:
            gripper.disconnect()
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    main()
