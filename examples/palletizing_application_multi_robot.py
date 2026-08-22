"""Set up the robot and suction gripper for the palletizing application.

This builds the existing static TF tree, places the selected robot at its
``robot_base`` frame, attaches the simulated suction gripper, runs all conveyors
until the lightbeam detects each box, picks it using the dynamic TF grasp frame,
and carries it with the TCP facing down to the next TF pallet frame. Once a box
is lifted clear, the conveyors move the next box to the lightbeam in parallel
while the robot completes the current placement and returns home.

Before running:
    1. Open the palletizing scene in Isaac Sim.
    2. Enable the Telekinesis Isaac Sim bridge extension.
    3. Import the robot at the configured prim path.
    4. Run::

           python palletizing_application_multi_robot.py --ur10e
           python palletizing_application_multi_robot.py --fanuc

This is an external Python application; it does not import ``omni`` or
``isaacsim``.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

from telekinesis.tf import tftree, tfutils
from telekinesis.synapse.robots.manipulators import (
    abstract_robot,
    fanuc,
    universal_robots,
)
from telekinesis.synapse.tools.suction_grippers import custom

from palletizing_isaacsim_backend import (
    DetectedBox,
    PalletizingIsaacSimBackend,
)
from palletizing_static_frames_visualization import (
    CONVEYOR_PRIM_PATH,
    PALLET_PRIM_PATH,
    PLACE_NUMX,
    PLACE_NUMY,
    PLACE_NUMZ,
    ROBOT_MOUNT_PRIM_PATH,
    build_static_frame_tree,
    get_pose_in_world,
)


BASE_URL = "http://127.0.0.1:8766"
REQUEST_TIMEOUT_SECONDS = 30.0

MOTION_SETTLE_SECONDS = 0.3
BOX_COUNT = 8

LIGHTBEAM_PRIM_PATH = "/World/palletizing_rough_scene/LightBeam_Sensor"
PALLETIZING_SCENE_ROOT = "/World/palletizing_rough_scene"
BOX_PRIM_NAME_PREFIX = "Cardbox_C2"
CONVEYOR_PRIM_PATHS = [
    CONVEYOR_PRIM_PATH,
    "/World/palletizing_rough_scene/ConveyorBelt_A11",
    "/World/palletizing_rough_scene/ConveyorBelt_A08_01",
]
CONVEYOR_RUN_VELOCITY = [-0.8, 0.0, 0.0]

# These prims must already exist in the open Isaac Sim scene. The application
# does not import the robot or gripper assets.
UR10E_PRIM_PATH = "/World/ur10e_robot"
FANUC_PRIM_PATH = "/World/fanuc_crx10ial"
SUCTION_GRIPPER_PRIM_PATH = "/World/defitech_modelled_surface_gripper_modelled"
# Centre of the visible suction-pad face relative to the active robot tool frame.
SUCTION_TCP_OFFSET = [0.0, 0.0, 0.075, 0.0, 0.0, 0.0]
# Suction ray origin relative to the Defitech gripper root. The USD currently
# authors this at +0.26088396 m, far above the visible pad.
SUCTION_ATTACHMENT_POINT_TRANSLATION = [0.0, 0.0, -0.073]

# Defitech gripper mounting orientation relative to the UR10e tool frame.
UR10E_T_SUCTION_GRIPPER = [0.0, 0.0, 0.0, 180.0, 0.0, 0.0]
# Tested link_6-to-gripper alignment for the imported Fanuc CRX-10iA/L.
FANUC_T_SUCTION_GRIPPER = [0.0, 0.0, 0.0, 0.0, -90.0, 0.0]


@dataclass(frozen=True)
class RobotConfig:
    """Robot-specific setup values used by the shared application.

    Attributes:
        robot_class: Synapse class used to create the robot client.
        prim_path: Default robot prim path in Isaac Sim.
        mount_T_robot_base: Robot-base pose relative to ``robot_mount``, using
            XYZ metres and Euler XYZ degrees.
        tool_mount_transform: Robot flange-to-gripper-root transform, using XYZ
            metres and Euler XYZ degrees.
        suction_tcp_offset: Active suction TCP relative to the robot tool
            frame, using XYZ metres and Euler XYZ degrees.
        home_joints: Clear home pose in joint degrees.
        joint_speed: Joint-motion speed in degrees per second.
        joint_acceleration: Joint-motion acceleration in degrees per second squared.
        cartesian_speed: TCP linear speed in metres per second.
        cartesian_acceleration: TCP linear acceleration in metres per second squared.
        contact_speed: Slower TCP speed used while approaching the box.
        carry_speed: TCP speed used while carrying a box.
        drive_stiffness: Optional simulation drive stiffness. ``None`` keeps
            the gains already authored in the robot asset.
        drive_damping: Optional simulation drive damping. ``None`` keeps the
            gains already authored in the robot asset.
    """

    robot_class: type
    prim_path: str
    mount_T_robot_base: list[float]
    tool_mount_transform: list[float]
    suction_tcp_offset: list[float]
    home_joints: list[float]
    joint_speed: float
    joint_acceleration: float
    cartesian_speed: float
    cartesian_acceleration: float
    contact_speed: float
    carry_speed: float
    drive_stiffness: float | None
    drive_damping: float | None


ROBOT_CONFIGS = {
    "ur10e": RobotConfig(
        robot_class=universal_robots.UniversalRobotsUR10E,
        prim_path=UR10E_PRIM_PATH,
        mount_T_robot_base=[0.0, 0.0, 0.0, 0.0, 0.0, 180.0],
        tool_mount_transform=UR10E_T_SUCTION_GRIPPER,
        suction_tcp_offset=SUCTION_TCP_OFFSET,
        home_joints=[0.0, -90.0, -60.0, -120.0, 90.0, 0.0],
        joint_speed=45.0,
        joint_acceleration=60.0,
        cartesian_speed=0.30,
        cartesian_acceleration=0.50,
        contact_speed=0.08,
        carry_speed=0.15,
        # The Telekinesis URDF import needs explicit gains to hold its pose.
        drive_stiffness=1.0e5,
        drive_damping=1.0e4,
    ),
    "fanuc": RobotConfig(
        robot_class=fanuc.FanucCRX10IAL,
        prim_path=FANUC_PRIM_PATH,
        # Keep the robot facing the palletizing work area on this mount.
        mount_T_robot_base=[0.0, 0.0, 0.0, 0.0, 0.0, 180.0],
        tool_mount_transform=FANUC_T_SUCTION_GRIPPER,
        suction_tcp_offset=SUCTION_TCP_OFFSET,
        home_joints=[45.0, 0.0, 0.0, 0.0, -90.0, 135.0],
        joint_speed=45.0,
        joint_acceleration=60.0,
        cartesian_speed=0.30,
        cartesian_acceleration=0.50,
        contact_speed=0.08,
        carry_speed=0.15,
        drive_stiffness=1.0e5,
        drive_damping=1.0e4,
    ),
}


def bridge_request(method: str, path: str, *, body: dict | None = None) -> dict | None:
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


def update_pick_frames(
    tree: tftree.TransformTree,
    detected_box: DetectedBox,
) -> None:
    """Update the runtime pick object and grasp from a stopped box.

    Args:
        tree: Palletizing transform tree containing ``pick_object`` and
            ``pick_grasp``.
        detected_box: Actual stopped-box measurement from Isaac Sim.

    Returns:
        None.

    Raises:
        ValueError: If a TF frame or measured pose is invalid.
    """
    tree.update(
        "pick_object",
        detected_box.conveyor_T_object,
        rot_type="deg",
    )
    tree.update(
        "pick_grasp",
        [0.0, 0.0, detected_box.size[2] / 2.0, 180.0, 0.0, 0.0],
        rot_type="deg",
    )


def place_robot_on_mount(
    tree: tftree.TransformTree,
    robot_prim_path: str,
) -> None:
    """Move the robot prim to the TF tree's ``robot_base`` frame.

    Args:
        tree: Palletizing static transform tree.
        robot_prim_path: USD path of the robot prim to move.

    Returns:
        None.

    Raises:
        requests.RequestException: If a bridge request fails.
        ValueError: If the transform cannot be converted to a pose.
    """
    bridge_request("PATCH", "/stage/simulation/timeline/stop")

    robot_base_pose_in_world = [
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
            "input_pose": {"pose": robot_base_pose_in_world},
        },
    )


def move_tcp_to_frame(
    robot: abstract_robot.AbstractManipulator,
    tree: tftree.TransformTree,
    frame_name: str,
    speed: float,
    acceleration: float,
) -> None:
    """Move the active robot TCP to a named TF frame and let it settle.

    Args:
        robot: Connected Synapse robot.
        tree: Palletizing transform tree containing the target frame.
        frame_name: Name of the target frame.
        speed: Cartesian speed in metres per second.
        acceleration: Cartesian acceleration in metres per second squared.

    Returns:
        None.

    Raises:
        RuntimeError: If Synapse cannot generate or execute the Cartesian move.
        ValueError: If the TF frame or pose is invalid.
    """
    target_pose = tree.lookup_transform("robot_base", frame_name, rot_type="deg")
    robot.set_cartesian_pose(
        target_pose,
        speed=speed,
        acceleration=acceleration,
    )
    time.sleep(MOTION_SETTLE_SECONDS)


def main() -> None:
    """Run the palletizing cycle with the selected robot and suction gripper.

    Returns:
        None.

    Raises:
        requests.RequestException: If an Isaac Sim bridge request fails.
        RuntimeError: If the robot or suction gripper cannot connect or attach.
        ValueError: If a configured transform is invalid.
    """
    parser = argparse.ArgumentParser(
        description="Set up a supported robot for the palletizing application.",
    )
    robot_selection = parser.add_mutually_exclusive_group()
    robot_selection.add_argument(
        "--ur10e",
        dest="robot_name",
        action="store_const",
        const="ur10e",
        help="Use the Universal Robots UR10e (default).",
    )
    robot_selection.add_argument(
        "--fanuc",
        dest="robot_name",
        action="store_const",
        const="fanuc",
        help="Use the Fanuc CRX-10iA/L.",
    )
    parser.set_defaults(robot_name="ur10e")
    parser.add_argument(
        "--robot-prim-path",
        help="Override the selected robot's default Isaac Sim prim path.",
    )
    parser.add_argument(
        "--gripper-prim-path",
        default=SUCTION_GRIPPER_PRIM_PATH,
        help="Override the suction-gripper prim path.",
    )
    args = parser.parse_args()

    config = ROBOT_CONFIGS[args.robot_name]
    robot_prim_path = args.robot_prim_path or config.prim_path
    scene_backend = PalletizingIsaacSimBackend(
        conveyor_paths=CONVEYOR_PRIM_PATHS,
        lightbeam_path=LIGHTBEAM_PRIM_PATH,
        scene_root=PALLETIZING_SCENE_ROOT,
        box_prim_name_prefix=BOX_PRIM_NAME_PREFIX,
        conveyor_run_velocity=CONVEYOR_RUN_VELOCITY,
    )

    bridge_request("GET", "/status")
    tree = build_static_frame_tree(
        get_pose_in_world(CONVEYOR_PRIM_PATH),
        get_pose_in_world(PALLET_PRIM_PATH),
        get_pose_in_world(ROBOT_MOUNT_PRIM_PATH),
    )
    tree.update("robot_base", config.mount_T_robot_base, rot_type="deg")

    print(f"Robot: {args.robot_name} ({robot_prim_path})")
    place_robot_on_mount(tree, robot_prim_path)

    robot = config.robot_class(name=args.robot_name)
    gripper = custom.CustomSuctionGripper()

    # CustomSuctionGripper.connect() requires the timeline to be playing.
    bridge_request("PATCH", "/stage/simulation/timeline/play")
    time.sleep(0.5)

    print("Connecting the robot and suction gripper...")
    robot.connect(simulation_prim_path=robot_prim_path)
    try:
        gripper.connect(simulation_prim_path=args.gripper_prim_path)
        try:
            gripper.set_attachment_point_properties(
                local_pose_0={
                    "translation": SUCTION_ATTACHMENT_POINT_TRANSLATION,
                },
            )
            robot.attach_tool(
                gripper,
                transform=config.tool_mount_transform,
            )
            robot.add_tcp(
                name="suction_tcp",
                transform=config.suction_tcp_offset,
                set_active=True,
            )
            # Imported cobots may need gain overrides after attachment restarts
            # physics. The built-in Isaac Sim UR10e deliberately skips this.
            if config.drive_stiffness is not None and config.drive_damping is not None:
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
                    body={
                        "stiffness": config.drive_stiffness,
                        "damping": config.drive_damping,
                    },
                )

            print("Moving to the home L pose...")
            robot.set_joint_positions(
                config.home_joints,
                speed=config.joint_speed,
                acceleration=config.joint_acceleration,
            )
            time.sleep(MOTION_SETTLE_SECONDS)

            if BOX_COUNT > PLACE_NUMX * PLACE_NUMY * PLACE_NUMZ:
                raise ValueError("BOX_COUNT exceeds the available pallet TF frames.")

            with ThreadPoolExecutor(max_workers=1) as conveyor_executor:
                detected_box_future = conveyor_executor.submit(
                    scene_backend.run_conveyors_until_lightbeam
                )

                for box_index in range(BOX_COUNT):
                    z_index, layer_index = divmod(
                        box_index,
                        PLACE_NUMX * PLACE_NUMY,
                    )
                    x_index, y_index = divmod(layer_index, PLACE_NUMY)
                    frame_suffix = f"{x_index}_{y_index}_{z_index}"
                    print(f"\n--- box {box_index + 1}/{BOX_COUNT} ---")

                    detected_box = detected_box_future.result()
                    update_pick_frames(tree, detected_box)
                    print(f"Picking {detected_box.prim_path.rsplit('/', 1)[-1]}...")
                    move_tcp_to_frame(
                        robot,
                        tree,
                        "pre_pick",
                        speed=config.cartesian_speed,
                        acceleration=config.cartesian_acceleration,
                    )
                    move_tcp_to_frame(
                        robot,
                        tree,
                        "pick_grasp",
                        speed=config.contact_speed,
                        acceleration=config.cartesian_acceleration,
                    )

                    gripper.grasp()
                    if not gripper.get_part_present():
                        raise RuntimeError(
                            "Suction did not detect a box. The robot was not lifted."
                        )

                    move_tcp_to_frame(
                        robot,
                        tree,
                        "pre_pick",
                        speed=config.cartesian_speed,
                        acceleration=config.cartesian_acceleration,
                    )

                    # Once the held box is clear, move the next box to the
                    # lightbeam while the robot completes the current placement.
                    if box_index + 1 < BOX_COUNT:
                        detected_box_future = conveyor_executor.submit(
                            scene_backend.run_conveyors_until_lightbeam
                        )

                    if (x_index, y_index) != (0, 0):
                        move_tcp_to_frame(
                            robot,
                            tree,
                            f"pre_place_0_0_{z_index}",
                            speed=config.carry_speed,
                            acceleration=config.cartesian_acceleration,
                        )

                    print(f"Placing at pallet frame {frame_suffix}...")
                    move_tcp_to_frame(
                        robot,
                        tree,
                        f"pre_place_{frame_suffix}",
                        speed=config.carry_speed,
                        acceleration=config.cartesian_acceleration,
                    )
                    move_tcp_to_frame(
                        robot,
                        tree,
                        f"place_grasp_{frame_suffix}",
                        speed=config.contact_speed,
                        acceleration=config.cartesian_acceleration,
                    )

                    gripper.release()
                    if gripper.get_part_present():
                        raise RuntimeError("Suction still reports a box after release.")

                    move_tcp_to_frame(
                        robot,
                        tree,
                        f"pre_place_{frame_suffix}",
                        speed=config.cartesian_speed,
                        acceleration=config.cartesian_acceleration,
                    )

                    robot.set_joint_positions(
                        config.home_joints,
                        speed=config.joint_speed,
                        acceleration=config.joint_acceleration,
                    )
                    time.sleep(MOTION_SETTLE_SECONDS)

            input(f"Placed {BOX_COUNT} boxes. Press Enter to disconnect...")
        finally:
            gripper.disconnect()
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    main()
