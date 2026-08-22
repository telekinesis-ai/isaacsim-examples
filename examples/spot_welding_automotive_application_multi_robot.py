"""Spot-weld cars stopped at the automotive lightbeam.

The user imports the KUKA and modeled welding gun into the open Isaac Sim
scene. This application places and configures them, recycles the two car slots,
stops each car at the lightbeam, performs one weld, and retracts safely.

Before running:
    1. Open the automotive spot-welding scene in Isaac Sim.
    2. Enable the Telekinesis Isaac Sim bridge and code-socket extensions.
    3. Import the KUKA and welding gun at the configured prim paths.
    4. Run ``python spot_welding_automotive_application_multi_robot.py --kuka``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import requests

from telekinesis.synapse.robots.manipulators import abstract_robot, kuka
from telekinesis.synapse.tools.welding_gun import SpotWeldingGun
from telekinesis.tf import tftree, tfutils

from spot_welding_automotive_isaacsim_backend import (
    SpotWeldingAutomotiveIsaacSimBackend,
)
from spot_welding_automotive_static_frames_visualization import (
    KUKA_TOOL0_T_WELD_TCP,
    build_static_frame_tree,
)


BASE_URL = "http://127.0.0.1:8766"
REQUEST_TIMEOUT_SECONDS = 30.0
MOTION_SETTLE_SECONDS = 0.5
# The gun joint travels from 0.0 to 0.1 m. At the calibrated car weld, the
# electrodes contact the panel at approximately 0.0122 m, or 0.122 normalized.
# Targeting 0.12 lets synchronous weld motion reach its destination instead of
# waiting for the unobstructed fully closed position at 1.0.
WELD_GUN_CLOSED_POSITION = 0.12

KUKA_PRIM_PATH = "/World/kuka_kr210"
WELDING_GUN_PRIM_PATH = "/World/spot_welding_gun_modelled"
SLEDGE_PRIM_PATHS = ["/World/sledge", "/World/sledge_01"]
SPAWN_TEMPLATE_ROOT = "/World/_spot_welding_spawn_templates"
WELD_CYCLE_COUNT = 8
LIGHTBEAM_PRIM_PATH = "/World/LightBeam_Sensor"
CONVEYOR_RUN_VELOCITIES = {
    "/World/ConveyorBelt_A05": [0.6, 0.0, 0.0],
    "/World/ConveyorBelt_A05_01": [-0.6, 0.0, 0.0],
    "/World/ConveyorBelt_A05_02": [0.6, 0.0, 0.0],
    "/World/ConveyorBelt_A05_03": [0.6, 0.0, 0.0],
}

SPARK_RELATIVE_PATHS = (
    "base_link/base_visual/mountplate/SpotWeldingTool_U20__U23_3/spark1",
    "base_link/base_visual/mountplate/SpotWeldingTool_U20__U23_3/spark2",
)


@dataclass(frozen=True)
class RobotConfig:
    """Robot-specific values used by the shared welding workflow."""

    robot_class: type
    prim_path: str
    mount_T_robot_base: list[float]
    tool_mount_frame: str
    weld_tcp_offset: list[float]
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
        mount_T_robot_base=[0.0, 0.0, 0.0, 0.0, 0.0, 90.0],
        tool_mount_frame="link_6",
        weld_tcp_offset=KUKA_TOOL0_T_WELD_TCP,
        home_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    """Send one request to the local Isaac Sim bridge."""
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
    """Place the already imported robot at the TF ``robot_base`` frame."""
    robot_pose = [
        float(value)
        for value in tfutils.transformation_matrix_to_pose(
            tree.lookup_transform("world", "robot_base"),
            rot_type="rotvec",
        )
    ]
    bridge_request("PATCH", "/stage/simulation/timeline/stop")
    bridge_request(
        "PUT",
        "/prims/poses",
        body={
            "prim_path": robot_prim_path,
            "input_pose": {"pose": robot_pose},
        },
    )
    print(f"Robot pose in world: {[round(value, 6) for value in robot_pose]}")


def apply_drive_gains(config: RobotConfig) -> None:
    """Apply the tested stable position-drive gains to the robot."""
    articulation = bridge_request(
        "PUT",
        "/articulations",
        body={"prim_path": config.prim_path},
    )
    if articulation is None:
        raise RuntimeError("The bridge returned no KUKA articulation information.")
    bridge_request(
        "POST",
        f"/articulations/{articulation['articulation_id']}/dof_gains",
        body={
            "stiffness": config.drive_stiffness,
            "damping": config.drive_damping,
        },
    )


def move_tcp_to_frame(
    robot: abstract_robot.AbstractManipulator,
    tree: tftree.TransformTree,
    frame_name: str,
    speed: float,
    acceleration: float,
) -> None:
    """Move the active robot TCP to one named TF target."""
    target_pose = list(
        tree.lookup_transform("robot_base", frame_name, rot_type="deg")
    )
    robot.set_cartesian_pose(
        target_pose,
        speed=speed,
        acceleration=acceleration,
    )
    time.sleep(MOTION_SETTLE_SECONDS)


def main() -> None:
    """Spot-weld cars by recycling the scene's two physical car slots."""
    parser = argparse.ArgumentParser(
        description="Run the automotive spot-welding application.",
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
    scene_backend = SpotWeldingAutomotiveIsaacSimBackend(
        conveyor_velocities=CONVEYOR_RUN_VELOCITIES,
        lightbeam_path=LIGHTBEAM_PRIM_PATH,
        sledge_paths=SLEDGE_PRIM_PATHS,
        sledge_template_source_path=SLEDGE_PRIM_PATHS[1],
        template_root=SPAWN_TEMPLATE_ROOT,
    )
    tree = build_static_frame_tree(
        robot_mount_T_robot_base=config.mount_T_robot_base,
    )

    bridge_request("GET", "/status")
    print(f"Robot: {args.robot_name} ({config.prim_path})")
    print("Placing the robot on the TF robot_mount frame...")
    place_robot_on_mount(tree, config.prim_path)

    scene_backend.hold_conveyors_during_setup()
    bridge_request("PATCH", "/stage/simulation/timeline/play")
    time.sleep(0.5)

    robot = config.robot_class(name=args.robot_name)
    gun_root = WELDING_GUN_PRIM_PATH.rstrip("/")
    gun = SpotWeldingGun(
        closed_position=WELD_GUN_CLOSED_POSITION,
        spark_prim_paths=tuple(
            f"{gun_root}/{relative_path}"
            for relative_path in SPARK_RELATIVE_PATHS
        ),
    )

    print("Connecting the robot...")
    robot.connect(simulation_prim_path=config.prim_path)
    try:
        print(f"Connecting the welding gun at {WELDING_GUN_PRIM_PATH}...")
        gun.connect(simulation_prim_path=WELDING_GUN_PRIM_PATH)
        try:
            print(f"Attaching the welding gun to {config.tool_mount_frame}...")
            robot.attach_tool(gun, mount_frame=config.tool_mount_frame)
            robot.add_tcp(
                name="weld_tcp",
                transform=config.weld_tcp_offset,
                set_active=True,
            )
            print("Active TCP: weld_tcp at the electrode contact frame")

            apply_drive_gains(config)
            print(
                "Applied KUKA drive gains: "
                f"stiffness={config.drive_stiffness:g}, "
                f"damping={config.drive_damping:g}"
            )
            gun.open()

            print(f"Moving to the KUKA home pose: {config.home_joints}")
            robot.set_joint_positions(
                config.home_joints,
                speed=config.joint_speed,
                acceleration=config.joint_acceleration,
            )
            time.sleep(MOTION_SETTLE_SECONDS)

            scene_backend.prepare_car_spawn_template()

            for cycle_index in range(1, WELD_CYCLE_COUNT + 1):
                pool_slot_index = (cycle_index - 1) % len(SLEDGE_PRIM_PATHS)
                sledge_path = SLEDGE_PRIM_PATHS[pool_slot_index]
                if cycle_index > len(SLEDGE_PRIM_PATHS):
                    scene_backend.spawn_car(sledge_path)

                print(f"\n--- car {cycle_index}/{WELD_CYCLE_COUNT} ---")
                world_T_car_pose = scene_backend.wait_for_car(
                    wait_for_clear=cycle_index > 1,
                    sledge_path=sledge_path,
                )
                conveyor_T_car = tree.lookup_transform(
                    "conveyor",
                    "world",
                ) @ tfutils.pose_to_transformation_matrix(
                    world_T_car_pose,
                    rot_type="deg",
                )
                tree.update("car", conveyor_T_car, rot_type="mat")
                print("Updated the dynamic car and weld frames.")

                print("Moving to pre_weld...")
                move_tcp_to_frame(
                    robot,
                    tree,
                    "pre_weld",
                    speed=config.cartesian_speed,
                    acceleration=config.cartesian_acceleration,
                )
                print("Moving to weld_tcp...")
                move_tcp_to_frame(
                    robot,
                    tree,
                    "weld_tcp",
                    speed=config.contact_speed,
                    acceleration=config.cartesian_acceleration,
                )

                print("Closing the gun and performing the spot weld...")
                gun.weld()
                print("Spot weld complete; the gun reopened.")

                print("Retracting to pre_weld...")
                move_tcp_to_frame(
                    robot,
                    tree,
                    "pre_weld",
                    speed=config.cartesian_speed,
                    acceleration=config.cartesian_acceleration,
                )

                print("Returning to the all-zero home pose...")
                robot.set_joint_positions(
                    config.home_joints,
                    speed=config.joint_speed,
                    acceleration=config.joint_acceleration,
                )
                time.sleep(MOTION_SETTLE_SECONDS)
                if cycle_index < WELD_CYCLE_COUNT:
                    print("Home reached; starting the next conveyor cycle.")

            input(
                f"All {WELD_CYCLE_COUNT} cars were welded. "
                "Press Enter to disconnect..."
            )
        finally:
            gun.disconnect()
    finally:
        robot.disconnect()
        robot.shutdown()


if __name__ == "__main__":
    main()
