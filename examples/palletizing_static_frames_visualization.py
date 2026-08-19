"""Build and visualize the palletizing cell's static TF frames.

Only the conveyor, pallet, and robot mount receive world poses from Isaac Sim.
The nominal pick target is defined relative to the conveyor; the application
updates it from each box's stopped pose at runtime. The eight place targets are
defined relative to the pallet as two centred 2-by-2 layers::

    world
    |-- conveyor
    |   `-- pick_object
    |       `-- pick_grasp
    |           `-- pre_pick
    |-- pallet
    |   |-- place_object_0_0_0
    |   |   `-- place_grasp_0_0_0
    |   |       `-- pre_place_0_0_0
    |   `-- ... place_object/grasp_1_1_1
    `-- robot_mount
        `-- robot_base

Object frames represent box centres. Grasp frames are at the top surfaces and
have their Z axes facing downward. This file only reads scene poses and draws
the TF tree in Rerun; it does not move objects or modify the Isaac Sim stage.
"""

from __future__ import annotations

import requests
import rerun as rr

from telekinesis.tf import tftree, tfutils


BASE_URL = "http://127.0.0.1:8766"
REQUEST_TIMEOUT_SECONDS = 30.0

CONVEYOR_PRIM_PATH = "/World/palletizing_rough_scene/ConveyorBelt_A08"
PALLET_PRIM_PATH = "/World/palletizing_rough_scene/pallet"
ROBOT_MOUNT_PRIM_PATH = "/World/palletizing_rough_scene/ur10_mount"
ROBOT_MOUNT_T_ROBOT_BASE = [0.0, 0.0, 0.0, 0.0, 0.0, 180.0]

BOX_SIZE = [0.513243397, 0.331865479, 0.259689436]
CONVEYOR_T_PICK_OBJECT = [
    -2.205667546,
    0.119918064,
    0.898990207,
    0.0,
    0.0,
    90.0,
]
OBJECT_T_GRASP = [0.0, 0.0, BOX_SIZE[2] / 2.0, 180.0, 0.0, 0.0]
# The pallet is about 475 mm lower than the conveyor. The base-layer offsets
# put pre-pick and pre-place at the same world height for level travel.
PICK_GRASP_T_PRE_PICK = [0.0, 0.0, -0.40, 0.0, 0.0, 0.0]
PLACE_GRASP_T_PRE_PLACE = [0.0, 0.0, -0.875, 0.0, 0.0, 0.0]

# The calibration box was resting directly on the pallet. Its USD root is near
# the bottom face; this TF frame is shifted to the logical box centre.
PALLET_T_FIRST_PLACE_OBJECT = [
    -0.307142031,
    -0.200925418,
    0.272350404,
    0.0,
    0.0,
    0.0,
]

# Mirror the calibrated first position across the pallet X and Y axes. This
# creates one centred 2-by-2 layer with the same box orientation in every cell.
PLACE_XSTEP = [-2.0 * PALLET_T_FIRST_PLACE_OBJECT[0], 0.0, 0.0]
PLACE_YSTEP = [0.0, -2.0 * PALLET_T_FIRST_PLACE_OBJECT[1], 0.0]
PLACE_ZSTEP = [0.0, 0.0, BOX_SIZE[2]]
PLACE_NUMX = 2
PLACE_NUMY = 2
PLACE_NUMZ = 2

TF_AXIS_LENGTH = 0.10


def bridge_request(
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> dict:
    """Send a read-only request to the local Isaac Sim bridge.

    Args:
        method: HTTP request method.
        path: Bridge endpoint path.
        params: Optional query parameters.

    Returns:
        Decoded JSON response from the bridge.

    Raises:
        requests.RequestException: If the bridge request or response fails.
    """
    response = requests.request(
        method,
        BASE_URL + path,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_pose_in_world(prim_path: str) -> list[float]:
    """Read a prim's rigid pose in the Isaac Sim world frame.

    Args:
        prim_path: USD path of the prim to query.

    Returns:
        XYZ in metres followed by a rotation vector in radians.

    Raises:
        requests.RequestException: If the bridge request fails.
        KeyError: If the bridge response does not contain ``pose``.
        TypeError: If the returned pose is not iterable.
        ValueError: If a pose value cannot be converted to ``float``.
    """
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


def build_static_frame_tree(
    conveyor_pose_in_world: list[float],
    pallet_pose_in_world: list[float],
    robot_mount_pose_in_world: list[float],
) -> tftree.TransformTree:
    """Build the palletizing cell's static TF tree.

    Args:
        conveyor_pose_in_world: Conveyor pose in world, using XYZ metres and a
            rotation vector in radians.
        pallet_pose_in_world: Pallet pose in world, using XYZ metres and a
            rotation vector in radians.
        robot_mount_pose_in_world: Robot-mount pose in world, using XYZ metres
            and a rotation vector in radians.

    Returns:
        Static palletizing transform tree.

    Raises:
        ValueError: If a supplied pose or TF frame is invalid.
    """
    tree = tftree.TransformTree("world")

    world_T_conveyor = tfutils.pose_to_transformation_matrix(
        conveyor_pose_in_world,
        rot_type="rotvec",
    )
    tree.add("world", "conveyor", world_T_conveyor, rot_type="mat")
    tree.add(
        "conveyor",
        "pick_object",
        CONVEYOR_T_PICK_OBJECT,
        rot_type="deg",
    )
    tree.add(
        "pick_object",
        "pick_grasp",
        OBJECT_T_GRASP,
        rot_type="deg",
    )
    tree.add(
        "pick_grasp",
        "pre_pick",
        PICK_GRASP_T_PRE_PICK,
        rot_type="deg",
    )

    world_T_pallet = tfutils.pose_to_transformation_matrix(
        pallet_pose_in_world,
        rot_type="rotvec",
    )
    tree.add("world", "pallet", world_T_pallet, rot_type="mat")

    for z_index in range(PLACE_NUMZ):
        for x_index in range(PLACE_NUMX):
            for y_index in range(PLACE_NUMY):
                place_xyz = [
                    PALLET_T_FIRST_PLACE_OBJECT[axis]
                    + x_index * PLACE_XSTEP[axis]
                    + y_index * PLACE_YSTEP[axis]
                    + z_index * PLACE_ZSTEP[axis]
                    for axis in range(3)
                ]
                object_frame = f"place_object_{x_index}_{y_index}_{z_index}"
                grasp_frame = f"place_grasp_{x_index}_{y_index}_{z_index}"
                pre_place_frame = f"pre_place_{x_index}_{y_index}_{z_index}"
                tree.add(
                    "pallet",
                    object_frame,
                    [*place_xyz, *PALLET_T_FIRST_PLACE_OBJECT[3:]],
                    rot_type="deg",
                )
                tree.add(
                    object_frame,
                    grasp_frame,
                    OBJECT_T_GRASP,
                    rot_type="deg",
                )
                grasp_T_pre_place = PLACE_GRASP_T_PRE_PLACE.copy()
                # The grasp frame points down. Shorten its negative-Z offset by
                # the stack height so every layer has one common travel height.
                grasp_T_pre_place[2] += z_index * PLACE_ZSTEP[2]
                tree.add(
                    grasp_frame,
                    pre_place_frame,
                    grasp_T_pre_place,
                    rot_type="deg",
                )

    world_T_robot_mount = tfutils.pose_to_transformation_matrix(
        robot_mount_pose_in_world,
        rot_type="rotvec",
    )
    tree.add("world", "robot_mount", world_T_robot_mount, rot_type="mat")
    tree.add(
        "robot_mount",
        "robot_base",
        ROBOT_MOUNT_T_ROBOT_BASE,
        rot_type="deg",
    )
    return tree


def main() -> None:
    """Read the three scene anchors and visualize their static TF children.

    Returns:
        None.

    Raises:
        requests.RequestException: If an Isaac Sim bridge request fails.
        KeyError: If a bridge pose response is missing required data.
        TypeError: If a bridge pose response has an invalid structure.
        ValueError: If a supplied pose or TF frame is invalid.
    """
    bridge_request("GET", "/status")
    tree = build_static_frame_tree(
        get_pose_in_world(CONVEYOR_PRIM_PATH),
        get_pose_in_world(PALLET_PRIM_PATH),
        get_pose_in_world(ROBOT_MOUNT_PRIM_PATH),
    )

    rr.init("palletizing_static_tf_tree", spawn=True)
    recording = rr.get_global_data_recording()
    tree.visualize_rerun(axis_len=TF_AXIS_LENGTH, recording_stream=recording)

    input("Rerun is showing the palletizing static frames. Press Enter...")


if __name__ == "__main__":
    main()
