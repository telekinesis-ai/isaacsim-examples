"""Build and visualize the static frame tree for the machine-tending cell.

Only the CNC machine and logical table frames receive world poses from Isaac
Sim. Every frame mounted inside them is defined relative to its source frame:

    world
    |-- cnc_machine
    |   `-- cnc_pedestal
    |       `-- cnc_object
    |           `-- cnc_grasp
    `-- table
        |-- robot_mount
        |   `-- robot_base
        |-- pick_grid
        |   |-- pick_slot_0_0
        |   |   `-- pick_object_0_0
        |   |       `-- pick_grasp_0_0
        |   `-- ... pick_slot/object/grasp_3_3
        `-- place_grid
            |-- place_slot_0_0
            |   `-- place_object_0_0
            |       `-- place_grasp_0_0
            `-- ... place_slot/object/grasp_3_3

The bridge supplies the CNC machine and table poses in the Isaac Sim world
frame. The CNC pedestal, robot mount, robot base, and grid origins are
calibrated transforms relative to their source frames.

Each grid origin coincides with slot ``0_0``. The remaining slots are generated
using the configured X and Y step vectors. User-defined translations are in
metres, and user-defined rotations are Euler XYZ angles in degrees.

The pick and place slot frames were calibrated at the centres of seated
cylinders, so their expected-object children use identity transforms. The CNC
pedestal is a support surface, so its expected-object frame is half an object
height above it. Every grasp frame is above its object centre with its Z axis
flipped to face downward. These are static expected targets, not live frames
that track the physical cylinders after pickup.

This script reads frame information and visualizes it in Rerun. It does not
move the robot or modify the Isaac Sim stage.
"""

from __future__ import annotations

import requests
import rerun as rr

from telekinesis.tf import tftree, tfutils


BASE_URL = "http://127.0.0.1:8766"
REQUEST_TIMEOUT_SECONDS = 30.0
CNC_PRIM_PATH = "/World/model_cnc_machine_tool"
CNC_MACHINE_T_PEDESTAL = [
    0.306580219,
    -0.361398900,
    1.443844948,
    0.0,
    0.0,
    0.0,
]
TABLE_FRAME_PRIM_PATH = "/World/table/table_frame"
TABLE_T_ROBOT_MOUNT = [
    -0.732800229,
    0.398314787,
    -0.036506316,
    0.0,
    0.0,
    89.868316437,
]
ROBOT_MOUNT_T_ROBOT_BASE = [0.0, 0.0, -0.005, 0.0, 0.0, 180.0]
TABLE_T_PICK_GRID = [
    -0.248248789,
    0.540517945,
    -0.126516634,
    0.0,
    0.0,
    0.0,
]
TABLE_T_PLACE_GRID = [
    -0.254255254,
    0.278978349,
    0.075566687,
    0.0,
    0.0,
    0.0,
]
PICK_GRID_XSTEP = [0.161751909, 0.000119909, 0.0]
PICK_GRID_YSTEP = [-0.000004609, 0.179967408, 0.0]
PLACE_GRID_XSTEP = [0.000266228, -0.180109580, 0.0]
PLACE_GRID_YSTEP = [0.161841134, 0.000461414, 0.0]
GRID_NUMX = 4
GRID_NUMY = 4
TF_AXIS_LENGTH = 0.05

OBJECT_HEIGHT = 0.135
GRASP_ABOVE_OBJECT_CENTER = 0.0475
CNC_OBJECT_FRAME = "cnc_object"
CNC_GRASP_FRAME = "cnc_grasp"
PICK_OBJECT_FRAME_PREFIX = "pick_object"
PICK_GRASP_FRAME_PREFIX = "pick_grasp"
PLACE_OBJECT_FRAME_PREFIX = "place_object"
PLACE_GRASP_FRAME_PREFIX = "place_grasp"
CNC_PEDESTAL_T_OBJECT = [0.0, 0.0, OBJECT_HEIGHT / 2, 0.0, 0.0, 0.0]
PICK_SLOT_T_OBJECT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PLACE_SLOT_T_OBJECT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
OBJECT_T_GRASP = [0.0, 0.0, GRASP_ABOVE_OBJECT_CENTER, 180.0, 0.0, 0.0]


def bridge_request(
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> dict:
    """Send a read-only request to the local Isaac Sim bridge.

    Args:
        method (str): HTTP request method.
        path (str): Bridge endpoint path.
        params (dict[str, str] | None): Optional query parameters.

    Returns:
        dict: Decoded JSON response from the bridge.

    Raises:
        requests.RequestException: If the request fails, the bridge returns an
            unsuccessful status, or its response is not valid JSON.
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
    """Read a prim pose in the Isaac Sim world frame.

    Args:
        prim_path (str): USD path of the prim to query.

    Returns:
        list[float]: Six-value pose containing XYZ in metres followed by a
            rotation vector in radians.

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


class Grid:
    """Represent a rectangular grid of TF slot frames.

    Attributes:
        name (str): Name of the grid frame.
        source_frame (str): Frame under which the grid is added.
        slot_frame_prefix (str): Prefix used for generated slot-frame names.
        source_T_grid (list[float]): Grid pose relative to the source frame.
            Translation is in metres and rotation is in Euler XYZ degrees.
        xstep (list[float]): XYZ translation in metres between slots in the X
            direction.
        ystep (list[float]): XYZ translation in metres between slots in the Y
            direction.
        numx (int): Number of slots in the X direction.
        numy (int): Number of slots in the Y direction.
    """

    def __init__(
        self,
        name: str,
        source_frame: str,
        slot_frame_prefix: str,
        source_T_grid: list[float],
        xstep: list[float],
        ystep: list[float],
        numx: int,
        numy: int,
    ) -> None:
        """Initialize a grid of TF slot frames.

        Args:
            name (str): Name of the grid frame.
            source_frame (str): Frame under which the grid will be added.
            slot_frame_prefix (str): Prefix used for slot-frame names.
            source_T_grid (list[float]): Grid pose relative to the source
                frame. Translation is in metres and rotation is in Euler XYZ
                degrees.
            xstep (list[float]): XYZ translation in metres for one step in the
                X direction.
            ystep (list[float]): XYZ translation in metres for one step in the
                Y direction.
            numx (int): Number of slots in the X direction.
            numy (int): Number of slots in the Y direction.

        Raises:
            ValueError: If the pose, steps, or grid dimensions are invalid.
        """
        if len(source_T_grid) != 6:
            raise ValueError("source_T_grid must contain six pose values.")
        if len(xstep) != 3:
            raise ValueError("xstep must contain three XYZ values.")
        if len(ystep) != 3:
            raise ValueError("ystep must contain three XYZ values.")
        if numx <= 0 or numy <= 0:
            raise ValueError("numx and numy must be greater than zero.")

        self.name: str = name
        self.source_frame: str = source_frame
        self.slot_frame_prefix: str = slot_frame_prefix
        self.source_T_grid: list[float] = source_T_grid
        self.xstep: list[float] = xstep
        self.ystep: list[float] = ystep
        self.numx: int = numx
        self.numy: int = numy

    def get_slot_frame(self, x_index: int, y_index: int) -> str:
        """Return the TF frame name for a slot.

        Args:
            x_index (int): Zero-based slot index in the grid X direction.
            y_index (int): Zero-based slot index in the grid Y direction.

        Returns:
            str: Name of the requested slot frame.

        Raises:
            IndexError: If either index is outside the grid.
        """
        if not 0 <= x_index < self.numx:
            raise IndexError(f"X index is outside the grid: {x_index}")

        if not 0 <= y_index < self.numy:
            raise IndexError(f"Y index is outside the grid: {y_index}")

        return f"{self.slot_frame_prefix}_{x_index}_{y_index}"

    def add_to_tf(self, tree: tftree.TransformTree) -> None:
        """Add the grid and all its slot frames to a TF tree.

        Args:
            tree (tftree.TransformTree): TF tree that will receive the grid and
                slot frames.

        Returns:
            None.

        Raises:
            ValueError: If a frame with the same name already exists.
        """
        tree.add(
            self.source_frame,
            self.name,
            self.source_T_grid,
            rot_type="deg",
        )

        for x_index in range(self.numx):
            for y_index in range(self.numy):
                slot_xyz = [
                    x_index * self.xstep[axis]
                    + y_index * self.ystep[axis]
                    for axis in range(3)
                ]

                tree.add(
                    self.name,
                    self.get_slot_frame(x_index, y_index),
                    [*slot_xyz, 0.0, 0.0, 0.0],
                    rot_type="deg",
                )


def build_static_frame_tree(
    cnc_machine_pose_in_world: list[float],
    table_pose_in_world: list[float],
) -> tuple[tftree.TransformTree, Grid, Grid]:
    """Build the machine-tending cell's static TF tree.

    Args:
        cnc_machine_pose_in_world (list[float]): CNC machine pose in the world
            frame, with XYZ in metres and rotation vector in radians.
        table_pose_in_world (list[float]): Logical table pose in the world
            frame, with XYZ in metres and rotation vector in radians.

    Returns:
        tuple[tftree.TransformTree, Grid, Grid]: Static TF tree, pick grid, and
            place grid.

    Raises:
        ValueError: If a supplied pose, grid configuration, or TF frame is
            invalid.
    """
    tree = tftree.TransformTree("world")
    world_T_cnc_machine = tfutils.pose_to_transformation_matrix(
        cnc_machine_pose_in_world,
        rot_type="rotvec",
    )
    tree.add("world", "cnc_machine", world_T_cnc_machine, rot_type="mat")
    tree.add(
        "cnc_machine",
        "cnc_pedestal",
        CNC_MACHINE_T_PEDESTAL,
        rot_type="deg",
    )
    tree.add(
        "cnc_pedestal",
        CNC_OBJECT_FRAME,
        CNC_PEDESTAL_T_OBJECT,
        rot_type="deg",
    )
    tree.add(
        CNC_OBJECT_FRAME,
        CNC_GRASP_FRAME,
        OBJECT_T_GRASP,
        rot_type="deg",
    )

    world_T_table = tfutils.pose_to_transformation_matrix(
        table_pose_in_world,
        rot_type="rotvec",
    )
    tree.add("world", "table", world_T_table, rot_type="mat")
    tree.add("table", "robot_mount", TABLE_T_ROBOT_MOUNT, rot_type="deg")
    tree.add(
        "robot_mount",
        "robot_base",
        ROBOT_MOUNT_T_ROBOT_BASE,
        rot_type="deg",
    )

    pick_grid = Grid(
        name="pick_grid",
        source_frame="table",
        slot_frame_prefix="pick_slot",
        source_T_grid=TABLE_T_PICK_GRID,
        xstep=PICK_GRID_XSTEP,
        ystep=PICK_GRID_YSTEP,
        numx=GRID_NUMX,
        numy=GRID_NUMY,
    )
    place_grid = Grid(
        name="place_grid",
        source_frame="table",
        slot_frame_prefix="place_slot",
        source_T_grid=TABLE_T_PLACE_GRID,
        xstep=PLACE_GRID_XSTEP,
        ystep=PLACE_GRID_YSTEP,
        numx=GRID_NUMX,
        numy=GRID_NUMY,
    )
    pick_grid.add_to_tf(tree)
    place_grid.add_to_tf(tree)

    for grid, object_prefix, grasp_prefix, slot_T_object in (
        (
            pick_grid,
            PICK_OBJECT_FRAME_PREFIX,
            PICK_GRASP_FRAME_PREFIX,
            PICK_SLOT_T_OBJECT,
        ),
        (
            place_grid,
            PLACE_OBJECT_FRAME_PREFIX,
            PLACE_GRASP_FRAME_PREFIX,
            PLACE_SLOT_T_OBJECT,
        ),
    ):
        for x_index in range(grid.numx):
            for y_index in range(grid.numy):
                slot_frame = grid.get_slot_frame(x_index, y_index)
                object_frame = f"{object_prefix}_{x_index}_{y_index}"
                grasp_frame = f"{grasp_prefix}_{x_index}_{y_index}"
                tree.add(
                    slot_frame,
                    object_frame,
                    slot_T_object,
                    rot_type="deg",
                )
                tree.add(
                    object_frame,
                    grasp_frame,
                    OBJECT_T_GRASP,
                    rot_type="deg",
                )

    return tree, pick_grid, place_grid


def main() -> None:
    """Build and display the verified static cell frames.

    Returns:
        None.

    Raises:
        requests.RequestException: If an Isaac Sim bridge request fails.
        KeyError: If a bridge pose response is missing required data.
        TypeError: If a bridge pose response has an invalid structure.
        ValueError: If a pose, grid configuration, or TF frame is invalid.
    """
    bridge_request("GET", "/status")

    cnc_machine_pose_in_world = get_pose_in_world(CNC_PRIM_PATH)
    table_pose_in_world = get_pose_in_world(TABLE_FRAME_PRIM_PATH)
    tree, _, _ = build_static_frame_tree(
        cnc_machine_pose_in_world,
        table_pose_in_world,
    )

    rr.init("machine_tending_tf_tree", spawn=True)
    recording = rr.get_global_data_recording()
    tree.visualize_rerun(axis_len=TF_AXIS_LENGTH, recording_stream=recording)

    input(
        "Rerun is showing the cell, slot, expected-object, and grasp frames. "
        "Press Enter..."
    )


if __name__ == "__main__":
    main()
