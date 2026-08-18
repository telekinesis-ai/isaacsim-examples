"""Build and visualize the static frame tree for the machine-tending cell.

Only the CNC machine and the logical table frame get global poses from Isaac
Sim. Every frame mounted inside them is stored relative to its physical parent:

    world
    |-- cnc_machine
    |   `-- cnc_pedestal
    `-- table
        |-- robot_mount
        |   `-- robot_base
        |-- pick_grid
        |   |-- pick_slot_0_0
        |   `-- ... pick_slot_3_3
        `-- place_grid
            |-- place_slot_0_0
            `-- ... place_slot_3_3

The bridge supplies only the global CNC and logical-table poses. The CNC
pedestal, robot mount, and grid origins are fixed calibrated child transforms.
Each grid origin is its first slot, so each ``slot_0_0`` has a zero transform
relative to its grid. This script is read-only.
"""

from __future__ import annotations

import math

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
    0.0,
]
ROBOT_MOUNT_T_ROBOT_BASE = [0.0, 0.0, -0.005, 0.0, 0.0, math.pi]
TABLE_T_PICK_GRID = [
    -0.249323199,
    0.541363956,
    -0.126516663,
    0.0,
    0.0,
    0.0,
]
PICK_GRID_COLUMN_STEP = [0.0, 0.178184612, 0.0]
PICK_GRID_ROW_STEP = [0.165020259, 0.000243775, 0.0]
TABLE_T_PLACE_GRID = [
    -0.255704683,
    0.284615655,
    0.075980730,
    0.0,
    0.0,
    0.0,
]
PLACE_GRID_COLUMN_STEP = [0.000083327, -0.181380000, 0.0]
PLACE_GRID_ROW_STEP = [0.161526667, -0.001556672, 0.0]
GRID_ROWS = 4
GRID_COLUMNS = 4
TF_AXIS_LENGTH = 0.05


def bridge_request(method: str, path: str, *, params=None):
    """Send a read-only request to the local Isaac Sim bridge."""
    response = requests.request(
        method,
        BASE_URL + path,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def world_pose(prim_path: str) -> list[float]:
    """Read a prim's world pose as metres plus rotation-vector radians."""
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
    """A reusable rectangular collection of TF slot frames."""

    def __init__(
        self,
        name: str,
        parent: str,
        slot_prefix: str,
        parent_T_grid: list[float],
        row_step: list[float],
        column_step: list[float],
        rows: int,
        columns: int,
    ) -> None:
        self.name = name
        self.parent = parent
        self.slot_prefix = slot_prefix
        self.parent_T_grid = parent_T_grid
        self.row_step = row_step
        self.column_step = column_step
        self.rows = rows
        self.columns = columns

    def slot_frame(self, row: int, column: int) -> str:
        """Return the TF frame name for one slot."""
        return f"{self.slot_prefix}_{row}_{column}"

    def add_to_tf(self, tree: tftree.TransformTree) -> None:
        """Add the grid origin and every generated slot to a TF tree."""
        tree.add(self.parent, self.name, self.parent_T_grid, rot_type="rad")

        for row in range(self.rows):
            for column in range(self.columns):
                slot_xyz = [
                    column * self.column_step[axis] + row * self.row_step[axis]
                    for axis in range(3)
                ]
                tree.add(
                    self.name,
                    self.slot_frame(row, column),
                    [*slot_xyz, 0.0, 0.0, 0.0],
                    rot_type="rad",
                )


def main() -> None:
    """Build and display the verified static cell frames."""
    bridge_request("GET", "/status")

    cnc_machine_world_pose = world_pose(CNC_PRIM_PATH)
    table_world_pose = world_pose(TABLE_FRAME_PRIM_PATH)

    tree = tftree.TransformTree("world")
    world_T_cnc_machine = tfutils.pose_to_transformation_matrix(
        cnc_machine_world_pose,
        rot_type="rotvec",
    )
    tree.add("world", "cnc_machine", world_T_cnc_machine, rot_type="mat")
    tree.add(
        "cnc_machine",
        "cnc_pedestal",
        CNC_MACHINE_T_PEDESTAL,
        rot_type="rad",
    )

    world_T_table = tfutils.pose_to_transformation_matrix(
        table_world_pose,
        rot_type="rotvec",
    )
    tree.add("world", "table", world_T_table, rot_type="mat")
    tree.add("table", "robot_mount", TABLE_T_ROBOT_MOUNT, rot_type="rad")

    tree.add(
        "robot_mount",
        "robot_base",
        ROBOT_MOUNT_T_ROBOT_BASE,
        rot_type="rad",
    )

    pick_grid = Grid(
        name="pick_grid",
        parent="table",
        slot_prefix="pick_slot",
        parent_T_grid=TABLE_T_PICK_GRID,
        row_step=PICK_GRID_ROW_STEP,
        column_step=PICK_GRID_COLUMN_STEP,
        rows=GRID_ROWS,
        columns=GRID_COLUMNS,
    )
    place_grid = Grid(
        name="place_grid",
        parent="table",
        slot_prefix="place_slot",
        parent_T_grid=TABLE_T_PLACE_GRID,
        row_step=PLACE_GRID_ROW_STEP,
        column_step=PLACE_GRID_COLUMN_STEP,
        rows=GRID_ROWS,
        columns=GRID_COLUMNS,
    )
    pick_grid.add_to_tf(tree)
    place_grid.add_to_tf(tree)

    rr.init("machine_tending_tf_tree", spawn=True)
    recording = rr.get_global_data_recording()
    tree.visualize_rerun(axis_len=TF_AXIS_LENGTH, recording_stream=recording)

    input("Rerun is showing the cell frames and both 16-slot grids. Press Enter...")


if __name__ == "__main__":
    main()
