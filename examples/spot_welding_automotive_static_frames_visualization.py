"""Build and visualize the automotive spot-welding TF frames.

The conveyors and robot mount are fixed scene anchors. The application will
update ``car`` from the sledge that stops at the lightbeam. The weld target is
fixed relative to that car::

    world
    |-- conveyor
    |   `-- car
    |       |-- pre_weld
    |       `-- weld_point
    |           `-- weld_tcp
    |-- conveyor_01
    |-- conveyor_02
    |-- conveyor_03
    `-- robot_mount
        `-- robot_base

``weld_point`` and ``weld_tcp`` currently coincide because the calibrated
car-to-weld transform was measured from the welding gun's physical
``contact_frame``. They remain separate so the physical task point and the
robot's active TCP have clear names in the application.

This script only visualizes the expected frames in Rerun. It does not modify
the Isaac Sim stage.
"""

from __future__ import annotations

import rerun as rr

from telekinesis.tf import tftree


# Fixed scene anchors in World: XYZ metres followed by Euler XYZ degrees.
CONVEYOR_POSE_IN_WORLD = [
    3.390086285,
    -2.178826791,
    0.878248396,
    0.0,
    0.0,
    0.0,
]
CONVEYOR_01_POSE_IN_WORLD = [
    -2.715278261,
    -2.181604811,
    0.878248191,
    0.0,
    0.0,
    180.0,
]
CONVEYOR_02_POSE_IN_WORLD = [
    9.584041870,
    -2.185192783,
    0.878249186,
    0.0,
    0.0,
    0.0,
]
CONVEYOR_03_POSE_IN_WORLD = [
    12.684274283,
    -2.185161104,
    0.878249186,
    0.0,
    0.0,
    0.0,
]
ROBOT_MOUNT_POSE_IN_WORLD = [
    5.336305992,
    -3.892470000,
    2.513500119,
    0.0,
    0.0,
    0.0,
]
ROBOT_MOUNT_T_ROBOT_BASE = [0.0, 0.0, 0.0, 0.0, 0.0, 90.0]

# Nominal stopped-car pose used only by this visualizer. The application will
# replace it with the pose measured after each lightbeam stop.
CONVEYOR_T_CAR_AT_STATION = [
    1.259216674,
    -0.005409020,
    0.279222426,
    0.0,
    0.0,
    0.0,
]

# Calibrated from /World/sledge to the welding gun's contact frame while the
# KUKA was manually positioned at the desired weld. This includes the exact
# taught TCP orientation; the application does not replace that rotation.
CAR_T_PRE_WELD = [
    2.582861231,
    0.178137887,
    0.938832835,
    155.433913960,
    -42.889793043,
    78.408749350,
]
CAR_T_WELD_POINT = [
    2.175266502,
    0.255436651,
    0.981135023,
    155.433913960,
    -42.889793043,
    78.408749350,
]

# The measured weld point already has the desired TCP position and rotation.
WELD_POINT_T_WELD_TCP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Fixed KUKA tool calibration used later by robot.add_tcp(). It is not added as
# another tree edge because tool0 moves with the robot during the application.
KUKA_TOOL0_T_WELD_TCP = [
    0.194476237,
    0.256590121,
    -0.603499164,
    90.000016418,
    -66.000008326,
    89.999968149,
]

TF_AXIS_LENGTH = 0.20


def build_static_frame_tree(
    conveyor_T_car: list[float] | None = None,
    robot_mount_T_robot_base: list[float] | None = None,
) -> tftree.TransformTree:
    """Build the calibrated automotive spot-welding transform tree."""
    tree = tftree.TransformTree("world")

    tree.add("world", "conveyor", CONVEYOR_POSE_IN_WORLD, rot_type="deg")
    tree.add("world", "conveyor_01", CONVEYOR_01_POSE_IN_WORLD, rot_type="deg")
    tree.add("world", "conveyor_02", CONVEYOR_02_POSE_IN_WORLD, rot_type="deg")
    tree.add("world", "conveyor_03", CONVEYOR_03_POSE_IN_WORLD, rot_type="deg")
    tree.add("world", "robot_mount", ROBOT_MOUNT_POSE_IN_WORLD, rot_type="deg")
    tree.add(
        "robot_mount",
        "robot_base",
        robot_mount_T_robot_base or ROBOT_MOUNT_T_ROBOT_BASE,
        rot_type="deg",
    )

    tree.add(
        "conveyor",
        "car",
        conveyor_T_car or CONVEYOR_T_CAR_AT_STATION,
        rot_type="deg",
    )
    tree.add("car", "pre_weld", CAR_T_PRE_WELD, rot_type="deg")
    tree.add("car", "weld_point", CAR_T_WELD_POINT, rot_type="deg")
    tree.add(
        "weld_point",
        "weld_tcp",
        WELD_POINT_T_WELD_TCP,
        rot_type="deg",
    )
    return tree


def main() -> None:
    """Visualize the calibrated spot-welding frame tree in Rerun."""
    tree = build_static_frame_tree()

    rr.init("spot_welding_automotive_static_tf_tree", spawn=True)
    recording = rr.get_global_data_recording()
    tree.visualize_rerun(axis_len=TF_AXIS_LENGTH, recording_stream=recording)

    input("Rerun is showing the spot-welding frames. Press Enter...")


if __name__ == "__main__":
    main()
