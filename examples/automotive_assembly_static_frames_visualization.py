"""Build and visualize the automotive roof-assembly TF frames.

The three conveyors, robot mount, and roof rack are fixed scene anchors. The
application updates the single ``car`` frame from whichever sledge stops at
the lightbeam. The roof placement point remains fixed relative to that car::

    world
    |-- conveyor
    |   `-- car
    |       `-- place
    |-- conveyor_01
    |-- conveyor_02
    |-- robot_mount
    `-- roof_rack
        `-- roof_object
            |-- roof_pick
            `-- roof_pre_pick

The pick frames are offset onto a solid section of the roof. The application
derives its pre-place target by adding a vertical approach offset to ``place``;
it does not store another TF frame. This script only visualizes the calibrated
tree in Rerun and does not modify the Isaac Sim scene.
"""

from __future__ import annotations

import rerun as rr

from telekinesis.tf import tftree, tfutils


# Scene-anchor poses in World, expressed as XYZ metres and Euler XYZ degrees.
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
ROBOT_MOUNT_POSE_IN_WORLD = [
    5.336305992,
    -3.892470000,
    2.513500119,
    0.0,
    0.0,
    0.0,
]
ROOF_RACK_POSE_IN_WORLD = [
    3.637855992,
    -4.373540000,
    -0.004530000,
    0.0,
    0.0,
    90.000001961,
]

# Nominal stopped pose used by the visualizer. The application replaces this
# transform with the live stopped-sledge pose during every cycle.
CONVEYOR_T_CAR_AT_STATION = [
    1.259216674,
    -0.005409020,
    0.279222426,
    0.0,
    0.0,
    0.0,
]

# Final roof-centre pose relative to either identical sledge.
CAR_T_PLACE_OBJECT = [
    -0.578176659,
    0.000941442,
    1.703993820,
    -0.066333326,
    0.039804808,
    0.227867210,
]

# Roof centre on the rack and centre-to-top suction transform.
ROOF_RACK_T_ROOF_OBJECT = [
    0.030787767,
    0.002079144,
    2.387986331,
    -0.132022064,
    -0.047301765,
    0.047465704,
]
ROOF_HEIGHT = 0.159531421
# Keep the suction ray origin just above the roof collider. At 30 mm the
# contact frame sat inside the curved roof's convex collision shape; 75 mm
# leaves roughly 5 mm of clear ray distance at the calibrated pick point.
PICK_SUCTION_CLEARANCE = 0.075
PLACE_RELEASE_CLEARANCE = 0.080
# The roof bounds centre is over an opening. Move the suction ray along
# roof-local +X so it lands on the solid roof surface.
ROOF_PICK_X_OFFSET = 0.100
ROOF_OBJECT_T_PICK_GRASP = [
    ROOF_PICK_X_OFFSET,
    0.0,
    ROOF_HEIGHT / 2.0 + PICK_SUCTION_CLEARANCE,
    180.0,
    0.0,
    0.0,
]
ROOF_OBJECT_T_PLACE_GRASP = [
    0.0,
    0.0,
    ROOF_HEIGHT / 2.0 + PLACE_RELEASE_CLEARANCE,
    180.0,
    0.0,
    0.0,
]
PICK_PRE_GRASP_CLEARANCE = 0.70
PLACE_PRE_GRASP_CLEARANCE = 0.50
PLACE_PRE_GRASP_OFFSET = PLACE_PRE_GRASP_CLEARANCE - PLACE_RELEASE_CLEARANCE
ROOF_OBJECT_T_PICK_PRE_GRASP = [
    ROOF_PICK_X_OFFSET,
    0.0,
    ROOF_HEIGHT / 2.0 + PICK_PRE_GRASP_CLEARANCE,
    180.0,
    0.0,
    0.0,
]
TF_AXIS_LENGTH = 0.15


def build_static_frame_tree() -> tftree.TransformTree:
    """Build the calibrated automotive roof-assembly transform tree.

    Returns:
        Static transform tree containing the scene and task frames.

    Raises:
        ValueError: If a configured pose or frame relationship is invalid.
    """
    tree = tftree.TransformTree("world")

    tree.add("world", "conveyor", CONVEYOR_POSE_IN_WORLD, rot_type="deg")
    tree.add("world", "conveyor_01", CONVEYOR_01_POSE_IN_WORLD, rot_type="deg")
    tree.add("world", "conveyor_02", CONVEYOR_02_POSE_IN_WORLD, rot_type="deg")
    tree.add("world", "robot_mount", ROBOT_MOUNT_POSE_IN_WORLD, rot_type="deg")
    tree.add("world", "roof_rack", ROOF_RACK_POSE_IN_WORLD, rot_type="deg")

    tree.add(
        "conveyor",
        "car",
        CONVEYOR_T_CAR_AT_STATION,
        rot_type="deg",
    )
    car_T_place = tfutils.pose_to_transformation_matrix(
        CAR_T_PLACE_OBJECT,
        rot_type="deg",
    ) @ tfutils.pose_to_transformation_matrix(
        ROOF_OBJECT_T_PLACE_GRASP,
        rot_type="deg",
    )
    tree.add("car", "place", car_T_place)

    tree.add(
        "roof_rack",
        "roof_object",
        ROOF_RACK_T_ROOF_OBJECT,
        rot_type="deg",
    )
    tree.add(
        "roof_object",
        "roof_pick",
        ROOF_OBJECT_T_PICK_GRASP,
        rot_type="deg",
    )
    tree.add(
        "roof_object",
        "roof_pre_pick",
        ROOF_OBJECT_T_PICK_PRE_GRASP,
        rot_type="deg",
    )
    return tree


def main() -> None:
    """Visualize the automotive roof-assembly static TF tree in Rerun.

    Returns:
        None.

    Raises:
        ValueError: If a configured pose or frame relationship is invalid.
    """
    tree = build_static_frame_tree()

    rr.init("automotive_assembly_static_tf_tree", spawn=True)
    recording = rr.get_global_data_recording()
    tree.visualize_rerun(axis_len=TF_AXIS_LENGTH, recording_stream=recording)

    input("Rerun is showing the automotive assembly frames. Press Enter...")


if __name__ == "__main__":
    main()
