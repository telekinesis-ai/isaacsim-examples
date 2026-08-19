"""Set up the robot and suction gripper for the palletizing application.

This builds the existing static TF tree, places the selected robot at its
``robot_base`` frame, attaches the simulated suction gripper, runs all conveyors
until the lightbeam detects each box, picks it using the dynamic TF grasp frame,
and carries it with the TCP facing down to the next TF pallet frame. After each
placement it retracts, returns empty to the home L pose, and restarts the
conveyors.

Before running:
    1. Open the palletizing scene in Isaac Sim.
    2. Enable the Telekinesis Isaac Sim bridge extension.
    3. Import the robot at the configured prim path.
    4. Run::

           python palletizing_application_multi_robot.py --ur10e

This is an external Python application; it does not import ``omni`` or
``isaacsim``.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass

import requests

from telekinesis.tf import tftree, tfutils
from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.suction_grippers import custom

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
ISAAC_SIM_CODE_HOST = "127.0.0.1"
ISAAC_SIM_CODE_PORT = 8226
CONVEYOR_WAIT_TIMEOUT_SECONDS = 120.0

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
LIGHTBEAM_MAX_FRAMES = 3600
DETECTED_BOX_RESULT_PREFIX = "__PALLETIZING_DETECTED_BOX__="

# These prims must already exist in the open Isaac Sim scene. The application
# does not import the robot or gripper assets.
UR10E_PRIM_PATH = "/World/ur10e_robot"
SUCTION_GRIPPER_PRIM_PATH = "/World/defitech_modelled_surface_gripper_modelled"
# Centre of the visible suction-pad face relative to the UR10e tool0 frame.
SUCTION_TCP_OFFSET = [0.0, 0.0, 0.075, 0.0, 0.0, 0.0]
# Suction ray origin relative to the Defitech gripper root. The USD currently
# authors this at +0.26088396 m, far above the visible pad.
SUCTION_ATTACHMENT_POINT_TRANSLATION = [0.0, 0.0, -0.073]

# Defitech gripper mounting orientation relative to the UR10e tool frame.
UR10E_T_SUCTION_GRIPPER = [0.0, 0.0, 0.0, 180.0, 0.0, 0.0]


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
    home_joints: list[float]
    joint_speed: float
    joint_acceleration: float
    cartesian_speed: float
    cartesian_acceleration: float
    contact_speed: float
    carry_speed: float
    drive_stiffness: float | None
    drive_damping: float | None


@dataclass(frozen=True)
class DetectedBox:
    """Pose and dimensions of the box stopped at the lightbeam.

    Attributes:
        prim_path: USD path of the detected box.
        conveyor_T_object: Logical box-centre pose relative to ``conveyor``,
            using XYZ metres and Euler XYZ degrees.
        size: Axis-aligned world dimensions in metres.
    """

    prim_path: str
    conveyor_T_object: list[float]
    size: list[float]


ROBOT_CONFIGS = {
    "ur10e": RobotConfig(
        robot_class=universal_robots.UniversalRobotsUR10E,
        prim_path=UR10E_PRIM_PATH,
        mount_T_robot_base=[0.0, 0.0, 0.0, 0.0, 0.0, 180.0],
        tool_mount_transform=UR10E_T_SUCTION_GRIPPER,
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


def execute_in_isaac_sim(code: str, timeout_seconds: float) -> str:
    """Execute scene-side Python in the currently open Isaac Sim process.

    This is the application's only direct dependency on Isaac Sim's Python
    execution socket. Scene-side operations unsupported by the bridge, such as
    conveyor control and lightbeam reads, use it here.

    Args:
        code: Python source to execute inside Isaac Sim.
        timeout_seconds: Maximum time to wait for the operation to finish.

    Returns:
        Text printed by the scene-side operation.

    Raises:
        RuntimeError: If Isaac Sim cannot be reached or rejects the operation.
        ValueError: If Isaac Sim returns an invalid response.
    """
    try:
        connection = socket.create_connection(
            (ISAAC_SIM_CODE_HOST, ISAAC_SIM_CODE_PORT),
            timeout=5.0,
        )
    except OSError as error:
        raise RuntimeError(
            "Could not reach Isaac Sim's Python code socket on port 8226. "
            "Enable the isaacsim.code_editor.vscode extension."
        ) from error

    with connection:
        connection.settimeout(timeout_seconds)
        connection.sendall(code.encode("utf-8"))
        response_bytes = bytearray()
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            response_bytes.extend(chunk)

    try:
        response = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Isaac Sim returned an invalid code-socket response."
        ) from error

    if response.get("status") != "ok":
        traceback_text = "\n".join(response.get("traceback", []))
        detail = traceback_text or response.get("evalue", "Unknown Isaac Sim error")
        raise RuntimeError(f"Isaac Sim scene-side operation failed:\n{detail}")
    return response.get("output", "")


def run_conveyors_until_lightbeam() -> DetectedBox:
    """Run all conveyors and measure the box stopped at the lightbeam.

    All configured conveyor surface velocities start together and are all set
    to zero when the lightbeam detects a box or if the operation fails. The
    stopped box nearest the sensor is measured after a short settling period.

    Returns:
        Detected box path, conveyor-relative centre pose, and dimensions.

    Raises:
        RuntimeError: If Isaac Sim cannot be reached, the scene prims are
            missing, no surface velocity exists, or detection times out.
    """
    isaac_code = f"""\
import json
import math

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.sensors.physx import _range_sensor
from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics

CONVEYOR_PATHS = {CONVEYOR_PRIM_PATHS!r}
LIGHTBEAM_PATH = {LIGHTBEAM_PRIM_PATH!r}
SCENE_ROOT = {PALLETIZING_SCENE_ROOT!r}
BOX_PREFIX = {BOX_PRIM_NAME_PREFIX!r}
RESULT_PREFIX = {DETECTED_BOX_RESULT_PREFIX!r}
RUN_VELOCITY = Gf.Vec3f(
    {CONVEYOR_RUN_VELOCITY[0]},
    {CONVEYOR_RUN_VELOCITY[1]},
    {CONVEYOR_RUN_VELOCITY[2]},
)
MAX_FRAMES = {LIGHTBEAM_MAX_FRAMES}
VELOCITY_ATTRIBUTE = "physxSurfaceVelocity:surfaceVelocity"


def rigid_transform(prim):
    source = Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    result = Gf.Transform()
    result.SetTranslation(source.GetTranslation())
    result.SetRotation(source.GetRotation())
    return result


def world_bounds(prim):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if aligned.IsEmpty():
        raise RuntimeError("No bounds available for " + str(prim.GetPath()))
    return aligned.GetMin(), aligned.GetMax()


def rotation_to_rpy(rotation):
    quaternion = rotation.GetQuat()
    w = float(quaternion.GetReal())
    x, y, z = [float(value) for value in quaternion.GetImaginary()]
    length = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = [value / length for value in (w, x, y, z)]
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def relative_pose(source_world, child_world):
    position = source_world.GetMatrix().GetInverse().Transform(
        child_world.GetTranslation()
    )
    rotation = source_world.GetRotation().GetInverse() * child_world.GetRotation()
    return [
        float(position[0]),
        float(position[1]),
        float(position[2]),
        *rotation_to_rpy(rotation),
    ]

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No stage is open in Isaac Sim.")

lightbeam = stage.GetPrimAtPath(LIGHTBEAM_PATH)
scene = stage.GetPrimAtPath(SCENE_ROOT)
if not lightbeam.IsValid():
    raise RuntimeError("Lightbeam not found: " + LIGHTBEAM_PATH)
if not scene.IsValid():
    raise RuntimeError("Palletizing scene root not found: " + SCENE_ROOT)

velocity_attributes = []
for conveyor_path in CONVEYOR_PATHS:
    conveyor = stage.GetPrimAtPath(conveyor_path)
    if not conveyor.IsValid():
        raise RuntimeError("Conveyor not found: " + conveyor_path)

    velocity_attribute = None
    for prim in Usd.PrimRange(conveyor):
        candidate = prim.GetAttribute(VELOCITY_ATTRIBUTE)
        if candidate.IsValid() and candidate.Get() is not None:
            velocity_attribute = candidate
            break
    if velocity_attribute is None:
        raise RuntimeError("No surface velocity found below " + conveyor_path)
    velocity_attributes.append(velocity_attribute)

boxes = [
    prim
    for prim in Usd.PrimRange(scene)
    if prim.GetName().startswith(BOX_PREFIX)
    and prim.HasAPI(UsdPhysics.RigidBodyAPI)
]
if not boxes:
    raise RuntimeError("No dynamic boxes beginning with " + BOX_PREFIX + " were found.")

timeline = omni.timeline.get_timeline_interface()
app = omni.kit.app.get_app()
sensor = _range_sensor.acquire_lightbeam_sensor_interface()
timeline.play()
for velocity_attribute in velocity_attributes:
    velocity_attribute.Set(RUN_VELOCITY)

# Boxes can fall asleep while the robot completes the previous cycle. Surface
# velocity alone does not reliably wake a sleeping rigid body.
physics_simulation = omni.physx.get_physx_simulation_interface()
stage_id = omni.usd.get_context().get_stage_id()
for box in boxes:
    body_path = PhysicsSchemaTools.sdfPathToInt(box.GetPath())
    physics_simulation.wake_up(stage_id, body_path)

clear_frames = 0
hit_frames = 0
detected_box = None
try:
    for frame in range(MAX_FRAMES):
        await app.next_update_async()
        hit_data = sensor.get_beam_hit_data(LIGHTBEAM_PATH)
        triggered = bool(
            hit_data is not None and np.asarray(hit_data, dtype=bool).any()
        )

        if frame < 5:
            continue
        if not triggered:
            clear_frames += 1
            hit_frames = 0
            continue
        if clear_frames < 3:
            continue

        hit_frames += 1
        if hit_frames >= 2:
            for velocity_attribute in velocity_attributes:
                velocity_attribute.Set(Gf.Vec3f(0.0, 0.0, 0.0))

            sensor_position = rigid_transform(lightbeam).GetTranslation()
            best_distance = math.inf
            for box in boxes:
                minimum, maximum = world_bounds(box)
                center = (minimum + maximum) * 0.5
                distance = sum(
                    float(center[axis] - sensor_position[axis]) ** 2
                    for axis in range(3)
                )
                if distance < best_distance:
                    best_distance = distance
                    detected_box = box

            break
    else:
        raise RuntimeError("Lightbeam was not triggered before timeout.")
finally:
    for velocity_attribute in velocity_attributes:
        velocity_attribute.Set(Gf.Vec3f(0.0, 0.0, 0.0))

if detected_box is None:
    raise RuntimeError("The lightbeam triggered, but no stopped box was identified.")

for _ in range(60):
    await app.next_update_async()

conveyor_world = rigid_transform(stage.GetPrimAtPath(CONVEYOR_PATHS[0]))
box_root_world = rigid_transform(detected_box)
minimum, maximum = world_bounds(detected_box)
center = (minimum + maximum) * 0.5
size = maximum - minimum

object_world = Gf.Transform()
object_world.SetTranslation(center)
object_world.SetRotation(box_root_world.GetRotation())

result = {{
    "prim_path": str(detected_box.GetPath()),
    "conveyor_T_object": relative_pose(conveyor_world, object_world),
    "size": [float(size[0]), float(size[1]), float(size[2])],
}}
print(RESULT_PREFIX + json.dumps(result))
"""

    print("Waiting for a box at the lightbeam...")
    output = execute_in_isaac_sim(isaac_code, CONVEYOR_WAIT_TIMEOUT_SECONDS)
    result_payload = None
    for line in output.splitlines():
        if line.startswith(DETECTED_BOX_RESULT_PREFIX):
            result_payload = json.loads(line.removeprefix(DETECTED_BOX_RESULT_PREFIX))

    if result_payload is None:
        raise RuntimeError("Isaac Sim did not return the detected box measurement.")
    return DetectedBox(
        prim_path=str(result_payload["prim_path"]),
        conveyor_T_object=[
            float(value) for value in result_payload["conveyor_T_object"]
        ],
        size=[float(value) for value in result_payload["size"]],
    )


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
    robot: universal_robots.UniversalRobotsUR10E,
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
    parser.add_argument(
        "--ur10e",
        dest="robot_name",
        action="store_const",
        const="ur10e",
        help="Use the Universal Robots UR10e (default).",
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
                transform=SUCTION_TCP_OFFSET,
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

            for box_index in range(BOX_COUNT):
                z_index, layer_index = divmod(
                    box_index,
                    PLACE_NUMX * PLACE_NUMY,
                )
                x_index, y_index = divmod(layer_index, PLACE_NUMY)
                frame_suffix = f"{x_index}_{y_index}_{z_index}"
                print(f"\n--- box {box_index + 1}/{BOX_COUNT} ---")

                detected_box = run_conveyors_until_lightbeam()
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
