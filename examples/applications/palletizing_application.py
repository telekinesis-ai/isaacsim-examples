"""Palletizing application for a UR10e with a suction gripper.

The cycle is packaged as :class:`PalletizingApplication`: ``setup`` prepares the
cell and connects the hardware, ``run`` palletizes the boxes, ``shutdown``
releases the hardware, and ``execute`` runs the three in order. The application
builds the cell's static TF tree, attaches the suction gripper, picks each box
from the conveyor using the TF grasp frame, and carries it with the TCP facing
down to the next TF pallet frame.

Without a robot IP the application runs in Isaac Sim: the scene is fetched from
the asset server and opened, the robot is imported from its URDF and placed at
the ``robot_base`` frame, the suction gripper loads itself from its own USD
asset when it connects, and the scene's conveyors run until the lightbeam
detects each box, so every pick uses that box's measured stopped pose. Once a
box is lifted clear, the conveyors move the next box to the lightbeam while the
robot completes the current placement and returns home.

With a robot IP the same cycle runs on hardware. The Isaac Sim scene setup and
conveyor control are skipped, the measured cell poses have to be supplied, and
every box is picked from the calibrated pick frame.

Before running in simulation:
    1. Enable the Telekinesis Isaac Sim bridge extension.
    2. Run::

           python palletizing_application.py

This is an external Python application; it does not import ``omni`` or
``isaacsim``.
"""

from __future__ import annotations

import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from telekinesis import datatypes, isaacsim_client
from telekinesis.tf import tftree, tfutils
from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.suction_grippers import abstract_gripper, custom


BASE_URL = "http://127.0.0.1:8766"
WEBSOCKET_BASE_URL = "ws://127.0.0.1:8766"
REQUEST_TIMEOUT_SECONDS = 30.0

SCENE_URL = (
    "https://assets.telekinesis.ai/usd/environments/palletizing/palletizing_rough_scene.zip"
)

BOX_COUNT = 8

# Isaac Sim scene elements the bridge does not expose yet. These are driven
# through the temporary code socket in PalletizingIsaacSimBackend below.
ISAAC_SIM_CODE_HOST = "127.0.0.1"
ISAAC_SIM_CODE_PORT = 8226
CONVEYOR_WAIT_TIMEOUT_SECONDS = 120.0
LIGHTBEAM_MAX_FRAMES = 3600
DETECTED_BOX_RESULT_PREFIX = "__PALLETIZING_DETECTED_BOX__="

LIGHTBEAM_PRIM_PATH = "/World/LightBeam_Sensor"
PALLETIZING_SCENE_ROOT = "/World"
BOX_PRIM_NAME_PREFIX = "Cardbox_C2"
CONVEYOR_PRIM_PATH = "/World/ConveyorBelt_A08"
CONVEYOR_PRIM_PATHS = [
    CONVEYOR_PRIM_PATH,
    "/World/ConveyorBelt_A11",
    "/World/ConveyorBelt_A08_01",
]
CONVEYOR_RUN_VELOCITY = [-0.8, 0.0, 0.0]

# Cell prims that receive world poses from Isaac Sim.
PALLET_PRIM_PATH = "/World/pallet"
ROBOT_MOUNT_PRIM_PATH = "/World/ur10_mount"
ROBOT_MOUNT_T_ROBOT_BASE = [0.0, 0.0, 0.0, 0.0, 0.0, 180.0]

# Static frame layout. The nominal pick target is defined relative to the
# conveyor; in simulation the application updates it from each box's stopped
# pose at runtime. The eight place targets are defined relative to the pallet as
# two centred 2-by-2 layers. Object frames represent box centres. Grasp frames
# are at the top surfaces and have their Z axes facing downward.
BOX_SIZE = [0.513243397, 0.331865479, 0.259689436]
CONVEYOR_T_PICK_OBJECT = [
    -2.205667546,
    0.119918064,
    0.898990207,
    0.0,
    0.0,
    90.0,
]
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
PLACE_NUMX = 2
PLACE_NUMY = 2
PLACE_NUMZ = 2

# Prim paths the application loads its own assets at: the robot from its URDF and
# the gripper from its USD asset.
ROBOT_PRIM_PATH = "/World/ur10e_robot"
SUCTION_GRIPPER_PRIM_PATH = "/World/defitech_modelled_surface_gripper_modelled"

# Defitech gripper mounting orientation relative to the UR10e tool flange.
FLANGE_TOOL_TRANSFORM = [0.0, 0.0, 0.0, 180.0, 0.0, 0.0]
# Centre of the visible suction-pad face relative to the robot tool flange.
FLANGE_TCP_TRANSFORM = [0.0, 0.0, 0.075, 0.0, 0.0, 0.0]


## ================== Isaac Sim scene backend ================== ##


class PalletizingIsaacSimBackend:
    """Control palletizing scene elements not yet exposed by the bridge.

    This isolates the temporary code-socket implementation used for conveyor
    control, lightbeam sensing, and live stopped-box measurement. These
    operations can later move behind the Isaac Sim bridge without changing the
    robot workflow.

    Attributes:
        conveyor_paths: Conveyor roots whose surface velocities run together.
        lightbeam_path: Isaac Sim path of the box-detection lightbeam.
        scene_root: Root containing the dynamic palletizing boxes.
        box_prim_name_prefix: Prim-name prefix used to identify boxes.
        conveyor_run_velocity: Running surface velocity in metres per second.
    """

    def __init__(
        self,
        *,
        conveyor_paths: list[str],
        lightbeam_path: str,
        scene_root: str,
        box_prim_name_prefix: str,
        conveyor_run_velocity: list[float],
    ) -> None:
        """Store the calibrated palletizing scene configuration.

        Args:
            conveyor_paths: Conveyor roots whose surface velocities run together.
            lightbeam_path: Isaac Sim path of the box-detection lightbeam.
            scene_root: Root containing the dynamic palletizing boxes.
            box_prim_name_prefix: Prim-name prefix used to identify boxes.
            conveyor_run_velocity: Running XYZ surface velocity in metres per
                second.

        Returns:
            None.

        Raises:
            ValueError: If the conveyor velocity does not contain three values.
        """
        if len(conveyor_run_velocity) != 3:
            raise ValueError("Conveyor run velocity must contain XYZ values.")

        self.conveyor_paths = conveyor_paths
        self.lightbeam_path = lightbeam_path
        self.scene_root = scene_root
        self.box_prim_name_prefix = box_prim_name_prefix
        self.conveyor_run_velocity = conveyor_run_velocity

    @staticmethod
    def _execute(code: str, timeout_seconds: float) -> str:
        """Execute scene-side Python in the currently open Isaac Sim process.

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
            # Isaac Sim's python_server buffers the payload and only executes it
            # once the client half-closes, so the write side must be shut down
            # before waiting for the reply.
            connection.shutdown(socket.SHUT_WR)
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

    def run_conveyors_until_lightbeam(self) -> list[float]:
        """Run all conveyors and measure the box stopped at the lightbeam.

        All configured conveyor surface velocities start together and are all
        set to zero when the lightbeam detects a box or if the operation fails.
        The stopped box nearest the sensor is measured after a short settling
        period.

        Returns:
            Logical box-centre pose relative to the conveyor, using XYZ metres
            and Euler XYZ degrees.

        Raises:
            RuntimeError: If Isaac Sim cannot be reached, the scene prims are
                missing, no surface velocity exists, or detection times out.
            ValueError: If Isaac Sim returns an invalid response.
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

CONVEYOR_PATHS = {self.conveyor_paths!r}
LIGHTBEAM_PATH = {self.lightbeam_path!r}
SCENE_ROOT = {self.scene_root!r}
BOX_PREFIX = {self.box_prim_name_prefix!r}
RESULT_PREFIX = {DETECTED_BOX_RESULT_PREFIX!r}
RUN_VELOCITY = Gf.Vec3f(
    {self.conveyor_run_velocity[0]},
    {self.conveyor_run_velocity[1]},
    {self.conveyor_run_velocity[2]},
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

object_world = Gf.Transform()
object_world.SetTranslation(center)
object_world.SetRotation(box_root_world.GetRotation())

result = {{
    "prim_path": str(detected_box.GetPath()),
    "conveyor_T_object": relative_pose(conveyor_world, object_world),
}}
print(RESULT_PREFIX + json.dumps(result))
"""

        print("Waiting for a box at the lightbeam...")
        output = self._execute(isaac_code, CONVEYOR_WAIT_TIMEOUT_SECONDS)
        result_payload = None
        for line in output.splitlines():
            if line.startswith(DETECTED_BOX_RESULT_PREFIX):
                result_payload = json.loads(
                    line.removeprefix(DETECTED_BOX_RESULT_PREFIX)
                )

        if result_payload is None:
            raise RuntimeError("Isaac Sim did not return the detected box measurement.")
        print(f"Detected {str(result_payload['prim_path']).rsplit('/', 1)[-1]}")
        return [float(value) for value in result_payload["conveyor_T_object"]]


## ================== Bridge access ================== ##


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
    response = requests.get(
        BASE_URL + "/prims/poses",
        params={
            "prim_path": prim_path,
            "coordinate_system": "world",
            "rotation_type": "cartesian",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return [float(value) for value in response.json()["pose"]]


## ================== Static cell frames ================== ##


def build_static_frame_tree(
    conveyor_pose_in_world: list[float],
    pallet_pose_in_world: list[float],
    robot_mount_pose_in_world: list[float],
    box_size: list[float],
) -> tftree.TransformTree:
    """Build the palletizing cell's static TF tree.

    The box height positions every grasp frame on a box's top surface and sets
    the vertical spacing between pallet layers, so the same tree serves any box
    the cell is set up for.

    Args:
        conveyor_pose_in_world: Conveyor pose in world, using XYZ metres and a
            rotation vector in radians.
        pallet_pose_in_world: Pallet pose in world, using XYZ metres and a
            rotation vector in radians.
        robot_mount_pose_in_world: Robot-mount pose in world, using XYZ metres
            and a rotation vector in radians.
        box_size: Box dimensions in metres.

    Returns:
        Static palletizing transform tree.

    Raises:
        ValueError: If a supplied pose or TF frame is invalid.
    """
    object_T_grasp = [0.0, 0.0, box_size[2] / 2.0, 180.0, 0.0, 0.0]
    place_zstep = [0.0, 0.0, box_size[2]]

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
        object_T_grasp,
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
                    + z_index * place_zstep[axis]
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
                    object_T_grasp,
                    rot_type="deg",
                )
                grasp_T_pre_place = PLACE_GRASP_T_PRE_PLACE.copy()
                # The grasp frame points down. Shorten its negative-Z offset by
                # the stack height so every layer has one common travel height.
                grasp_T_pre_place[2] += z_index * place_zstep[2]
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


## ================== Application ================== ##


class PalletizingApplication:
    """Palletize conveyor boxes with one robot and one suction gripper.

    The robot and the gripper are supplied by the caller, already created but
    not yet connected. Leaving ``robot_ip`` unset runs the cell in Isaac Sim:
    the scene is opened, the robot is imported and placed on its mount, and the
    scene's conveyors deliver each box to the lightbeam, where its stopped pose
    is measured. Giving a ``robot_ip`` connects to hardware instead, in which
    case the cell poses have to be supplied and every box is picked from the
    calibrated pick frame.

    Typical use::

        application = PalletizingApplication(robot, gripper)
        application.execute()

    :meth:`execute` runs setup, the palletizing cycles, and shutdown. The three
    are also public so a caller that wants the cycles under its own supervision
    can drive them separately.

    Attributes:
        tree: Palletizing cell transform tree, built by :meth:`setup`.
    """

    # Clear home pose in joint degrees, and the motion limits the cycle runs at.
    HOME_JOINTS = [0.0, -90.0, -60.0, -120.0, 90.0, 0.0]
    JOINT_SPEED = 45.0
    JOINT_ACCELERATION = 60.0
    CARTESIAN_SPEED = 0.30
    CARTESIAN_ACCELERATION = 0.50
    # Slower speeds for approaching a box and for carrying one.
    CONTACT_SPEED = 0.08
    CARRY_SPEED = 0.15
    # The Telekinesis URDF import needs explicit gains to hold its pose.
    DRIVE_STIFFNESS = 1.0e5
    DRIVE_DAMPING = 1.0e4

    def __init__(
        self,
        robot: universal_robots.UniversalRobotsUR10E,
        gripper: abstract_gripper.AbstractSuctionGripper,
        *,
        box_size: list[float] = BOX_SIZE,
        flange_tool_transform: list[float] = FLANGE_TOOL_TRANSFORM,
        flange_tcp_transform: list[float] = FLANGE_TCP_TRANSFORM,
        box_count: int = BOX_COUNT,
        robot_ip: str | None = None,
        gripper_ip: str | None = None,
        robot_prim_path: str = ROBOT_PRIM_PATH,
        gripper_prim_path: str = SUCTION_GRIPPER_PRIM_PATH,
        scene_url: str = SCENE_URL,
        conveyor_pose_in_world: list[float] | None = None,
        pallet_pose_in_world: list[float] | None = None,
        robot_mount_pose_in_world: list[float] | None = None,
    ) -> None:
        """Store the hardware and the palletizing cell configuration.

        Args:
            robot: Robot to palletize with, not yet connected.
            gripper: Suction gripper, not yet connected.
            box_size: Box dimensions in metres. Sets the height of every grasp
                frame and the spacing between pallet layers.
            flange_tool_transform: Gripper root relative to the robot tool
                flange, using XYZ metres and Euler XYZ degrees.
            flange_tcp_transform: Suction TCP relative to the robot tool flange,
                using XYZ metres and Euler XYZ degrees.
            box_count: Number of boxes to palletize.
            robot_ip: Robot controller address. Leave as ``None`` to run the
                cell in Isaac Sim.
            gripper_ip: Suction-gripper address, used with ``robot_ip``.
            robot_prim_path: USD path the robot is imported at in simulation.
            gripper_prim_path: USD path the gripper is loaded at in simulation.
            scene_url: Asset-server URL of the palletizing scene bundle.
            conveyor_pose_in_world: Measured conveyor pose, using XYZ metres and
                a rotation vector in radians. Read from Isaac Sim when ``None``.
            pallet_pose_in_world: Measured pallet pose, in the same format.
                Read from Isaac Sim when ``None``.
            robot_mount_pose_in_world: Measured robot-mount pose, in the same
                format. Read from Isaac Sim when ``None``.

        Returns:
            None.
        """
        self.robot = robot
        self.gripper = gripper

        self.box_size = box_size
        self.flange_tool_transform = flange_tool_transform
        self.flange_tcp_transform = flange_tcp_transform
        self.box_count = box_count

        self.robot_ip = robot_ip
        self.gripper_ip = gripper_ip
        self.robot_prim_path = robot_prim_path
        self.gripper_prim_path = gripper_prim_path
        self.scene_url = scene_url

        self.conveyor_pose_in_world = conveyor_pose_in_world
        self.pallet_pose_in_world = pallet_pose_in_world
        self.robot_mount_pose_in_world = robot_mount_pose_in_world

        # Built by setup(): the cell frames, and in simulation the Isaac Sim
        # client and the conveyor backend.
        self.tree: tftree.TransformTree | None = None
        self.client: isaacsim_client.IsaacSimClient | None = None
        self.scene_backend: PalletizingIsaacSimBackend | None = None

    @property
    def is_simulated(self) -> bool:
        """Whether the cell runs in Isaac Sim rather than on real hardware."""
        return self.robot_ip is None

    # ------------------------------------------------------------------ #
    # Setup / teardown
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """Prepare the cell, build its frames, and connect the hardware.

        In simulation the scene bundle is downloaded once and cached, then
        opened in place of whatever stage is open. It holds the palletizing cell
        only, so the robot is imported from the URDF its Synapse class fetched,
        and the gripper is loaded from its own USD asset when it connects. The
        scene has to be open before the cell poses can be read from it.

        Returns:
            None.

        Raises:
            requests.RequestException: If an Isaac Sim bridge request fails.
            RuntimeError: If the robot or the suction gripper cannot connect or
                attach, or if the robot articulation cannot be read back.
            ValueError: If a configured transform is invalid, or a cell pose is
                missing while running on hardware.
        """
        if self.is_simulated:
            self.client = isaacsim_client.IsaacSimClient(
                api_key="",
                base_url=BASE_URL,
                websocket_base_url=WEBSOCKET_BASE_URL,
            )
            self.scene_backend = PalletizingIsaacSimBackend(
                conveyor_paths=CONVEYOR_PRIM_PATHS,
                lightbeam_path=LIGHTBEAM_PRIM_PATH,
                scene_root=PALLETIZING_SCENE_ROOT,
                box_prim_name_prefix=BOX_PRIM_NAME_PREFIX,
                conveyor_run_velocity=CONVEYOR_RUN_VELOCITY,
            )
            bridge_request("GET", "/status")

            scene = datatypes.USD.from_url(self.scene_url)
            print(f"Opening the palletizing scene: {scene.path}")
            self.client.stage.open_scene(scene.path.as_posix())
            print(f"Importing the robot at {self.robot_prim_path}...")
            self.client.articulation.create(
                self.robot_prim_path,
                str(self.robot.urdf_path),
            )

        cell_poses = []
        for pose_in_world, prim_path in (
            (self.conveyor_pose_in_world, CONVEYOR_PRIM_PATH),
            (self.pallet_pose_in_world, PALLET_PRIM_PATH),
            (self.robot_mount_pose_in_world, ROBOT_MOUNT_PRIM_PATH),
        ):
            if pose_in_world is None and not self.is_simulated:
                raise ValueError(
                    "Running on hardware requires the measured conveyor, pallet "
                    "and robot-mount poses."
                )
            cell_poses.append(
                pose_in_world
                if pose_in_world is not None
                else get_pose_in_world(prim_path)
            )
        self.tree = build_static_frame_tree(*cell_poses, self.box_size)

        print("Connecting the robot and suction gripper...")
        if self.is_simulated:
            self.place_robot_on_mount()
            # SuctionGripper.connect() requires the timeline to be playing.
            bridge_request("PATCH", "/stage/simulation/timeline/play")
            time.sleep(0.5)

            self.robot.connect(simulation_prim_path=self.robot_prim_path)
            self.gripper.connect(simulation_prim_path=self.gripper_prim_path)
            # The gripper USD authors the suction ray origin at +0.26088396 m,
            # far above the visible pad. Put it back on the pad face.
            self.gripper.set_attachment_point_properties(
                local_pose_0={"translation": [0.0, 0.0, -0.073]},
            )
        else:
            self.robot.connect(ip=self.robot_ip)
            self.gripper.connect(ip=self.gripper_ip)

        self.robot.attach_tool(self.gripper, transform=self.flange_tool_transform)
        self.robot.add_tcp(
            name="suction_tcp",
            transform=self.flange_tcp_transform,
            set_active=True,
        )

        # The imported robot may need gain overrides after attachment restarts
        # physics. The built-in Isaac Sim UR10e deliberately skips this.
        if self.is_simulated:
            articulation = bridge_request(
                "PUT",
                "/articulations",
                body={"prim_path": self.robot_prim_path},
            )
            if articulation is None:
                raise RuntimeError("No robot articulation information returned")
            bridge_request(
                "POST",
                f"/articulations/{articulation['articulation_id']}/dof_gains",
                body={
                    "stiffness": self.DRIVE_STIFFNESS,
                    "damping": self.DRIVE_DAMPING,
                },
            )

        print("Moving to the home L pose...")
        self.move_to_home()

    def shutdown(self) -> None:
        """Disconnect the suction gripper and the robot.

        Safe to call whether or not :meth:`setup` finished: a gripper that never
        connected is left alone, and the robot returns to offline mode from any
        state.

        Returns:
            None.
        """
        if self.gripper.is_connected:
            self.gripper.disconnect()
        self.robot.disconnect()
        self.robot.shutdown()

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def execute(self) -> None:
        """Set up the cell, palletize every box, and shut down.

        Returns:
            None.

        Raises:
            requests.RequestException: If an Isaac Sim bridge request fails.
            RuntimeError: If the hardware cannot be set up, a move fails, or the
                suction gripper does not report the expected box.
            ValueError: If a configured transform is invalid, or ``box_count``
                exceeds the available pallet frames.
        """
        try:
            self.setup()
            self.run()
        finally:
            self.shutdown()

    def run(self) -> None:
        """Palletize ``box_count`` boxes, one pick-and-place cycle each.

        In simulation the conveyors run on their own thread, so once a picked
        box is clear the next one travels to the lightbeam while the robot
        completes the current placement and returns home. A real cell runs its
        own conveyors, so every box is picked from the calibrated pick frame.
        Boxes fill the pallet cell by cell and layer by layer.

        Returns:
            None.

        Raises:
            RuntimeError: If a move fails or the suction gripper does not report
                the expected box.
            ValueError: If ``box_count`` exceeds the available pallet frames, or
                a TF frame is invalid.
        """
        if self.box_count > PLACE_NUMX * PLACE_NUMY * PLACE_NUMZ:
            raise ValueError("box_count exceeds the available pallet TF frames.")

        with ThreadPoolExecutor(max_workers=1) as conveyor_executor:
            detected_box_future = (
                conveyor_executor.submit(
                    self.scene_backend.run_conveyors_until_lightbeam
                )
                if self.scene_backend is not None
                else None
            )

            for box_index in range(self.box_count):
                z_index, layer_index = divmod(box_index, PLACE_NUMX * PLACE_NUMY)
                x_index, y_index = divmod(layer_index, PLACE_NUMY)
                print(f"\n--- box {box_index + 1}/{self.box_count} ---")

                self.pick(
                    detected_box_future.result()
                    if detected_box_future is not None
                    else None
                )

                # Once the held box is clear, move the next box to the lightbeam
                # while the robot completes the current placement.
                if self.scene_backend is not None and box_index + 1 < self.box_count:
                    detected_box_future = conveyor_executor.submit(
                        self.scene_backend.run_conveyors_until_lightbeam
                    )

                self.place(x_index, y_index, z_index)
                self.move_to_home()

        input(f"Placed {self.box_count} boxes. Press Enter to disconnect...")

    # ------------------------------------------------------------------ #
    # One pick-and-place cycle
    # ------------------------------------------------------------------ #

    def pick(self, conveyor_T_object: list[float] | None = None) -> None:
        """Pick the next box off the conveyor and lift it clear.

        Args:
            conveyor_T_object: Measured box-centre pose relative to the
                conveyor, using XYZ metres and Euler XYZ degrees. It replaces
                the nominal ``pick_object`` frame, so the grasp follows the box
                actually at the sensor. Leave as ``None`` to pick from the
                calibrated frame.

        Returns:
            None.

        Raises:
            RuntimeError: If a Cartesian move fails, or the suction gripper
                reports no box after grasping.
            ValueError: If a TF frame or the measured pose is invalid.
        """
        if conveyor_T_object is not None:
            self.tree.update("pick_object", conveyor_T_object, rot_type="deg")

        print("Picking the box at the conveyor...")
        self.move_tcp_to_frame("pre_pick", self.CARTESIAN_SPEED)
        self.move_tcp_to_frame("pick_grasp", self.CONTACT_SPEED)

        self.gripper.grasp()
        if not self.gripper.get_part_present():
            raise RuntimeError(
                "Suction did not detect a box. The robot was not lifted."
            )

        self.move_tcp_to_frame("pre_pick", self.CARTESIAN_SPEED)

    def place(self, x_index: int, y_index: int, z_index: int) -> None:
        """Place the held box in one pallet cell and retreat above it.

        Args:
            x_index: Pallet cell index along the pallet X axis.
            y_index: Pallet cell index along the pallet Y axis.
            z_index: Stack layer index.

        Returns:
            None.

        Raises:
            RuntimeError: If a Cartesian move fails, or the suction gripper
                still reports a box after releasing.
            ValueError: If a TF frame is invalid.
        """
        frame_suffix = f"{x_index}_{y_index}_{z_index}"

        # Cells other than the first are reached across the layer's own travel
        # height, so a carried box stays level instead of cutting a diagonal.
        if (x_index, y_index) != (0, 0):
            self.move_tcp_to_frame(f"pre_place_0_0_{z_index}", self.CARRY_SPEED)

        print(f"Placing at pallet frame {frame_suffix}...")
        self.move_tcp_to_frame(f"pre_place_{frame_suffix}", self.CARRY_SPEED)
        self.move_tcp_to_frame(f"place_grasp_{frame_suffix}", self.CONTACT_SPEED)

        self.gripper.release()
        if self.gripper.get_part_present():
            raise RuntimeError("Suction still reports a box after release.")

        self.move_tcp_to_frame(f"pre_place_{frame_suffix}", self.CARTESIAN_SPEED)

    # ------------------------------------------------------------------ #
    # Motion and scene helpers
    # ------------------------------------------------------------------ #

    def move_to_home(self) -> None:
        """Move the robot to its home joint pose.

        Returns:
            None.

        Raises:
            RuntimeError: If Synapse cannot execute the joint move.
        """
        self.robot.set_joint_positions(
            self.HOME_JOINTS,
            speed=self.JOINT_SPEED,
            acceleration=self.JOINT_ACCELERATION,
        )

    def move_tcp_to_frame(self, frame_name: str, speed: float) -> None:
        """Move the active robot TCP to a named TF frame.

        Every move runs at the configured Cartesian acceleration; only the speed
        differs between approach, contact, and carry motions.

        Args:
            frame_name: Name of the target frame in the cell transform tree.
            speed: Cartesian speed in metres per second.

        Returns:
            None.

        Raises:
            RuntimeError: If Synapse cannot generate or execute the Cartesian move.
            ValueError: If the TF frame or pose is invalid.
        """
        target_pose = self.tree.lookup_transform(
            "robot_base",
            frame_name,
            rot_type="deg",
        )
        self.robot.set_cartesian_pose(
            target_pose,
            speed=speed,
            acceleration=self.CARTESIAN_ACCELERATION,
        )

    def place_robot_on_mount(self) -> None:
        """Move the robot prim to the transform tree's ``robot_base`` frame.

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
                self.tree.lookup_transform("world", "robot_base"),
                rot_type="rotvec",
            )
        ]
        bridge_request(
            "PUT",
            "/prims/poses",
            body={
                "prim_path": self.robot_prim_path,
                "input_pose": {"pose": robot_base_pose_in_world},
            },
        )


def main() -> None:
    """Run the palletizing application in Isaac Sim.

    Returns:
        None.

    Raises:
        requests.RequestException: If an Isaac Sim bridge request fails.
        RuntimeError: If the robot or suction gripper cannot connect or attach.
        ValueError: If a configured transform is invalid.
    """
    robot = universal_robots.UniversalRobotsUR10E(name="ur10e")
    gripper = custom.SuctionGripper()

    application = PalletizingApplication(robot, gripper)
    application.execute()


if __name__ == "__main__":
    main()
