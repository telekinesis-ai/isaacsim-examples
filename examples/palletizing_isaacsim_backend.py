"""Isaac Sim scene operations for the palletizing example.

This module isolates the temporary code-socket implementation used for
conveyor control, lightbeam sensing, and live stopped-box measurement. These
operations can later move behind the Isaac Sim bridge without changing the
robot workflow.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass


ISAAC_SIM_CODE_HOST = "127.0.0.1"
ISAAC_SIM_CODE_PORT = 8226
CONVEYOR_WAIT_TIMEOUT_SECONDS = 120.0
LIGHTBEAM_MAX_FRAMES = 3600
DETECTED_BOX_RESULT_PREFIX = "__PALLETIZING_DETECTED_BOX__="


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


class PalletizingIsaacSimBackend:
    """Control palletizing scene elements not yet exposed by the bridge.

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

    def run_conveyors_until_lightbeam(self) -> DetectedBox:
        """Run all conveyors and measure the box stopped at the lightbeam.

        All configured conveyor surface velocities start together and are all
        set to zero when the lightbeam detects a box or if the operation fails.
        The stopped box nearest the sensor is measured after a short settling
        period.

        Returns:
            Detected box path, conveyor-relative centre pose, and dimensions.

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
        output = self._execute(isaac_code, CONVEYOR_WAIT_TIMEOUT_SECONDS)
        result_payload = None
        for line in output.splitlines():
            if line.startswith(DETECTED_BOX_RESULT_PREFIX):
                result_payload = json.loads(
                    line.removeprefix(DETECTED_BOX_RESULT_PREFIX)
                )

        if result_payload is None:
            raise RuntimeError("Isaac Sim did not return the detected box measurement.")
        return DetectedBox(
            prim_path=str(result_payload["prim_path"]),
            conveyor_T_object=[
                float(value) for value in result_payload["conveyor_T_object"]
            ],
            size=[float(value) for value in result_payload["size"]],
        )
