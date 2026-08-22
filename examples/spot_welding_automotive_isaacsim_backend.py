"""Isaac Sim scene operations used by automotive spot welding."""

from __future__ import annotations

import json
import socket


ISAAC_SIM_CODE_HOST = "127.0.0.1"
ISAAC_SIM_CODE_PORT = 8226
REQUEST_TIMEOUT_SECONDS = 30.0
CONVEYOR_WAIT_TIMEOUT_SECONDS = 120.0
LIGHTBEAM_MAX_FRAMES = 3600
CAR_POSE_RESULT_PREFIX = "CAR_POSE_RESULT:"


class SpotWeldingAutomotiveIsaacSimBackend:
    """Handle scene operations that are not currently available in the bridge."""

    def __init__(
        self,
        conveyor_velocities: dict[str, list[float]],
        lightbeam_path: str,
        sledge_paths: list[str],
        sledge_template_source_path: str,
        template_root: str,
    ) -> None:
        self.conveyor_velocities = conveyor_velocities
        self.lightbeam_path = lightbeam_path
        self.sledge_paths = sledge_paths
        self.sledge_template_source_path = sledge_template_source_path
        self.template_root = template_root
        self.sledge_template_path = f"{template_root}/sledge_01"

    @staticmethod
    def _execute(code: str, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> str:
        """Execute Python in the open Isaac Sim process through its code socket."""
        try:
            connection = socket.create_connection(
                (ISAAC_SIM_CODE_HOST, ISAAC_SIM_CODE_PORT),
                timeout=5.0,
            )
        except OSError as error:
            raise RuntimeError(
                "Could not reach Isaac Sim's code socket on port 8226. "
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

        response = json.loads(response_bytes.decode("utf-8"))
        if response.get("status") != "ok":
            traceback_text = "\n".join(response.get("traceback", []))
            raise RuntimeError(
                "Isaac Sim scene-side operation failed:\n"
                + (traceback_text or response.get("evalue", "Unknown error"))
            )
        return response.get("output", "")

    def hold_conveyors_during_setup(self) -> None:
        """Hold all conveyors and both cars still while the robot is prepared."""
        code = f'''
from pxr import Gf, Usd, UsdPhysics
import omni.usd

CONVEYOR_PATHS = {list(self.conveyor_velocities)!r}
SLEDGE_PATHS = {self.sledge_paths!r}
VELOCITY_ATTRIBUTE = "physxSurfaceVelocity:surfaceVelocity"

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No stage is open.")

for conveyor_path in CONVEYOR_PATHS:
    conveyor = stage.GetPrimAtPath(conveyor_path)
    if not conveyor.IsValid():
        raise RuntimeError("Conveyor not found: " + conveyor_path)
    velocity_found = False
    for prim in Usd.PrimRange(conveyor):
        attribute = prim.GetAttribute(VELOCITY_ATTRIBUTE)
        if attribute.IsValid() and attribute.Get() is not None:
            attribute.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            velocity_found = True
    if not velocity_found:
        raise RuntimeError("No surface velocity found below " + conveyor_path)

for sledge_path in SLEDGE_PATHS:
    sledge = stage.GetPrimAtPath(sledge_path)
    if not sledge.IsValid():
        raise RuntimeError("Sledge not found: " + sledge_path)
    for prim in Usd.PrimRange(sledge):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body = UsdPhysics.RigidBodyAPI(prim)
            rigid_body.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

print("All conveyors are held at zero during robot setup.")
'''
        output = self._execute(code)
        if output.strip():
            print(output.strip())

    def prepare_car_spawn_template(self) -> None:
        """Copy the second car into one inactive reusable spawn template."""
        code = f'''
import omni.kit.app
import omni.kit.commands
import omni.usd
from pxr import Usd, UsdPhysics

SOURCE_PATH = {self.sledge_template_source_path!r}
TEMPLATE_ROOT = {self.template_root!r}
TEMPLATE_PATH = {self.sledge_template_path!r}

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No stage is open in Isaac Sim.")
if not stage.GetPrimAtPath(SOURCE_PATH).IsValid():
    raise RuntimeError("Car spawn source not found: " + SOURCE_PATH)

if stage.GetPrimAtPath(TEMPLATE_ROOT).IsValid():
    stage.RemovePrim(TEMPLATE_ROOT)
stage.DefinePrim(TEMPLATE_ROOT, "Scope")

omni.kit.commands.execute(
    "CopyPrimCommand",
    path_from=SOURCE_PATH,
    path_to=TEMPLATE_PATH,
)
template = stage.GetPrimAtPath(TEMPLATE_PATH)
if not template.IsValid():
    raise RuntimeError("Car spawn template was not created: " + TEMPLATE_PATH)

rigid_body_count = sum(
    prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in Usd.PrimRange(template)
)
collider_count = sum(
    prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(template)
)
if rigid_body_count == 0 or collider_count == 0:
    raise RuntimeError(
        "Car spawn template is not physical: "
        + str(rigid_body_count)
        + " rigid bodies, "
        + str(collider_count)
        + " colliders"
    )

template.SetActive(False)
await omni.kit.app.get_app().next_update_async()
print("Prepared inactive car spawn template from " + SOURCE_PATH)
'''
        output = self._execute(code)
        if output.strip():
            print(output.strip())

    def spawn_car(self, sledge_path: str) -> None:
        """Replace one completed car pool slot from the inactive template."""
        code = f'''
import omni.kit.app
import omni.kit.commands
import omni.physx
import omni.usd
from pxr import PhysicsSchemaTools, Usd, UsdPhysics

SLEDGE_PATH = {sledge_path!r}
TEMPLATE_PATH = {self.sledge_template_path!r}

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No stage is open in Isaac Sim.")

template = stage.GetPrimAtPath(TEMPLATE_PATH)
if not template.IsValid():
    raise RuntimeError("Car spawn template not found: " + TEMPLATE_PATH)

if stage.GetPrimAtPath(SLEDGE_PATH).IsValid():
    stage.RemovePrim(SLEDGE_PATH)

template.SetActive(True)
omni.kit.commands.execute(
    "CopyPrimCommand",
    path_from=TEMPLATE_PATH,
    path_to=SLEDGE_PATH,
)
template.SetActive(False)

spawned_car = stage.GetPrimAtPath(SLEDGE_PATH)
if not spawned_car.IsValid():
    raise RuntimeError("Car was not spawned: " + SLEDGE_PATH)
spawned_car.SetActive(True)

await omni.kit.app.get_app().next_update_async()
rigid_bodies = [
    prim
    for prim in Usd.PrimRange(spawned_car)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI)
]
if not rigid_bodies:
    raise RuntimeError("Spawned car has no rigid body: " + SLEDGE_PATH)

physics_simulation = omni.physx.get_physx_simulation_interface()
stage_id = omni.usd.get_context().get_stage_id()
for rigid_body in rigid_bodies:
    physics_simulation.wake_up(
        stage_id,
        PhysicsSchemaTools.sdfPathToInt(rigid_body.GetPath()),
    )

for _ in range(10):
    await omni.kit.app.get_app().next_update_async()

print("Spawned replacement car: " + SLEDGE_PATH)
'''
        output = self._execute(code)
        if output.strip():
            print(output.strip())

    def wait_for_car(
        self,
        *,
        wait_for_clear: bool,
        sledge_path: str,
    ) -> list[float]:
        """Run every conveyor until the lightbeam detects and stops one car."""
        code = f'''
import json
import math
import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.sensors.physx import _range_sensor
from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics

CONVEYOR_VELOCITIES = {self.conveyor_velocities!r}
LIGHTBEAM_PATH = {self.lightbeam_path!r}
SLEDGE_PATHS = {self.sledge_paths!r}
STOPPED_SLEDGE_PATH = {sledge_path!r}
WAIT_FOR_CLEAR = {wait_for_clear!r}
MAX_FRAMES = {LIGHTBEAM_MAX_FRAMES}
RESULT_PREFIX = {CAR_POSE_RESULT_PREFIX!r}
VELOCITY_ATTRIBUTE = "physxSurfaceVelocity:surfaceVelocity"

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No stage is open.")

if not stage.GetPrimAtPath(LIGHTBEAM_PATH).IsValid():
    raise RuntimeError("Lightbeam not found: " + LIGHTBEAM_PATH)

sledges = []
for sledge_path in SLEDGE_PATHS:
    sledge = stage.GetPrimAtPath(sledge_path)
    if not sledge.IsValid():
        raise RuntimeError("Sledge not found: " + sledge_path)
    sledges.append(sledge)

velocity_attributes = []
for conveyor_path, velocity in CONVEYOR_VELOCITIES.items():
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
    velocity_attributes.append((velocity_attribute, Gf.Vec3f(*velocity)))

timeline = omni.timeline.get_timeline_interface()
app = omni.kit.app.get_app()
sensor = _range_sensor.acquire_lightbeam_sensor_interface()
timeline.play()
for velocity_attribute, velocity in velocity_attributes:
    velocity_attribute.Set(velocity)

physics_simulation = omni.physx.get_physx_simulation_interface()
stage_id = omni.usd.get_context().get_stage_id()
for sledge in sledges:
    for prim in Usd.PrimRange(sledge):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            physics_simulation.wake_up(
                stage_id,
                PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
            )

clear_frames = 0
try:
    for _ in range(MAX_FRAMES):
        await app.next_update_async()
        hit_data = sensor.get_beam_hit_data(LIGHTBEAM_PATH)
        triggered = bool(
            hit_data is not None and np.asarray(hit_data, dtype=bool).any()
        )
        if not triggered:
            clear_frames += 1
            continue
        if WAIT_FOR_CLEAR and clear_frames < 3:
            continue

        for velocity_attribute, _ in velocity_attributes:
            velocity_attribute.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        for sledge in sledges:
            for prim in Usd.PrimRange(sledge):
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                rigid_body = UsdPhysics.RigidBodyAPI(prim)
                rigid_body.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                physics_simulation.put_to_sleep(
                    stage_id,
                    PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
                )
        break
    else:
        raise RuntimeError("Lightbeam was not triggered before timeout.")
finally:
    for velocity_attribute, _ in velocity_attributes:
        velocity_attribute.Set(Gf.Vec3f(0.0, 0.0, 0.0))

for _ in range(60):
    await app.next_update_async()

stopped_sledge = stage.GetPrimAtPath(STOPPED_SLEDGE_PATH)
if not stopped_sledge.IsValid():
    raise RuntimeError("Cannot measure the stopped car frame.")


def rotation_to_rpy(rotation):
    quaternion = rotation.GetQuat()
    w = float(quaternion.GetReal())
    x, y, z = [float(value) for value in quaternion.GetImaginary()]
    length = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = [value / length for value in (w, x, y, z)]
    return [
        math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))),
        math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))),
        math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))),
    ]


car_transform = Gf.Transform(
    UsdGeom.XformCache().GetLocalToWorldTransform(stopped_sledge)
)
world_T_car = [
    *[float(value) for value in car_transform.GetTranslation()],
    *rotation_to_rpy(car_transform.GetRotation()),
]

print("Lightbeam hit; all conveyors and sledges stopped.")
print(
    RESULT_PREFIX
    + json.dumps(
        {{
            "sledge_path": STOPPED_SLEDGE_PATH,
            "world_T_car": world_T_car,
        }}
    )
)
'''
        print("Starting all conveyors; waiting for a sledge at the lightbeam...")
        output = self._execute(code, CONVEYOR_WAIT_TIMEOUT_SECONDS)
        result_payload = None
        for line in output.splitlines():
            if line.startswith(CAR_POSE_RESULT_PREFIX):
                result_payload = json.loads(
                    line.removeprefix(CAR_POSE_RESULT_PREFIX)
                )
            elif line:
                print(line)

        if result_payload is None:
            raise RuntimeError("Isaac Sim did not return the stopped car pose.")
        if result_payload.get("sledge_path") != sledge_path:
            raise RuntimeError("Isaac Sim measured a different sledge than requested.")
        return [float(value) for value in result_payload["world_T_car"]]
