"""Isaac Sim scene operations for the automotive assembly example.

This module isolates the temporary code-socket implementation used for
conveyors, lightbeam sensing, live car measurement, and runtime spawning.
The application depends only on :class:`AutomotiveIsaacSimBackend`, so these
operations can later move behind the Isaac Sim bridge without changing the
robot workflow.
"""

from __future__ import annotations

import json
import socket


ISAAC_SIM_CODE_HOST = "127.0.0.1"
ISAAC_SIM_CODE_PORT = 8226
REQUEST_TIMEOUT_SECONDS = 30.0
CONVEYOR_WAIT_TIMEOUT_SECONDS = 120.0
LIGHTBEAM_MAX_FRAMES = 3600
CAR_POSE_RESULT_PREFIX = "CAR_POSE_RESULT:"


class AutomotiveIsaacSimBackend:
    """Control automotive scene elements not yet exposed by the bridge.

    Attributes:
        conveyor_velocities: Surface velocity for every running conveyor.
        lightbeam_path: Isaac Sim prim path of the station lightbeam.
        sledge_paths: Two alternating runtime car paths.
        roof_paths: Two alternating runtime roof paths.
        sledge_template_source_path: Car whose model and startup pose are
            copied for every replacement car.
        template_root: Inactive runtime template container.
    """

    def __init__(
        self,
        *,
        conveyor_velocities: dict[str, list[float]],
        lightbeam_path: str,
        sledge_paths: list[str],
        roof_paths: list[str],
        sledge_template_source_path: str,
        template_root: str,
    ) -> None:
        """Store the scene paths used by the Isaac Sim adapter.

        Args:
            conveyor_velocities: Conveyor prim paths and running velocities.
            lightbeam_path: Prim path of the station lightbeam.
            sledge_paths: Alternating runtime car paths.
            roof_paths: Alternating runtime roof paths.
            sledge_template_source_path: Car copied for all later spawns.
            template_root: Inactive template container path.

        Returns:
            None.

        Raises:
            ValueError: If the car and roof pools do not have matching sizes.
        """
        if len(sledge_paths) != len(roof_paths):
            raise ValueError("Car and roof pools must have matching sizes.")

        self.conveyor_velocities = conveyor_velocities
        self.lightbeam_path = lightbeam_path
        self.sledge_paths = sledge_paths
        self.roof_paths = roof_paths
        self.sledge_template_source_path = sledge_template_source_path
        self.template_root = template_root
        self.sledge_template_path = f"{template_root}/sledge_01"
        self.roof_template_path = f"{template_root}/roof"

    @staticmethod
    def _execute(code: str, timeout_seconds: float) -> str:
        """Execute Python in the currently open Isaac Sim process.

        Args:
            code: Python source to execute inside Isaac Sim.
            timeout_seconds: Maximum response wait time.

        Returns:
            Text printed by the scene-side operation.

        Raises:
            RuntimeError: If Isaac Sim cannot be reached or rejects the code.
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
            raise ValueError("Isaac Sim returned an invalid response.") from error

        if response.get("status") != "ok":
            traceback_text = "\n".join(response.get("traceback", []))
            detail = traceback_text or response.get(
                "evalue",
                "Unknown Isaac Sim error",
            )
            raise RuntimeError(f"Isaac Sim scene-side operation failed:\n{detail}")
        return response.get("output", "")

    def prepare_spawn_templates(self) -> None:
        """Capture the sledge_01 startup pose and roof as inactive templates.

        Returns:
            None.

        Raises:
            RuntimeError: If a source or copy lacks required physics.
            ValueError: If Isaac Sim returns an invalid response.
        """
        isaac_code = f"""\
import omni.kit.commands
import omni.kit.app
import omni.usd
from pxr import Usd, UsdPhysics

SLEDGE_SOURCE_PATH = {self.sledge_template_source_path!r}
ROOF_SOURCE_PATH = {self.roof_paths[0]!r}
SECOND_ROOF_PATH = {self.roof_paths[1]!r}
TEMPLATE_ROOT = {self.template_root!r}
SLEDGE_TEMPLATE_PATH = {self.sledge_template_path!r}
ROOF_TEMPLATE_PATH = {self.roof_template_path!r}

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No stage is open in Isaac Sim.")

for source_path in (SLEDGE_SOURCE_PATH, ROOF_SOURCE_PATH):
    if not stage.GetPrimAtPath(source_path).IsValid():
        raise RuntimeError("Spawn source not found: " + source_path)

if stage.GetPrimAtPath(TEMPLATE_ROOT).IsValid():
    stage.RemovePrim(TEMPLATE_ROOT)
stage.DefinePrim(TEMPLATE_ROOT, "Scope")

for source_path, template_path in (
    (SLEDGE_SOURCE_PATH, SLEDGE_TEMPLATE_PATH),
    (ROOF_SOURCE_PATH, ROOF_TEMPLATE_PATH),
):
    omni.kit.commands.execute(
        "CopyPrimCommand",
        path_from=source_path,
        path_to=template_path,
    )
    template = stage.GetPrimAtPath(template_path)
    if not template.IsValid():
        raise RuntimeError("Spawn template was not created: " + template_path)
    rigid_body_count = sum(
        prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in Usd.PrimRange(template)
    )
    collider_count = sum(
        prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(template)
    )
    if rigid_body_count == 0 or collider_count == 0:
        raise RuntimeError(
            "Spawn template is not physical: "
            + template_path
            + " ("
            + str(rigid_body_count)
            + " rigid bodies, "
            + str(collider_count)
            + " colliders)"
        )
    template.SetActive(False)

# Recreate roof 1 as well, so the first pick uses the exact same freshly
# spawned physics state as every later pick.
roof_template = stage.GetPrimAtPath(ROOF_TEMPLATE_PATH)
for runtime_path, active in (
    (ROOF_SOURCE_PATH, True),
    (SECOND_ROOF_PATH, False),
):
    if stage.GetPrimAtPath(runtime_path).IsValid():
        stage.RemovePrim(runtime_path)
    roof_template.SetActive(True)
    omni.kit.commands.execute(
        "CopyPrimCommand",
        path_from=ROOF_TEMPLATE_PATH,
        path_to=runtime_path,
    )
    roof_template.SetActive(False)
    spawned_roof = stage.GetPrimAtPath(runtime_path)
    if not spawned_roof.IsValid():
        raise RuntimeError("Roof was not created: " + runtime_path)
    spawned_roof.SetActive(active)

await omni.kit.app.get_app().next_update_async()

print("Prepared templates and recreated roof 1 from the same roof template.")
"""
        output = self._execute(isaac_code, REQUEST_TIMEOUT_SECONDS)
        if output:
            print(output.strip())

    def spawn_cycle_assets(self, sledge_path: str, roof_path: str) -> None:
        """Replace one old car/roof pair from the inactive templates.

        Args:
            sledge_path: Runtime path for the replacement car.
            roof_path: Runtime path for the replacement roof.

        Returns:
            None.

        Raises:
            RuntimeError: If a template or spawned rigid body is missing.
            ValueError: If Isaac Sim returns an invalid response.
        """
        isaac_code = f"""\
import omni.kit.commands
import omni.kit.app
import omni.physx
import omni.usd
from pxr import PhysicsSchemaTools, Usd, UsdPhysics

SLEDGE_PATH = {sledge_path!r}
ROOF_PATH = {roof_path!r}
SLEDGE_TEMPLATE_PATH = {self.sledge_template_path!r}
ROOF_TEMPLATE_PATH = {self.roof_template_path!r}

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No stage is open in Isaac Sim.")

for template_path in (SLEDGE_TEMPLATE_PATH, ROOF_TEMPLATE_PATH):
    if not stage.GetPrimAtPath(template_path).IsValid():
        raise RuntimeError("Spawn template not found: " + template_path)

for runtime_path in (SLEDGE_PATH, ROOF_PATH):
    if stage.GetPrimAtPath(runtime_path).IsValid():
        stage.RemovePrim(runtime_path)

for template_path, runtime_path in (
    (SLEDGE_TEMPLATE_PATH, SLEDGE_PATH),
    (ROOF_TEMPLATE_PATH, ROOF_PATH),
):
    template = stage.GetPrimAtPath(template_path)
    template.SetActive(True)
    omni.kit.commands.execute(
        "CopyPrimCommand",
        path_from=template_path,
        path_to=runtime_path,
    )
    template.SetActive(False)
    spawned = stage.GetPrimAtPath(runtime_path)
    if not spawned.IsValid():
        raise RuntimeError("Runtime asset was not spawned: " + runtime_path)
    spawned.SetActive(True)

await omni.kit.app.get_app().next_update_async()
physics_simulation = omni.physx.get_physx_simulation_interface()
stage_id = omni.usd.get_context().get_stage_id()
for runtime_path in (SLEDGE_PATH, ROOF_PATH):
    runtime_prim = stage.GetPrimAtPath(runtime_path)
    rigid_bodies = [
        prim
        for prim in Usd.PrimRange(runtime_prim)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if not rigid_bodies:
        raise RuntimeError("Spawned asset has no rigid body: " + runtime_path)
    for rigid_body in rigid_bodies:
        physics_simulation.wake_up(
            stage_id,
            PhysicsSchemaTools.sdfPathToInt(rigid_body.GetPath()),
        )

for _ in range(10):
    await omni.kit.app.get_app().next_update_async()

print("Spawned car and roof: " + SLEDGE_PATH + ", " + ROOF_PATH)
"""
        output = self._execute(isaac_code, REQUEST_TIMEOUT_SECONDS)
        if output:
            print(output.strip())

    def activate_roof(self, roof_path: str) -> None:
        """Activate and wake a prepared roof on the rack.

        Args:
            roof_path: Runtime roof prim to activate.

        Returns:
            None.

        Raises:
            RuntimeError: If the roof or its rigid body is missing.
            ValueError: If Isaac Sim returns an invalid response.
        """
        isaac_code = f"""\
import omni.kit.app
import omni.physx
import omni.usd
from pxr import PhysicsSchemaTools, Usd, UsdPhysics

ROOF_PATH = {roof_path!r}
stage = omni.usd.get_context().get_stage()
roof = stage.GetPrimAtPath(ROOF_PATH) if stage is not None else None
if roof is None or not roof.IsValid():
    raise RuntimeError("Prepared roof not found: " + ROOF_PATH)

roof.SetActive(True)
await omni.kit.app.get_app().next_update_async()
roof = stage.GetPrimAtPath(ROOF_PATH)
rigid_bodies = [
    prim for prim in Usd.PrimRange(roof)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI)
]
if not rigid_bodies:
    raise RuntimeError("Prepared roof has no rigid body: " + ROOF_PATH)

physics_simulation = omni.physx.get_physx_simulation_interface()
stage_id = omni.usd.get_context().get_stage_id()
for rigid_body in rigid_bodies:
    physics_simulation.wake_up(
        stage_id,
        PhysicsSchemaTools.sdfPathToInt(rigid_body.GetPath()),
    )

print("Activated physical roof: " + ROOF_PATH)
"""
        output = self._execute(isaac_code, REQUEST_TIMEOUT_SECONDS)
        if output:
            print(output.strip())

    def attach_placed_roof(self, *, roof_path: str, sledge_path: str) -> None:
        """Fix a released roof to its car at the current relative pose.

        Args:
            roof_path: Root prim of the roof that was just released.
            sledge_path: Root prim of the car carrying that roof.

        Returns:
            None.

        Raises:
            RuntimeError: If either asset lacks exactly one rigid body.
            ValueError: If Isaac Sim returns an invalid response.
        """
        isaac_code = f"""\
import omni.kit.app
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOF_PATH = {roof_path!r}
SLEDGE_PATH = {sledge_path!r}
JOINT_PATH = ROOF_PATH + "/PlacedRoofFixedJoint"

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No stage is open in Isaac Sim.")


def rigid_bodies_below(root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError("Asset not found: " + root_path)
    return [
        prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]


sledge_bodies = rigid_bodies_below(SLEDGE_PATH)
roof_bodies = rigid_bodies_below(ROOF_PATH)
if len(sledge_bodies) != 1 or len(roof_bodies) != 1:
    raise RuntimeError(
        "Expected one rigid body below each asset; found "
        + str(len(sledge_bodies))
        + " for the car and "
        + str(len(roof_bodies))
        + " for the roof."
    )

sledge_body = sledge_bodies[0]
roof_body = roof_bodies[0]
xform_cache = UsdGeom.XformCache()
world_T_sledge = xform_cache.GetLocalToWorldTransform(sledge_body)
world_T_roof = xform_cache.GetLocalToWorldTransform(roof_body)
sledge_T_roof = world_T_roof * world_T_sledge.GetInverse()
relative = Gf.Transform(sledge_T_roof)
relative_translation = relative.GetTranslation()
relative_quaternion = relative.GetRotation().GetQuat()

if stage.GetPrimAtPath(JOINT_PATH).IsValid():
    stage.RemovePrim(JOINT_PATH)
fixed_joint = UsdPhysics.FixedJoint.Define(stage, JOINT_PATH)
fixed_joint.CreateBody0Rel().SetTargets([sledge_body.GetPath()])
fixed_joint.CreateBody1Rel().SetTargets([roof_body.GetPath()])
fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*relative_translation))
fixed_joint.CreateLocalRot0Attr().Set(
    Gf.Quatf(
        float(relative_quaternion.GetReal()),
        Gf.Vec3f(*relative_quaternion.GetImaginary()),
    )
)
fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
fixed_joint.CreateLocalRot1Attr().Set(
    Gf.Quatf(1.0, Gf.Vec3f(0.0)),
)
fixed_joint.CreateCollisionEnabledAttr(False)
fixed_joint.CreateExcludeFromArticulationAttr(True)

# Match velocities before PhysX closes the fixed constraint.
sledge_rigid_body = UsdPhysics.RigidBodyAPI(sledge_body)
roof_rigid_body = UsdPhysics.RigidBodyAPI(roof_body)
roof_rigid_body.GetVelocityAttr().Set(sledge_rigid_body.GetVelocityAttr().Get())
roof_rigid_body.GetAngularVelocityAttr().Set(
    sledge_rigid_body.GetAngularVelocityAttr().Get()
)

await omni.kit.app.get_app().next_update_async()
print("Fixed placed roof to car: " + ROOF_PATH + " -> " + SLEDGE_PATH)
"""
        output = self._execute(isaac_code, REQUEST_TIMEOUT_SECONDS)
        if output:
            print(output.strip())

    def hold_conveyors_during_setup(self) -> None:
        """Set conveyor and car velocities to zero during robot setup.

        Returns:
            None.

        Raises:
            RuntimeError: If a conveyor or velocity attribute is missing.
            ValueError: If Isaac Sim returns an invalid response.
        """
        isaac_code = f"""\
import omni.usd
from pxr import Gf, Usd, UsdPhysics

CONVEYOR_PATHS = {list(self.conveyor_velocities)!r}
SLEDGE_PATHS = {self.sledge_paths!r}
VELOCITY_ATTRIBUTE = "physxSurfaceVelocity:surfaceVelocity"

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No stage is open in Isaac Sim.")

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
    velocity_attribute.Set(Gf.Vec3f(0.0, 0.0, 0.0))

for sledge_path in SLEDGE_PATHS:
    sledge = stage.GetPrimAtPath(sledge_path)
    if not sledge.IsValid():
        raise RuntimeError("Sledge not found: " + sledge_path)
    for prim in Usd.PrimRange(sledge):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_body = UsdPhysics.RigidBodyAPI(prim)
        rigid_body.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

print("All conveyors are held at zero during robot setup.")
"""
        output = self._execute(isaac_code, REQUEST_TIMEOUT_SECONDS)
        if output:
            print(output.strip())

    def wait_for_car(self, *, wait_for_clear: bool, sledge_path: str) -> list[float]:
        """Run conveyors until the lightbeam detects and stops a car.

        Args:
            wait_for_clear: Require the previous sensor hit to clear first.
            sledge_path: Car whose stopped World pose should be returned.

        Returns:
            Stopped car pose in World as XYZ metres and Euler XYZ degrees.

        Raises:
            RuntimeError: If a required prim is missing or detection times out.
            ValueError: If Isaac Sim returns an invalid result.
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
    raise RuntimeError("No stage is open in Isaac Sim.")

if not stage.GetPrimAtPath(LIGHTBEAM_PATH).IsValid():
    raise RuntimeError("Lightbeam not found: " + LIGHTBEAM_PATH)

sledges = []
for runtime_sledge_path in SLEDGE_PATHS:
    sledge = stage.GetPrimAtPath(runtime_sledge_path)
    if not sledge.IsValid():
        raise RuntimeError("Sledge not found: " + runtime_sledge_path)
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


def rigid_transform(prim):
    source = Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    result = Gf.Transform()
    result.SetTranslation(source.GetTranslation())
    result.SetRotation(source.GetRotation())
    return result


def rotation_to_rpy(rotation):
    quaternion = rotation.GetQuat()
    w = float(quaternion.GetReal())
    x, y, z = [float(value) for value in quaternion.GetImaginary()]
    length = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = [value / length for value in (w, x, y, z)]
    return [
        math.degrees(
            math.atan2(
                2.0 * (w * x + y * z),
                1.0 - 2.0 * (x * x + y * y),
            )
        ),
        math.degrees(
            math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        ),
        math.degrees(
            math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
        ),
    ]


car_world = rigid_transform(stopped_sledge)
world_T_car = [
    *[float(value) for value in car_world.GetTranslation()],
    *rotation_to_rpy(car_world.GetRotation()),
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
"""

        print("Starting all conveyors; waiting for a sledge at the lightbeam...")
        output = self._execute(
            isaac_code,
            CONVEYOR_WAIT_TIMEOUT_SECONDS,
        )
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
            raise RuntimeError(
                "Isaac Sim measured a different sledge than requested."
            )
        return [float(value) for value in result_payload["world_T_car"]]
