"""
UR10e palletizing demo — Isaac Sim + Telekinesis Synapse.

Demonstrates a palletizing workflow for the Universal Robots UR10e.
The robot mounts on a stand, moves to its home configuration, and waits
for a box to arrive at a lightbeam sensor before stopping the conveyor.

Components not yet supported by Synapse (conveyor, suction gripper,
lightbeam sensor, box spawner, pallet) are wrapped as thin classes.
When Synapse adds native support, replace a class body with the relevant
import — the pipeline in ``main()`` stays unchanged.

Workflow (complete in Isaac Sim before running):
  1. Open the palletizing scene USD.
  2. Import the UR10e URDF via the URDF Importer (Fix Base = ON).
     The robot spawns at the world origin; this script repositions it.
  3. Update the prim-path constants below to match the open stage.
  4. Run via the Isaac Sim VS Code Extension or the Kit Script Editor.

You manage the timeline: ``main()`` plays it before connecting to any
articulation. The interface never stops/pauses the timeline, so connecting
a robot never restarts the simulation.
"""

from __future__ import annotations

from typing import Optional

import omni.kit.app
import omni.timeline
import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from telekinesis.synapse.robots.manipulators import universal_robots


# ===========================================================================
# Scene prim paths — right-click any prim in the Stage panel → Copy Prim Path.
# ===========================================================================

# Articulation-root path of the imported UR10e URDF.
ROBOT_PRIM_PATH: str = "/World/ur10e_robot"

# Prim on which the robot base should rest.
STAND_PRIM_PATH: str = "/World/palletizing_rough_scene/ur10_mount"

# Vertical offset above the stand origin where the robot base sits (m).
# Zero when the stand origin is flush with its top surface.
STAND_TOP_OFFSET_Z: float = 0.0

# Yaw (deg) about world Z applied after positioning so the robot faces the
# conveyor. Set to 0.0 if the URDF default orientation already faces the belt.
ROBOT_YAW_DEG: float = 0.0

# Position-drive gains. Imported URDFs default to soft drives; raising these
# keeps the arm steady under gravity. Must be set before play().
# UR10e (33 kg): 1e5 / 1e4 is sufficient.
ROBOT_DRIVE_STIFFNESS: float = 1.0e5
ROBOT_DRIVE_DAMPING: float = 1.0e4

# UR10e ready pose — elbow-up configuration (degrees).
HOME_Q: list[float] = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]

# All conveyor belt prims that carry boxes. start()/stop() control them together.
CONVEYOR_PRIM_PATHS: list[str] = [
    "/World/palletizing_rough_scene/ConveyorBelt_A08",
    "/World/palletizing_rough_scene/ConveyorBelt_A11",
    "/World/palletizing_rough_scene/ConveyorBelt_A08_01",
]

# Isaac LightBeam sensor that detects a box arriving in the pick zone.
LIGHTBEAM_PRIM_PATH: str = "/World/palletizing_rough_scene/LightBeam_Sensor"

# USD path of the suction gripper added from Isaac assets.
GRIPPER_PRIM_PATH: str = "/World/suction_gripper"

# Maximum physics frames to wait for the lightbeam before giving up.
MONITOR_MAX_STEPS: int = 3000


# ===========================================================================
# Component classes
#
# Components not yet supported by Synapse are wrapped here as thin classes.
# When Synapse adds native support, replace the class body with a proper
# import — the rest of the pipeline stays unchanged.
# ===========================================================================

class SuctionGripper:
    """Placeholder for a suction-cup end-effector.

    Attach to the robot flange, then call ``grip()`` / ``release()`` to
    toggle the suction surface. Replace this stub with the Synapse suction
    class once it becomes available.

    Args:
        simulation_prim_path: USD path of the gripper prim in the open stage.
    """

    def __init__(self, simulation_prim_path: str) -> None:
        self._prim_path: str = simulation_prim_path

    def connect(self) -> None:
        """Bind to the gripper prim in the open stage."""
        raise NotImplementedError

    def attach_to(self, robot, parent_mount: str = "flange") -> None:
        """Couple the gripper to the robot flange with a fixed joint.

        Args:
            robot: Connected Synapse manipulator instance.
            parent_mount: Link name on the robot to attach to.
        """
        raise NotImplementedError

    def grip(self) -> None:
        """Activate suction to grasp a box."""
        raise NotImplementedError

    def release(self) -> None:
        """Deactivate suction to release a box."""
        raise NotImplementedError


class Conveyor:
    """Controls one or more conveyor belts via PhysX surface velocity.

    Isaac Sim conveyors can be driven by setting a ``physxSurfaceVelocity``
    attribute on a belt collider rather than through an OmniGraph node.
    This class discovers that attribute under each given prim path and
    exposes a simple ``start()`` / ``stop()`` interface.

    Args:
        stage: The open USD stage.
        conveyor_prim_paths: Root prim paths of each belt segment.

    Raises:
        RuntimeError: If no surface-velocity attribute is found under any
            of the provided paths.
    """

    SURFACE_VELOCITY_ATTR: str = "physxSurfaceVelocity:surfaceVelocity"

    def __init__(self, stage, conveyor_prim_paths: list[str]) -> None:
        self._stage = stage
        self._attrs: list = []
        self._run_velocities: list[Gf.Vec3f] = []

        for path in conveyor_prim_paths:
            attr = self._find_surface_velocity_attr(path)
            if attr is None:
                print(f"[WARN] No surface velocity found under {path!r} — skipping.")
                continue
            self._attrs.append(attr)
            self._run_velocities.append(Gf.Vec3f(attr.Get()))

        if not self._attrs:
            raise RuntimeError(
                "No 'physxSurfaceVelocity:surfaceVelocity' attribute found under "
                f"any of {conveyor_prim_paths}. Apply PhysX Surface Velocity to "
                "the belt collider(s) in the scene."
            )
        print(f"Conveyor: controlling {len(self._attrs)} belt(s).")

    def _find_surface_velocity_attr(self, root_path: str):
        """Traverse descendants of *root_path* and return the first prim that
        carries a surface-velocity attribute, or ``None`` if absent."""
        root = self._stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            attr = prim.GetAttribute(self.SURFACE_VELOCITY_ATTR)
            if attr and attr.IsValid() and attr.Get() is not None:
                return attr
        return None

    def start(self) -> None:
        """Restore each belt to its configured surface velocity."""
        for attr, vel in zip(self._attrs, self._run_velocities):
            attr.Set(vel)

    def stop(self) -> None:
        """Zero the surface velocity on every belt to halt box transport."""
        for attr in self._attrs:
            attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


class LightBeamSensor:
    """Wraps an Isaac Sim LightBeam sensor prim.

    Triggers when any beam ray reports a hit (i.e. an object is within the
    sensor's range), indicating a box has arrived in the pick zone.

    Args:
        prim_path: USD path of the IsaacLightBeam prim.
    """

    def __init__(self, prim_path: str) -> None:
        from isaacsim.sensors.physx import _range_sensor  # only available inside Kit
        self._prim_path: str = prim_path
        self._iface = _range_sensor.acquire_lightbeam_sensor_interface()

    def is_triggered(self) -> bool:
        """Return ``True`` when at least one beam ray detects an object."""
        hit = self._iface.get_beam_hit_data(self._prim_path)
        return bool(hit is not None and np.asarray(hit).any())


class BoxSpawner:
    """Spawns boxes onto the conveyor. Not yet implemented."""

    def spawn(self) -> None:
        """Spawn one box at the conveyor entry point."""
        raise NotImplementedError


class Pallet:
    """Tracks stack positions on a pallet (fill order: X → Y → Z).

    Args:
        starting_position: World position [x, y, z] of the first slot centre.
        box_size: Box dimensions [x, y, z] in metres.
        counts: Number of slots along each axis (x_count, y_count, z_count).
    """

    def __init__(
        self,
        starting_position: list[float],
        box_size: list[float],
        counts: tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        self._start = starting_position
        self._box_size = box_size
        self._counts = counts

    def next_slot(self) -> Optional[list[float]]:
        """Return the next free slot position, or ``None`` when the pallet is full."""
        raise NotImplementedError


# ===========================================================================
# Internal helpers
# ===========================================================================

def _world_translation(stage, prim_path: str) -> np.ndarray:
    """Return the world-frame (x, y, z) translation of *prim_path*.

    Args:
        stage: Open USD stage.
        prim_path: Path of the prim to query.

    Returns:
        NumPy array of shape (3,) in metres.

    Raises:
        RuntimeError: If the prim does not exist in the stage.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim {prim_path!r} not found in the stage.")
    t = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])])


def set_robot_drive_gains(stage) -> None:
    """Apply position-drive gains to every revolute joint under the robot root.

    Imported URDFs often default to soft drives that sag or oscillate under
    gravity. This must run **before** ``play()``; PhysX reads drive parameters
    only at simulation initialisation.

    Args:
        stage: Open USD stage.
    """
    count = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(ROBOT_PRIM_PATH):
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.CreateStiffnessAttr().Set(ROBOT_DRIVE_STIFFNESS)
            drive.CreateDampingAttr().Set(ROBOT_DRIVE_DAMPING)
            count += 1
    print(f"Drive gains set on {count} joint(s) "
          f"(stiffness={ROBOT_DRIVE_STIFFNESS:g}, damping={ROBOT_DRIVE_DAMPING:g}).")


def position_robot_on_stand(stage) -> None:
    """Reposition the robot base onto the stand and apply the configured yaw.

    The URDF importer places the robot at the world origin. This function
    moves the base to the stand's world position before the timeline starts.
    Repositioning a live articulation corrupts its internal state, so this
    must run **before** ``play()``.

    The URDF importer authors an ``xformOp:orient`` (quaternion) op which
    makes ``XformCommonAPI.SetTranslate`` silently no-op. The translate op
    is therefore written directly.

    Args:
        stage: Open USD stage.

    Raises:
        RuntimeError: If the robot or stand prim is not found in the stage.
    """
    mount_xyz = _world_translation(stage, STAND_PRIM_PATH)
    mount_xyz[2] += STAND_TOP_OFFSET_Z

    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not robot_prim.IsValid():
        raise RuntimeError(
            f"Robot prim {ROBOT_PRIM_PATH!r} not found in the stage. "
            "Import the URDF first (URDF Importer → Fix Base ON)."
        )

    xform = UsdGeom.Xformable(robot_prim)
    translate_op = next(
        (op for op in xform.GetOrderedXformOps()
         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(mount_xyz[0]), float(mount_xyz[1]), float(mount_xyz[2])))

    if ROBOT_YAW_DEG:
        yaw = Gf.Rotation(Gf.Vec3d(0, 0, 1), ROBOT_YAW_DEG).GetQuat()
        orient_op = next(
            (op for op in xform.GetOrderedXformOps()
             if op.GetOpType() == UsdGeom.XformOp.TypeOrient),
            None,
        )
        if orient_op is None:
            xform.AddOrientOp().Set(Gf.Quatf(yaw))
        else:
            cur = orient_op.Get()
            orient_op.Set(
                Gf.Quatf(yaw) * cur if isinstance(cur, Gf.Quatf)
                else yaw * Gf.Quatd(cur)
            )

    actual = UsdGeom.XformCache().GetLocalToWorldTransform(
        robot_prim
    ).ExtractTranslation()
    print(
        f"Robot base → {[round(float(v), 3) for v in actual]}  "
        f"(target {[round(float(v), 3) for v in mount_xyz]})"
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Run the UR10e palletizing demo."""
    stage = omni.usd.get_context().get_stage()

    # Reposition and configure drives before the timeline starts.
    position_robot_on_stand(stage)
    set_robot_drive_gains(stage)

    omni.timeline.get_timeline_interface().play()

    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(simulation_prim_path=ROBOT_PRIM_PATH)

    app = omni.kit.app.get_app()
    conveyor = Conveyor(stage, CONVEYOR_PRIM_PATHS)
    sensor = LightBeamSensor(LIGHTBEAM_PRIM_PATH)

    try:
        robot.set_joint_positions(HOME_Q)
        print("Robot at home:", [round(v, 1) for v in robot.state.joint_positions])

        conveyor.start()
        print("Conveyor running — waiting for box at lightbeam...")
        for _ in range(MONITOR_MAX_STEPS):
            app.update()
            if sensor.is_triggered():
                conveyor.stop()
                print("Box detected — conveyor stopped.")
                break
        else:
            print("[WARN] Lightbeam not triggered within step budget.")

        # Next: wait for box to settle → read box world pose → convert to
        # robot base frame → suction pick → place on pallet → restart conveyor.

    finally:
        robot.disconnect()
        print("Done.")


if __name__ in ("__main__", "isaacsim.code_editor.vscode.extension"):
    main()
