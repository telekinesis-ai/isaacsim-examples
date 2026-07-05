"""
Multi-brand machine-tending demo — Isaac Sim + Telekinesis Synapse.

Spawns a small collaborative or industrial robot on the mount point in a
warehouse digital-twin scene next to a CNC machine and drives it to a
gentle ready pose. Change one line to switch between all supported brands.

Scene built using assets from Extwin Synthesis — see repository README for
credits and asset table.

Workflow (complete in Isaac Sim before running):
  1. Open ``assets/environments/cnc_machine_tending/cnc_machine_tending.usd``.
  2. Add the robot from the Content Browser (or import via URDF Importer).
     Confirm ``prim_path`` in the registry matches the Stage tree.
  3. Set ``ACTIVE_ROBOT`` to the desired brand key (see registry below).
  4. Run via the Isaac Sim VS Code Extension or the Kit Script Editor.

You manage the timeline: ``main()`` plays it before connecting to any
articulation. The interface never stops/pauses the timeline, so connecting
a robot never restarts the simulation.
"""

from __future__ import annotations

import math

import omni.kit.app
import omni.timeline
import omni.usd
import numpy as np
from pxr import UsdGeom, UsdPhysics, Gf

from telekinesis.synapse.robots.manipulators import (
    fanuc,
    franka_robotics,
    motoman,
    neura_robotics,
    universal_robots,
)


# ===========================================================================
# ← Change this one line to switch the active robot brand.
# ===========================================================================
ACTIVE_ROBOT: str = "ur10e"


# ===========================================================================
# Robot registry — small robots for machine tending.
#
# Each entry:
#   (class, prim_path, home_q, stiffness, damping, yaw_offset_deg)
#
#   class          — Synapse manipulator class.
#   prim_path      — USD articulation-root path (right-click → Copy Prim Path).
#   home_q         — Joint positions (degrees) for a gentle ready pose.
#                    None → use robot.default_joint_configuration.
#   stiffness      — PhysX position-drive stiffness (N·m/rad).
#   damping        — PhysX position-drive damping (N·m·s/rad).
#   yaw_offset_deg — Extra yaw (deg) added to the mount's own yaw.
#
# NOTE: prim_path depends on how the robot was added to the stage.
# Verify it in the Stage tree before running.
# ===========================================================================
ROBOT_REGISTRY: dict = {
    "ur10e":   (universal_robots.UniversalRobotsUR10E, "/World/ur10e_robot",    [0.0, -90.0, -90.0, -90.0, 90.0, 0.0],    1.0e5, 1.0e4, 0.0),
    "motoman": (motoman.MotomanMH5,                    "/World/motoman_mh5",    [0.0,   0.0,   0.0,   0.0, -90.0, 0.0],   1.0e5, 1.0e4, 0.0),
    "franka":  (franka_robotics.FrankaRoboticsPanda,   "/World/franka",         None,                                      1.0e5, 1.0e4, 0.0),
    "neura":   (neura_robotics.NeuraRoboticsMAiRA7M,   "/World/maira7M",        [0.0,  30.0,   0.0,  60.0,  0.0, 90.0, 0.0], 1.0e6, 1.0e5, 0.0),
    "fanuc":   (fanuc.FanucCRX10IAL,                   "/World/fanuc_crx10ial", [0.0,   0.0,   0.0,   0.0, -90.0, 0.0],   1.0e5, 1.0e4, 0.0),
}


# ===========================================================================
# Scene prim paths
# ===========================================================================

MOUNT_PRIM_PATH: str  = "/World/sortbot_housing/Mount_point"
MOUNT_OFFSET_Z: float = -0.005


# ===========================================================================
# Helpers
# ===========================================================================

def _world_translation(stage, prim_path: str) -> np.ndarray:
    """Return the world-frame (x, y, z) translation of *prim_path* (metres)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim {prim_path!r} not found in the stage.")
    t = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])])


def _mount_yaw_deg(stage, prim_path: str) -> float:
    """Return the yaw (deg, Z-axis) of *prim_path* so the robot faces the mount."""
    m = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(prim_path))
    x = m.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    return math.degrees(math.atan2(float(x[1]), float(x[0])))


# ===========================================================================
# Pre-play setup (must run BEFORE play())
# ===========================================================================

def set_robot_drive_gains(stage, robot_prim_path: str,
                          stiffness: float, damping: float) -> None:
    """Stiffen every revolute joint under the robot so it resists gravity.

    PhysX reads drive parameters only at sim init, so this must run before
    play().
    """
    count = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(robot_prim_path):
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.CreateStiffnessAttr().Set(stiffness)
            drive.CreateDampingAttr().Set(damping)
            count += 1
    print(f"Drive gains set on {count} joint(s) "
          f"(stiffness={stiffness:g}, damping={damping:g}).")


def position_robot_on_mount(stage, robot_prim_path: str,
                            yaw_offset_deg: float = 0.0) -> None:
    """Move the robot base onto the mount point, matching the mount's yaw.

    Must run before play() — repositioning a live articulation corrupts its
    internal state.
    """
    mount_xyz = _world_translation(stage, MOUNT_PRIM_PATH)
    mount_xyz[2] += MOUNT_OFFSET_Z
    yaw_deg = _mount_yaw_deg(stage, MOUNT_PRIM_PATH) + yaw_offset_deg

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(
            f"Robot prim {robot_prim_path!r} not found. "
            "Add it to the stage first (Content Browser or URDF Importer)."
        )

    xform = UsdGeom.Xformable(robot_prim)
    translate_op = next(
        (op for op in xform.GetOrderedXformOps()
         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None,
    )
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*[float(v) for v in mount_xyz]))

    if yaw_deg:
        yaw = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg).GetQuat()
        orient_op = next(
            (op for op in xform.GetOrderedXformOps()
             if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None,
        )
        if orient_op is None:
            xform.AddOrientOp().Set(Gf.Quatf(yaw))
        else:
            cur = orient_op.Get()
            orient_op.Set(
                Gf.Quatf(yaw) * cur if isinstance(cur, Gf.Quatf)
                else yaw * Gf.Quatd(cur)
            )

    actual = UsdGeom.XformCache().GetLocalToWorldTransform(robot_prim).ExtractTranslation()
    print(f"Robot base → {[round(float(v), 3) for v in actual]}  "
          f"(target {[round(float(v), 3) for v in mount_xyz]}, yaw={round(yaw_deg, 1)}°)")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Run the machine-tending demo for the robot selected by ACTIVE_ROBOT."""
    robot_cls, robot_prim_path, home_q, stiffness, damping, yaw_offset_deg = (
        ROBOT_REGISTRY[ACTIVE_ROBOT]
    )
    print(f"Active robot : {ACTIVE_ROBOT} — {robot_cls.__name__} @ {robot_prim_path}")

    stage = omni.usd.get_context().get_stage()

    # --- Pre-play: all static USD edits before physics starts. ---
    position_robot_on_mount(stage, robot_prim_path, yaw_offset_deg)
    set_robot_drive_gains(stage, robot_prim_path, stiffness, damping)

    omni.timeline.get_timeline_interface().play()

    # Step a few frames so PhysX creates the simulation view before connecting.
    # Without this, set_joint_positions is called before the view is ready and
    # silently does nothing — the arm spawns but never moves to home_q.
    app = omni.kit.app.get_app()
    for _ in range(10):
        app.update()

    robot = robot_cls()
    robot.connect(simulation_prim_path=robot_prim_path)

    if home_q is None:
        home_q = robot.default_joint_configuration.tolist()

    try:
        robot.set_joint_positions(home_q)
        print("Robot at home:", [round(v, 1) for v in robot.state.joint_positions])

    finally:
        robot.disconnect()
        print("Done.")


if __name__ in ("__main__", "isaacsim.code_editor.vscode.extension"):
    main()
