"""
Multi-brand automotive assembly demo — Isaac Sim + Telekinesis Synapse.

Demonstrates large industrial robots mounted on a pedestal in a vehicle
factory environment. Each robot is repositioned onto the pedestal, rotated
to face the work cell, moved to its home configuration, and coupled to the
suction gripper.

Workflow (complete in Isaac Sim before running):
  1. Open the assembly scene USD (factory environment with pedestal).
  2. Import the robot URDF via the URDF Importer (Fix Base = ON).
  3. Set ``ACTIVE_ROBOT`` to the desired brand key (see registry below).
  4. Run via the Isaac Sim VS Code Extension or the Kit Script Editor.

You manage the timeline: ``main()`` plays it before connecting to any
articulation. The interface never stops/pauses the timeline, so connecting
a robot never restarts the simulation.
"""

from __future__ import annotations

import omni.timeline
import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

from telekinesis.synapse.robots.manipulators import (
    abb,
    kuka,
    neura_robotics,
)


# ===========================================================================
# ← Change this one line to switch the active robot brand.
# ===========================================================================
ACTIVE_ROBOT: str = "kuka"


# ===========================================================================
# Robot registry — large industrial arms suitable for automotive assembly.
#
# Each entry: (class, prim_path, home_q, stiffness, damping, yaw_deg, flange_link)
#
#   class        — Synapse manipulator class.
#   prim_path    — USD articulation-root path; must match the imported URDF.
#   home_q       — Joint positions in degrees for the ready pose.
#                  None → use robot.default_joint_configuration.
#   stiffness    — PhysX position-drive stiffness (N·m/rad). Heavy industrial
#                  arms require high values to resist gravity.
#   damping      — PhysX position-drive damping (N·m·s/rad).
#   yaw_deg      — Rotation about world Z (degrees) to face the work cell.
#   flange_link  — Relative path from robot root to the flange link used to
#                  attach the gripper. None → no gripper attached.
# ===========================================================================
ROBOT_REGISTRY: dict = {
    "kuka":  (kuka.KukaKR210L150,             "/World/kuka_kr210",          None, 1.0e6, 1.0e5, 145.0, "link_6/tool0"),
    "neura": (neura_robotics.NeuraRoboticsMAiRA7M, "/World/maira7M",        None, 1.0e6, 1.0e5, 145.0, None),
    "abb":   (abb.AbbIRB7600150350,           "/World/abb_irb7600_150_350", None, 1.0e7, 1.0e6, 145.0, "link_6"),
}


# ===========================================================================
# Scene prim paths — right-click any prim in the Stage panel → Copy Prim Path.
# ===========================================================================

STAND_PRIM_PATH:    str   = "/World/high_pedestal"
STAND_TOP_OFFSET_Z: float = 2.51   # pedestal origin is at its base; this is its height (m)

GRIPPER_PRIM_PATH: str = "/World/suction_gripper"
GRIPPER_BODY_PATH: str = "/World/suction_gripper"

# Orientation correction applied on top of the flange pose (XYZ Euler degrees).
# Adjust if the gripper face does not align with the expected approach direction.
GRIPPER_MOUNT_POS:       tuple = (0.0, 0.0, 0.0)
GRIPPER_MOUNT_ROT_EULER: tuple = (-90.0, -5.0, -90.0)


# ===========================================================================
# Helpers
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


def set_robot_drive_gains(
    stage,
    robot_prim_path: str,
    stiffness: float,
    damping: float,
) -> None:
    """Apply position-drive gains to every revolute joint under the robot root.

    Imported URDFs often default to soft drives that sag or oscillate under
    gravity. Must run **before** ``play()``; PhysX reads drive parameters
    only at simulation initialisation.

    Args:
        stage: Open USD stage.
        robot_prim_path: Root prim path of the robot articulation.
        stiffness: Position-drive stiffness in N·m/rad.
        damping: Position-drive damping in N·m·s/rad.
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


def setup_gripper_physics(stage) -> None:
    """Ensure the suction gripper body has a non-zero mass so the fixed joint holds.

    A rigid body with zero mass is treated as infinitely heavy (static) by PhysX,
    which prevents the joint from constraining it. Must run **before** ``play()``.

    Args:
        stage: Open USD stage.
    """
    prim = stage.GetPrimAtPath(GRIPPER_BODY_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"Gripper body prim {GRIPPER_BODY_PATH!r} not found.")
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(2.0)
    print(f"Gripper physics: mass=2.0 kg on {GRIPPER_BODY_PATH}.")


def position_robot_on_pedestal(
    stage,
    robot_prim_path: str,
    yaw_deg: float = 0.0,
) -> None:
    """Reposition the robot base onto the pedestal and apply yaw rotation.

    The URDF importer places the robot at the world origin. This function
    moves the base to the pedestal's world position and rotates it to face
    the work cell. Must run **before** ``play()``.

    Args:
        stage: Open USD stage.
        robot_prim_path: Root prim path of the robot articulation.
        yaw_deg: Rotation about world Z in degrees to face the work cell.

    Raises:
        RuntimeError: If the robot or pedestal prim is not found.
    """
    mount_xyz = _world_translation(stage, STAND_PRIM_PATH)
    mount_xyz[2] += STAND_TOP_OFFSET_Z

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(
            f"Robot prim {robot_prim_path!r} not found in the stage. "
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

    if yaw_deg:
        yaw = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg).GetQuat()
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
        f"(target {[round(float(v), 3) for v in mount_xyz]}, yaw={yaw_deg}°)"
    )


def attach_gripper_to_robot(
    stage,
    robot_prim_path: str,
    flange_link: str,
) -> None:
    """Weld the suction gripper to the robot's flange rigid body.

    Three steps are required:
    1. Disable any world-anchor joints on the gripper so it is free to follow the arm.
    2. Pre-position the gripper at the flange, preserving the gripper's own scale,
       so no snap impulse fires when the joint activates.
    3. Create a maximal fixed joint with local frames that satisfy the constraint
       at the current pose, then filter robot/gripper collision pairs.

    Must run **after** ``play()`` so world transforms are live.

    Args:
        stage: Open USD stage.
        robot_prim_path: Root prim path of the robot articulation.
        flange_link: Relative path from the robot root to the flange link
                     (e.g. ``"link_6/tool0"``).
    """
    flange_prim        = stage.GetPrimAtPath(f"{robot_prim_path}/{flange_link}")
    gripper_root       = stage.GetPrimAtPath(GRIPPER_PRIM_PATH)
    gripper_mount_prim = stage.GetPrimAtPath(GRIPPER_BODY_PATH)

    # Walk up from the flange to the nearest rigid-body ancestor.
    body0 = flange_prim
    while body0.IsValid() and body0.GetPath().pathString.startswith(robot_prim_path):
        if body0.HasAPI(UsdPhysics.RigidBodyAPI):
            break
        body0 = body0.GetParent()
    robot_mount = body0.GetPath().pathString

    # Disable world-anchor joints and kinematic flag on the gripper.
    for p in Usd.PrimRange(gripper_root):
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            if not j.GetBody0Rel().GetTargets() or not j.GetBody1Rel().GetTargets():
                j.CreateJointEnabledAttr().Set(False)
        if p.HasAttribute("physics:kinematicEnabled"):
            p.GetAttribute("physics:kinematicEnabled").Set(False)

    # Pre-position: adopt the flange pose while keeping the gripper's own scale
    # and applying the orientation correction from GRIPPER_MOUNT_ROT_EULER.
    flange_xf   = Gf.Transform(omni.usd.get_world_transform_matrix(body0))
    mount_scale = Gf.Transform(omni.usd.get_world_transform_matrix(gripper_mount_prim)).GetScale()
    ex, ey, ez  = GRIPPER_MOUNT_ROT_EULER
    correction  = (Gf.Rotation(Gf.Vec3d(1, 0, 0), ex)
                   * Gf.Rotation(Gf.Vec3d(0, 1, 0), ey)
                   * Gf.Rotation(Gf.Vec3d(0, 0, 1), ez))

    desired = Gf.Transform()
    desired.SetScale(mount_scale)
    desired.SetRotation(correction * flange_xf.GetRotation())
    desired.SetTranslation(flange_xf.GetTranslation() + Gf.Vec3d(*GRIPPER_MOUNT_POS))

    world_T_root   = omni.usd.get_world_transform_matrix(gripper_root)
    world_T_mount  = omni.usd.get_world_transform_matrix(gripper_mount_prim)
    world_T_parent = omni.usd.get_world_transform_matrix(gripper_root.GetParent())
    parent_T_root_new = (
        (world_T_mount * world_T_root.GetInverse()).GetInverse()
        * desired.GetMatrix()
        * world_T_parent.GetInverse()
    )
    UsdGeom.Xformable(gripper_root).MakeMatrixXform().Set(parent_T_root_new)

    # Maximal fixed joint with local frames computed from current poses — no snap.
    fixed = UsdPhysics.FixedJoint.Define(
        stage, Sdf.Path(GRIPPER_PRIM_PATH).AppendChild("SynapseFixedJoint").pathString
    )
    fixed.CreateBody0Rel().SetTargets([Sdf.Path(robot_mount)])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(GRIPPER_BODY_PATH)])
    cache = UsdGeom.XformCache()
    rel   = Gf.Transform(cache.GetLocalToWorldTransform(gripper_mount_prim)
                         * cache.GetLocalToWorldTransform(body0).GetInverse())
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(rel.GetTranslation()))
    fixed.CreateLocalRot0Attr().Set(Gf.Quatf(rel.GetRotation().GetQuat()))
    fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    fixed.CreateExcludeFromArticulationAttr().Set(True)

    UsdPhysics.FilteredPairsAPI.Apply(
        stage.GetPrimAtPath(robot_prim_path)
    ).CreateFilteredPairsRel().AddTarget(Sdf.Path(GRIPPER_PRIM_PATH))

    print(f"Gripper attached: {robot_mount} → {GRIPPER_BODY_PATH}.")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Run the automotive assembly demo for the robot selected by ``ACTIVE_ROBOT``."""
    robot_cls, robot_prim_path, home_q, stiffness, damping, yaw_deg, flange_link = (
        ROBOT_REGISTRY[ACTIVE_ROBOT]
    )
    print(f"Active robot : {ACTIVE_ROBOT} — {robot_cls.__name__} @ {robot_prim_path}")

    stage = omni.usd.get_context().get_stage()

    position_robot_on_pedestal(stage, robot_prim_path, yaw_deg)
    set_robot_drive_gains(stage, robot_prim_path, stiffness, damping)
    if flange_link is not None:
        setup_gripper_physics(stage)

    omni.timeline.get_timeline_interface().play()

    robot = robot_cls()
    robot.connect(simulation_prim_path=robot_prim_path)

    if home_q is None:
        home_q = robot.default_joint_configuration.tolist()

    try:
        robot.set_joint_positions(home_q)
        print("Robot at home:", [round(v, 1) for v in robot.state.joint_positions])

        if flange_link is not None:
            attach_gripper_to_robot(stage, robot_prim_path, flange_link)
            print("Gripper attached — ready.")

    finally:
        robot.disconnect()
        print("Done.")


if __name__ in ("__main__", "isaacsim.code_editor.vscode.extension"):
    main()
