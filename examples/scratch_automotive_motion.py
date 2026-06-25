"""
Automotive assembly pick-and-carry — Isaac Sim + Telekinesis Synapse.

Extends the assembly demo with a joint-space pick-and-carry sequence:
the robot reaches the roof piece, grips it with a simulated suction weld,
lifts it, and carries it over the vehicle body.

Workflow (complete in Isaac Sim before running):
  1. Open the assembly scene USD.
  2. Import the robot URDF via the URDF Importer (Fix Base = ON).
  3. Jog the arm to each waypoint pose, read ``robot.state.joint_positions``,
     and paste the values into ``ROOF_PICK_Q``, ``ROOF_LIFT_Q``, ``ROOF_CARRY_Q``.
  4. Set ``ACTIVE_ROBOT`` and run via the Isaac Sim VS Code Extension.

You manage the timeline: ``main()`` plays it before connecting to any
articulation.
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
#   stiffness    — PhysX position-drive stiffness (N·m/rad).
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
STAND_TOP_OFFSET_Z: float = 2.51

GRIPPER_PRIM_PATH: str = "/World/suction_gripper"
GRIPPER_BODY_PATH: str = "/World/suction_gripper/suction_gripper/A6"

GRIPPER_MOUNT_POS:       tuple = (0.0, 0.0, 0.0)
GRIPPER_MOUNT_ROT_EULER: tuple = (-90.0, -5.0, -90.0)

ROOF_ROOT_PATH: str = "/World/RoofPiece"
SLEDGE_PATH:    str = "/World/sledge"

# Joint-space waypoints (degrees). Jog the arm in the GUI, read
# robot.state.joint_positions, and paste the values here.
ROOF_PICK_Q:  list = [55.2,  28.5,  19.3, -1.3,  42.3,  19.8]
ROOF_LIFT_Q:  list = [24.1,  34.6, -31.5, -8.4,  89.3,  66.8]
ROOF_CARRY_Q: list = [ 9.2,  44.6, -31.5, -3.7,  73.3,  69.1]
MOVE_SPEED_DEG: float = 20.0


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


def _resolve_rigid_body(stage, root_path: str):
    """Return the first prim under *root_path* that has RigidBodyAPI.

    Args:
        stage: Open USD stage.
        root_path: Root prim path to search under.

    Raises:
        RuntimeError: If no rigid body is found.
    """
    root = stage.GetPrimAtPath(root_path)
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
    raise RuntimeError(f"No RigidBodyAPI found under {root_path!r}.")


def set_robot_drive_gains(
    stage,
    robot_prim_path: str,
    stiffness: float,
    damping: float,
) -> None:
    """Apply position-drive gains to every revolute joint under the robot root.

    Must run **before** ``play()``.

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

    Must run **before** ``play()``.

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


def setup_roofpiece_physics(stage) -> None:
    """Set mass on the roof piece and pin it kinematic until gripped.

    Must run **before** ``play()``.

    Args:
        stage: Open USD stage.
    """
    body = _resolve_rigid_body(stage, ROOF_ROOT_PATH)
    if not body.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI.Apply(body)
    UsdPhysics.MassAPI(body).CreateMassAttr().Set(5.0)
    body.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    print(f"RoofPiece physics: mass=5.0 kg, kinematic on {body.GetPath()}.")


def position_robot_on_pedestal(
    stage,
    robot_prim_path: str,
    yaw_deg: float = 0.0,
) -> None:
    """Reposition the robot base onto the pedestal and apply yaw rotation.

    Must run **before** ``play()``.

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

    Must run **after** ``play()`` so world transforms are live.

    Args:
        stage: Open USD stage.
        robot_prim_path: Root prim path of the robot articulation.
        flange_link: Relative path from the robot root to the flange link.
    """
    flange_prim        = stage.GetPrimAtPath(f"{robot_prim_path}/{flange_link}")
    gripper_root       = stage.GetPrimAtPath(GRIPPER_PRIM_PATH)
    gripper_mount_prim = stage.GetPrimAtPath(GRIPPER_BODY_PATH)

    body0 = flange_prim
    while body0.IsValid() and body0.GetPath().pathString.startswith(robot_prim_path):
        if body0.HasAPI(UsdPhysics.RigidBodyAPI):
            break
        body0 = body0.GetParent()
    robot_mount = body0.GetPath().pathString

    for p in Usd.PrimRange(gripper_root):
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            if not j.GetBody0Rel().GetTargets() or not j.GetBody1Rel().GetTargets():
                j.CreateJointEnabledAttr().Set(False)
        if p.HasAttribute("physics:kinematicEnabled"):
            p.GetAttribute("physics:kinematicEnabled").Set(False)

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


def weld_bodies(stage, body0_path: str, body1_path: str, joint_name: str) -> None:
    """Create a maximal fixed joint between two rigid bodies with no snap impulse.

    Local frames are computed from current world poses so the constraint is
    satisfied at the moment of creation.

    Args:
        stage: Open USD stage.
        body0_path: USD path of the first rigid body.
        body1_path: USD path of the second rigid body.
        joint_name: Name for the joint prim (created under body0).
    """
    cache = UsdGeom.XformCache()
    t0  = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body0_path))
    t1  = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body1_path))
    rel = Gf.Transform(t1 * t0.GetInverse())

    fixed = UsdPhysics.FixedJoint.Define(
        stage, Sdf.Path(body0_path).AppendChild(joint_name).pathString
    )
    fixed.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(rel.GetTranslation()))
    fixed.CreateLocalRot0Attr().Set(Gf.Quatf(rel.GetRotation().GetQuat()))
    fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    fixed.CreateExcludeFromArticulationAttr().Set(True)
    print(f"Welded {body0_path} ↔ {body1_path}.")


def pick_and_carry_roof(robot, stage) -> None:
    """Reach the roof piece, grip it, lift, and carry it over the vehicle body.

    Uses pure joint-space moves (``set_joint_positions`` → ``move_j``) so no
    IK is involved and all waypoints are guaranteed to execute. Tune
    ``ROOF_PICK_Q``, ``ROOF_LIFT_Q``, and ``ROOF_CARRY_Q`` by jogging the arm
    in the GUI and reading ``robot.state.joint_positions``.

    Args:
        robot: Connected Synapse manipulator instance.
        stage: Open USD stage.
    """
    roof_body = _resolve_rigid_body(stage, ROOF_ROOT_PATH)
    roof_path = roof_body.GetPath().pathString

    robot.set_joint_positions(ROOF_PICK_Q, speed=MOVE_SPEED_DEG)

    if roof_body.HasAttribute("physics:kinematicEnabled"):
        roof_body.GetAttribute("physics:kinematicEnabled").Set(False)
    weld_bodies(stage, GRIPPER_BODY_PATH, roof_path, "RoofGripJoint")
    UsdPhysics.FilteredPairsAPI.Apply(
        stage.GetPrimAtPath(GRIPPER_PRIM_PATH)
    ).CreateFilteredPairsRel().AddTarget(Sdf.Path(ROOF_ROOT_PATH))
    print("Roof gripped.")

    robot.set_joint_positions(ROOF_LIFT_Q,  speed=MOVE_SPEED_DEG)
    robot.set_joint_positions(ROOF_CARRY_Q, speed=MOVE_SPEED_DEG)
    print("Roof carried over vehicle body.")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Run the automotive assembly pick-and-carry for the robot in ``ACTIVE_ROBOT``."""
    robot_cls, robot_prim_path, home_q, stiffness, damping, yaw_deg, flange_link = (
        ROBOT_REGISTRY[ACTIVE_ROBOT]
    )
    print(f"Active robot : {ACTIVE_ROBOT} — {robot_cls.__name__} @ {robot_prim_path}")

    stage = omni.usd.get_context().get_stage()

    position_robot_on_pedestal(stage, robot_prim_path, yaw_deg)
    set_robot_drive_gains(stage, robot_prim_path, stiffness, damping)
    if flange_link is not None:
        setup_gripper_physics(stage)
        setup_roofpiece_physics(stage)

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
            pick_and_carry_roof(robot, stage)

    finally:
        robot.disconnect()
        print("Done.")


if __name__ in ("__main__", "isaacsim.code_editor.vscode.extension"):
    main()
