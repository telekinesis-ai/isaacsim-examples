"""
All-robots automotive assembly demo — Isaac Sim + Telekinesis Synapse.

Spawns 9 industrial manipulators on the automotive factory pedestals.
ABB IRB7600 with suction gripper on the RobotController pedestal.
Heavy industrial arms (Kuka, ABB, Neura) on the remaining 8 pedestals.
9 Unitree G1 humanoids and 11 Idealworks iw.hub robots placed around the factory floor.

Toggle VISUAL_MODE at the top of this file to switch between two modes:

  VISUAL_MODE = False  (default — physics simulation)
    Import robots via Isaac Utils → URDF Importer (Fix Base = ON).
    Physics articulation, joint drives, and gripper weld all work.
    Use the bundled Kuka, ABB, and Unitree URDFs under assets/robots/.
    Import Neura MAiRA from telekinesis-urdfs.

  VISUAL_MODE = True  (static scene — full colors, no physics)
    Add the Kuka, ABB, and Unitree top-level USDA files from their
    assets/robots/<robot>/usd/ directories. Colors are correct
    (6.0.1-generated), Physics defaults to none, and Play is not called.
    Gripper is positioned visually only — no physics joint is created.
    Neura MAiRA has no visual USD; import via URDF Importer regardless of mode.

Workflow:
  1. Open assets/environments/all_robots_automotive/automotive_warehouse.usd
  2. Add ONE base prim per brand (drag USD or import URDF) at the paths in ARM_BASE_PRIMS.
     Add ONE G1 humanoid at HUMANOID_BASE_PATH and ONE iw.hub at IWHUB_BASE_PATH.
     The script duplicates all remaining instances automatically.
  3. Run via the Isaac Sim VS Code Extension or the Kit Script Editor.
"""

from __future__ import annotations

import omni.timeline
import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

# ---------------------------------------------------------------------------
# Mode toggle — see module docstring for full explanation.
# ---------------------------------------------------------------------------
VISUAL_MODE: bool = False

from telekinesis.synapse.robots.manipulators import (
    abb,
    kuka,
    neura_robotics,
)


# ===========================================================================
# Scene pedestal Link1 paths — mount points where robot bases are placed.
# ===========================================================================
_ROOT = "/World/Factory/Samples/Welding_Assembly_Animated_Adjust/Welding_Assembly_Animated_Export/root"

PEDESTAL_LINK1 = {
    "p11":        f"{_ROOT}/RobotPedestal_U20__U23_11/Link1",
    "p10":        f"{_ROOT}/RobotPedestal_U20__U23_10/Link1",
    "p9":         f"{_ROOT}/RobotPedestal_U20__U23_9/Link1",
    "p4":         f"{_ROOT}/RobotPedestal_U20__U23_4/Link1",
    "p3":         f"{_ROOT}/RobotPedestal_U20__U23_3/Link1",
    "p2":         f"{_ROOT}/RobotPedestal_U20__U23_2/Link1",
    "p7":         f"{_ROOT}/RobotPedestal_U20__U23_7/Link1",
    "p_base":     f"{_ROOT}/RobotPedestal/Link1",
    "controller": f"{_ROOT}/RobotController/Link1",
}


# ===========================================================================
# Gripper config — Kuka on RobotController pedestal only.
# ===========================================================================
GRIPPER_PRIM_PATH:     str   = "/World/suction_gripper"
GRIPPER_BODY_PATH:     str   = "/World/suction_gripper/suction_gripper/A6"
GRIPPER_MOUNT_POS:     tuple = (0.0, 0.0, 0.0)
GRIPPER_MOUNT_ROT_EULER: tuple = (-90.0, -5.0, -90.0)


# ===========================================================================
# Arm robot base prims — import exactly ONE of each brand via URDF Importer
# (Fix Base = ON). The script duplicates the remaining instances automatically.
# ===========================================================================
ARM_BASE_PRIMS: dict = {
    kuka.KukaKR210L150:                  "/World/kuka_kr210",
    abb.AbbIRB7600150350:                "/World/abb_irb7600_150_350",
    neura_robotics.NeuraRoboticsMAiRA7M: "/World/maira7M",
}


# ===========================================================================
# Robot registry — one entry per pedestal.
#
# Keys in PEDESTAL_LINK1 above map 1-to-1 to keys here.
#
# Each entry: (class, prim_path, home_q, stiffness, damping, yaw_deg, flange_link)
#
#   class        — Synapse manipulator class.
#   prim_path    — USD prim path. Base prims must match ARM_BASE_PRIMS above.
#                  Copy prims are created automatically by spawn_arm_robots().
#   home_q       — Joint positions in degrees. None → default configuration.
#   stiffness    — PhysX drive stiffness (N·m/rad).
#   damping      — PhysX drive damping (N·m·s/rad).
#   yaw_deg      — Rotation about world Z (degrees) to face the work cell.
#   flange_link  — Relative path from robot root to the flange link for gripper
#                  attachment. None → no gripper.
# ===========================================================================
ROBOT_REGISTRY: dict = {
    #  pedestal key      class                              prim_path                          home_q  stiff    damp     yaw    flange
    "p11":        (kuka.KukaKR210L150,                  "/World/kuka_kr210",                  None, 1.0e6, 1.0e5,  164.0, None),
    "p10":        (kuka.KukaKR210L150,                  "/World/kuka_kr210_01",               None, 1.0e6, 1.0e5,  118.0, None),
    "p9":         (abb.AbbIRB7600150350,                "/World/abb_irb7600_150_350",          None, 1.0e7, 1.0e6,    0.0, None),
    "p4":         (abb.AbbIRB7600150350,                "/World/abb_irb7600_150_350_01",       None, 1.0e7, 1.0e6,   42.0, None),
    "p3":         (neura_robotics.NeuraRoboticsMAiRA7M, "/World/maira7M",                     None, 1.0e6, 1.0e5,    0.0, None),
    "p2":         (neura_robotics.NeuraRoboticsMAiRA7M, "/World/maira7M_01",                  None, 1.0e6, 1.0e5,    0.0, None),
    "p7":         (kuka.KukaKR210L150,                  "/World/kuka_kr210_02",               None, 1.0e6, 1.0e5, -148.0, None),
    "p_base":     (kuka.KukaKR210L150,                  "/World/kuka_kr210_03",               None, 1.0e6, 1.0e5,  174.0, None),
    "controller": (abb.AbbIRB7600150350,                "/World/abb_irb7600_150_350_02",       None, 1.0e7, 1.0e6,  145.0, "link_6"),
}


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
    print(f"  Drive gains set on {count} joint(s) "
          f"(stiffness={stiffness:g}, damping={damping:g}).")


def setup_gripper_physics(stage) -> None:
    """Ensure the suction gripper body has a non-zero mass so the fixed joint holds.

    Args:
        stage: Open USD stage.
    """
    prim = stage.GetPrimAtPath(GRIPPER_BODY_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"Gripper body prim {GRIPPER_BODY_PATH!r} not found.")
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(2.0)
    print(f"  Gripper physics: mass=2.0 kg on {GRIPPER_BODY_PATH}.")


def position_robot_on_pedestal(
    stage,
    robot_prim_path: str,
    link1_path: str,
    yaw_deg: float = 0.0,
) -> None:
    """Reposition the robot base onto the pedestal Link1 mount point.

    Args:
        stage: Open USD stage.
        robot_prim_path: Root prim path of the robot articulation.
        link1_path: World path of the pedestal's Link1 Xform (mount point).
        yaw_deg: Rotation about world Z in degrees to face the work cell.

    Raises:
        RuntimeError: If the robot or Link1 prim is not found.
    """
    mount_xyz = _world_translation(stage, link1_path)

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(
            f"Robot prim {robot_prim_path!r} not found. "
            "Import the URDF first (URDF Importer → Fix Base ON) and update ROBOT_REGISTRY."
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
        f"  Robot base → {[round(float(v), 3) for v in actual]}  "
        f"(target {[round(float(v), 3) for v in mount_xyz]}, yaw={yaw_deg}°)"
    )


def attach_gripper_to_robot(
    stage,
    robot_prim_path: str,
    flange_link: str,
) -> None:
    """Weld the suction gripper to the robot's flange rigid body.

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

    print(f"  Gripper attached: {robot_mount} → {GRIPPER_BODY_PATH}.")


# ===========================================================================
# Visual-mode gripper helpers.
# ===========================================================================

def _find_visual_flange(stage, robot_prim_path: str):
    """Return the first 'flange' or 'tool0' prim found under the robot root.

    Visual USDs nest links inside Geometry/base_link/... rather than exposing
    them as direct children of the articulation root. This search finds the
    correct mount prim regardless of nesting depth.

    Args:
        stage: Open USD stage.
        robot_prim_path: Root prim path of the robot.

    Returns:
        The flange/tool0 Usd.Prim, or None if not found.
    """
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root.IsValid():
        return None
    for name in ("flange", "tool0"):
        for prim in Usd.PrimRange(root):
            if prim.GetName() == name:
                return prim
    return None


def attach_gripper_visually(stage, robot_prim_path: str) -> None:
    """Move the suction gripper to the robot flange transform without physics.

    Locates the flange (or tool0) prim in the visual USD hierarchy and applies
    its world transform — plus the mount correction — to the gripper root.
    No mass, FixedJoint, or collision filtering is created.

    Args:
        stage: Open USD stage.
        robot_prim_path: Root prim path of the robot.
    """
    flange_prim = _find_visual_flange(stage, robot_prim_path)
    if flange_prim is None:
        print(f"  [SKIP] No flange or tool0 found under {robot_prim_path}.")
        return

    gripper_root       = stage.GetPrimAtPath(GRIPPER_PRIM_PATH)
    gripper_mount_prim = stage.GetPrimAtPath(GRIPPER_BODY_PATH)

    if not gripper_root.IsValid():
        raise RuntimeError(f"Gripper root {GRIPPER_PRIM_PATH!r} not found.")
    if not gripper_mount_prim.IsValid():
        raise RuntimeError(f"Gripper mount {GRIPPER_BODY_PATH!r} not found.")

    ex, ey, ez  = GRIPPER_MOUNT_ROT_EULER
    correction  = (Gf.Rotation(Gf.Vec3d(1, 0, 0), ex)
                   * Gf.Rotation(Gf.Vec3d(0, 1, 0), ey)
                   * Gf.Rotation(Gf.Vec3d(0, 0, 1), ez))

    flange_xf   = Gf.Transform(omni.usd.get_world_transform_matrix(flange_prim))
    mount_scale = Gf.Transform(
        omni.usd.get_world_transform_matrix(gripper_mount_prim)
    ).GetScale()

    desired = Gf.Transform()
    desired.SetScale(mount_scale)
    desired.SetRotation(correction * flange_xf.GetRotation())
    desired.SetTranslation(flange_xf.GetTranslation() + Gf.Vec3d(*GRIPPER_MOUNT_POS))

    # Align the gripper's actual mount body with the flange.  Aligning the
    # gripper root directly loses the root-to-mount offset and can move the
    # visible geometry far away from the robot.
    world_T_root = omni.usd.get_world_transform_matrix(gripper_root)
    world_T_mount = omni.usd.get_world_transform_matrix(gripper_mount_prim)
    world_T_parent = omni.usd.get_world_transform_matrix(gripper_root.GetParent())
    parent_T_root_new = (
        (world_T_mount * world_T_root.GetInverse()).GetInverse()
        * desired.GetMatrix()
        * world_T_parent.GetInverse()
    )
    UsdGeom.Xformable(gripper_root).MakeMatrixXform().Set(parent_T_root_new)

    print(f"  Gripper positioned visually at {flange_prim.GetPath()}.")


# ===========================================================================
# Arm robot spawning — duplicates base prims into all registry positions.
# ===========================================================================

def spawn_arm_robots(stage) -> None:
    """Duplicate each arm base prim into all registry positions for that brand.

    Args:
        stage: Open USD stage.

    Raises:
        RuntimeError: If a base prim from ARM_BASE_PRIMS is not in the stage.
    """
    import omni.kit.commands

    for robot_cls, base_path in ARM_BASE_PRIMS.items():
        if not stage.GetPrimAtPath(base_path).IsValid():
            raise RuntimeError(
                f"Base robot prim {base_path!r} not found. "
                f"Import one {robot_cls.__name__} via URDF Importer "
                f"(Fix Base ON) and confirm its prim path matches ARM_BASE_PRIMS."
            )

    print("\n=== Spawning Arm Robots ===")
    for _, entry in ROBOT_REGISTRY.items():
        robot_cls, prim_path, *_ = entry
        base_path = ARM_BASE_PRIMS[robot_cls]
        if prim_path == base_path:
            print(f"  [BASE] {prim_path}")
            continue
        if not stage.GetPrimAtPath(prim_path).IsValid():
            omni.kit.commands.execute(
                "CopyPrimCommand",
                path_from=base_path,
                path_to=prim_path,
            )
            print(f"  Duplicated {base_path} → {prim_path}")
        else:
            print(f"  [EXISTS] {prim_path}")


# ===========================================================================
# Humanoid config — import ONE g1_29dof_with_hand via URDF Importer first.
# Script duplicates it and places all 9 at the positions below.
# ===========================================================================
HUMANOID_BASE_PATH = "/World/g1_29dof_with_hand"

HUMANOID_CONFIGS = [
    # (prim_path,                          pos_xyz,                     scale, yaw_deg)
    ("/World/g1_29dof_with_hand",    (-1.790,  7.608, 1.113), 1.4,  -28.0),
    ("/World/g1_29dof_with_hand_01", (-4.268,  2.278, 1.113), 1.4,    0.0),
    ("/World/g1_29dof_with_hand_02", (-10.043,-2.771, 1.113), 1.4,   85.0),
    ("/World/g1_29dof_with_hand_03", (-1.916, -9.250, 1.113), 1.4,   39.0),
    ("/World/g1_29dof_with_hand_04", (5.011,   6.740, 1.113), 1.4,  145.0),
    ("/World/g1_29dof_with_hand_05", (8.970,   4.049, 1.113), 1.4,  -91.0),
    ("/World/g1_29dof_with_hand_06", (9.825,   9.763, 1.113), 1.4, -143.0),
    ("/World/g1_29dof_with_hand_07", (4.585,  14.739, 1.113), 1.4, -105.0),
    ("/World/g1_29dof_with_hand_08", (6.690,  -3.535, 1.113), 1.4,  140.0),
]


def spawn_humanoids(stage) -> None:
    """Duplicate the base humanoid prim and position all 9 instances.

    Args:
        stage: Open USD stage.
    """
    import omni.kit.commands

    base_prim = stage.GetPrimAtPath(HUMANOID_BASE_PATH)
    if not base_prim.IsValid():
        raise RuntimeError(
            f"Base humanoid {HUMANOID_BASE_PATH!r} not found. "
            "Import the g1_29dof_with_hand URDF first (Fixed Base ON)."
        )

    # Duplicate base into each destination path (skip base itself)
    for prim_path, _, _, _ in HUMANOID_CONFIGS[1:]:
        if not stage.GetPrimAtPath(prim_path).IsValid():
            omni.kit.commands.execute(
                "CopyPrimCommand",
                path_from=HUMANOID_BASE_PATH,
                path_to=prim_path,
            )
            print(f"  Duplicated → {prim_path}")

    # Position, scale, and orient all 9
    for prim_path, pos, scale, yaw_deg in HUMANOID_CONFIGS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"  [SKIP] {prim_path} not found")
            continue

        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()

        xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
        yaw_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg).GetQuat()
        xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(yaw_quat))
        xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(scale, scale, scale))

        print(f"  Placed {prim_path} → pos={pos}, yaw={yaw_deg}°, scale={scale}")


# ===========================================================================
# Idealworks iw.hub config - add ONE base iw.hub prim to the stage first.
# Script duplicates it and places all 11 at the positions below.
# ===========================================================================
IWHUB_BASE_PATH = "/World/iw_hub"

IWHUB_CONFIGS = [
    # (prim_path,             pos_xyz,                    scale, yaw_deg)
    ("/World/iw_hub",    (-3.550,   8.940, 0.087), 1.0,   0.0),
    ("/World/iw_hub_01", (-6.323,   6.269, 0.087), 1.0,  90.0),
    ("/World/iw_hub_02", (-6.323,   1.721, 0.087), 1.0,  90.0),
    ("/World/iw_hub_03", (-6.323,  -4.299, 0.087), 1.0,  90.0),
    ("/World/iw_hub_04", (-6.323,  -9.957, 0.087), 1.0,  90.0),
    ("/World/iw_hub_05", (-1.529, -13.000, 0.087), 1.0, 180.0),
    ("/World/iw_hub_06", (5.571,    8.940, 0.087), 1.0,   0.0),
    ("/World/iw_hub_07", (7.991,    7.014, 0.087), 1.0,  90.0),
    ("/World/iw_hub_08", (7.991,    0.695, 0.087), 1.0,  90.0),
    ("/World/iw_hub_09", (7.991,   -4.442, 0.087), 1.0,  90.0),
    ("/World/iw_hub_10", (5.631,   12.438, 0.087), 1.0,  90.0),
]


def spawn_iwhubs(stage) -> None:
    """Duplicate the base iw.hub prim and position all 11 instances.

    Args:
        stage: Open USD stage.
    """
    import omni.kit.commands

    base_prim = stage.GetPrimAtPath(IWHUB_BASE_PATH)
    if not base_prim.IsValid():
        raise RuntimeError(
            f"Base iw.hub {IWHUB_BASE_PATH!r} not found. "
            "Add or reference one iw.hub asset at that prim path first."
        )

    for prim_path, _, _, _ in IWHUB_CONFIGS[1:]:
        if not stage.GetPrimAtPath(prim_path).IsValid():
            omni.kit.commands.execute(
                "CopyPrimCommand",
                path_from=IWHUB_BASE_PATH,
                path_to=prim_path,
            )
            print(f"  Duplicated -> {prim_path}")

    for prim_path, pos, scale, yaw_deg in IWHUB_CONFIGS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"  [SKIP] {prim_path} not found")
            continue

        xformable = UsdGeom.Xformable(prim)
        desired = Gf.Transform()
        desired.SetTranslation(Gf.Vec3d(pos[0], pos[1], pos[2]))
        desired.SetRotation(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg))
        desired.SetScale(Gf.Vec3d(scale, scale, scale))
        xformable.MakeMatrixXform().Set(desired.GetMatrix())

        print(f"  Placed {prim_path} -> pos={pos}, yaw={yaw_deg} deg, scale={scale}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Spawn all robots onto their pedestals and attach the gripper.

    Behaviour is controlled by VISUAL_MODE at the top of this file.
    """
    stage = omni.usd.get_context().get_stage()

    # --- Duplicate arm robots from base prims ---
    spawn_arm_robots(stage)

    # --- Duplicate and place humanoids ---
    print("\n=== Spawning Humanoids ===")
    spawn_humanoids(stage)

    # --- Duplicate and place Idealworks iw.hub robots ---
    print("\n=== Spawning Idealworks iw.hub Robots ===")
    spawn_iwhubs(stage)

    # --- Position all arm robots; set drive gains in physics mode only ---
    for pedestal_key, entry in ROBOT_REGISTRY.items():
        robot_cls, robot_prim_path, _home_q, stiffness, damping, yaw_deg, flange_link = entry
        link1_path = PEDESTAL_LINK1[pedestal_key]
        print(f"\n[{pedestal_key}] {robot_cls.__name__} @ {robot_prim_path}")
        position_robot_on_pedestal(stage, robot_prim_path, link1_path, yaw_deg)
        if not VISUAL_MODE:
            set_robot_drive_gains(stage, robot_prim_path, stiffness, damping)

    if VISUAL_MODE:
        # Static scene: position gripper visually, skip Play entirely.
        for pedestal_key, entry in ROBOT_REGISTRY.items():
            robot_cls, robot_prim_path, *_, flange_link = entry
            if flange_link is not None:
                print(f"\n[{pedestal_key}] Positioning gripper visually on {robot_cls.__name__}...")
                attach_gripper_visually(stage, robot_prim_path)
        print("\nAll robots positioned (visual mode — Play skipped).")
    else:
        # Physics mode: set up gripper mass, start simulation, then weld.
        for pedestal_key, entry in ROBOT_REGISTRY.items():
            _, robot_prim_path, *_, flange_link = entry
            if flange_link is not None:
                setup_gripper_physics(stage)

        omni.timeline.get_timeline_interface().play()

        for pedestal_key, entry in ROBOT_REGISTRY.items():
            robot_cls, robot_prim_path, *_, flange_link = entry
            if flange_link is not None:
                print(f"\n[{pedestal_key}] Attaching gripper to {robot_cls.__name__}...")
                robot = robot_cls()
                robot.connect(simulation_prim_path=robot_prim_path)
                try:
                    attach_gripper_to_robot(stage, robot_prim_path, flange_link)
                finally:
                    robot.disconnect()

        print("\nAll robots positioned. Gripper attached.")


if __name__ in ("__main__", "isaacsim.code_editor.vscode.extension"):
    main()
