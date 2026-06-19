# Isaac Sim / USD physics debug script
# Run in Isaac Sim through VS Code extension or Script Editor.

from pxr import Usd, UsdGeom, UsdPhysics, Sdf
import omni.usd


# -----------------------------
# EDIT THESE PATHS
# -----------------------------

ROBOT_ROOT = "/World/robot_assembly/kuka_kr120r2500pro"
GRIPPER_ROOT = "/World/robot_assembly/suction_gripper"

# Optional: set to "" if you do not know it.
EXPECTED_WRIST_LINK = ""  # example: "/World/high_pedestal_assembly/kuka_kr120r2500pro/kuka_kr120r2500pro/link_6"
EXPECTED_TOOL0 = ""       # example: ".../link_6/tool0"

# Print only under these roots. Add "/World" if you want everything.
SCAN_ROOTS = [
    ROBOT_ROOT,
    GRIPPER_ROOT,
]


# -----------------------------
# Helpers
# -----------------------------

stage = omni.usd.get_context().get_stage()

if stage is None:
    raise RuntimeError("No USD stage is open.")


def get_prim(path):
    if not path:
        return None
    prim = stage.GetPrimAtPath(path)
    return prim if prim and prim.IsValid() else None


def has_api(prim, api_cls):
    if not prim or not prim.IsValid():
        return False
    try:
        return prim.HasAPI(api_cls)
    except Exception:
        return False


def is_rigid_body(prim):
    return has_api(prim, UsdPhysics.RigidBodyAPI)


def is_collision(prim):
    return has_api(prim, UsdPhysics.CollisionAPI)


def has_mass_api(prim):
    return has_api(prim, UsdPhysics.MassAPI)


def get_rigid_enabled(prim):
    if not is_rigid_body(prim):
        return None
    api = UsdPhysics.RigidBodyAPI(prim)
    attr = api.GetRigidBodyEnabledAttr()
    value = attr.Get()
    # In many USD schemas, unauthored enabled attr behaves like true.
    return True if value is None else bool(value)


def get_mass_value(prim):
    if not has_mass_api(prim):
        return None
    api = UsdPhysics.MassAPI(prim)
    attr = api.GetMassAttr()
    return attr.Get()


def get_density_value(prim):
    if not has_mass_api(prim):
        return None
    api = UsdPhysics.MassAPI(prim)
    attr = api.GetDensityAttr()
    return attr.Get()


def get_xform_reset(prim):
    if not prim or not prim.IsValid():
        return None
    if not prim.IsA(UsdGeom.Xformable):
        return None
    try:
        return UsdGeom.Xformable(prim).GetResetXformStack()
    except Exception:
        return None


def get_type(prim):
    return prim.GetTypeName() if prim and prim.IsValid() else "<invalid>"


def path_exists(path):
    prim = get_prim(str(path))
    return prim is not None


def nearest_rigid_ancestor(prim):
    p = prim.GetParent()
    while p and p.IsValid() and p.GetPath() != Sdf.Path.absoluteRootPath:
        if is_rigid_body(p) and get_rigid_enabled(p):
            return p
        p = p.GetParent()
    return None


def all_rigid_ancestors(prim):
    result = []
    p = prim.GetParent()
    while p and p.IsValid() and p.GetPath() != Sdf.Path.absoluteRootPath:
        if is_rigid_body(p):
            result.append(p)
        p = p.GetParent()
    return result


def traverse_from(path):
    root = get_prim(path)
    if not root:
        print(f"[WARN] Root does not exist: {path}")
        return []
    return list(Usd.PrimRange(root))


def rel_targets(prim, rel_name):
    rel = prim.GetRelationship(rel_name)
    if not rel:
        return []
    return list(rel.GetTargets())


def is_joint_prim(prim):
    if not prim or not prim.IsValid():
        return False

    # Most USD physics joints derive from UsdPhysics.Joint.
    try:
        if prim.IsA(UsdPhysics.Joint):
            return True
    except Exception:
        pass

    # Fallback by type name.
    t = str(prim.GetTypeName())
    return "Joint" in t or t.startswith("Physics")


def is_fixed_joint(prim):
    try:
        return prim.IsA(UsdPhysics.FixedJoint)
    except Exception:
        return str(prim.GetTypeName()) == "PhysicsFixedJoint"


def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_sub(title):
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)


# -----------------------------
# Basic existence checks
# -----------------------------

print_header("USD PHYSICS DEBUG REPORT")

print(f"Stage root layer: {stage.GetRootLayer().identifier}")

for label, path in [
    ("ROBOT_ROOT", ROBOT_ROOT),
    ("GRIPPER_ROOT", GRIPPER_ROOT),
    ("EXPECTED_WRIST_LINK", EXPECTED_WRIST_LINK),
    ("EXPECTED_TOOL0", EXPECTED_TOOL0),
]:
    if not path:
        print(f"{label}: <not set>")
    else:
        prim = get_prim(path)
        print(f"{label}: {path}")
        print(f"  exists: {bool(prim)}")
        if prim:
            print(f"  type: {get_type(prim)}")
            print(f"  RigidBodyAPI: {is_rigid_body(prim)}")
            print(f"  CollisionAPI: {is_collision(prim)}")
            print(f"  MassAPI: {has_mass_api(prim)}")
            print(f"  resetXformStack: {get_xform_reset(prim)}")


# -----------------------------
# Search for useful named prims
# -----------------------------

print_sub("Named prim hints: link_6 / tool0 / flange / gripper")

name_keywords = ["link_6", "tool0", "flange", "gripper", "suction"]
for root_path in SCAN_ROOTS:
    for prim in traverse_from(root_path):
        name_lower = prim.GetName().lower()
        if any(k.lower() in name_lower for k in name_keywords):
            print(f"{prim.GetPath()} | type={get_type(prim)} | rigid={is_rigid_body(prim)} | collider={is_collision(prim)} | mass={has_mass_api(prim)}")


# -----------------------------
# Rigid bodies
# -----------------------------

print_sub("Rigid bodies found")

rigid_bodies = []

for root_path in SCAN_ROOTS:
    for prim in traverse_from(root_path):
        if is_rigid_body(prim):
            rigid_bodies.append(prim)
            ancestor = nearest_rigid_ancestor(prim)
            reset = get_xform_reset(prim)
            enabled = get_rigid_enabled(prim)
            mass_value = get_mass_value(prim)
            density_value = get_density_value(prim)

            print(f"\nRigid body: {prim.GetPath()}")
            print(f"  type: {get_type(prim)}")
            print(f"  enabled: {enabled}")
            print(f"  resetXformStack: {reset}")
            print(f"  MassAPI: {has_mass_api(prim)}")
            print(f"  mass: {mass_value}")
            print(f"  density: {density_value}")

            if ancestor:
                print(f"  [PROBLEM?] Has enabled rigid-body ancestor: {ancestor.GetPath()}")
                print("             This can cause nested rigid body / xformstack reset errors.")
            else:
                print("  rigid-body ancestor: none")

if not rigid_bodies:
    print("No rigid bodies found under SCAN_ROOTS.")


# -----------------------------
# Colliders
# -----------------------------

print_sub("Colliders found")

colliders = []

for root_path in SCAN_ROOTS:
    for prim in traverse_from(root_path):
        if is_collision(prim):
            colliders.append(prim)
            owner = prim
            p = prim
            nearest_body = None
            while p and p.IsValid() and p.GetPath() != Sdf.Path.absoluteRootPath:
                if is_rigid_body(p):
                    nearest_body = p
                    break
                p = p.GetParent()

            print(f"\nCollider: {prim.GetPath()}")
            print(f"  type: {get_type(prim)}")
            print(f"  nearest rigid-body owner: {nearest_body.GetPath() if nearest_body else '<none: static collider>'}")
            print(f"  MassAPI: {has_mass_api(prim)}")
            print(f"  mass: {get_mass_value(prim)}")
            print(f"  density: {get_density_value(prim)}")

if not colliders:
    print("No colliders found under SCAN_ROOTS.")


# -----------------------------
# Mass APIs
# -----------------------------

print_sub("MassAPI prims found")

mass_prims = []

for root_path in SCAN_ROOTS:
    for prim in traverse_from(root_path):
        if has_mass_api(prim):
            mass_prims.append(prim)
            print(f"\nMassAPI: {prim.GetPath()}")
            print(f"  type: {get_type(prim)}")
            print(f"  RigidBodyAPI: {is_rigid_body(prim)}")
            print(f"  CollisionAPI: {is_collision(prim)}")
            print(f"  mass: {get_mass_value(prim)}")
            print(f"  density: {get_density_value(prim)}")

if not mass_prims:
    print("No MassAPI prims found under SCAN_ROOTS.")


# -----------------------------
# Joints
# -----------------------------

print_sub("Physics joints found")

joints = []

# Search wider around /World because joints are often authored outside the robot/gripper root.
world = get_prim("/World")
joint_search_roots = ["/World"] if world else SCAN_ROOTS

for root_path in joint_search_roots:
    for prim in traverse_from(root_path):
        if is_joint_prim(prim):
            # Only print actual USD physics joint-like prims.
            body0_targets = rel_targets(prim, "physics:body0")
            body1_targets = rel_targets(prim, "physics:body1")

            if not body0_targets and not body1_targets and "Joint" not in str(get_type(prim)):
                continue

            joints.append(prim)

            print(f"\nJoint: {prim.GetPath()}")
            print(f"  type: {get_type(prim)}")
            print(f"  fixed joint: {is_fixed_joint(prim)}")

            for label, targets in [("Body0", body0_targets), ("Body1", body1_targets)]:
                if not targets:
                    print(f"  [PROBLEM] {label}: <EMPTY>")
                    continue

                for t in targets:
                    target_prim = get_prim(str(t))
                    print(f"  {label}: {t}")
                    print(f"    exists: {bool(target_prim)}")
                    if target_prim:
                        print(f"    type: {get_type(target_prim)}")
                        print(f"    RigidBodyAPI: {is_rigid_body(target_prim)}")
                        print(f"    RigidBody enabled: {get_rigid_enabled(target_prim)}")
                        print(f"    CollisionAPI: {is_collision(target_prim)}")
                        print(f"    MassAPI: {has_mass_api(target_prim)}")
                        if target_prim.GetName().lower() == "tool0":
                            print("    [LIKELY PROBLEM] This target is named tool0. tool0 is often only an Xform, not a rigid body.")
                        if not is_rigid_body(target_prim):
                            print("    [PROBLEM] Joint target is not a RigidBodyAPI prim.")

            # Print local joint frame attributes if authored.
            for attr_name in [
                "physics:localPos0",
                "physics:localRot0",
                "physics:localPos1",
                "physics:localRot1",
                "physics:jointEnabled",
                "physics:breakForce",
                "physics:breakTorque",
            ]:
                attr = prim.GetAttribute(attr_name)
                if attr and attr.HasAuthoredValueOpinion():
                    print(f"  {attr_name}: {attr.Get()}")

if not joints:
    print("No physics joints found under /World.")


# -----------------------------
# Specific diagnosis
# -----------------------------

print_sub("Likely gripper fixed-joint diagnosis")

gripper = get_prim(GRIPPER_ROOT)

if not gripper:
    print(f"[PROBLEM] GRIPPER_ROOT does not exist: {GRIPPER_ROOT}")
else:
    print(f"Gripper root: {gripper.GetPath()}")
    print(f"  type: {get_type(gripper)}")
    print(f"  RigidBodyAPI: {is_rigid_body(gripper)}")
    print(f"  CollisionAPI on root: {is_collision(gripper)}")
    print(f"  MassAPI on root: {has_mass_api(gripper)}")
    print(f"  mass on root: {get_mass_value(gripper)}")

    ancestor = nearest_rigid_ancestor(gripper)
    if ancestor:
        print(f"  [PROBLEM] Gripper is nested below enabled rigid body: {ancestor.GetPath()}")
        print("            For fixed-joint method, gripper should usually be outside the robot articulation hierarchy.")
    else:
        print("  Gripper has no rigid-body ancestor: OK for separate fixed-joint method.")

    # Find joints referencing the gripper.
    referencing = []
    for j in joints:
        b0 = [str(x) for x in rel_targets(j, "physics:body0")]
        b1 = [str(x) for x in rel_targets(j, "physics:body1")]
        if str(gripper.GetPath()) in b0 or str(gripper.GetPath()) in b1:
            referencing.append(j)

    if not referencing:
        print("  [PROBLEM] No joint has Body0/Body1 directly targeting the gripper root.")
        print("            If the gripper drops, this is very likely the reason.")
    else:
        print("  Joints directly targeting gripper root:")
        for j in referencing:
            print(f"    {j.GetPath()} | type={get_type(j)}")


# -----------------------------
# Summary recommendations
# -----------------------------

print_sub("What to look for in the output")

print("""
For a separate rigid gripper fixed-jointed to the robot, you usually want:

1. Gripper root:
   RigidBodyAPI = True
   MassAPI = True
   Collider = on root or children
   No enabled rigid-body ancestor

2. Fixed joint:
   Body0 = actual robot rigid link, usually link_6
   Body1 = gripper rigid body root
   Neither Body0 nor Body1 should be EMPTY
   Body0 should not be tool0 unless tool0 actually has RigidBodyAPI

3. Bad signs:
   - Body0: <EMPTY>
   - Body1: <EMPTY>
   - Body0 or Body1 points to tool0 with RigidBodyAPI=False
   - Gripper is inside /link_6/tool0 while it also has RigidBodyAPI
   - Gripper visual child has RigidBodyAPI but gripper root does not
   - Multiple nested rigid bodies without resetXformStack
""")

print("\nDONE.")