"""
Manipulator — extension-mode URDF load & movement demo (manufacturer-agnostic).

Imports an arm from a local URDF file, starts the simulation, and moves every
joint to a target pose. Nothing is hardcoded to a specific robot: the joint names
are read from the imported articulation (``dof_names``) and the target is sized
to the articulation's DOF count, so the same script works for any serial
manipulator — just change ``MANIPULATOR_URDF``.

Run inside a live Isaac Sim session via the VS Code extension (not standalone).
The stage should be empty before running.
"""

import asyncio
import pathlib

import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdLux

_ROOT = pathlib.Path(r'D:\Telekinesis\Code\isaacsim-examples\assets').parent
_MODEL_DIR = _ROOT / "assets"

# Adjust this path to the manipulator URDF you want to load.
MANIPULATOR_URDF = (
    _MODEL_DIR
    / "example-robot-data"
    / "robots"
    / "kuka"
    / "kr120_description"
    / "urdf"
    / "kr120r2500pro.urdf"
)

# Where to place the imported manipulator in the stage.
MANIPULATOR_PRIM_PATH = "/World/manipulator"

# Target angle applied to every joint (degrees). Generic so it works regardless
# of how many joints the manipulator has.
TARGET_DEG = 30.0
WAIT_STEPS = 120


def import_urdf_at(stage, urdf_path, dest_prim_path, import_config):
    """Import ``urdf_path`` and place it exactly at ``dest_prim_path``.

    The URDF importer parents the import under the default prim (or stage root),
    so we import then move the result to the requested path. Returns the new
    articulation-root path.
    """
    parent = Sdf.Path(dest_prim_path).GetParentPath()
    if parent.pathString not in ("", "/") and not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, parent.pathString)

    _, imported = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        get_articulation_root=True,
    )

    # Climb to the top-level imported prim (importer may return a nested root).
    top = Sdf.Path(imported)
    while top.GetParentPath() not in (parent, Sdf.Path("/"), Sdf.Path()):
        top = top.GetParentPath()

    if top.pathString != dest_prim_path:
        omni.kit.commands.execute(
            "MovePrim", path_from=top.pathString, path_to=dest_prim_path
        )

    # Articulation root after the move (preserve any nested suffix).
    return dest_prim_path + imported[len(top.pathString):]


async def main():
    app = omni.kit.app.get_app()
    stage = omni.usd.get_context().get_stage()

    # !! Very important: timeline has to be stopped when importing
    omni.timeline.get_timeline_interface().stop()

    if not stage.GetPrimAtPath("/groundPlane").IsValid():
        PhysicsSchemaTools.addGroundPlane(
            stage, "/groundPlane", "Z", 15, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.7)
        )
    if not stage.GetPrimAtPath("/DistantLight").IsValid():
        distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        distant_light.CreateIntensityAttr(1000)

    if not MANIPULATOR_URDF.is_file():
        raise FileNotFoundError(f"Cannot find manipulator URDF: {MANIPULATOR_URDF}")

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.distance_scale = 1.0

    manipulator_prim_path = import_urdf_at(
        stage, MANIPULATOR_URDF, MANIPULATOR_PRIM_PATH, import_config
    )
    print(f"Imported manipulator at: {manipulator_prim_path}")

    await app.next_update_async()
    await app.next_update_async()

    omni.timeline.get_timeline_interface().play()

    robot = None
    for _ in range(60):
        try:
            art = SingleArticulation(prim_path=manipulator_prim_path, name="manipulator")
            art.initialize()
            if art.num_dof and art.num_dof > 0:
                robot = art
                break
        except Exception:
            pass
        await app.next_update_async()

    if robot is None:
        raise RuntimeError("Manipulator articulation did not become valid.")

    print(f"num_dof: {robot.num_dof}")
    print(f"dof_names: {robot.dof_names}")

    # Drive every joint of the articulation (joint names come from the asset).
    joint_indices = list(range(robot.num_dof))
    target_rad = np.deg2rad(np.full(robot.num_dof, TARGET_DEG))

    robot.apply_action(
        ArticulationAction(
            joint_positions=target_rad.tolist(),
            joint_indices=joint_indices,
        )
    )
    print(f"Commanded joints (rad): {target_rad}")

    for _ in range(WAIT_STEPS):
        await app.next_update_async()

    print("\nDone. Manipulator imported and joints moved.")


asyncio.ensure_future(main())
