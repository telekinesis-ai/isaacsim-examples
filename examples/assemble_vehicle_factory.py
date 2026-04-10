"""Assemble an industrial robot setup in the factory environment.

This example loads the local factory USD scene, disables a pre-authored robot
instance, imports a KUKA KR120 from URDF, and places it into the scene using an
Xform wrapper for pose control.

The script focuses on scene composition and robot placement. It does not include
task logic such as welding trajectories or closed-loop manipulation control.
"""

# Launch the SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import pathlib

# Any Omniverse level imports must occur after the `SimulationApp` class is
# instantiated (because APIs are provided by the extension/runtime plugin
# system, it must be loaded before they will be available to import).
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Sdf, UsdLux, UsdGeom


def euler_xyz_to_quat_wxyz(
    orientation: np.ndarray,
    degrees: bool = False,
) -> np.ndarray:
    """Convert XYZ Euler angles to a quaternion in ``[w, x, y, z]`` order.

    Args:
        orientation (np.ndarray): Euler angles ``[rx, ry, rz]``.
        degrees (bool): If ``True``, interpret angles as degrees.

    Returns:
        np.ndarray: Quaternion with shape ``(4,)`` in ``[w, x, y, z]`` order.
    """
    rx, ry, rz = orientation
    if degrees:
        rx, ry, rz = np.deg2rad([rx, ry, rz])

    hx = rx * 0.5
    hy = ry * 0.5
    hz = rz * 0.5

    cx, sx = np.cos(hx), np.sin(hx)
    cy, sy = np.cos(hy), np.sin(hy)
    cz, sz = np.cos(hz), np.sin(hz)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    return np.array([w, x, y, z], dtype=np.float64)


def main():
    """Create the factory scene, place the imported robot, and run simulation.

    High-level flow:
        1. Initialize `/World` and add the factory USD environment.
        2. Disable an existing robot prim in the referenced factory scene.
        3. Import KUKA KR120 from local URDF data and parent it under factory.
        4. Set robot world pose and step the world when timeline Play is active.
    """

    # ----------------------------- Setup stage -----------------------------

    world = World(stage_units_in_meters=1.0)

    # Get stage handle
    stage = omni.usd.get_context().get_stage()

    # Ensure a stable /World default prim exists. This helps keep the stage
    # structure predictable.
    world_path = Sdf.Path("/World")
    world_prim = stage.GetPrimAtPath(world_path)
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, world_path).GetPrim()
    stage.SetDefaultPrim(world_prim)

    # Add lighting
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(1000)

    # Resolve local factory USD from repository root to avoid Windows
    # backslash-escape issues in string literals.
    root_dir = pathlib.Path(__file__).resolve().parent.parent
    factory_usd_path = (
        root_dir / "models" / "environment" / "Factory" / "Factory.usd"
    )

    add_reference_to_stage(
        usd_path=str(factory_usd_path),
        prim_path="/World/factory",
    )

    # ----------------------------- Setup robot -----------------------------

    # Disable the pre-authored UR10 from the referenced workcell so the
    # imported UR10e is the only active robot in the stage.
    stage = get_current_stage()
    kuka_u20_prim_path = "/World/factory/Samples/Welding_Assembly_Animated_Adjust/Welding_Assembly_Animated_Export/root/RobotController/Link1"
    kuka_u20_prim = stage.GetPrimAtPath(kuka_u20_prim_path)
    kuka_u20_prim.SetActive(False)

    # Import Kuka from URDF
    model_dir = root_dir / "models"
    _, kuka_import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    kuka_import_config.merge_fixed_joints = False
    kuka_import_config.convex_decomp = False
    kuka_import_config.import_inertia_tensor = True
    kuka_import_config.fix_base = True
    kuka_import_config.distance_scale = 1
    kuka_urdf_path = (
        model_dir
        / "example-robot-data"
        / "robots"
        / "kuka"
        / "kr120_description"
        / "urdf"
        / "kr120r2500pro.urdf"
    )
    _, kuka_prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(kuka_urdf_path),
        import_config=kuka_import_config,
        get_articulation_root=False,
    )

    # Re-parent the imported robot under the conveyor workcell while preserving
    # its world-space transform.
    omni.kit.commands.execute(
        "ParentPrimsCommand",
        parent_path="/World/factory",
        child_paths=[kuka_prim_path],
        keep_world_transform=True,
    )

    # Resolve the imported robot's new path after re-parenting and define the
    # flange/tool mount used for gripper assembly.
    kuka_prim_path = "/World/factory/" + kuka_prim_path.split("/")[-1]
    kuka_mount_path = f"{kuka_prim_path}/tool0"

    # Add a wrapper XFormPrim for the robot to facilitate scene-level control and set its local pose.
    robot_xfrom_name = "kuka_xform"
    robot_xform = world.scene.add(
        SingleXFormPrim(kuka_prim_path, name=robot_xfrom_name)
    )
    robot_xform.set_world_pose(
        position=np.array([2.60, 5.25, 2.52]),
    )


    # ----------------------------- simulation -----------------------------


    # Flush a few frames so assembled USD prims are fully realized before physics
    # and articulation handles are created.
    simulation_app.update()
    simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()

    # Legacy placeholder kept to avoid changing runtime behavior.
    stage = 0

    while simulation_app.is_running():
        # Pump Kit app events/UI.
        simulation_app.update()

        # Gate control loop by timeline Play state.
        if not timeline.is_playing():
            continue

        # Advance physics and render.
        world.step(render=True)

    world.pause()
    simulation_app.close()


if __name__ == "__main__":
    main()
