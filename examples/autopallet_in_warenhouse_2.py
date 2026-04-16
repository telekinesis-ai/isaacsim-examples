"""
Warehouse scene composition demo for Isaac Sim.

This script creates an auto-palletizing sandbox using a UR10-based workcell,
warehouse assets, and a suction gripper.

Current content:
    1. Create `/World` as default prim (if missing) and add a distant light.
    2. Add warehouse background + UR10 conveyor workcell.
    3. Add a target box and a 3x3 grid of cardboard boxes.
    4. Start simulation loop gated by timeline Play state.
"""

# Launch the SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import pathlib
import numpy as np

# Any Omniverse level imports must occur after the `SimulationApp` class is
# instantiated (because APIs are provided by the extension/runtime plugin
# system, it must be loaded before they will be available to import).
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.manipulators import SingleManipulator
import isaacsim.robot_motion.motion_generation.interface_config_loader as interface_config_loader
from isaacsim.robot_motion.motion_generation.lula.kinematics import (
    LulaKinematicsSolver,
)
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import (
    ArticulationKinematicsSolver,
)
import omni.usd
import omni.timeline
import omni.kit.commands
from pxr import Sdf, UsdLux, UsdGeom

# Robot Assembler is packaged as an extension. It must be enabled before
# importing, otherwise the import may fail if the extension is not loaded yet.
import omni.kit.app

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate(
    "isaacsim.robot_setup.assembler", True
)
# Helper used to assemble (attach) robot components, e.g. mounting grippers.
from isaacsim.robot_setup.assembler import RobotAssembler

root_dir = pathlib.Path(__file__).resolve().parent.parent


def add_reference_with_pose(
    asset_path: str,
    prim_path: str,
    position=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
):
    """
    Add a USD reference to the stage and place it in world space.

    Args:
        asset_path (str): Relative path under Isaac assets root.
            Example: ``/Isaac/Environments/Simple_Warehouse/full_warehouse.usd``.
        prim_path (str): Destination prim path on stage.
        position (np.ndarray): World translation with shape ``(3,)``.
        orientation (np.ndarray): World quaternion in ``[w, x, y, z]``.
    """
    assets_root_path = get_assets_root_path()
    stand_usd_path = assets_root_path + asset_path

    add_reference_to_stage(
        usd_path=stand_usd_path,
        prim_path=prim_path,
    )

    # Wrapper object name for debug readability in object registry.
    xfrom_name = prim_path.split("/")[-1] + "_xform"
    xform = SingleXFormPrim(prim_path, name=xfrom_name)
    xform.set_world_pose(
        position=position,
        orientation=orientation,
    )


def euler_xyz_to_quat_wxyz(
    orientation: np.ndarray,
    degrees: bool = False,
) -> np.ndarray:
    """
    Convert Euler angles (x, y, z) to quaternion in wxyz order.

    Args:
        orientation: Euler angles ``[rx, ry, rz]`` in radians by default.
        degrees: if ``True``, treat input Euler angles as degrees.

    Returns:
        np.ndarray of shape (4,), quaternion in [w, x, y, z]
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
    """
    Build a warehouse scene, replace the embedded robot with imported UR10e,
    mount a suction gripper, and run the simulation loop.

    High-level flow:
        1. Create `/World` and environments assets.
        2. Import UR10e from local URDF data.
        3. Attach a suction gripper using `RobotAssembler`.
        4. Step the world only when timeline Play is active.
    """

    # ----------------------------- Setup stage -----------------------------

    world = World(stage_units_in_meters=1.0)

    # Get stage handle
    stage = omni.usd.get_context().get_stage()

    # Ensure /World is the default prim.
    world_path = Sdf.Path("/World")
    world_prim = UsdGeom.Xform.Define(stage, world_path).GetPrim()
    stage.SetDefaultPrim(world_prim)

    # Add lighting
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(1000)

    # Get assets root path for referencing USD assets.
    assets_root_path = get_assets_root_path()

    # Add warehouse environments.
    stand_usd_path = (
        assets_root_path
        + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
    )
    add_reference_to_stage(
        usd_path=stand_usd_path,
        prim_path="/World/warenhouse",
    )

    # Add robot workcell with long conveyor and rotate it about +Z.
    stand_usd_path = "/Isaac/Samples/Leonardo/Stage/ur10_bin_stacking_long_conveyor.usd"
    ur10_with_conveyor_prim_path = "/World/ur10_with_long_conveyor"
    add_reference_with_pose(
        asset_path=stand_usd_path,
        prim_path=ur10_with_conveyor_prim_path,
        position=np.array([-17.0, 4.0, 1.18]),
        orientation=euler_xyz_to_quat_wxyz(
            orientation=np.array([0, 0, -np.pi / 2])
        )
    )

    # Add one target box near conveyor output region.
    add_reference_with_pose(
        asset_path="/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd",
        prim_path="/World/target_box",
        position=np.array([-15.0, 4.0, 0.9]),
    )

    # Add a 3x3 box grid as palletizing source objects.
    for i in range(9):
        box_name = f"box_{i}"

        root_position = np.array([-17.7, 2.85, 0.52])
        row_spacing = 0.26
        col_spacing = 0.38

        row = i // 3
        col = i % 3

        box_position = root_position + np.array(
            [col * col_spacing, row * row_spacing, 0.0]
        )

        add_reference_with_pose(
            asset_path="/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd",
            prim_path=f"/World/box_grid/{box_name}",
            position=box_position,
        )


    # ----------------------------- Setup robot -----------------------------


    # Disable the pre-authored UR10 from the referenced workcell so the
    # imported UR10e is the only active robot in the stage.
    stage = get_current_stage()
    ur10_prim_path = "/World/ur10_with_long_conveyor/ur10"
    ur10_prim = stage.GetPrimAtPath(ur10_prim_path)
    ur10_prim.SetActive(False)

    # Import UR10e from URDF
    model_dir = root_dir / "assets"
    _, ur10e_import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    ur10e_import_config.merge_fixed_joints = False
    ur10e_import_config.convex_decomp = False
    ur10e_import_config.import_inertia_tensor = True
    ur10e_import_config.fix_base = True
    ur10e_import_config.distance_scale = 1.0
    ur10e_urdf_path = (
        model_dir
        / "example-robot-data"
        / "robots"
        / "universal_robots"
        / "urdf"
        / "ur10e.urdf"
    )
    _, ur10e_prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(ur10e_urdf_path),
        import_config=ur10e_import_config,
        get_articulation_root=False,
    )

    # Re-parent the imported robot under the conveyor workcell while preserving
    # its world-space transform.
    omni.kit.commands.execute(
        "ParentPrimsCommand",
        parent_path=ur10_with_conveyor_prim_path,
        child_paths=[ur10e_prim_path],
        keep_world_transform=True,
    )

    # Resolve the imported robot's new path after re-parenting and define the
    # tool mount used for gripper assembly.
    ur10e_prim_path = (
        ur10_with_conveyor_prim_path + "/" + ur10e_prim_path.split("/")[-1]
    )
    ur10e_mount_path = f"{ur10e_prim_path}/tool0"

    # Set the robot's local pose and default state.
    robot_xfrom_name = "ur10e_xform"
    robot_xform = world.scene.add(
        SingleXFormPrim(ur10e_prim_path, name=robot_xfrom_name)
    )
    robot_xform.set_local_pose(
        translation=np.array([0.0, 0.0, 0.0]),
        orientation=euler_xyz_to_quat_wxyz(orientation=np.array([0, 0, np.pi])),
    )
    position, orientation = robot_xform.get_world_pose()
    robot_xform.set_default_state(
        position=position,
        orientation=orientation,
    )

    # Add the suction gripper USD under a dedicated mount prim.
    gripper_usd_path = (
        assets_root_path
        + "/Isaac/Robots/UniversalRobots/ur10/grippers/short_gripper.usd"
    )
    gripper_prim_path = ur10_with_conveyor_prim_path + "/suction_gripper"
    add_reference_to_stage(
        usd_path=gripper_usd_path,
        prim_path=gripper_prim_path,
    )

    # Set the gripper's local pose relative to the robot flange.
    gripper_xform_name = "suction_gripper_xform"
    gripper_xform = world.scene.add(
        SingleXFormPrim(
            prim_path=gripper_prim_path,
            name=gripper_xform_name,
        )
    )
    gripper_xform.set_local_pose(
        orientation=euler_xyz_to_quat_wxyz(
            orientation=np.array([0, np.pi / 2, 0])
        )
    )

    # This gripper asset is a single rigid prim, so the same path can be used
    # for both the end-effector root and the surface-gripper attachment prim.
    gripper = SurfaceGripper(
        end_effector_prim_path=gripper_prim_path,
        surface_gripper_path=gripper_prim_path+"/SurfaceGripper",
    )

    # Make sure stage updates after import/reference.
    simulation_app.update()
    stage = omni.usd.get_context().get_stage()

    # Assemble robot with gripper
    assembly_namespace = "Gripper"
    variant_name = "ur10e_with_suction_gripper"
    assembler = RobotAssembler()
    # Attach gripper to tool0 and bake assembly edits into the stage.
    assembler.begin_assembly(
        stage,
        ur10e_prim_path,
        ur10e_mount_path,
        gripper_prim_path,
        gripper_prim_path,
        assembly_namespace,
        variant_name,
    )
    assembler.assemble()
    assembler.finish_assemble()

    # ----------------------------- simulation -----------------------------

    # Flush a few frames so assembled USD prims are fully realized before physics
    # and articulation handles are created.
    simulation_app.update()
    simulation_app.update()

    # Robot wrapper and world reset
    robot = world.scene.add(
        SingleManipulator(
            prim_path=ur10e_prim_path,
            name="ur10e_with_rg6",
            end_effector_prim_path=gripper_prim_path,
            gripper=gripper,
        )
    )
    # Update after robot wrapper
    world.reset()
    simulation_app.update()

    # IK solver
    # Load Isaac Sim built-in Lula config for UR10e to avoid hardcoded file paths.
    kinematics_config = (
        interface_config_loader.load_supported_lula_kinematics_solver_config(
            "UR10e"
        )
    )
    lula_solver = LulaKinematicsSolver(**kinematics_config)
    # Solve IK on UR10e tool0 frame.
    ik_solver = ArticulationKinematicsSolver(
        robot,
        lula_solver,
        "tool0",
    )

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
