"""
UR10e + OnRobot RG6 pick-and-place example in Isaac Sim.

This script builds a minimal scene, imports UR10e and RG6 from URDF, assembles
them with `RobotAssembler`, and runs a staged pick-and-place routine on a cube.

Workflow:
    1. Scene setup:
        - Create a stage with ground, light, and one dynamic cube target.

    2. Asset import and assembly:
        - Import UR10e and RG6 from local assets.
        - Attach RG6 base link to UR10e `tool0`.

    3. Robot control setup:
        - Create `SingleManipulator` + `ParallelGripper`.
        - Initialize Lula IK via
          `load_supported_lula_kinematics_solver_config("UR10e")`.

    4. Pick-and-place loop:
        - Wait for timeline Play.
        - Execute staged motion: pre-pick, pick, close, lift, pre-place, place,
          open.
"""

# Launch the SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import pathlib
import numpy as np

# Any Omniverse level imports must occur after the `SimulationApp` class is
# instantiated (because APIs are provided by the extension/runtime plugin
# system, it must be loaded before they will be available to import).
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
import isaacsim.robot_motion.motion_generation.interface_config_loader as interface_config_loader
from isaacsim.robot_motion.motion_generation.lula.kinematics import (
    LulaKinematicsSolver,
)
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import (
    ArticulationKinematicsSolver,
)
from isaacsim.storage.native import get_assets_root_path
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdLux, UsdGeom, Usd

# Robot Assembler is packaged as an extension. It must be enabled before
# importing, otherwise the import may fail if the extension is not loaded yet.
import omni.kit.app

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate(
    "isaacsim.robot_setup.assembler", True
)

# Helper used to assemble (attach) robot components, e.g. mounting grippers.
from isaacsim.robot_setup.assembler import RobotAssembler

# ----------------------------- Setup stage -----------------------------

world = World(stage_units_in_meters=1.0)

# Get stage handle
stage = omni.usd.get_context().get_stage()

# Ensure /World exists and is the default prim.
world_path = Sdf.Path("/World")
world_prim = stage.GetPrimAtPath(world_path)
stage.SetDefaultPrim(world_prim)

# Add a ground plane
PhysicsSchemaTools.addGroundPlane(
    stage, "/groundPlane", "Z", 15, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.7)
)

# Add lighting
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(1000)

# Add cube through world.scene so reset/init is managed by World.
cube = world.scene.add(
    DynamicCuboid(
        name="cube",
        prim_path="/World/Cube",
        position=np.array([-0.7, -0.7, 0.0]) / get_stage_units(),
        orientation=np.array([0, 0, 1, 0]),  # wxyz
        scale=np.array([1.0, 1.0, 1.0]),
        size=float(0.0515 / get_stage_units()),
        color=np.array([0, 0, 1]),
    )
)

# Ensure a stable /World default prim exists. This helps keep the stage
# structure predictable.
world_path = Sdf.Path("/World")
world_prim = stage.GetPrimAtPath(world_path)
if not world_prim.IsValid():
    world_prim = UsdGeom.Xform.Define(stage, world_path).GetPrim()
stage.SetDefaultPrim(world_prim)


# ----------------------------- Setup robot -----------------------------


root_dir = pathlib.Path(__file__).resolve().parent.parent
model_dir = root_dir / "models"

# Import UR10e from URDF
status, ur10e_import_config = omni.kit.commands.execute(
    "URDFCreateImportConfig"
)
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
status, ur10e_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=str(ur10e_urdf_path),
    import_config=ur10e_import_config,
    get_articulation_root=True,
)

# The base robot prim is the parent of the articulation root returned by importer.
ur10e_base = Sdf.Path(ur10e_prim_path).GetParentPath().pathString
ur10e_base_mount = f"{ur10e_base}/tool0"

# Add Robotiq 2F-85 as a separate referenced robot on the stage
assets_root_path = get_assets_root_path()
robotiq_usd_path = (
    assets_root_path + "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
)
add_reference_to_stage(
    usd_path=robotiq_usd_path,
    prim_path="/World",
)
robotiq_prim_path = "/World/Robotiq_2F_85"
robotiq_mount_path = f"{robotiq_prim_path}/base_link"

# Configure RG6 as a parallel gripper with one driving joint before assembly.
gripper = ParallelGripper(
    end_effector_prim_path=robotiq_mount_path,
    joint_prim_names=["finger_joint"],
    joint_opened_positions=np.array([0.0]),
    joint_closed_positions=np.array([0.8]),
    action_deltas=np.array([-0.8]),
    use_mimic_joints=True,
)

# Make sure stage updates after import/reference.
simulation_app.update()
stage = omni.usd.get_context().get_stage()

# Assemble robot with gripper
assembly_namespace = "Gripper"
variant_name = "ur10e_with_robotiq_2f_85"
assembler = RobotAssembler()
assembler.begin_assembly(
    stage,
    ur10e_base,
    ur10e_base_mount,
    robotiq_prim_path,
    robotiq_mount_path,
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
        end_effector_prim_path=robotiq_mount_path,
        gripper=gripper,
    )
)
world.reset()

# Update after robot wrapper
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

# Robot home position
ARM_HOME = np.array([-0.0, -1.2, 1.1, 0.0, 0.0, 0.0], dtype=np.float32)
ARM_JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
HOME_TOL = 2e-2

# Cube position
cube_pos, _ = cube.get_world_pose()
# Convert cube pose to the mirrored convention used by this script.
cube_pos[:2] *= -1

# Target position
target_pos = np.array([-cube_pos[0], cube_pos[1], cube_pos[2]])
# Convert target pose to the mirrored convention used by this script.
target_pos[:2] *= -1

# Tcp_offset: tool0 to gripper TCP compensation.
tcp_offset = np.array([0, 0, 0.172])
# Offset: safe approach height above object/target.
offset = np.array([0, 0, 0.1])

timeline = omni.timeline.get_timeline_interface()
stage_id = 0
was_playing = False

# Main loop
while simulation_app.is_running():
    simulation_app.update()

    is_playing = timeline.is_playing()

    # ---------------------- timeline edge detection ----------------------

    if is_playing and not was_playing:
        # User just clicked Play: start from initial state.
        world.reset()
        stage_id = 0
        was_playing = is_playing
        continue

    elif (not is_playing) and was_playing:
        # User just clicked Stop: immediately restore initial scene.
        was_playing = is_playing
        continue

    was_playing = is_playing

    # ----------------------------- task logic -----------------------------

    # Do not advance your state machine until user presses Play.
    if not is_playing:
        continue

    # Stage 0: move to home and open gripper.
    if stage_id < 20:
        arm_action = ArticulationAction(
            joint_positions=ARM_HOME,
            joint_indices=ARM_JOINT_INDICES,
        )
        robot.apply_action(arm_action)
        robot.gripper.open()

        current_q = robot.get_joint_positions()[ARM_JOINT_INDICES]
        if np.allclose(current_q, ARM_HOME, atol=HOME_TOL):
            stage_id += 1

    # Stage 20-49: pre-pick approach.
    elif stage_id < 50:
        target = cube_pos + offset + tcp_offset
        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target,
            target_orientation=np.array([0, 0, 1, 0]),
        )
        if success:
            robot.apply_action(action)
            stage_id += 1

    # Stage 50-99: descend to pick.
    elif stage_id < 100:
        target = cube_pos + tcp_offset
        if stage_id == 50:
            print("pick_tcp_offset", tcp_offset)
            print("pick_cube_pos", cube_pos)
            print("pick_target", target)

        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target,
            target_orientation=np.array([0, 0, 1, 0]),
        )
        if success:
            robot.apply_action(action)
            stage_id += 1

    # Stage 100-149: close gripper.
    elif stage_id < 150:
        robot.gripper.close()
        stage_id += 1

    # Stage 150-199: lift after grasp.
    elif stage_id < 200:
        target = cube_pos + offset + tcp_offset
        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target,
            target_orientation=np.array([0, 0, 1, 0]),
        )
        if success:
            robot.apply_action(action)
            stage_id += 1

    # Stage 200-249: back to home.
    elif stage_id < 250:
        arm_action = ArticulationAction(
            joint_positions=ARM_HOME,
            joint_indices=ARM_JOINT_INDICES,
        )
        robot.apply_action(arm_action)

        current_q = robot.get_joint_positions()[ARM_JOINT_INDICES]
        if np.allclose(current_q, ARM_HOME, atol=HOME_TOL):
            stage_id += 1

    # Stage 250-299: move above place target.
    elif stage_id < 300:
        target = target_pos + offset + tcp_offset
        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target,
            target_orientation=np.array([0, 0, 1, 0]),
        )
        if success:
            robot.apply_action(action)
            stage_id += 1

    # Stage 300-349: descend to place pose.
    elif stage_id < 350:
        target = target_pos + tcp_offset + np.array([0, 0, 0.05])
        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target,
            target_orientation=np.array([0, 0, 1, 0]),
        )
        if success:
            robot.apply_action(action)
            stage_id += 1

    # Stage 350-399: open gripper.
    elif stage_id < 400:
        robot.gripper.open()
        stage_id += 1

    # Stage 400+: return home once.
    elif stage_id == 400:
        arm_action = ArticulationAction(
            joint_positions=ARM_HOME,
            joint_indices=ARM_JOINT_INDICES,
        )
        robot.apply_action(arm_action)

        current_q = robot.get_joint_positions()[ARM_JOINT_INDICES]
        if np.allclose(current_q, ARM_HOME, atol=HOME_TOL):
            stage_id += 1

    world.step(render=True)

world.pause()
simulation_app.close()
