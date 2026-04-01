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

import pathlib
import numpy as np

# Launch the SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# Any Omniverse level imports must occur after the `SimulationApp` class is
# instantiated (because APIs are provided by the extension/runtime plugin
# system, it must be loaded before they will be available to import).
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
import isaacsim.robot_motion.motion_generation.interface_config_loader as interface_config_loader
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
from isaacsim.core.utils.stage import get_stage_units
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdLux

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


# Get stage handle
stage = omni.usd.get_context().get_stage()

# Add a ground plane
PhysicsSchemaTools.addGroundPlane(
    stage, "/groundPlane", "Z", 15, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.7)
)

# Add lighting
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(1000)

# Add cube
cube = DynamicCuboid(
    name="cube",
    prim_path="/Cube",
    position=np.array([-0.7, -0.7, 0.0]) / get_stage_units(),
    orientation=np.array([0, 0, 1, 0]),  # (w,x,y,z)
    scale=np.array([1.0, 1.0, 1.0]),
    size=float(0.0515 / get_stage_units()),
    color=np.array([0, 0, 1]),
)


# ----------------------------- Setup robot -----------------------------


root_dir = pathlib.Path(__file__).resolve().parent.parent
model_dir = root_dir / "models"

# Import URDF, prim_path contains the path the path to the usd prim in the stage.
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
    urdf_path=ur10e_urdf_path,
    import_config=ur10e_import_config,
    get_articulation_root=True,
)

# Import onrobot rg6，prim_path contains the path the path to the usd prim in
# the stage.
status, gripper_import_config = omni.kit.commands.execute(
    "URDFCreateImportConfig"
)
gripper_import_config.merge_fixed_joints = False
gripper_import_config.convex_decomp = False
gripper_import_config.import_inertia_tensor = True
gripper_import_config.fix_base = False
gripper_import_config.distance_scale = 1.0
gripper_urdf_path = (
    model_dir
    / "example-robot-data"
    / "end_effectors"
    / "onrobot"
    / "onrobot_rg_description"
    / "urdf"
    / "onrobot_rg6_model.urdf"
)
status, gripper_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=gripper_urdf_path,
    import_config=gripper_import_config,
    get_articulation_root=True,
)

# Ensure imported prims are available on stage before path queries.
simulation_app.update()

# Prim path to the base robot
ur10e_base = Sdf.Path(ur10e_prim_path).GetParentPath().pathString
# Prim path to the mount point of the base robot
ur10e_base_mount = f"{ur10e_base}/tool0"
# Prim path to the attach robot
onrobot_rg6_attach = Sdf.Path(gripper_prim_path).GetParentPath().pathString
# Prim path to the mount point of the attach robot
onrobot_rg6_attach_mount = f"{onrobot_rg6_attach}/onrobot_rg6_base_link"
# Assembly namespace
assembly_namespace = "Gripper"
variant_name = "ur10e_with_rg6"

# Assemble ur10e and onrobot rg6
assembler = RobotAssembler()
# Begin the Assembly process
assembler.begin_assembly(
    stage,
    ur10e_base,
    ur10e_base_mount,
    onrobot_rg6_attach,
    onrobot_rg6_attach_mount,
    assembly_namespace,
    variant_name,
)
# Perform any Additional transformations on the Attach robot pose here directly
# through USD.
assembler.assemble()
# This function will finish the assembly process by adding the attachment link
# to the parent robot joint and link lists, and then either merge the session
# layer into the current stage, or save a configuration file, and remove the
# session layer from the stage.
assembler.finish_assemble()


# ----------------------------- simulation -----------------------------


# Flush a few frames so assembled USD prims are fully realized before physics
# and articulation handles are created.
simulation_app.update()
simulation_app.update()

# Build physics context and initialize PhysX tensor views.
simulation_context = SimulationContext()
simulation_context.initialize_physics()

# One more frame flush after physics init to make handles stable.
simulation_app.update()
simulation_app.update()

# RG6 base link path attached under UR10e.
ee_path = f"{ur10e_base}/onrobot_rg6_model/onrobot_rg6_base_link"

# Configure RG6 as a parallel gripper with one driving joint.
gripper = ParallelGripper(
    end_effector_prim_path=ee_path,
    joint_prim_names=["finger_joint"],
    joint_opened_positions=np.array([0.0]),
    joint_closed_positions=np.array([0.8]),
    action_deltas=np.array([-0.8]),
    use_mimic_joints=True,
)

robot = SingleManipulator(
    prim_path=ur10e_prim_path,
    name="ur10e_with_rg6",
    end_effector_prim_path=ee_path,
    gripper=gripper,
)
# Important: initialize() binds articulation callbacks for gripper actions.
robot.initialize()

# Optional handle for the UR10e tool0 frame (useful for debug/inspection).
tool0_prim = XFormPrim(f"{ur10e_base}/tool0")

articulation_controller = robot.get_articulation_controller()

# IK solver
# Load Isaac Sim built-in Lula config for UR10e to avoid hardcoded file paths.
kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("UR10e")
lula_solver = LulaKinematicsSolver(**kinematics_config)

# Solve IK on UR10e tool0 frame.
ee_frame_name = "tool0"

ik_solver = ArticulationKinematicsSolver(
    robot,
    lula_solver,
    ee_frame_name,
)

# task targets 
cube_pos, _ = cube.get_world_pose()
# Convert cube pose to the mirrored convention used by this script.
cube_pos[:2] *= -1
target_pos = np.array([-cube_pos[0], cube_pos[1], cube_pos[2]])
target_pos[:2] *= -1
# tcp_offset: tool0 -> gripper TCP compensation.
tcp_offset = np.array([0, 0, 0.25])
# offset: safe approach height above object/target.
offset = np.array([0, 0, 0.2])

timeline = omni.timeline.get_timeline_interface()

stage = 0

while simulation_app.is_running():
    # Pump Kit app events/UI.
    simulation_app.update()

    # Gate control loop by timeline Play state.
    if not timeline.is_playing():
        continue

    # Stage 0: move to an initial arm pose and open gripper.
    if stage == 0:
        robot.set_joint_positions([[-0.0,-1.2,1.1,0.0,0.0,0.0,-0.0, 0.0,0.0,0.0,0.0,0.0]])
        robot.gripper.open()
        stage += 1

    # Stage 1-49: pre-pick approach above cube.
    if 0 < stage < 50:
        target = cube_pos + offset + tcp_offset

        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target,
            target_orientation=np.array([0, 0, 1, 0]),
        )

        if success:
            articulation_controller.apply_action(action)
            stage += 1


    # Stage 50-99: descend to pick pose.
    elif stage < 100:
        target = cube_pos + tcp_offset

        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target,
            target_orientation=np.array([0, 0, 1, 0]),
        )

        if success:
            articulation_controller.apply_action(action)
            stage += 1

    # Stage 100-149: close gripper.
    elif stage < 150:
        action = robot.gripper.forward(action="close")
        articulation_controller.apply_action(action)
        stage += 1

    # Stage 150-199: lift object after grasp.
    elif stage < 200:
        target = cube_pos + np.array([0, 0, 0.05]) + tcp_offset

        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target,
            target_orientation=np.array([0, 0, 1, 0]),
        )

        if success:
            articulation_controller.apply_action(action)
            stage += 1

    # Stage 200-249: move above place target.
    elif stage < 250:
        target = target_pos + offset + tcp_offset

        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target,
            target_orientation=np.array([0, 0, 1, 0]),
        )

        if success:
            articulation_controller.apply_action(action)
            stage += 1

    # Stage 250-299: descend to place pose.
    elif stage < 300:
        target = target_pos + tcp_offset

        action, success = ik_solver.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=np.array([0, 0, 1, 0]),
        )

        if success:
            articulation_controller.apply_action(action)
            stage += 1

    # Stage 300-349: open gripper to release.
    elif stage < 350:
        action = robot.gripper.forward(action="open")
        articulation_controller.apply_action(action)
        stage += 1

    # Stage 350: one-time completion message.
    elif stage == 350:
        stage += 1

    # Advance physics and render.
    simulation_context.step(render=True)
