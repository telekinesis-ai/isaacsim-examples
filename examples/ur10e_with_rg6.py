"""
UR10e + OnRobot RG6 assembly example in Isaac Sim.

This script builds a minimal Isaac Sim scene, imports a UR10e arm and an
OnRobot RG6 gripper from URDF, and assembles them into one robot setup using
`RobotAssembler`.

Workflow:
    1. Scene setup:
        - Create a USD stage with a ground plane and distant light.

    2. Asset import:
        - Import UR10e and RG6 from local URDF assets.
        - Move the imported RG6 prim to `/onrobot_rg6_model` so it has a stable
          attachment path.

    3. Robot assembly:
        - Enable the `isaacsim.robot_setup.assembler` extension.
        - Attach RG6 base link to UR10e `tool0` via `RobotAssembler`.

    4. Simulation startup:
        - Initialize physics with `SimulationContext`.
        - Create an `Articulation` handle for the UR10e.
        - Run a Play-gated simulation loop (`timeline.is_playing()`).
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
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
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

# Configure the RG6 as a 1-DOF parallel gripper driven by finger_joint.
gripper = ParallelGripper(
    end_effector_prim_path=f"{ur10e_base}/onrobot_rg6_model/onrobot_rg6_base_link",
    joint_prim_names=["finger_joint"],
    joint_opened_positions=np.array([0.0]),
    joint_closed_positions=np.array([0.8]),
    action_deltas=np.array([-0.8]),
    use_mimic_joints=True,
)

robot = SingleManipulator(
    prim_path=ur10e_prim_path,
    name="ur10e_with_rg6",
    end_effector_prim_path=f"{ur10e_base}/onrobot_rg6_model/onrobot_rg6_base_link",
    gripper=gripper,
)
# Initialize wires gripper callbacks (apply_action/getters/setters).
robot.initialize()

timeline = omni.timeline.get_timeline_interface()

TRIAL_IDX = 0

# Main simulation loop
while simulation_app.is_running():

    # Pump Kit app/UI/events every frame.
    simulation_app.update()

    # Only control robot when user presses Play in the UI timeline.
    if not timeline.is_playing():
        continue

    # One-shot close command.
    if TRIAL_IDX == 0:
        # Close the gripper
        robot.gripper.close()
        TRIAL_IDX = 1
        continue

    # Step physics + render after commands are issued.
    simulation_context.step(render=True)
