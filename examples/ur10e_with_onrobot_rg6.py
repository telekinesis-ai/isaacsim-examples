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
        - Create an `Articulation` handle for the UR10e and print DOF names.
        - Run a Play-gated simulation loop (`timeline.is_playing()`).
"""

import pathlib
import numpy as np
from loguru import logger

# Launch the SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# Any Omniverse level imports must occur after the `SimulationApp` class is
# instantiated (because APIs are provided by the extension/runtime plugin
# system, it must be loaded before they will be available to import).
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
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
variant_name = "ur10e_with_onrobot_rg6"

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


# Update the simulation to ensure the robot is fully imported before we try to
# interact with it.
simulation_app.update()
simulation_app.update()

# Create a `SimulationContext` to interact with the physics scene and get/set
# robot state.
simulation_context = SimulationContext()

# Initialize physics for getting any articulation.
simulation_context.initialize_physics()

simulation_app.update()
simulation_app.update()

# Articulation
art = Articulation(ur10e_prim_path)
art.initialize()
finger_idx = art.get_dof_index("finger_joint")

# Wait for manual Play in the Isaac Sim UI; do not auto-start.
timeline = omni.timeline.get_timeline_interface()

TRIAL_IDX = 0

# Main simulation loop
while simulation_app.is_running():
    simulation_app.update()

    if not timeline.is_playing():
        continue

    # Initialization control is performed only once when the device first enters
    # Play mode, avoiding repeated resets every frame.
    if TRIAL_IDX == 0:

        # Set the finger joints to close.
        positions = np.array([[0.5]])
        art.set_joint_positions(
            positions,
            joint_indices=np.array([finger_idx])
        )

        # Get all joint positions.
        joint_positions = art.get_joint_positions()
        logger.info("Joint positions: {}", joint_positions)

        TRIAL_IDX = 1
        continue

    simulation_context.step(render=True)
