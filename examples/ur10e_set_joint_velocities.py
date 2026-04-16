"""
UR10e articulation control example in Isaac Sim.

This script constructs a minimal simulation scene, imports a UR10e robot from
URDF, initializes an articulation interface, and demonstrates both state-setting
and controller-based commands within a Play-gated simulation loop.

Overview:
    The example covers three stages of interaction with an articulation:

    1. Scene setup:
        - Create a USD stage with ground plane and lighting.
        - Import a UR10e robot from a URDF file.

    2. Articulation initialization:
        - Resolve articulation root from imported USD.
        - Initialize physics and articulation handle.
        - Build DOF name-to-index mapping.

    3. Runtime control (executed once on Play):
        - Direct state setting (teleport): set_joint_velocities
"""

# Launch the SimulationApp
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import pathlib
import numpy as np
from loguru import logger

# Any Omniverse level imports must occur after the `SimulationApp` class is
# instantiated (because APIs are provided by the extension/runtime plugin
# system, it must be loaded before they will be available to import).
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdLux


# ----------------------------- Setup stage -----------------------------


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
model_dir = root_dir / "assets"

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


# ----------------------------- simulation -----------------------------


# Update the simulation to ensure the robot is fully imported before we try to
# interact with it.
simulation_app.update()

# Create a `SimulationContext` to interact with the physics scene and get/set
# robot state.
simulation_context = SimulationContext()

# Initialize physics for getting any articulation.
simulation_context.initialize_physics()

# Articulation 
art = Articulation(ur10e_prim_path)
art.initialize()
wrist_3_joint_index = art.get_dof_index("wrist_3_joint")
wrist_2_joint_index = art.get_dof_index("wrist_2_joint")

# Wait for manual Play in the Isaac Sim UI; do not auto-start.
timeline = omni.timeline.get_timeline_interface()

TRIAL_IDX = 0

while simulation_app.is_running():
    simulation_app.update()

    if not timeline.is_playing():
        continue

    # Initialization control is performed only once when the device first enters
    # Play mode, avoiding repeated resets every frame.
    if TRIAL_IDX == 0:

        # Set the joint velocities of articulations in the view.
        # Set the velocities for all the articulation joints to the indicated
        # values.
        velocities = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        art.set_joint_velocities(velocities)

        # Set only the two joints : wrist_3_joint and wrist_2_joint to 0.7.
        velocities = np.array([[0.7, 0.7]])
        art.set_joint_velocities(
            velocities,
            joint_indices=np.array([wrist_3_joint_index, wrist_2_joint_index])
        )

        # Get all joint velocities.
        joint_velocities = art.get_joint_velocities()
        logger.info("Joint velocities: {}", joint_velocities)

        TRIAL_IDX = 1
        continue

    simulation_context.step(render=True)
