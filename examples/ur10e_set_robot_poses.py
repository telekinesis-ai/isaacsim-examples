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
        - Direct state setting (teleport): set_local_poses / set_world_poses.
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

        # Set prim poses in the view with respect to the local frame (the
        # prim's parent frame).
        art.set_local_poses(
            translations=np.array([[2.0, 2.0, 0.0]]),
            orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            indices=np.array([0]),
        )

        # Set poses of prims in the view with respect to the world's frame.
        art.set_world_poses(
            positions=np.array([[2.0, 2.0, 2.0]]),
            orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            indices=np.array([0]),
        )

        # Get world pose and local pose
        world_poses = art.get_world_poses()
        local_poses = art.get_local_poses()
        logger.info("World poses: {}", world_poses)
        logger.info("Local poses: {}", local_poses)

        TRIAL_IDX = 1
        continue

    simulation_context.step(render=True)
