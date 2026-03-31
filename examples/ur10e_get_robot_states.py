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
        - State queries:
            * joint states, forces, poses, velocities
"""

import pathlib
import sys
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

logger.remove()
logger.add(
    sys.stdout,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level:<8} | {name}:{function}:{line} - {message}"
    ),
)

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

        # ------------------------ Get the robot state ------------------------

        # 1. Get the joint postitions.
        # Get all joint positions.
        joint_positions = art.get_joint_positions()
        # Get the two joint positions: wrist_3_joint and wrist_2_joint.
        joint_positions = art.get_joint_positions(
            joint_indices=np.array([wrist_3_joint_index, wrist_2_joint_index])
        )

        # 2. Get the joint velocities.
        # Get all joint velocities.
        joint_velocities = art.get_joint_velocities()
        # Get the two joint velocites: wrist_3_joint and wrist_2_joint.
        joint_velocities = art.get_joint_velocities(
            joint_indices=np.array([wrist_3_joint_index, wrist_2_joint_index])
        )

        # 3. Returns the efforts computed/measured by the physics solver of the
        # joint forces in the DOF motion direction.
        # Get all measured joint efforts.
        measured_joint_efforts = art.get_measured_joint_efforts()
        # Get the two measured joint efforts: wrist_3_joint and wrist_2_joint.
        measured_joint_efforts = art.get_measured_joint_efforts(
            joint_indices=np.array([wrist_3_joint_index, wrist_2_joint_index])
        )

        # 4. Get the joint efforts of articulations in the view.
        # Get all joint efforts.
        applied_joint_efforts = art.get_applied_joint_efforts()
        # Get the two joint efforts: wrist_3_joint and wrist_2_joint.
        applied_joint_efforts = art.get_applied_joint_efforts(
            joint_indices=np.array([wrist_3_joint_index, wrist_2_joint_index])
        )

        # 5. Get the measured joint reaction forces and torques (link incoming
        # joint forces and torques) to external loads.
        # Get all measured joint forces
        measured_joint_forces = art.get_measured_joint_forces()
        # Get the two measured joint forces: wrist_3_joint and wrist_2_joint.
        measured_joint_forces = art.get_measured_joint_forces(
            joint_indices=np.array([wrist_3_joint_index, wrist_2_joint_index])
        )

        # 6. Get the poses of the prims in the view with respect to the world's
        # frame.
        world_poses = art.get_world_poses()

        # 7. Get prim poses in the view with respect to the local frame (the prim's
        # parent frame).
        local_poses = art.get_local_poses()

        # 8. Get the linear and angular velocities of prims in the view.
        velocities = art.get_velocities()

        # 9. Get the linear velocities of prims in the view.
        linear_velocities = art.get_linear_velocities()

        # 10. Get the angular velocities of prims in the view.
        angular_velocities = art.get_angular_velocities()

        # 11. Get the default joint states defined with the ``set_joints_default_
        # state`` method.
        joints_default_state = art.get_joints_default_state()

        # 12. Get the current joint states (positions and velocities).
        joints_state = art.get_joints_state()
   
        # 13. Get the dof index and build a "joint name -> DOF index" map for
        # fast name-based control commands.
        dof_names = art.dof_names
        dof_name_to_index = {
            name: art.get_dof_index(name)
            for name in art.dof_names
        }

        # logger the robot state
        logger.info("DOF indices: {}", dof_name_to_index)
        logger.info("Joint positions: {}", joint_positions)
        logger.info("Joint velocities: {}", joint_velocities)
        logger.info("Measured joint efforts: {}", measured_joint_efforts)
        logger.info("Applied joint efforts: {}", applied_joint_efforts)
        logger.info("Measured joint forces: {}", measured_joint_forces)
        logger.info("World poses: {}", world_poses)
        logger.info("Local poses: {}", local_poses)
        logger.info("Velocities: {}", velocities)
        logger.info("Linear velocities: {}", linear_velocities)
        logger.info("Angular velocities: {}", angular_velocities)
        logger.info("Joints default position: {}", joints_default_state.positions)
        logger.info("Joints default velocity: {}", joints_default_state.velocities)
        logger.info("Joints default effort: {}", joints_default_state.efforts)
        logger.info("Joints position: {}", joints_state.positions)
        logger.info("Joints velocity: {}", joints_state.velocities)

        TRIAL_IDX = 1
        continue

    simulation_context.step(render=True)
