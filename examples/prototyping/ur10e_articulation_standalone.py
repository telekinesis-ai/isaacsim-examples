"""
UR10e (no gripper) example in Isaac Sim.

This script builds a minimal Isaac Sim scene, imports a UR10e arm from URDF,
and drives it as a plain articulation (no gripper / no assembly).

Workflow:
    1. Scene setup:
        - Create a USD stage with a ground plane and distant light.

    2. Asset import:
        - Import UR10e from a local URDF asset.

    3. Simulation startup:
        - Initialize physics with `SimulationContext`.
        - Create a `SingleArticulation` handle for the UR10e.
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
from isaacsim.core.prims import SingleArticulation
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdLux

from isaacsim.core.utils.types import ArticulationAction


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

# Ensure imported prims are available on stage before path queries.
simulation_app.update()


# ----------------------------- simulation -----------------------------


# Flush a few frames so imported USD prims are fully realized before physics
# and articulation handles are created.
simulation_app.update()
simulation_app.update()

# Build physics context and initialize PhysX tensor views.
simulation_context = SimulationContext()
simulation_context.initialize_physics()

# One more frame flush after physics init to make handles stable.
simulation_app.update()
simulation_app.update()

# Plain articulation handle for the UR10e (no end effector / gripper).
robot = SingleArticulation(
    prim_path=ur10e_prim_path,
    name="ur10e_no_gripper",
)
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

    # One-shot pose command.
    if TRIAL_IDX == 0:
        robot.apply_action(
            ArticulationAction(
                joint_positions=[90, 90, 0, 0, 90, 0],
                joint_indices=np.arange(robot.num_dof).tolist(),
            )
        )
        TRIAL_IDX = 1
        continue

    # Step physics + render after commands are issued.
    simulation_context.step(render=True)
