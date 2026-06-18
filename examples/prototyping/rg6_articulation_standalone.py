"""
OnRobot RG6 gripper — standalone articulation demo in Isaac Sim.

This script builds a minimal Isaac Sim scene and imports *only* the OnRobot RG6
gripper from URDF (no UR arm, no assembly). The gripper is anchored in space
(`fix_base=True`) and driven directly through the native `SingleArticulation`
API by commanding its single actuated joint, `finger_joint`.

Per the Isaac Sim 6.0.0 articulation-controller docs, the recommended joint
control flow is: create a `SingleArticulation`, `initialize()` it once the sim
is playing, package targets in an `ArticulationAction`, and send them with
`apply_action(...)`. No `ParallelGripper`/`SingleManipulator` wrapper is needed.

Workflow:
    1. Scene setup: ground plane + distant light.
    2. Asset import: RG6 from local URDF, base fixed in place.
    3. Simulation: initialize physics, grab a SingleArticulation handle, and run
       a Play-gated loop that toggles the gripper open/close.
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


# ----------------------------- Setup gripper -----------------------------


root_dir = pathlib.Path(__file__).resolve().parent.parent
model_dir = root_dir.parent.parent / "telekinesis-urdfs/src/telekinesis_urdfs/models"

# Import onrobot rg6. prim_path contains the path to the usd prim in the stage.
# fix_base=True anchors the gripper base in space (standalone — there is no arm
# to attach it to, so without this it would fall under gravity).
status, gripper_import_config = omni.kit.commands.execute(
    "URDFCreateImportConfig"
)
gripper_import_config.merge_fixed_joints = False
gripper_import_config.convex_decomp = False
gripper_import_config.import_inertia_tensor = True
gripper_import_config.fix_base = True
gripper_import_config.distance_scale = 1.0
gripper_urdf_path = model_dir / "example-robot-data/tools/onrobot/onrobot_rg_description/urdf/onrobot_rg6_model.urdf"


if not gripper_urdf_path.is_file():
    raise FileNotFoundError(f"Cant load file {gripper_urdf_path}")

status, gripper_prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=gripper_urdf_path,
    import_config=gripper_import_config,
    get_articulation_root=True,
)

# Ensure imported prims are available on stage before handles are created.
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

# Native articulation handle for the RG6 (articulation root from the import).
gripper = SingleArticulation(prim_path=gripper_prim_path, name="rg6")
gripper.initialize()

print(f"num_dof: {gripper.num_dof}")
print(f"dof_names: {gripper.dof_names}")

# `finger_joint` is the single actuated joint; the other 5 knuckle/finger joints
# are URDF mimic joints. Driving finger_joint relies on the importer having
# created PhysX mimic-joint constraints so the followers move automatically. If
# they do NOT follow, command all DOFs explicitly with the ±1 multipliers, or
# fall back to ParallelGripper(use_mimic_joints=True).
finger_idx = gripper.dof_names.index("finger_joint")

# finger_joint limits are ~[-0.628, 0.628] rad. Use a closed/open pair within range.
OPEN_RAD = 0.0
CLOSE_RAD = 0.5

timeline = omni.timeline.get_timeline_interface()

TRIAL_IDX = 0

# Main simulation loop
while simulation_app.is_running():

    # Pump Kit app/UI/events every frame.
    simulation_app.update()

    # Only control the gripper when the user presses Play in the UI timeline.
    if not timeline.is_playing():
        continue

    # One-shot close command.
    if TRIAL_IDX == 0:
        gripper.apply_action(
            ArticulationAction(
                joint_positions=[CLOSE_RAD],
                joint_indices=[finger_idx],
            )
        )
        TRIAL_IDX = 1
        continue

    # Step physics + render after commands are issued.
    simulation_context.step(render=True)
