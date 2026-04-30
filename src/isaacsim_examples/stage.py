"""Scene-level helpers: stage setup, camera framing, screenshots."""

import pathlib

import numpy as np
from loguru import logger

# These imports are only available after SimulationApp has been created.
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import open_stage as _open_stage
from isaacsim.core.utils.viewports import set_camera_view
import omni.kit.viewport.utility as vp_util
import omni.timeline
import omni.usd
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdLux, UsdShade


def render_frames(simulation_app, n: int = 120):
    """Run *n* simulation update ticks so the viewport renders and physics settles."""
    for _ in range(n):
        simulation_app.update()


def open_usd_stage(usd_path, simulation_app):
    """Open a pre-built USD environment and initialise simulation.

    Loads the USD file, ensures a ``/World`` default prim and lighting
    exist, creates a :class:`World`, plays the timeline, and initialises
    physics.  After this call the scene is ready for
    :meth:`SimManipulator.connect`.

    Args:
        usd_path: Path to the ``.usd`` file.
        simulation_app: The ``SimulationApp`` singleton.

    Returns:
        The USD stage.
    """
    _open_stage(str(usd_path))

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Ensure /World default prim exists
    world_path = Sdf.Path("/World")
    world_prim = stage.GetPrimAtPath(world_path)
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, world_path).GetPrim()
    stage.SetDefaultPrim(world_prim)

    # Lighting
    distant_light = UsdLux.DistantLight.Define(
        stage, Sdf.Path("/DistantLight")
    )
    distant_light.CreateIntensityAttr(1000)

    # Let assembled USD prims fully realise, then start physics
    render_frames(simulation_app, 10)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    render_frames(simulation_app, 60)

    world.reset()

    return stage


def setup_stage(simulation_app=None):
    """Create ground plane, lighting, and a dark glossy floor material.

    When *simulation_app* is provided the function also creates a
    :class:`SimulationContext`, plays the timeline, and initialises
    physics — so the scene is ready for :meth:`SimManipulator.connect`.

    Args:
        simulation_app: Optional ``SimulationApp`` singleton.  Pass it
            to have the full simulation boilerplate handled here.

    Returns:
        stage: The current USD stage.
    """

    ground_path = "/World/groundPlane"

    stage = omni.usd.get_context().get_stage()

    PhysicsSchemaTools.addGroundPlane(
        stage,
        ground_path,
        "Z",
        15,
        Gf.Vec3f(0, 0, 0),
        Gf.Vec3f(0.02, 0.02, 0.02),
    )

    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distant_light.CreateIntensityAttr(600)

    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome_light.CreateIntensityAttr(250)

    # Dark glossy material for the floor
    material_path = "/World/Materials/DarkGlossyFloor"
    material = UsdShade.Material.Define(stage, material_path)

    shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.02, 0.02, 0.02)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.01)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.25, 0.25, 0.25)
    )

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    ground_prim = stage.GetPrimAtPath(ground_path)
    UsdShade.MaterialBindingAPI(ground_prim).Bind(material)

    if simulation_app is not None:
        sim_context = SimulationContext()
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        render_frames(simulation_app, 60)
        sim_context.initialize_physics()

    world_prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
    stage.SetDefaultPrim(world_prim)

    return stage


def frame_robot(simulation_app, eye=None, target=None, n_frames: int = 30):
    """Position the viewport camera. Defaults work for origin-mounted manipulators."""
    if eye is None:
        eye = np.array([1.3, -1.5, 1.3])
    if target is None:
        target = np.array([0.1, 0.1, 0.5])
    set_camera_view(eye=eye, target=target)
    render_frames(simulation_app, n_frames)


def take_screenshot(simulation_app, robot_name: str,
                    output_dir: str = "screenshots", n_frames: int = 30):
    """Capture the active viewport to *output_dir/{robot_name}.png*."""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    file_path = str(out / f"{robot_name}.png")

    viewport = vp_util.get_active_viewport()
    vp_util.capture_viewport_to_file(viewport, file_path)
    render_frames(simulation_app, n_frames)
    logger.info(f"Screenshot saved to {file_path}")
