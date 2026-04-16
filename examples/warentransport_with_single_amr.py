"""
Warehouse transport scene composition demo for Isaac Sim.

This script builds a warehouse environment with single autonomous mobile
robot (AMR) and runs a timeline-gated simulation loop.
"""

# Launch the SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import pathlib

# Any Omniverse level imports must occur after the `SimulationApp` class is
# instantiated (because APIs are provided by the extension/runtime plugin
# system, it must be loaded before they will be available to import).
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Sdf, UsdLux, UsdGeom


def add_reference_with_pose(
    asset_path,
    prim_path,
    position=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
):
    """
    Reference a USD asset onto stage and set its world pose.

    Args:
        asset_path (str): Relative path under Isaac assets root.
            Example: ``/Isaac/Environments/Simple_Warehouse/full_warehouse.usd``.
        prim_path (str): Destination prim path on stage.
        position (np.ndarray): World translation with shape ``(3,)``.
        orientation (np.ndarray): World quaternion in ``[w, x, y, z]``.
    """
    assets_root_path = get_assets_root_path()
    print(assets_root_path)
    stand_usd_path = assets_root_path + asset_path

    add_reference_to_stage(
        usd_path=stand_usd_path,
        prim_path=prim_path,
    )

    # Wrapper object name for debug readability in object registry.
    xfrom_name = prim_path.split("/")[-1] + "_xform"
    xform = SingleXFormPrim(prim_path, name=xfrom_name)
    xform.set_world_pose(
        position=position,
        orientation=orientation,
    )


def euler_xyz_to_quat_wxyz(
    orientation: np.ndarray,
    degrees: bool = False,
) -> np.ndarray:
    """
    Convert Euler angles (x, y, z) to quaternion in wxyz order.

    Args:
        orientation: Euler angles ``[rx, ry, rz]`` in radians by default.
        degrees: if ``True``, treat input Euler angles as degrees.

    Returns:
        np.ndarray of shape (4,), quaternion in [w, x, y, z]
    """
    rx, ry, rz = orientation
    if degrees:
        rx, ry, rz = np.deg2rad([rx, ry, rz])

    hx = rx * 0.5
    hy = ry * 0.5
    hz = rz * 0.5

    cx, sx = np.cos(hx), np.sin(hx)
    cy, sy = np.cos(hy), np.sin(hy)
    cz, sz = np.cos(hz), np.sin(hz)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    return np.array([w, x, y, z], dtype=np.float64)


def main():
    """Build the warehouse scene and run the timeline-gated sim loop."""

    # ----------------------------- Setup stage -----------------------------

    world = World(stage_units_in_meters=1.0)

    # Get stage handle
    stage = omni.usd.get_context().get_stage()

    # Ensure a stable /World default prim exists. This helps keep the stage
    # structure predictable.
    world_path = Sdf.Path("/World")
    world_prim = stage.GetPrimAtPath(world_path)
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, world_path).GetPrim()
    stage.SetDefaultPrim(world_prim)

    # Add lighting
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(1000)

    # Add warehouse background.
    root_dir = pathlib.Path(__file__).resolve().parent.parent
    factory_usd_path = f"{root_dir}\\assets\\environments\\IsaacWarehouse\\IsaacWarehouse.usd"
    add_reference_to_stage(
        usd_path=str(factory_usd_path),
        prim_path="/World/factory",
    )

    # ----------------------------- Setup robot -----------------------------

    # Add amr to scene and set pose.
    add_reference_with_pose(
        asset_path="/Isaac/Robots/Clearpath/Dingo/dingo.usd",
        prim_path="/World/amr",
        position=np.array([13.7, -9.39, 0.016]),
        orientation=euler_xyz_to_quat_wxyz(
            orientation=np.array([0.0, 0.0, 90.0]), degrees=True
        ),
    )


    # ----------------------------- simulation -----------------------------


    # Flush a few frames so assembled USD prims are fully realized before physics
    # and articulation handles are created.
    simulation_app.update()
    simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()

    # Legacy placeholder kept to avoid changing runtime behavior.
    stage = 0

    while simulation_app.is_running():
        # Pump Kit app events/UI.
        simulation_app.update()

        # Gate control loop by timeline Play state.
        if not timeline.is_playing():
            continue

        # Advance physics and render.
        world.step(render=True)

    world.pause()
    simulation_app.close()


if __name__ == "__main__":
    main()
