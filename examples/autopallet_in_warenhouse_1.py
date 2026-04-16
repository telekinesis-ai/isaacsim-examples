"""
Autopallet conveyor-bin spawning demo for Isaac Sim.

This script loads a conveyor scene, removes the UR10 robot from the referenced
stage, and continuously spawns KLT bins with randomized poses and fixed
conveyor-direction velocity.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import random
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import RigidPrim, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
from isaacsim.cortex.framework.cortex_utils import get_assets_root_path_or_die
import isaacsim.cortex.framework.math_util as math_util


class SceneAssets:
    """Container for resolved USD asset paths used by this scene."""

    def __init__(self):
        """Resolve Nucleus root and build paths for all required assets."""
        self.assets_root_path = get_assets_root_path_or_die()

        self.ur10_table_usd = (
            self.assets_root_path
            + "/Isaac/Samples/Leonardo/Stage/ur10_bin_stacking_short_suction.usd"
        )
        self.small_klt_usd = (
            self.assets_root_path + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        )
        self.background_usd = (
            self.assets_root_path
            + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        )


def random_bin_spawn_transform():
    """Generate randomized spawn transform for a bin.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            position (3,), orientation quaternion (4,) as (w, x, y, z).
    """
    x = random.uniform(-0.15, 0.15)
    y = 1.5
    z = -0.15
    position = np.array([x, y, z], dtype=np.float32)

    rz = random.random() * 0.02 - 0.01
    rw = random.random() * 0.02 - 0.01
    norm = np.sqrt(rz**2 + rw**2)

    if norm < 1e-8:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        quat = math_util.Quaternion([rw / norm, 0, 0, rz / norm])

        if random.random() > 0.5:
            quat = quat * math_util.Quaternion([0, 0, 1, 0])

        quat = np.array(quat.vals, dtype=np.float32)

    return position, quat


class BinConveyorTask(BaseTask):
    """Task that manages one active bin moving through the conveyor area."""

    def __init__(self, assets, env_path="/World/Ur10Table"):
        """Initialize task state.

        Args:
            assets: SceneAssets with reusable USD paths.
            env_path: Root prim path for the loaded environments.
        """
        super().__init__(name="bin_conveyor_task")
        self.assets = assets
        self.env_path = env_path

        self.bins = []
        self.on_conveyor = None

    def _spawn_bin(self, rigid_bin: RigidPrim):
        """Apply spawn pose, velocities, and visibility to the new bin."""
        position, orientation = random_bin_spawn_transform()
        rigid_bin.set_world_poses(
            positions=np.array([position], dtype=np.float32),
            orientations=np.array([orientation], dtype=np.float32),
        )
        rigid_bin.set_linear_velocities(
            np.array([[0.0, -0.30, 0.0]], dtype=np.float32)
        )
        rigid_bin.set_angular_velocities(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        )
        rigid_bin.set_visibilities(np.array([True], dtype=bool))

    def post_reset(self):
        """Clear tracked bin objects after world reset."""
        for rigid_bin in self.bins:
            try:
                self.scene.remove_object(rigid_bin.name)
            except Exception:
                pass

        self.bins.clear()
        self.on_conveyor = None

    def pre_step(self, time_step_index, simulation_time):
        """Spawn a new bin when there is no bin within conveyor bounds.

        Args:
            time_step_index: Current simulation step index (unused).
            simulation_time: Current simulation time in seconds (unused).
        """
        spawn_new = False

        if self.on_conveyor is None:
            spawn_new = True
        else:
            positions, _ = self.on_conveyor.get_world_poses()
            x, y, z = positions[0]
            is_on_conveyor = (y > 0.0) and (-0.4 < x < 0.4)

            if not is_on_conveyor:
                spawn_new = True

        if spawn_new:
            name = f"bin_{len(self.bins)}"
            prim_path = f"{self.env_path}/bins/{name}"

            add_reference_to_stage(
                usd_path=self.assets.small_klt_usd,
                prim_path=prim_path,
            )

            rigid_bin = self.scene.add(
                RigidPrim(
                    prim_path,
                    name=name,
                )
            )

            self._spawn_bin(rigid_bin)
            self.on_conveyor = rigid_bin
            self.bins.append(rigid_bin)


def setup_scene(assets: SceneAssets):
    """Load stage references and adjust scene prim layout."""
    env_path = "/World/Ur10Table"

    # Load main stage (table/conveyor and related props).
    add_reference_to_stage(
        usd_path=assets.ur10_table_usd,
        prim_path=env_path,
    )

    robot_prim_path = f"{env_path}/ur10"
    if is_prim_path_valid(robot_prim_path):
        delete_prim(robot_prim_path)

    # Load warehouse background.
    add_reference_to_stage(
        usd_path=assets.background_usd,
        prim_path="/World/Background",
    )

    # Align background transform for this scene.
    XFormPrim(
        "/World/Background",
        positions=np.array([[10.00, 2.00, -1.18180]], dtype=np.float32),
        orientations=np.array([[0.7071, 0.0, 0.0, 0.7071]], dtype=np.float32),
    )


def main():
    """Build world, register task, and run simulation loop."""
    world = World(stage_units_in_meters=1.0)

    assets = SceneAssets()
    setup_scene(assets)

    task = BinConveyorTask(assets=assets)
    world.add_task(task)

    world.reset()

    # Step physics and render until app closes.
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
