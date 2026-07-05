# Contributing Guide — Build Your Own Environment

This guide walks through building an Isaac Sim scene from scratch, importing a robot, and running a Synapse-powered application following the approach used for the palletizing demo in this repository.

---

## Prerequisites

- Isaac Sim installed and running:
  - **5.1.0** → Python 3.10, conda env: `conda create -n isaacsim python=3.10`
  - **6.0.0** → Python 3.12, conda env: `conda create -n isaacsim python=3.12`
- `telekinesis-synapse` installed (see [README](README.md))
- `telekinesis-urdfs` cloned and installed
- Isaac Sim VS Code Extension installed and connected

---

## 1. Scene Structure

> The following steps walk through building the **palletizing scene** included in this repository. The same approach applies to any industrial scene — swap the warehouse environment, conveyor, and sensor for the props relevant to your application, and follow the same structure.

A well-structured scene separates environment (static props) from active components (robot, conveyor, sensor). The palletizing scene follows this layout:

```
World/
├── Environment/          ← static warehouse background
├── ConveyorBelt_A08/     ← active conveyor (physics + surface velocity)
├── WireMachineGuard_*/   ← safety fencing (static colliders)
├── Pallet_A1/            ← target pallet (static)
├── LightBeam_Sensor      ← detection sensor
├── ur10_mount/           ← robot stand (static)
└── <robot>/              ← imported robot URDF (added by user at runtime)
```

Keep the robot **out of the scene USD**. Users import their own URDF at runtime the script repositions it automatically.

---

## 2. Building the Scene Step by Step

### 2.1 Warehouse Environment

1. In Isaac Sim, open the **Content** browser.
2. Navigate to:
   ```
   Isaac Sim / Environments / Modular_Warehouse / Props /
   ```
3. Drag `sm_warehouse_a06_h10m_straight90_01` into the viewport.
4. Position it to frame your scene place the working area (conveyor + robot) near the centre.

---

### 2.2 Conveyor Belt

1. In the **Content** browser, navigate to:
   ```
   Isaac Sim / Props / Conveyors /
   ```
2. Drag `ConveyorBelt_A08` into the viewport and position it in the scene.
3. In the **Stage** panel, expand the `ConveyorBelt_A08` prim and select `Rollers`.
4. In the **Property** panel, enable the following under **Extra Properties**:
   - `Collision Enabled` ✓
   - `Kinematic Enabled` ✓
   - `Rigid Body Enabled` ✓
5. Under **Physics → Rigid Body**, enable:
   - `Velocities in Local Space` ✓
6. At the top of the **Property** panel, click **Add** and select **Surface Velocity** from the list.
7. Scroll down to the **Surface Velocity** section and enable:
   - `Surface Velocity Enabled` ✓
   - `Surface Velocity Local Space` ✓
8. Set **Surface Linear Velocity** assign a value (e.g. `0.8`) to either X, Y, or Z depending on the belt's orientation. To find the correct axis, place a cardboard box on the belt and press **Play** to see which direction it moves. Adjust the axis until boxes travel toward the robot.

---

### 2.3 Robot Stand (optional for compact arms)

For smaller collaborative robots (UR10e, Franka, Fanuc CRX, Motoman MH5), add a stand so the robot reaches the belt comfortably.

1. In the **Content** browser, navigate to:
   ```
   Isaac Sim / Props / Mounts /
   ```
   Select a suitable mount and drag it into the viewport.
2. Add it to the stage and position it next to the conveyor belt at the pick zone.
3. Note the prim path (right-click → **Copy Prim Path**) you will set this as `STAND_PRIM_PATH` in the script.

For large industrial robots (Kuka KR210, ABB IRB7600) the stand is not needed the script places them directly on the floor.

---

### 2.4 Safety Fencing

1. Open the **NVIDIA Assets** browser (`Window → NVIDIA Assets`).
2. Search for `WireMachineGuard`.
3. Drag `WireMachineGuard_A04_01` into the viewport.
4. Duplicate and position to enclose the work cell.

---

### 2.5 Pallet

1. In the **NVIDIA Assets** browser, search for `Pallet`.
2. Drag `Pallet_A1` into the viewport and position it where boxes should be stacked.

---

### 2.6 Lightbeam Sensor

The lightbeam sensor detects when a box arrives in the pick zone and signals the script to stop the conveyor.

**Adding the sensor:**

1. In Isaac Sim, go to `Create → Sensors → Lightbeam Sensor → Generic`.
2. Position the sensor so its beam crosses the conveyor path just before the pick point.

**Configuring the sensor** in the **Property** panel under **Raw USD Properties**, set:

| Property | Value |
|----------|-------|
| `enabled` | ✓ |
| `maxRange` | `1.2` |
| `minRange` | `0.4` |
| `numRays` | `1` |
| `curtainAxis` | `x=0, y=0, z=1` |

`curtainAxis z=1` means the beam sweeps vertically, which faces boxes travelling horizontally along the belt keep this default.

**Verifying the sensor** run this script in the Isaac Sim Script Editor while the simulation is playing and move a box in front of the sensor:

```python
from isaacsim.sensors.physx import _range_sensor
import numpy as np

lb  = _range_sensor.acquire_lightbeam_sensor_interface()
path = "/World/LightBeam_Sensor"  # update to your sensor prim path

hit   = lb.get_beam_hit_data(path)
depth = lb.get_linear_depth_data(path)
print("hit (1=detected, 0=clear):", hit)
print("depth (m):", depth)
```

Expected values:

| State | `hit` | `depth` |
|-------|-------|---------|
| No box in front | `[0]` | `[~1.2]` (beam reaches max range) |
| Box in front | `[1]` | `[< 1.2]` (beam hits the box) |

---

### 2.7 Cardboard Boxes

1. In the **NVIDIA Assets** browser, navigate to `NVIDIA Assets / Containers /` and select `Cardbox_C2`.
2. In the **Property** panel set **Scale** to `x=0.8, y=0.8, z=0.8` for a compact size that fits the belt.
3. Drag the box onto the start of the conveyor belt. Right-click the prim → `Add → Physics → Rigid Body with Collider Preset`.
4. Duplicate (`Ctrl+D`) to place multiple boxes at the belt entry.

---

## 3. Importing a Robot URDF

Robot URDFs are sourced from [telekinesis-urdfs](https://github.com/telekinesis-ai/telekinesis-urdfs). The user imports their robot at runtime the scene USD does **not** include a robot.

**Steps:**

1. In Isaac Sim, go to `Isaac Utils → Workflows → URDF Importer`.
2. Set **Input File** to the robot's URDF from `telekinesis-urdfs`, e.g.:
   ```
   <telekinesis-urdfs>/src/telekinesis_urdfs/models/example-robot-data/
   robots/universal_robots/urdf/ur10e.urdf
   ```
3. Enable **Fix Base** ✓ required for all manipulators.
4. Set **Reference Model** as needed.
5. Leave stiffness and damping at defaults the application script sets them via code before the simulation starts (see `set_robot_drive_gains()` in `examples/palletizing.py`).
6. Click **Import**. The robot spawns at the world origin the script repositions it.
7. Right-click the robot's root prim in the Stage panel → **Copy Prim Path**. Set this as `ROBOT_PRIM_PATH` (or the matching key in `ROBOT_REGISTRY`) in the script.

---

## 4. Saving the Scene

Save the scene **without the robot** so it can be reused across all robot brands:

1. Remove the robot prim from the stage (or undo the import).
2. `File → Save As` → save to `assets/environments/palletizing/` with a descriptive name:
   - `palletizing_stand.usd` for compact robots using the stand
   - `palletizing_floor.usd` for large robots placed on the floor

---

## 5. Testing the Scene

Once the scene is built and a robot URDF is imported, test it with the provided script:

1. Open `examples/palletizing.py` in VS Code.
2. Set `ACTIVE_ROBOT` to your robot brand and verify the prim paths match your stage.
3. Run via the **Isaac Sim VS Code Edition** extension → **Run**.

See [README.md](README.md) for the full run instructions and supported robot list.

---

## 6. Guidelines

- **No prototyping or test files in the public repo.** Only clean, working examples belong here.
- **Assets in one place.** Scene USDs go under `assets/environments/<application>/` not in the root or in `examples/`.
- **Scene USDs must not contain the robot.** Users import their own URDF; the script handles placement.
- **All public functions need docstrings and type hints.** See `examples/palletizing.py` for reference style.
