<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="100%"
        src="https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/Telekinesis+Banner.png"
      >
    </a>
  </p>

  <p align="center">
    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/v/telekinesis-ai" />
    </a>
    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/pyversions/telekinesis-ai" />
    </a>
    <a href="https://pypi.org/project/telekinesis-ai/">
      <img src="https://img.shields.io/pypi/l/telekinesis-ai" />
    </a>
    <a href="https://docs.telekinesis.ai">
      <img src="https://img.shields.io/badge/docs-telekinesis.ai-blue" />
    </a>
  </p>

  <h2>Any robot. Any task. One Physical AI platform.</h2>

  <p>
    <a href="https://docs.telekinesis.ai/">Telekinesis Docs</a>
    &nbsp;•&nbsp;
    <a href="https://discord.gg/S5v8bYAnc6">Discord</a>
    &nbsp;•&nbsp;
    <a href="https://www.linkedin.com/company/telekinesis-ai/">LinkedIn</a>
    &nbsp;•&nbsp;
    <a href="https://x.com/telekinesis_ai">X</a>
    &nbsp;•&nbsp;
    <a href="https://telekinesis.ai/">Website</a>

</p>
</div>


# isaacsim-examples

isaacsim-examples provides Isaac Sim examples for loading and controlling the robot, including set/get robot state demos and robot/end-effector model assets.


## Getting Started

> **Already have a Telekinesis or Synapse environment?** You can reuse it — skip the environment setup below and go straight to [Installation](#installation). See the [quickstart guide](https://docs.telekinesis.ai/getting-started/quickstart.html) for reference.

### Conda Environment

It is highly recommended to install a Miniconda environment before setting up the project. You can install Miniconda by following instructions from [here](https://docs.conda.io/en/latest/miniconda.html#installing).

Create a new Conda environment:

```bash
conda create -n isaacsim-examples python=3.11
conda activate isaacsim-examples
```

## Installation

1. Clone the repository:
    ```bash
    cd path/to/working_directory
    git clone https://github.com/telekinesis-ai/isaacsim-examples.git
    ```

2. On Windows, enable long path support to avoid installation errors:

    ```bash
    # PowerShell (run as administrator)
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    ```

3. Install `telekinesis-urdfs`:
    ```bash
    cd /path/to/working/directory
    git clone --depth 1 https://github.com/telekinesis-ai/telekinesis-urdfs.git
    cd telekinesis-urdfs
    pip install .
    ```

    > **Note:** `telekinesis-urdfs` is a large repository. The initial clone and wheel build are expected to take several minutes — do not interrupt the process.

4. Install `isaacsim`:
    ```bash
    pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
    ```

5. Install `telekinesis-synapse`:
    ```bash
    pip install telekinesis-synapse
    ```

## Examples

| File | Description |
|------|-------------|
| `examples/test_palletizing.py` | Multi-brand palletizing demo mounts a robot, moves to home, runs the conveyor and stops it when a box reaches the lightbeam sensor. Change one line to switch between all 7 supported brands. |
| `examples/add_physics_to_prim.py` | Adds rigid body and SDF collider physics to any prim in the open stage. |
| `examples/remove_timeline.py` | Strips animation curves from a USD stage and exports a static version useful when imported assets animate unexpectedly during physics simulation. |
| `examples/examine_tree.py` | Prints the full prim tree of the open stage useful for inspecting scene structure and finding prim paths. |


## Palletizing Scene Demo

Demonstrates a palletizing workflow across all 7 supported robot brands in Isaac Sim. A robot mounts on a stand or floor anchor, moves to its home configuration, and stops the conveyor when a box arrives at the lightbeam sensor.

### Scene USDs

Two ready-to-use scenes are provided under `assets/environments/palletizing/`:

| File | Robots |
|------|--------|
| `palletizing_with_stand.usd` | UR10e, Franka Panda, Fanuc CRX-10iA/L, Motoman MH5, Neura MAiRA 7M compact and medium arms on the stand |
| `palletizing_floor_mount.usd` | Kuka KR210 L150, ABB IRB7600 large industrial arms placed directly on the floor |

### How to Run

1. Open Isaac Sim.
2. Open the scene USD that matches your robot (`File → Open`).
3. Import the robot URDF via `Isaac Utils → URDF Importer` with **Fix Base = ON**. The robot spawns at the world origin the script repositions it automatically.
4. Open `examples/test_palletizing.py` in VS Code.
5. Set `ACTIVE_ROBOT` at the top of the file to your robot brand:

   ```python
   ACTIVE_ROBOT = "ur10e"   # or "franka", "fanuc", "motoman", "kuka", "neura", "abb"

## Support

For issues and questions:
- Create an issue in the Github repository
- Contact the development team
