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


<!-- ## Table of Contents -->

## Getting Started

### Conda Environment
It is highly recommended to install a Miniconda environment before setting up the project. You can install Miniconda by following instructions from [here](https://docs.conda.io/en/latest/miniconda.html#installing).

Create a new Conda environment called `isaacsim-examples`:

```bash
conda create -n isaacsim-examples python=3.11
```

To activate the environment:

```bash
conda activate isaacsim-examples
```

## Installation

1. Clone the repository:
    ```bash
    cd path/to/working_directory
    git clone https://github.com/telekinesis-ai/isaacsim-examples.git
    ```


2. Install the package:
    ```bash
    cd isaacsim-examples
    pip install .
    ```

    If you want to install the package in editable mode:
    ```bash
    pip install -e .
    ```


3. On Windows, it may be necessary to enable long path support to avoid installation errors due to OS limitations：

    You can also copy this code to a file which can set this for you, or use the PowerShell command as administrator from a terminal window with elevated privileges:.reg

    ```bash
    # PowerShell
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    ```
    Or
    ```bash
    # Registry (.reg) file
    Windows Registry Editor Version 5.00

    [HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem]
    "LongPathsEnabled"=dword:00000001
    ```


4. Install `isaacsim`：
    ```bash
    pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
    ```

5. Install `telekinesis-urdfs`:
    ```bash
    git clone --depth 1 https://github.com/telekinesis-ai/telekinesis-urdfs.git
    cd telekinesis-urdfs
    pip install .
    ```

    > **Note:** `telekinesis-urdfs` is a large repository containing robot model data. The initial clone and wheel build are expected to take several minutes — do not interrupt the process.

6. Install `synapse` as part of `telekinesis-ai`:
    ```bash
    pip install telekinesis[synapse]
    ```

## Examples

To be able to run the examples, follow the below steps:
1. Activate the environment:
    ```bash
    conda activate isaacsim-examples
    ```
2. Run the example

   **Robot control exmaple**

    ```bash
    python examples/ur10e_set_joint_positions.py
    ```
    Expected output: Some logs and joint positions
    ```bash
    ...
    ...
    Joint positions: [[1.5 1.5 1.5 0.0 0.5 0.5]]
    ```

    **Robot and Gripper control Example**

    ```bash
    python examples/ur10e_with_rg6.py
    ```
    Note: The first time Isaac Sim is imported, a prompt will ask you to accept the EULA at runtime. After the EULA is accepted, you will not see it again. If the EULA is not accepted, the execution will be terminated. Running Isaac Sim for the first time also take longer to loading.
    

## Support

For issues and questions:
- Create an issue in the Github repository
- Contact the development team
