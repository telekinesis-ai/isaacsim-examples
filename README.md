# isaacsim-examples

isaacsim-examples provides Isaac Sim examples for loading and controlling the robot, including set/get robot state demos and robot/end-effector model assets.


<!-- ## Table of Contents -->

## Getting Started

### Conda Environment
It is highly recommended to install a Miniconda environment before setting up the project. You can install Miniconda by following instructions from [here](https://docs.conda.io/en/latest/miniconda.html#installing).

Create a new Conda environment called `isaacsim-examples`:
    ```
    conda create -n isaacsim-examples python=3.11
    ```

To activate the environment:
    ```
    conda activate isaacsim-examples
    ```

## Installation

1. Clone the repository:
    ```bash
    cd path/to/working_directory
    git clone -b develop https://gitlab.com/telekinesis/isaacsim-examples.git
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

    You can also copy this code to a file which can set this for you, or use the PowerShell command from a terminal window with elevated privileges:.reg

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


## Dependencies

## Examples

To be able to run the examples, follow the below steps:
1. Activate the environment:
    ```bash
    conda activate isaacsim-examples
    ```
2. Run the example
    ```bash
    python examples/ur10e_set_joint_positions.py
    ```
    Expected output: Some logs and joint positions：
    ```bash
    ...
    ...
    Joint positions: [[1.5 1.5 1.5 0.  0.5 0.5]]
    ```
    Note: The first time isaacsim is imported, a prompt asks you to accept the EULA at runtime. After the EULA is accepted, you will not see it again. If the EULA is not accepted, the execution will be terminated.
    

## License


## Support

For issues and questions:
- Create an issue in the GitLab repository
- Contact the development team
