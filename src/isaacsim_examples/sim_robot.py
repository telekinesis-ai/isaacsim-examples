"""Simulated manipulator — Isaac Sim backend for synapse robots.

Extends :class:`AbstractManipulator` so that any code written against the
synapse manipulator interface works identically in simulation and on real
hardware.  Kinematics (FK / IK) are inherited from the base class via
Pinocchio; joint-level commands are routed to an Isaac Sim articulation.

The public API follows the ``AbstractManipulator`` convention:
joints in **degrees**, Cartesian poses in **meters + degrees**.
Isaac Sim internally uses radians; conversion happens at the boundary
of each public method.

.. note::

    Synapse imports **must** happen before ``SimulationApp`` is created
    (DLL version conflicts).  All Omniverse / Isaac Sim imports are
    deferred to the module body, which is loaded *after* ``SimulationApp``.

TODO: -Once URDF paths are dynamically fetched rather than hardcoded in
      derived robot classes, this constructor could accept config
      parameters directly (urdf_path, frame_names, …) instead of
      requiring a robot instance.
      - where is rot type for set cartesian or set joint
"""

import copy
import pathlib
from typing import Any, Optional

import numpy as np
from loguru import logger
from typing_extensions import override

from telekinesis.synapse.robots.manipulators.abstract_robot import AbstractManipulator
try:
    import telekinesis_urdfs
except ImportError as e:
    raise ImportError(
        "telekinesis-urdfs is not installed. "
        "Install it from: https://github.com/telekinesis-ai/telekinesis-urdfs"
    ) from e
# Omniverse imports (available after SimulationApp has been created)
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation, SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
import omni.kit.commands


_PHYSICS_HZ = 60
"""Default Isaac Sim physics step rate."""

_MIN_MOTION_STEPS = 30
"""Minimum number of simulation steps for any motion command."""


class SimManipulator(AbstractManipulator):
    """Simulated manipulator backed by NVIDIA Isaac Sim.

    This class bridges the synapse :class:`AbstractManipulator` interface
    with Isaac Sim's physics engine.  It accepts any concrete synapse robot
    instance, copies its kinematic configuration, and builds its own
    Pinocchio model from the same URDF.  The ``SimulationApp`` acts as the
    hardware backend — analogous to RTDE for a Universal Robots arm.

    Usage::

        from telekinesis.synapse.robots.manipulators import universal_robots
        robot = universal_robots.UniversalRobotsUR10E()

        sim = SimManipulator(robot, simulation_app)
        sim.connect()

        sim.set_joint_positions([0, -90, -90, 0, 90, 0])
        pose = sim.get_cartesian_pose()
        sim.set_cartesian_pose([0.5, 0.2, 0.3, 180, 0, 0])

        sim.disconnect()

    The caller must ensure that ``SimulationContext`` (or ``World``) is
    created, physics is initialised, and the timeline is playing **before**
    calling :meth:`connect`.
    """

    _DEFAULT_HOLD_STIFFNESS = 1e4
    _DEFAULT_HOLD_DAMPING = 1e3

    def __init__(
        self,
        robot: AbstractManipulator,
        simulation_app,
        fix_base: bool = True,
        merge_fixed_joints: bool = False,
        motion_stiffness: float = 1e4,
        motion_damping: float = 1e3,
    ):
        """Initialise the simulated manipulator.

        Copies the kinematic configuration from *robot* and builds an
        independent Pinocchio model from the same URDF.  The Isaac Sim
        articulation is **not** created here — call :meth:`connect` to
        import the robot into the scene.

        Args:
            robot: A fully-constructed synapse manipulator whose
                configuration (URDF, frame names, TCP, joint limits) will
                be used to set up this instance.
            simulation_app: The ``SimulationApp`` singleton.
            fix_base: Fix the robot base to the world frame during URDF
                import.
            merge_fixed_joints: Merge fixed joints during URDF import.
            motion_stiffness: PD position gain used during smooth motion.
            motion_damping: PD velocity gain used during smooth motion.
        """
        super().__init__()

        self.default_tcp = robot.default_tcp
        self.active_tcp = robot.active_tcp
        self._default_joint_configuration = robot._default_joint_configuration.copy()
        self.urdf_path = robot.urdf_path
        self.srdf_path = robot.srdf_path
        # Extract robot description
        try:
            robot_description = telekinesis_urdfs.load(type(robot).__name__.lower())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load robot description for '{robot.__class__.__name__}'. "
                "Ensure telekinesis-urdfs is installed: "
                "https://github.com/telekinesis-ai/telekinesis-urdfs"
            ) from e
        model_root_dir = robot_description.root_dir


        self._build_pinocchio_model(
            urdf_path=str(self.urdf_path),
            model_dir=str(model_root_dir),
            srdf_path=str(self.srdf_path),
        )


        self._simulation_app = simulation_app
        self._fix_base = fix_base
        self._merge_fixed_joints = merge_fixed_joints
        self._base_motion_stiffness = motion_stiffness
        self._base_motion_damping = motion_damping

        self._single_manipulator: SingleManipulator | None = None
        self._prim_path: str | None = None
        self._connected: bool = False

        self._motion_kps: np.ndarray | None = None
        self._motion_kds: np.ndarray | None = None
        self._hold_kps: np.ndarray | None = None
        self._hold_kds: np.ndarray | None = None

        self._current_joints = np.array(self.ndof)
    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    @override
    def connect(self, ip: str = "") -> None:
        """Import the robot URDF into Isaac Sim and initialise the articulation.

        This is the simulation equivalent of connecting to physical
        hardware.  The *ip* parameter is accepted for interface
        compatibility but is unused.

        Raises:
            RuntimeError: If already connected or if the URDF import /
                articulation initialisation fails.
            FileNotFoundError: If the URDF file does not exist.
        """
        if self._connected:
            raise RuntimeError(
                "Already connected. Call disconnect() before reconnecting.")

        self.ip = ip

        urdf = pathlib.Path(str(self.urdf_path))
        if not urdf.exists():
            raise FileNotFoundError(f"URDF not found: {urdf}")

        self._prim_path = self._import_urdf()
        self._render(5)

        sim_context = SimulationContext.instance()
        if sim_context is not None:
            sim_context.stop()
            self._render(3)

        # SingleArticulation (no gripper yet); attach_tool() upgrades to SingleManipulator.
        self._single_manipulator = SingleArticulation(
            prim_path=self._prim_path,
            name="sim_manipulator",
        )

        if sim_context is not None:
            sim_context.reset()
            self._render(5)
        self._single_manipulator.initialize()

        art_ndof = self._single_manipulator.num_dof

        self._motion_kps = np.full(art_ndof, self._base_motion_stiffness)
        self._motion_kds = np.full(art_ndof, self._base_motion_damping)
        self._hold_kps = np.full(art_ndof, self._DEFAULT_HOLD_STIFFNESS)
        self._hold_kds = np.full(art_ndof, self._DEFAULT_HOLD_DAMPING)
        self._apply_hold_gains()

        self._connected = True

        if self.ndof != art_ndof:
            logger.warning(
                f"DOF mismatch: Pinocchio model has {self.ndof} joints, "
                f"Isaac Sim articulation has {art_ndof}. "
                f"Joint arrays will be padded/truncated automatically."
            )

        logger.info(
            f"SimManipulator connected: {urdf.stem} | "
            f"pinocchio ndof={self.ndof}, articulation num_dof={art_ndof}"
        )

    @override
    def disconnect(self) -> None:
        """Remove the robot from the Isaac Sim stage and release resources.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected:
            raise RuntimeError("Not connected. Nothing to disconnect.")

        self._render(5)

        root_prim = "/" + self._prim_path.strip("/").split("/")[0]
        omni.kit.commands.execute(
            "DeletePrims",
            paths=[root_prim, "/visuals", "/colliders", "/meshes"],
        )
        self._render(10)

        self._single_manipulator = None
        self._prim_path = None
        self._connected = False

        logger.info(f"SimManipulator disconnected: {self.urdf_path}")

    @override
    def set_joint_positions(
        self,
        joint_positions: list[float] | np.ndarray,
        speed: float = 40,
        acceleration: float = 80.0,
        asynchronous: bool = False,
    ) -> None:
        """Move the robot to the specified joint positions.

        Joint positions are specified in **degrees**, consistent with the
        ``AbstractManipulator`` convention.  Motion is executed via PD
        control in the physics simulation.

        Args:
            joint_positions: Target joint positions in degrees.
            speed: Approximate joint speed in deg/s.  Used to compute
                the number of simulation steps for the motion.
            acceleration: Approximate joint acceleration in deg/s².
                Scales the PD gains during motion.
            asynchronous: If ``True``, set PD targets and return
                immediately without waiting for the motion to complete.

        Raises:
            RuntimeError: If not connected.
            TypeError: If *joint_positions* is not a list or ndarray.
            ValueError: If the array shape is wrong or values are out
                of joint limits.
        """
        self._check_connected()
        logger.debug(f"Set joints deg {joint_positions}")
        if not isinstance(joint_positions, (list, np.ndarray)):
            raise TypeError(
                "joint_positions must be a list or numpy array, "
                f"got {type(joint_positions).__name__}.")

        joint_positions = np.asarray(joint_positions, dtype=float).flatten()
        self._current_joints = joint_positions
        if joint_positions.shape[0] != self.ndof:
            raise ValueError(
                f"Expected {self.ndof} joint positions, "
                f"got {joint_positions.shape[0]}.")

        if not self.in_joint_limits(joint_positions, rot_type="deg"):
            raise ValueError("Joint positions are outside limits.")

        target_rad = np.deg2rad(joint_positions)
        arm_indices = np.arange(self.ndof).tolist()

        if asynchronous:
            self._single_manipulator.apply_action(
                ArticulationAction(joint_positions=target_rad.tolist(), joint_indices=arm_indices)
            )
            return

        current_rad = self._single_manipulator.get_joint_positions()[:self.ndof]
        steps = self._compute_motion_steps(current_rad, target_rad, speed)
        self._move_joints(target_rad, steps)

    @override
    def get_joint_positions(self) -> list[float]:
        """Return the current joint positions in degrees.

        Only the first ``ndof`` values from the articulation are returned
        (extra DOFs from the URDF import are excluded).

        Raises:
            RuntimeError: If not connected.
        """
        self._check_connected()

        positions_rad = self._single_manipulator.get_joint_positions()[:self.ndof]
        return np.rad2deg(positions_rad).tolist()

    @override
    def set_cartesian_pose(
        self,
        cartesian_pose: list[float] | np.ndarray,
        speed: float = 40.0,
        acceleration: float = 80.0,
        asynchronous: bool = False,
    ) -> None:
        """Move the end-effector to a Cartesian pose via IK.

        The pose is given as ``[x, y, z, rx, ry, rz]`` in metres and
        degrees.  Internally this solves inverse kinematics (inherited
        from ``AbstractManipulator``) and then calls
        :meth:`set_joint_positions` with the result.

        Args:
            cartesian_pose: Target pose ``[x, y, z, rx, ry, rz]``
                in metres and degrees.
            speed: Approximate joint speed in deg/s.
            acceleration: Approximate joint acceleration in deg/s².
            asynchronous: If ``True``, set PD targets and return
                immediately.

        Raises:
            RuntimeError: If not connected or if IK fails.
            TypeError: If *cartesian_pose* is not a list or ndarray.
            ValueError: If the pose does not have 6 elements.
        """
        self._check_connected()

        if not isinstance(cartesian_pose, (list, np.ndarray)):
            raise TypeError(
                "cartesian_pose must be a list or numpy array, "
                f"got {type(cartesian_pose).__name__}.")

        cartesian_pose = np.asarray(cartesian_pose, dtype=float).flatten()

        if cartesian_pose.shape[0] != 6:
            raise ValueError(
                "cartesian_pose must have 6 elements [x, y, z, rx, ry, rz], "
                f"got {cartesian_pose.shape[0]}.")

        q_deg = self.inverse_kinematics(
            target_pose=cartesian_pose,
            rot_type="deg",
            q_init=self._current_joints,
            solver='multi_start_clik'
        )

        self.set_joint_positions(
            q_deg,
            speed=speed,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )

    @override
    def get_cartesian_pose(self) -> list[float]:
        """Return the current end-effector pose in metres and degrees.

        Uses forward kinematics on the current joint positions.

        Returns:
            Pose as ``[x, y, z, rx, ry, rz]`` in metres and degrees.

        Raises:
            RuntimeError: If not connected.
        """
        self._check_connected()

        q_deg = self.get_joint_positions()
        pose = self.forward_kinematics(q_deg, rot_type="deg")
        return pose.tolist() if isinstance(pose, np.ndarray) else list(pose)


    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def single_manipulator(self) -> SingleManipulator | None:
        """The Isaac Sim SingleManipulator handle (``None`` before connect)."""
        return self._single_manipulator

    @property
    def prim_path(self) -> str | None:
        """USD prim path of the articulation root."""
        return self._prim_path

    @property
    def simulation_app(self):
        """The ``SimulationApp`` singleton."""
        return self._simulation_app

    # ------------------------------------------------------------------
    # Private helpers — Isaac Sim interaction
    # ------------------------------------------------------------------

    def _check_connected(self) -> None:
        """Raise ``RuntimeError`` if not connected."""
        if not self._connected:
            raise RuntimeError(
                "Not connected. Call connect() before sending commands.")

    def _import_urdf(self) -> str:
        """Import the robot URDF into the Isaac Sim stage."""
        status, import_config = omni.kit.commands.execute(
            "URDFCreateImportConfig"
        )
        if not status:
            raise RuntimeError("Failed to create URDF import config.")

        import_config.merge_fixed_joints = self._merge_fixed_joints
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = self._fix_base
        import_config.distance_scale = 1.0
        import_config.density = 0.0

        import_config.default_drive_type = (
            _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        )
        import_config.default_drive_strength = self._DEFAULT_HOLD_STIFFNESS
        import_config.default_position_drive_damping = self._DEFAULT_HOLD_DAMPING

        urdf_path = str(self.urdf_path)
        status, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
            get_articulation_root=True,
        )
        if not status:
            raise RuntimeError(f"Failed to import URDF: {urdf_path}")

        logger.info(f"Imported URDF: {urdf_path}")
        logger.info(f"Articulation prim path: {prim_path}")
        return prim_path

    def _pad_joint_positions(self, positions: np.ndarray) -> np.ndarray:
        """Pad or truncate *positions* to match the articulation DOF count."""
        positions = np.asarray(positions, dtype=float).flatten()
        art_ndof = self._single_manipulator.num_dof
        if len(positions) == art_ndof:
            return positions
        padded = np.zeros(art_ndof)
        n = min(len(positions), art_ndof)
        padded[:n] = positions[:n]
        return padded

    def _move_joints(self, arm_positions_rad: np.ndarray, steps: int) -> None:
        """Smooth motion to arm target over *steps* sim steps.

        Only arm joints (first ``ndof``) are commanded; gripper joints are
        left at their current positions via ``joint_indices``.
        """
        arm_start = self._single_manipulator.get_joint_positions()[:self.ndof]
        delta = arm_positions_rad[:self.ndof] - arm_start
        arm_indices = np.arange(self.ndof).tolist()

        for i in range(1, steps + 1):
            alpha = i / steps
            waypoint = (arm_start + alpha * delta).tolist()
            self._single_manipulator.apply_action(
                ArticulationAction(joint_positions=waypoint, joint_indices=arm_indices)
            )
            self._simulation_app.update()

    def _teleport_joints(self, positions_rad: np.ndarray) -> None:
        """Instantly set arm joint positions (no physics-driven motion)."""
        positions = np.asarray(positions_rad, dtype=float).flatten()[:self.ndof]
        arm_indices = np.arange(self.ndof).tolist()
        self._single_manipulator.set_joint_positions(positions, joint_indices=arm_indices)
        self._render(5)

    def _compute_motion_steps(
        self,
        current_rad: np.ndarray,
        target_rad: np.ndarray,
        speed_deg_s: float,
    ) -> int:
        """Estimate the number of simulation steps for a motion."""
        max_delta_deg = float(np.max(np.abs(np.rad2deg(target_rad - current_rad))))
        if max_delta_deg < 0.1:
            return _MIN_MOTION_STEPS

        duration_s = max_delta_deg / max(speed_deg_s, 1.0)
        steps = int(duration_s * _PHYSICS_HZ)
        return max(steps, _MIN_MOTION_STEPS)

    def _apply_hold_gains(self) -> None:
        """No-op: SingleManipulator drive gains are set at URDF import time."""

    def _apply_motion_gains(self) -> None:
        """No-op: SingleManipulator drive gains are set at URDF import time."""

    def _render(self, n: int = 5) -> None:
        """Step the simulation app *n* times to let the viewport update."""
        for _ in range(n):
            self._simulation_app.update()

    # ------------------------------------------------------------------
    # Unimplemented abstract methods — TODO: Isaac Sim backend
    #
    # Stubs that satisfy AbstractManipulator's abstract interface so this
    # class (and its concrete subclasses) can be instantiated. Each method
    # raises NotImplementedError until a real Isaac Sim implementation is
    # provided.
    # ------------------------------------------------------------------

    @override
    def set_cartesian_pose_in_joint_space(
        self,
        cartesian_pose: list[float] | np.ndarray,
        speed: float = 60,
        acceleration: float = 80,
        asynchronous: bool = False,
    ) -> None:
        raise NotImplementedError(
            "set_cartesian_pose_in_joint_space is not implemented for SimManipulator yet.")

    @override
    def set_joint_position_in_cartesian_space(
        self,
        joint_positions: list[float] | np.ndarray,
        speed: float = 1.05,
        acceleration: float = 1.4,
        asynchronous: bool = False,
    ) -> None:
        raise NotImplementedError(
            "set_joint_position_in_cartesian_space is not implemented for SimManipulator yet.")

    @override
    def move_until_contact(
        self,
        cartesian_speed: list[float] | np.ndarray,
        direction: list[float] | np.ndarray,
        acceleration: float = 1.4,
    ) -> None:
        raise NotImplementedError(
            "move_until_contact is not implemented for SimManipulator yet.")

    @override
    def move_in_force_mode(
        self,
        task_frame: list[float] | np.ndarray,
        selection_vector: list[int] | np.ndarray,
        force: list[float] | np.ndarray,
        force_mode_type: int,
        limits_speed: list[float] | np.ndarray,
    ) -> None:
        raise NotImplementedError(
            "move_in_force_mode is not implemented for SimManipulator yet.")

    @override
    def move_joint_path(self, path: list[list[float]], asynchronous: bool = False) -> None:
        raise NotImplementedError(
            "move_joint_path is not implemented for SimManipulator yet.")

    @override
    def move_cartesian_path(self, path: list[list[float]], asynchronous: bool = False) -> None:
        raise NotImplementedError(
            "move_cartesian_path is not implemented for SimManipulator yet.")

    @override
    def move_path(self, path: Any, asynchronous: bool = False) -> None:
        raise NotImplementedError(
            "move_path is not implemented for SimManipulator yet.")

    @override
    def servo_joint(
        self,
        q: list[float],
        speed: float,
        acceleration: float,
        time: float,
        lookahead_time: float,
        gain: float,
    ) -> None:
        raise NotImplementedError(
            "servo_joint is not implemented for SimManipulator yet.")

    @override
    def servo_cartesian(
        self,
        pose: list[float],
        speed: float,
        acceleration: float,
        time: float,
        lookahead_time: float,
        gain: float,
    ) -> None:
        raise NotImplementedError(
            "servo_cartesian is not implemented for SimManipulator yet.")

    @override
    def servo_circular(
        self,
        pose: list[float],
        speed: float = 0.25,
        acceleration: float = 1.2,
        blend: float = 0.0,
    ) -> None:
        raise NotImplementedError(
            "servo_circular is not implemented for SimManipulator yet.")

    @override
    def servo_stop(self, deceleration: float = 10.0) -> None:
        raise NotImplementedError(
            "servo_stop is not implemented for SimManipulator yet.")

    @override
    def speed_joint(
        self,
        qd: list[float],
        acceleration: float = 0.5,
        time: float = 0.0,
    ) -> None:
        raise NotImplementedError(
            "speed_joint is not implemented for SimManipulator yet.")

    @override
    def speed_cartesian(
        self,
        xd: list[float],
        acceleration: float = 0.25,
        time: float = 0.0,
    ) -> None:
        raise NotImplementedError(
            "speed_cartesian is not implemented for SimManipulator yet.")

    @override
    def speed_stop(self, deceleration: float = 10.0) -> None:
        raise NotImplementedError(
            "speed_stop is not implemented for SimManipulator yet.")

    @override
    def stop_force_mode(self) -> None:
        raise NotImplementedError(
            "stop_force_mode is not implemented for SimManipulator yet.")

    @override
    def set_force_mode_damping(self, damping: float) -> None:
        raise NotImplementedError(
            "set_force_mode_damping is not implemented for SimManipulator yet.")

    @override
    def set_force_mode_gain_scaling(self, scaling: float) -> None:
        raise NotImplementedError(
            "set_force_mode_gain_scaling is not implemented for SimManipulator yet.")

    @override
    def tool_contact(self, direction: list[float]) -> bool:
        raise NotImplementedError(
            "tool_contact is not implemented for SimManipulator yet.")

    @override
    def zero_ft_sensor(self) -> None:
        raise NotImplementedError(
            "zero_ft_sensor is not implemented for SimManipulator yet.")

    @override
    def set_external_force_torque(self, wrench: list[float]) -> None:
        raise NotImplementedError(
            "set_external_force_torque is not implemented for SimManipulator yet.")

    @override
    def stop_cartesian_motion(self, stopping_speed: float = 0.5) -> None:
        raise NotImplementedError(
            "stop_cartesian_motion is not implemented for SimManipulator yet.")

    @override
    def stop_joint_motion(self, stopping_speed: float = 0.5) -> None:
        raise NotImplementedError(
            "stop_joint_motion is not implemented for SimManipulator yet.")

    @override
    def trigger_protective_stop(self) -> None:
        raise NotImplementedError(
            "trigger_protective_stop is not implemented for SimManipulator yet.")

    @override
    def start_jog(
        self,
        speeds: list[float],
        feature: int = 0,
        acc: float = 0.5,
        custom_frame: list[float] | None = None,
    ) -> None:
        raise NotImplementedError(
            "start_jog is not implemented for SimManipulator yet.")

    @override
    def stop_jog(self) -> None:
        raise NotImplementedError(
            "stop_jog is not implemented for SimManipulator yet.")

    @override
    def get_timestamp(self) -> float:
        raise NotImplementedError(
            "get_timestamp is not implemented for SimManipulator yet.")

    @override
    def get_robot_mode(self) -> str:
        raise NotImplementedError(
            "get_robot_mode is not implemented for SimManipulator yet.")

    @override
    def get_robot_status(self) -> str:
        raise NotImplementedError(
            "get_robot_status is not implemented for SimManipulator yet.")

    @override
    def get_safety_mode(self) -> str:
        raise NotImplementedError(
            "get_safety_mode is not implemented for SimManipulator yet.")

    @override
    def get_runtime_state(self) -> str:
        raise NotImplementedError(
            "get_runtime_state is not implemented for SimManipulator yet.")

    @override
    def is_protective_stopped(self) -> bool:
        raise NotImplementedError(
            "is_protective_stopped is not implemented for SimManipulator yet.")

    @override
    def is_emergency_stopped(self) -> bool:
        raise NotImplementedError(
            "is_emergency_stopped is not implemented for SimManipulator yet.")

    @override
    def is_program_running_on_controller(self) -> bool:
        raise NotImplementedError(
            "is_program_running_on_controller is not implemented for SimManipulator yet.")

    @override
    def is_steady(self) -> bool:
        raise NotImplementedError(
            "is_steady is not implemented for SimManipulator yet.")

    @override
    def is_pose_within_safety_limits(self, pose: list[float]) -> bool:
        raise NotImplementedError(
            "is_pose_within_safety_limits is not implemented for SimManipulator yet.")

    @override
    def is_joints_within_safety_limits(self, q: list[float]) -> bool:
        raise NotImplementedError(
            "is_joints_within_safety_limits is not implemented for SimManipulator yet.")

    @override
    def get_actual_joint_velocities(self) -> list[float]:
        raise NotImplementedError(
            "get_actual_joint_velocities is not implemented for SimManipulator yet.")

    @override
    def get_actual_joint_currents(self) -> list[float]:
        raise NotImplementedError(
            "get_actual_joint_currents is not implemented for SimManipulator yet.")

    @override
    def get_joint_temperatures(self) -> list[float]:
        raise NotImplementedError(
            "get_joint_temperatures is not implemented for SimManipulator yet.")

    @override
    def get_joint_torques(self) -> list[float]:
        raise NotImplementedError(
            "get_joint_torques is not implemented for SimManipulator yet.")

    @override
    def get_actual_joint_positions_history(self, steps: int = 0) -> list[float]:
        raise NotImplementedError(
            "get_actual_joint_positions_history is not implemented for SimManipulator yet.")

    @override
    def get_actual_tcp_speed(self) -> list[float]:
        raise NotImplementedError(
            "get_actual_tcp_speed is not implemented for SimManipulator yet.")

    @override
    def get_actual_tcp_force(self) -> list[float]:
        raise NotImplementedError(
            "get_actual_tcp_force is not implemented for SimManipulator yet.")

    @override
    def get_target_waypoint(self) -> list[float]:
        raise NotImplementedError(
            "get_target_waypoint is not implemented for SimManipulator yet.")

    @override
    def get_tcp_offset(self) -> list[float]:
        raise NotImplementedError(
            "get_tcp_offset is not implemented for SimManipulator yet.")

    @override
    def get_ft_raw_wrench(self) -> list[float]:
        raise NotImplementedError(
            "get_ft_raw_wrench is not implemented for SimManipulator yet.")

    @override
    def get_digital_in_state(self, input_id: int) -> bool:
        raise NotImplementedError(
            "get_digital_in_state is not implemented for SimManipulator yet.")

    @override
    def get_digital_out_state(self, output_id: int) -> bool:
        raise NotImplementedError(
            "get_digital_out_state is not implemented for SimManipulator yet.")

    @override
    def get_payload(self) -> float:
        raise NotImplementedError(
            "get_payload is not implemented for SimManipulator yet.")

    @override
    def get_payload_cog(self) -> list[float]:
        raise NotImplementedError(
            "get_payload_cog is not implemented for SimManipulator yet.")

    @override
    def set_payload(self, mass: float, cog: list[float] | None = None) -> None:
        raise NotImplementedError(
            "set_payload is not implemented for SimManipulator yet.")

    @override
    def set_tcp(self, tcp_offset: list[float]) -> None:
        raise NotImplementedError(
            "set_tcp is not implemented for SimManipulator yet.")

    @override
    def get_step_time(self) -> float:
        raise NotImplementedError(
            "get_step_time is not implemented for SimManipulator yet.")

    @override
    def get_speed_scaling(self) -> float:
        raise NotImplementedError(
            "get_speed_scaling is not implemented for SimManipulator yet.")

    @override
    def get_async_operation_progress(self) -> str:
        raise NotImplementedError(
            "get_async_operation_progress is not implemented for SimManipulator yet.")

    @override
    def get_freedrive_status(self) -> int:
        raise NotImplementedError(
            "get_freedrive_status is not implemented for SimManipulator yet.")

    @override
    def start_freedrive_mode(
        self,
        free_axes: list[int] | None = None,
        feature: list[float] | None = None,
    ) -> None:
        raise NotImplementedError(
            "start_freedrive_mode is not implemented for SimManipulator yet.")

    @override
    def stop_freedrive_mode(self) -> None:
        raise NotImplementedError(
            "stop_freedrive_mode is not implemented for SimManipulator yet.")

    @override
    def start_teach_mode(self) -> None:
        raise NotImplementedError(
            "start_teach_mode is not implemented for SimManipulator yet.")

    @override
    def stop_teach_mode(self) -> None:
        raise NotImplementedError(
            "stop_teach_mode is not implemented for SimManipulator yet.")

    @override
    def get_on_robot_inverse_kinematics(
        self,
        pose: list[float],
        qnear: Optional[list[float]] = None,
        max_position_error: float = 1e-10,
        max_orientation_error: float = 1e-10,
    ) -> list[float]:
        raise NotImplementedError(
            "get_on_robot_inverse_kinematics is not implemented for SimManipulator yet.")

    @override
    def get_on_robot_forward_kinematics(
        self,
        q: Optional[list[float]] = None,
        tcp_offset: Optional[list[float]] = None,
    ) -> list[float]:
        raise NotImplementedError(
            "get_on_robot_forward_kinematics is not implemented for SimManipulator yet.")

    @override
    def get_pose_transform(
        self,
        source_pose: list[float],
        relative_transform: list[float],
    ) -> list[float]:
        raise NotImplementedError(
            "get_pose_transform is not implemented for SimManipulator yet.")

    @override
    def send_custom_script_function(self, function_name: str, script: str) -> bool:
        raise NotImplementedError(
            "send_custom_script_function is not implemented for SimManipulator yet.")

    @override
    def send_custom_script(self, script: str) -> bool:
        raise NotImplementedError(
            "send_custom_script is not implemented for SimManipulator yet.")

    @override
    def send_custom_script_file(self, file_path: str) -> bool:
        raise NotImplementedError(
            "send_custom_script_file is not implemented for SimManipulator yet.")

    @override
    def set_custom_script_file(self, file_path: str) -> None:
        raise NotImplementedError(
            "set_custom_script_file is not implemented for SimManipulator yet.")

    @override
    def set_watchdog(self, min_frequency: float = 10.0) -> bool:
        raise NotImplementedError(
            "set_watchdog is not implemented for SimManipulator yet.")

    @override
    def kick_watchdog(self) -> bool:
        raise NotImplementedError(
            "kick_watchdog is not implemented for SimManipulator yet.")

    @override
    def unlock_protective_stop(self) -> None:
        raise NotImplementedError(
            "unlock_protective_stop is not implemented for SimManipulator yet.")

    @override
    def ft_rtde_input_enable(
        self,
        enable: bool,
        sensor_mass: float = 0.0,
        sensor_measuring_offset: list[float] | None = None,
        sensor_cog: list[float] | None = None,
    ) -> bool:
        raise NotImplementedError(
            "ft_rtde_input_enable is not implemented for SimManipulator yet.")

    @override
    def enable_external_ft_sensor(
        self,
        enable: bool,
        sensor_mass: float = 0.0,
        sensor_measuring_offset: list[float] | None = None,
        sensor_cog: list[float] | None = None,
    ) -> bool:
        raise NotImplementedError(
            "enable_external_ft_sensor is not implemented for SimManipulator yet.")

    @override
    def set_gravity(self, direction: list[float]) -> bool:
        raise NotImplementedError(
            "set_gravity is not implemented for SimManipulator yet.")

    @override
    def get_actual_tool_flange_pose(self) -> list[float]:
        raise NotImplementedError(
            "get_actual_tool_flange_pose is not implemented for SimManipulator yet.")

    @override
    def get_inverse_kinematics_has_solution(
        self,
        pose: list[float],
        qnear: list[float] | None = None,
        max_position_error: float = 1e-10,
        max_orientation_error: float = 1e-10,
    ) -> bool:
        raise NotImplementedError(
            "get_inverse_kinematics_has_solution is not implemented for SimManipulator yet.")

    @override
    def get_mass_matrix(
        self,
        q: list[float] | None = None,
        include_rotors_inertia: bool = False,
    ) -> list[float]:
        raise NotImplementedError(
            "get_mass_matrix is not implemented for SimManipulator yet.")

    @override
    def get_coriolis_and_centrifugal_torques(
        self,
        q: list[float] | None = None,
        qd: list[float] | None = None,
    ) -> list[float]:
        raise NotImplementedError(
            "get_coriolis_and_centrifugal_torques is not implemented for SimManipulator yet.")

    @override
    def get_target_joint_accelerations(self) -> list[float]:
        raise NotImplementedError(
            "get_target_joint_accelerations is not implemented for SimManipulator yet.")

    @override
    def get_jacobian(
        self,
        q: list[float] | None = None,
        tcp: list[float] | None = None,
    ) -> list[float]:
        raise NotImplementedError(
            "get_jacobian is not implemented for SimManipulator yet.")

    @override
    def get_jacobian_time_derivative(
        self,
        q: list[float] | None = None,
        qd: list[float] | None = None,
        tcp: list[float] | None = None,
    ) -> list[float]:
        raise NotImplementedError(
            "get_jacobian_time_derivative is not implemented for SimManipulator yet.")

    @override
    def start_contact_detection(self, direction: list[float] | None = None) -> bool:
        raise NotImplementedError(
            "start_contact_detection is not implemented for SimManipulator yet.")

    @override
    def read_contact_detection(self) -> bool:
        raise NotImplementedError(
            "read_contact_detection is not implemented for SimManipulator yet.")

    @override
    def stop_contact_detection(self) -> bool:
        raise NotImplementedError(
            "stop_contact_detection is not implemented for SimManipulator yet.")

    @override
    def direct_torque(self, torques: list[float], friction_comp: bool = True) -> bool:
        raise NotImplementedError(
            "direct_torque is not implemented for SimManipulator yet.")

    @override
    def set_target_payload(
        self,
        mass: float,
        cog: list[float] | None = None,
        inertia: list[float] | None = None,
    ) -> bool:
        raise NotImplementedError(
            "set_target_payload is not implemented for SimManipulator yet.")

    @override
    def get_target_joint_positions(self) -> list[float]:
        raise NotImplementedError(
            "get_target_joint_positions is not implemented for SimManipulator yet.")

    @override
    def get_target_joint_velocities(self) -> list[float]:
        raise NotImplementedError(
            "get_target_joint_velocities is not implemented for SimManipulator yet.")

    @override
    def get_target_joint_currents(self) -> list[float]:
        raise NotImplementedError(
            "get_target_joint_currents is not implemented for SimManipulator yet.")

    @override
    def get_target_joint_moments(self) -> list[float]:
        raise NotImplementedError(
            "get_target_joint_moments is not implemented for SimManipulator yet.")

    @override
    def get_target_tcp_pose(self) -> list[float]:
        raise NotImplementedError(
            "get_target_tcp_pose is not implemented for SimManipulator yet.")

    @override
    def get_target_tcp_speed(self) -> list[float]:
        raise NotImplementedError(
            "get_target_tcp_speed is not implemented for SimManipulator yet.")

    @override
    def get_target_speed_fraction(self) -> float:
        raise NotImplementedError(
            "get_target_speed_fraction is not implemented for SimManipulator yet.")

    @override
    def get_actual_main_voltage(self) -> float:
        raise NotImplementedError(
            "get_actual_main_voltage is not implemented for SimManipulator yet.")

    @override
    def get_actual_robot_voltage(self) -> float:
        raise NotImplementedError(
            "get_actual_robot_voltage is not implemented for SimManipulator yet.")

    @override
    def get_actual_robot_current(self) -> float:
        raise NotImplementedError(
            "get_actual_robot_current is not implemented for SimManipulator yet.")

    @override
    def get_actual_joint_voltage(self) -> list[float]:
        raise NotImplementedError(
            "get_actual_joint_voltage is not implemented for SimManipulator yet.")

    @override
    def get_actual_current_as_torque(self) -> list[float]:
        raise NotImplementedError(
            "get_actual_current_as_torque is not implemented for SimManipulator yet.")

    @override
    def get_actual_digital_input_bits(self) -> int:
        raise NotImplementedError(
            "get_actual_digital_input_bits is not implemented for SimManipulator yet.")

    @override
    def get_actual_digital_output_bits(self) -> int:
        raise NotImplementedError(
            "get_actual_digital_output_bits is not implemented for SimManipulator yet.")

    @override
    def get_standard_analog_input(self, index: int) -> float:
        raise NotImplementedError(
            "get_standard_analog_input is not implemented for SimManipulator yet.")

    @override
    def get_standard_analog_output(self, index: int) -> float:
        raise NotImplementedError(
            "get_standard_analog_output is not implemented for SimManipulator yet.")

    @override
    def get_output_int_register(self, output_id: int) -> int:
        raise NotImplementedError(
            "get_output_int_register is not implemented for SimManipulator yet.")

    @override
    def get_output_double_register(self, output_id: int) -> float:
        raise NotImplementedError(
            "get_output_double_register is not implemented for SimManipulator yet.")

    @override
    def get_actual_execution_time(self) -> float:
        raise NotImplementedError(
            "get_actual_execution_time is not implemented for SimManipulator yet.")

    @override
    def get_actual_tool_accelerometer(self) -> list[float]:
        raise NotImplementedError(
            "get_actual_tool_accelerometer is not implemented for SimManipulator yet.")

    @override
    def get_actual_momentum(self) -> float:
        raise NotImplementedError(
            "get_actual_momentum is not implemented for SimManipulator yet.")

    @override
    def get_safety_status_bits(self) -> int:
        raise NotImplementedError(
            "get_safety_status_bits is not implemented for SimManipulator yet.")

    @override
    def get_speed_scaling_combined(self) -> float:
        raise NotImplementedError(
            "get_speed_scaling_combined is not implemented for SimManipulator yet.")

    @override
    def get_joint_control_output(self) -> list[float]:
        raise NotImplementedError(
            "get_joint_control_output is not implemented for SimManipulator yet.")

    @override
    def get_joint_mode(self) -> list[int]:
        raise NotImplementedError(
            "get_joint_mode is not implemented for SimManipulator yet.")

    @override
    def get_payload_inertia(self) -> list[float]:
        raise NotImplementedError(
            "get_payload_inertia is not implemented for SimManipulator yet.")

    @override
    def start_file_recording(
        self, filename: str, variables: list[str] | None = None
    ) -> bool:
        raise NotImplementedError(
            "start_file_recording is not implemented for SimManipulator yet.")

    @override
    def stop_file_recording(self) -> bool:
        raise NotImplementedError(
            "stop_file_recording is not implemented for SimManipulator yet.")

    @override
    def get_controller_frequency(self) -> float:
        raise NotImplementedError(
            "get_controller_frequency is not implemented for SimManipulator yet.")

    @override
    def move_with_cartesian_velocity(
        self,
        xd: list[float],
        acceleration: float = 0.25,
        time: float = 0.0,
    ) -> None:
        """Command Cartesian velocity (speed controller).

        The TCP moves at the given velocity until :meth:`stop_speed_motion` is called
        or ``time`` elapses.

        Args:
            xd (list[float]): TCP velocity ``[vx, vy, vz, vrx, vry, vrz]``.
                vx, vy, vz in m/s; vrx, vry, vrz in deg/s.
            acceleration (float): TCP acceleration [m/s²]. Defaults to
                ``0.25``.
            time (float): Duration [s]. ``0.0`` means run indefinitely.
                Defaults to ``0.0``.

        Raises:
            NotImplementedError: Not implemented for this manipulator brand.
        """
        raise NotImplementedError(
            "move_with_cartesian_velocity must be implemented by the derived class."
        )

    @override
    def move_with_joint_speed(
        self,
        qd: list[float],
        acceleration: float = 0.5,
        time: float = 0.0,
    ) -> None:
        """Command joint-space velocities (speed controller).

        The robot moves at the given joint velocities until :meth:`stop_speed_motion`
        is called or ``time`` elapses.

        Args:
            qd (list[float]): Joint velocities ``[j1…j6]`` [deg/s].
            acceleration (float): Joint acceleration [deg/s²]. Defaults to
                ``0.5``.
            time (float): Duration [s]. ``0.0`` means run indefinitely.
                Defaults to ``0.0``.

        Raises:
            NotImplementedError: Not implemented for this manipulator brand.
        """
        raise NotImplementedError("move_with_joint_speed must be implemented by the derived class.")

    @override
    def stop_speed_motion(self, deceleration: float = 10.0) -> None:
        """Stop an active speed controller.

        Stops whichever speed mode is currently active (joint or Cartesian).

        Args:
            deceleration (float): Deceleration rate. Units depend on the active
                speed mode:
                - After ``move_with_joint_speed``: joint deceleration [deg/s²].
                - After ``move_with_cartesian_velocity``: tool deceleration
                  [m/s²] — note the deceleration is passed as-is after
                  deg/s² → rad/s² conversion, which is incorrect for Cartesian
                  mode; pass a value scaled accordingly.
                Defaults to ``10.0``.

        Raises:
            NotImplementedError: Not implemented for this manipulator brand.
        """
        raise NotImplementedError("stop_speed_motion must be implemented by the derived class.")
