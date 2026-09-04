"""Palletizing application for a UR10e with a suction gripper.

The cycle is packaged as :class:`PalletizingApplication`: ``setup`` prepares the
cell and connects the hardware, ``run`` palletizes the boxes, ``shutdown``
releases the hardware, and ``execute`` runs the three in order. The application
builds the cell's static TF tree, attaches the suction gripper, picks each box
from the conveyor's calibrated pick pose, and carries it with the TCP facing
down to the next free pallet pose.

Without a robot IP the application runs in Isaac Sim: the scene is fetched from
the asset server and opened, the robot is imported from its URDF and placed at
the ``robot_base`` frame, the suction gripper loads itself from its own USD
asset when it connects, and the scene's conveyors run until the lightbeam
reports a box at the pick pose. Once a box is lifted clear, the conveyors move
the next box to the lightbeam while the robot completes the current placement
and returns home.

With a robot IP the same cycle runs on hardware. The Isaac Sim scene setup and
conveyor control are skipped and the measured cell poses have to be supplied.
Either way every box is picked from the same calibrated pick pose, because a
light barrier reports only that a box is standing at it, not where it stands.

Before running in simulation:
    1. Enable the Telekinesis Isaac Sim bridge extension.
    2. Run::

           python palletizing_application.py

This is an external Python application; it does not import ``omni`` or
``isaacsim``.
"""

from __future__ import annotations

import atexit
import dataclasses
import faulthandler
import json
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib import parse

from loguru import logger

from telekinesis import datatypes, isaacsim_client
from telekinesis.medulla import conveyors, sensors
from telekinesis.tf import tftree, tfutils
from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.suction_grippers import abstract_gripper, isaacsim


## ================== Configuration ================== ##


@dataclasses.dataclass
class RobotConfig:
    """The robot: where it stands, how it is reached, and how fast it moves.

    Attributes:
        ip: Robot controller address. Leave as ``None`` to run the cell in
            Isaac Sim instead of on hardware.
        prim_path: USD path the robot is imported at in simulation.
        robot_base_offset: Robot base relative to the robot mount, using XYZ
            metres and Euler XYZ degrees. The mount prim in a scene is rarely
            authored facing the way the robot stands on it, so this turns the
            measured mount pose into the base the robot is commanded in.
        home_joint_positions: Clear home pose in joint degrees.
        joint_speed: Speed of the home move, in degrees per second.
        joint_acceleration: Acceleration of the home move, in degrees per second
            squared.
        cartesian_speed: Speed of the approach and retreat moves, in metres per
            second.
        cartesian_acceleration: Acceleration of every Cartesian move, in metres
            per second squared.
        cartesian_place_speed: Speed of the last move onto a box and onto the
            pallet, in metres per second.
        cartesian_carry_speed: Speed the robot travels at while holding a box,
            in metres per second.
        drive_stiffness: Position gain applied to every joint drive once the
            gripper is attached. A robot imported from a URDF needs explicit
            gains to hold its pose. ``None`` keeps the gains the stage was
            authored with, which is what a robot already tuned in the scene
            needs.
        drive_damping: Velocity gain applied together with ``drive_stiffness``,
            and ignored when that is ``None``.
    """

    ip: str | None = None
    prim_path: str = "/World/ur10e_robot"
    robot_base_offset: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 180.0]
    )
    home_joint_positions: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, -90.0, -60.0, -120.0, 90.0, 0.0]
    )
    joint_speed: float = 45.0
    joint_acceleration: float = 60.0
    cartesian_speed: float = 0.30
    cartesian_acceleration: float = 0.50
    # Slower speeds for setting a box down and for carrying one.
    cartesian_place_speed: float = 0.08
    cartesian_carry_speed: float = 0.15
    drive_stiffness: float | None = 1.0e5
    drive_damping: float | None = 1.0e4

    def __post_init__(self) -> None:
        """Check the transforms and the motion limits.

        Returns:
            None.

        Raises:
            ValueError: If a transform is not six values, or a speed or an
                acceleration is not positive.
        """
        if len(self.robot_base_offset) != 6:
            raise ValueError(
                "robot_base_offset must be six values: XYZ metres and Euler "
                "XYZ degrees."
            )
        for name in (
            "joint_speed",
            "joint_acceleration",
            "cartesian_speed",
            "cartesian_acceleration",
            "cartesian_place_speed",
            "cartesian_carry_speed",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")


@dataclasses.dataclass
class GripperConfig:
    """The suction gripper: how it is reached and where it sits on the robot.

    Attributes:
        ip: Suction-gripper address, used when the robot config has an IP.
        prim_path: USD path the gripper loads its own USD asset at in
            simulation.
        tool_mount_transform: Gripper root relative to the robot tool flange,
            using XYZ metres and Euler XYZ degrees.
        suction_tcp_offset: Suction TCP relative to the robot tool flange, using
            XYZ metres and Euler XYZ degrees. This is the frame every pick and
            place pose is reached with.
    """

    ip: str | None = None
    prim_path: str = "/World/defitech_modelled_surface_gripper_modelled"
    # Defitech gripper mounting orientation relative to the UR10e tool flange.
    tool_mount_transform: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 180.0, 0.0, 0.0]
    )
    # Centre of the visible suction-pad face relative to the robot tool flange.
    suction_tcp_offset: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.075, 0.0, 0.0, 0.0]
    )

    def __post_init__(self) -> None:
        """Check the gripper mounting transforms.

        Returns:
            None.

        Raises:
            ValueError: If a transform is not six values.
        """
        for name in ("tool_mount_transform", "suction_tcp_offset"):
            if len(getattr(self, name)) != 6:
                raise ValueError(
                    f"{name} must be six values: XYZ metres and Euler XYZ degrees."
                )


@dataclasses.dataclass
class PalletLayout:
    """The cell's calibrated geometry and the number of boxes to palletize.

    The pick target is one calibrated pose relative to the conveyor, because a
    light barrier reports only that a box is standing at it. The place targets
    are defined relative to the pallet as stacked layers of centred cells.

    Object frames represent box centres. Grasp frames sit on the box top
    surfaces with their Z axes facing downward, which is why the stand-off
    offsets are negative in Z.

    Attributes:
        box_size: Box dimensions in metres. Sets the height of every grasp frame
            and the spacing between pallet layers.
        box_count: Number of boxes to palletize, at most one per pallet cell.
        pick_pose: Where the gripper takes a box off the conveyor, relative to
            the conveyor, using XYZ metres and Euler XYZ degrees. It sits on the
            top surface of a box stopped at the lightbeam and faces down, so it
            is calibrated for ``box_size``: a box of another height needs its
            own value.
        pre_pick_offset: How far the gripper stands off from ``pick_pose`` on
            the way in and on the way out, relative to ``pick_pose`` and in the
            same format. The pick pose faces down, so a stand-off above the
            conveyor is negative in Z.
        pre_place_offset: How far the gripper stands off from a place pose on
            the way in and on the way out, in the same format. The pallet sits
            lower than the conveyor, so the base-layer stand-offs are set to put
            pre-pick and pre-place at one common world height, which keeps a
            carried box level.
        first_place_pose: Centre of the first box placed on the pallet, relative
            to the pallet and in the same format. Every other place pose is this
            one stepped by the box spacings below.
        box_spacing_x: How far apart neighbouring boxes stand along the pallet X
            axis, as an XYZ offset in metres from one box centre to the next.
            Usually one box footprint plus any gap. ``None`` mirrors
            ``first_place_pose`` across the pallet X axis, which spaces the
            boxes so the layer ends up centred on the pallet.
        box_spacing_y: The same along the pallet Y axis. ``None`` mirrors
            ``first_place_pose`` the same way.
        place_numx: Number of boxes per layer along the pallet X axis.
        place_numy: Number of boxes per layer along the pallet Y axis.
        place_numz: Number of stacked layers. Layers are spaced by the box
            height, so they are not configured separately.
        conveyor_pose_in_world: Measured pose of the conveyor the pick pose is
            referenced to, using XYZ metres and Euler XYZ degrees. Read from the
            open Isaac Sim scene when left as ``None``, which running on
            hardware cannot do.
        pallet_pose_in_world: Measured pallet pose, in the same format.
        robot_mount_pose_in_world: Measured robot-mount pose, in the same
            format.
    """

    box_size: list[float] = dataclasses.field(
        default_factory=lambda: [0.513243397, 0.331865479, 0.259689436]
    )
    box_count: int = 8
    pick_pose: list[float] = dataclasses.field(
        default_factory=lambda: [
            -2.205667546,
            0.119918064,
            1.028834925,
            180.0,
            0.0,
            90.0,
        ]
    )
    pre_pick_offset: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, -0.40, 0.0, 0.0, 0.0]
    )
    pre_place_offset: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, -0.875, 0.0, 0.0, 0.0]
    )
    # The calibration box was resting directly on the pallet. Its USD root is
    # near the bottom face; this TF frame is shifted to the logical box centre.
    first_place_pose: list[float] = dataclasses.field(
        default_factory=lambda: [
            -0.307142031,
            -0.200925418,
            0.272350404,
            0.0,
            0.0,
            0.0,
        ]
    )
    box_spacing_x: list[float] | None = None
    box_spacing_y: list[float] | None = None
    place_numx: int = 2
    place_numy: int = 2
    place_numz: int = 2
    conveyor_pose_in_world: list[float] | None = None
    pallet_pose_in_world: list[float] | None = None
    robot_mount_pose_in_world: list[float] | None = None

    def __post_init__(self) -> None:
        """Derive the unset pallet steps and check the layout.

        Returns:
            None.

        Raises:
            ValueError: If a transform or a pose has the wrong number of values,
                a cell count is below one, or ``box_count`` exceeds the pallet
                cells the layout provides.
        """
        if len(self.box_size) != 3:
            raise ValueError("box_size must be three values in metres.")
        for name in (
            "pick_pose",
            "pre_pick_offset",
            "pre_place_offset",
            "first_place_pose",
        ):
            if len(getattr(self, name)) != 6:
                raise ValueError(
                    f"{name} must be six values: XYZ metres and Euler XYZ degrees."
                )
        for name in ("place_numx", "place_numy", "place_numz"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one.")

        # Mirror the calibrated first box across the pallet axes, which spaces
        # the boxes so the layer ends up centred with one box orientation.
        if self.box_spacing_x is None:
            self.box_spacing_x = [-2.0 * self.first_place_pose[0], 0.0, 0.0]
        if self.box_spacing_y is None:
            self.box_spacing_y = [0.0, -2.0 * self.first_place_pose[1], 0.0]
        for name in ("box_spacing_x", "box_spacing_y"):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"{name} must be three values in metres.")

        if self.box_count < 1:
            raise ValueError("box_count must be at least one.")
        if self.box_count > self.place_numx * self.place_numy * self.place_numz:
            raise ValueError("box_count exceeds the pallet cells the layout provides.")

        for name in (
            "conveyor_pose_in_world",
            "pallet_pose_in_world",
            "robot_mount_pose_in_world",
        ):
            pose = getattr(self, name)
            if pose is not None and len(pose) != 6:
                raise ValueError(
                    f"{name} must be six values: XYZ metres and Euler XYZ degrees."
                )


@dataclasses.dataclass
class SimulationConfig:
    """Isaac Sim wiring: the bridge, the scene, and the prims the cell drives.

    Ignored when the application runs on hardware.

    Attributes:
        base_url: HTTP address of the Isaac Sim bridge.
        websocket_base_url: Websocket address of the Isaac Sim bridge.
        scene_url: Asset-server URL of the palletizing scene bundle.
        pallet_prim_path: Prim the pallet pose is read from.
        robot_mount_prim_path: Prim the robot-mount pose is read from.
        lightbeam_prim_path: Light barrier that reports a box at the pick pose.
        conveyor_prim_paths: Belts of the conveyor line, in the order they are
            started. The first one is the belt the pick pose and the cell's
            conveyor pose are measured against.
        cargo_root: Prim whose rigid bodies are woken every time the belts
            start, because a belt cannot pick up cargo that fell asleep while it
            was stopped.
        conveyor_velocity: Signed speed along the belts' installed travel
            direction, in metres per second. Flip the sign if the boxes run away
            from the lightbeam.
        beam_poll_seconds: Interval between light-barrier reads while waiting
            for a box.
        beam_timeout_seconds: How long a box has to reach the lightbeam before
            the cycle gives up.
        timeline_settle_seconds: How long to let physics run after the timeline
            starts, before the belts, the light barrier and the gripper are
            connected to a stage that is still coming up.
    """

    base_url: str = "http://127.0.0.1:8766"
    websocket_base_url: str = "ws://127.0.0.1:8766"
    scene_url: str = (
        "https://assets.telekinesis.ai/usd/environments/palletizing/"
        "palletizing_rough_scene.zip"
    )
    pallet_prim_path: str = "/World/pallet"
    robot_mount_prim_path: str = "/World/ur10_mount"
    lightbeam_prim_path: str = "/World/LightBeam_Sensor"
    conveyor_prim_paths: list[str] = dataclasses.field(
        default_factory=lambda: [
            "/World/ConveyorBelt_A08",
            "/World/ConveyorBelt_A11",
            "/World/ConveyorBelt_A08_01",
        ]
    )
    cargo_root: str = "/World"
    conveyor_velocity: float = 0.8
    beam_poll_seconds: float = 0.05
    beam_timeout_seconds: float = 120.0
    timeline_settle_seconds: float = 0.5

    def __post_init__(self) -> None:
        """Check the conveyor line and the light-barrier timings.

        Returns:
            None.

        Raises:
            ValueError: If no belt is named, the conveyor velocity is zero, or a
                light-barrier timing is not positive.
        """
        if not self.conveyor_prim_paths:
            raise ValueError("conveyor_prim_paths must name at least one belt.")
        if self.conveyor_velocity == 0.0:
            raise ValueError(
                "conveyor_velocity must be non-zero: a stopped belt never "
                "delivers a box to the lightbeam."
            )
        for name in ("beam_poll_seconds", "beam_timeout_seconds"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.timeline_settle_seconds < 0.0:
            raise ValueError("timeline_settle_seconds cannot be negative.")


## ================== Application ================== ##


class PalletizingApplication:
    """Palletize conveyor boxes with one robot and one suction gripper.

    The robot and the gripper are supplied by the caller, already created but
    not yet connected. Leaving the robot config's ``ip`` unset runs the cell in
    Isaac Sim: the scene is opened, the robot is imported and placed on its
    mount, and the scene's conveyors deliver each box to the lightbeam. Giving
    it an ``ip`` connects to hardware instead, in which case the cell poses have
    to be supplied and the real line delivers the boxes. Either way every box is
    picked from the same calibrated pick pose.

    Typical use::

        application = PalletizingApplication(robot, gripper)
        application.execute()

    :meth:`execute` runs setup, the palletizing cycles, and shutdown. The three
    are also public so a caller that wants the cycles under its own supervision
    can drive them separately.

    Attributes:
        robot_config: The robot: where it stands, how it is reached, and the
            motion limits the cycle runs at.
        gripper_config: The suction gripper: how it is reached and where it sits
            on the robot.
        layout: Calibrated cell geometry and the number of boxes to palletize.
            :meth:`setup` fills in any measured cell pose left unset by reading
            it from the open Isaac Sim scene.
        simulation: Isaac Sim bridge, scene and prim paths, unused on hardware.
        tree: Palletizing cell transform tree, built by :meth:`setup`.
    """

    def __init__(
        self,
        robot: universal_robots.UniversalRobotsUR10E,
        gripper: abstract_gripper.AbstractSuctionGripper,
        *,
        robot_config: RobotConfig | None = None,
        gripper_config: GripperConfig | None = None,
        layout: PalletLayout | None = None,
        simulation_config: SimulationConfig | None = None,
    ) -> None:
        """Store the hardware and the palletizing cell configuration.

        Args:
            robot: Robot to palletize with, not yet connected.
            gripper: Suction gripper, not yet connected.
            robot_config: Where the robot stands, how it is reached, and the
                motion limits the cycle runs at. Leaving its ``ip`` unset runs
                the cell in Isaac Sim. The defaults are calibrated for a UR10e.
            gripper_config: How the suction gripper is reached and where it sits
                on the robot tool flange. The defaults are calibrated for the
                Defitech gripper.
            layout: Calibrated cell geometry and the number of boxes to
                palletize. Leaving its measured cell poses unset reads them from
                the open Isaac Sim scene, which running on hardware cannot do.
            simulation_config: Isaac Sim bridge, scene and prim paths. Ignored when the
                robot config has an ``ip``.

        Returns:
            None.
        """
        self.robot = robot
        self.gripper = gripper

        # Built here rather than defaulted in the signature: a configuration is
        # mutable, and one instance in a default argument would be shared by
        # every application constructed without that argument.
        self.robot_config = robot_config if robot_config is not None else RobotConfig()
        self.gripper_config = (
            gripper_config if gripper_config is not None else GripperConfig()
        )
        self.layout = layout if layout is not None else PalletLayout()
        self.simulation_config = simulation_config if simulation_config is not None else SimulationConfig()

        # Built by setup(): the cell frames, and in simulation the Isaac Sim
        # client, the conveyor line and the light barrier at the pick pose.
        self.tree: tftree.TransformTree | None = None
        self.client: isaacsim_client.IsaacSimClient | None = None
        self.conveyors: list[conveyors.Conveyor] = []
        self.lightbeam: sensors.LightBeamSensor | None = None

        # Set once the cell should stop, so a conveyor wait on its own thread
        # returns within one poll instead of sitting out its whole timeout.
        self._stopping = threading.Event()
        self._is_shut_down = False

    @property
    def is_simulated(self) -> bool:
        """Whether the cell runs in Isaac Sim rather than on real hardware."""
        return self.robot_config.ip is None

    def _get_pose_in_world(self, prim_path: str) -> list[float]:
        """Read a prim's rigid pose in the Isaac Sim world frame.

        Args:
            prim_path: USD path of the prim to query.

        Returns:
            XYZ in metres followed by Euler XYZ angles in degrees.

        Raises:
            isaacsim_client.HTTPError: If the bridge request fails.
            KeyError: If the bridge response does not contain ``pose``.
            ValueError: If a pose value cannot be converted to ``float``.
        """
        query = parse.urlencode(
            {
                "prim_path": prim_path,
                "coordinate_system": "world",
                "rotation_type": "cartesian",
            }
        )
        response = self.client.get(f"/prims/poses?{query}")
        pose_in_rotvec = [float(value) for value in response.json()["pose"]]
        return list(tfutils.convert_pose_format(pose_in_rotvec, "rotvec", "deg"))

    def _build_frame_tree(self) -> tftree.TransformTree:
        """Build the cell's static TF tree from the layout and the robot config.

        The layout's box height positions every grasp frame on a box's top
        surface and sets the vertical spacing between pallet layers, so the same
        layout serves any box the cell is set up for.

        Returns:
            Static palletizing transform tree.

        Raises:
            ValueError: If a measured cell pose is missing, or a configured pose
                is invalid.
        """
        layout = self.layout
        for name in (
            "conveyor_pose_in_world",
            "pallet_pose_in_world",
            "robot_mount_pose_in_world",
        ):
            if getattr(layout, name) is None:
                raise ValueError(f"layout.{name} is required to build the frame tree.")

        # Grasp frames sit on a box's top surface facing down, so a box's centre
        # is half a box below its grasp frame.
        object_to_grasp = [0.0, 0.0, layout.box_size[2] / 2.0, 180.0, 0.0, 0.0]
        box_spacing_z = [0.0, 0.0, layout.box_size[2]]

        tree = tftree.TransformTree("world")

        tree.add("world", "conveyor", layout.conveyor_pose_in_world, rot_type="deg")
        tree.add("conveyor", "pick_pose", layout.pick_pose, rot_type="deg")
        tree.add("pick_pose", "pre_pick", layout.pre_pick_offset, rot_type="deg")

        tree.add("world", "pallet", layout.pallet_pose_in_world, rot_type="deg")
        for z_index in range(layout.place_numz):
            for x_index in range(layout.place_numx):
                for y_index in range(layout.place_numy):
                    place_xyz = [
                        layout.first_place_pose[axis]
                        + x_index * layout.box_spacing_x[axis]
                        + y_index * layout.box_spacing_y[axis]
                        + z_index * box_spacing_z[axis]
                        for axis in range(3)
                    ]
                    object_frame = f"place_object_{x_index}_{y_index}_{z_index}"
                    place_frame = f"place_pose_{x_index}_{y_index}_{z_index}"
                    pre_place_frame = f"pre_place_{x_index}_{y_index}_{z_index}"
                    tree.add(
                        "pallet",
                        object_frame,
                        [*place_xyz, *layout.first_place_pose[3:]],
                        rot_type="deg",
                    )
                    tree.add(object_frame, place_frame, object_to_grasp, rot_type="deg")

                    pre_place_offset = layout.pre_place_offset.copy()
                    # The place frame points down. Shorten its negative-Z offset
                    # by the stack height so every layer has one common travel
                    # height.
                    pre_place_offset[2] += z_index * box_spacing_z[2]
                    tree.add(
                        place_frame,
                        pre_place_frame,
                        pre_place_offset,
                        rot_type="deg",
                    )

        tree.add(
            "world",
            "robot_mount",
            layout.robot_mount_pose_in_world,
            rot_type="deg",
        )
        tree.add(
            "robot_mount",
            "robot_base",
            self.robot_config.robot_base_offset,
            rot_type="deg",
        )
        return tree

    # ------------------------------------------------------------------ #
    # Setup / teardown
    # ------------------------------------------------------------------ #

    def _setup(self) -> None:
        """Prepare the cell, build its frames, and connect the hardware.

        In simulation the scene bundle is downloaded once and cached, then
        opened in place of whatever stage is open. It holds the palletizing cell
        only, so the robot is imported from the URDF its Synapse class fetched,
        and the gripper is loaded from its own USD asset when it connects. The
        scene has to be open before the cell poses can be read from it, and the
        conveyors and the lightbeam are connected once the robot stands on its
        mount and the timeline plays.

        Returns:
            None.

        Raises:
            isaacsim_client.HTTPError: If an Isaac Sim bridge request fails.
            RuntimeError: If the robot, the suction gripper, a conveyor or the
                lightbeam cannot connect, or the gripper cannot attach.
            ValueError: If a configured transform is invalid, or a cell pose is
                missing while running on hardware.
        """
        logger.info("Setting up the palletizing cell...")
        if self.is_simulated:
            self.client = isaacsim_client.IsaacSimClient(
                api_key="",
                base_url=self.simulation_config.base_url,
                websocket_base_url=self.simulation_config.websocket_base_url,
            )
            self.client.get("/status")

            scene = datatypes.USD.from_url(
                self.simulation_config.scene_url, use_cache=True)  # TODO use cache this is to fix
            print(f"Opening the palletizing scene: {scene.path}")
            self.client.stage.open_scene(scene.path.as_posix())
            print(f"Importing the robot at {self.robot_config.prim_path}...")
            self.client.articulation.create(
                self.robot_config.prim_path,
                str(self.robot.urdf_path),
            )

        # The measured poses are kept on the layout, so a caller can read back
        # what the scene reported and reuse it as a hardware calibration.
        for field_name, prim_path in (
            ("conveyor_pose_in_world", self.simulation_config.conveyor_prim_paths[0]),
            ("pallet_pose_in_world", self.simulation_config.pallet_prim_path),
            ("robot_mount_pose_in_world", self.simulation_config.robot_mount_prim_path),
        ):
            if getattr(self.layout, field_name) is not None:
                continue
            if not self.is_simulated:
                raise ValueError(
                    "Running on hardware requires the measured conveyor, pallet "
                    "and robot-mount poses on the layout."
                )
            setattr(self.layout, field_name, self._get_pose_in_world(prim_path))
        self.tree = self._build_frame_tree()

        print("Connecting the cell hardware...")
        if self.is_simulated:
            self._place_robot_on_mount()
            # SuctionGripper.connect() requires the timeline to be playing, and
            # so do the belts and the light barrier: a belt carries nothing and
            # a beam has no reading while physics is stopped.
            self.client.patch("/stage/simulation/timeline/play", "{}")
            time.sleep(self.simulation_config.timeline_settle_seconds)

            for prim_path in self.simulation_config.conveyor_prim_paths:
                belt = conveyors.Conveyor(name=prim_path.rsplit("/", 1)[-1])
                belt.connect(
                    simulation_prim_path=prim_path,
                    cargo_root=self.simulation_config.cargo_root,
                )
                self.conveyors.append(belt)
            self.lightbeam = sensors.LightBeamSensor(name="pick_barrier")
            self.lightbeam.connect(
                simulation_prim_path=self.simulation_config.lightbeam_prim_path
            )

            self.robot.connect(simulation_prim_path=self.robot_config.prim_path)
            self.gripper.connect(
                simulation_prim_path=self.gripper_config.prim_path
            )
        else:
            self.robot.connect(ip=self.robot_config.ip)
            self.gripper.connect(ip=self.gripper_config.ip)

        self.robot.attach_tool(
            self.gripper,
            transform=self.gripper_config.tool_mount_transform,
        )
        self.robot.add_tcp(
            name="suction_tcp",
            transform=self.gripper_config.suction_tcp_offset,
            set_active=True,
        )

        # A robot imported from a URDF needs gain overrides after attachment
        # restarts physics. A robot already tuned in the scene, such as the
        # built-in Isaac Sim UR10e, leaves the gains unset and skips this.
        if self.is_simulated and self.robot_config.drive_stiffness is not None:
            # The robot prim already exists, so this only registers it and hands
            # back the id the gains are addressed by.
            articulation = self.client.articulation.create(self.robot_config.prim_path)
            self.client.post(
                f"/articulations/{articulation['articulation_id']}/dof_gains",
                json.dumps(
                    {
                        "stiffness": self.robot_config.drive_stiffness,
                        "damping": self.robot_config.drive_damping,
                    }
                ),
            )

        print("Moving to the home L pose...")
        self._move_to_home()

    def shutdown(self) -> None:
        """Stop the conveyor line and disconnect the cell hardware.

        The line is stopped first: disconnecting a belt does not stop it, so a
        cell abandoned mid-cycle would otherwise keep carrying boxes into it.

        Every step is attempted even when an earlier one fails, and a failure is
        logged with its traceback rather than raised, so that a gripper which
        will not release cannot leave the robot connected, and so that teardown
        never replaces the failure that caused it.

        Safe to call whether or not :meth:`_setup` finished, and safe to call
        twice: hardware that is not connected is left alone, and the second call
        returns without touching anything.

        Returns:
            None.
        """
        if self._is_shut_down:
            return
        self._is_shut_down = True

        self._stopping.set()
        self._stop_conveyor_line()

        for belt in self.conveyors:
            try:
                if belt.is_connected:
                    belt.disconnect()
            except Exception:
                logger.exception(f"Could not disconnect the conveyor {belt.name}.")

        try:
            if self.lightbeam is not None and self.lightbeam.is_connected:
                self.lightbeam.disconnect()
        except Exception:
            logger.exception("Could not disconnect the lightbeam sensor.")

        try:
            if self.gripper.is_connected:
                self.gripper.disconnect()
        except Exception:
            logger.exception("Could not disconnect the suction gripper.")

        try:
            if self.robot.is_connected:
                self.robot.disconnect()
        except Exception:
            logger.exception("Could not disconnect the robot.")

        try:
            self.robot.shutdown()
        except Exception:
            logger.exception("Could not release the robot.")

    def request_stop(self, signal_number: int, frame: object) -> None:
        """Ask the cell to stop, from a signal handler.

        Releases a conveyor wait that is still polling, so teardown is not held
        up by the rest of its timeout, then raises ``KeyboardInterrupt`` so the
        running cycle unwinds through :meth:`execute`'s teardown. Registered for
        SIGINT and SIGTERM, the second of which would otherwise end the process
        with the belts still running.

        Args:
            signal_number: Number of the signal that arrived.
            frame: Stack frame the signal interrupted.

        Returns:
            None.

        Raises:
            KeyboardInterrupt: Always, to unwind the running cycle.
        """
        self._stopping.set()
        raise KeyboardInterrupt(f"stopping the cell on signal {signal_number}")

    def _stop_conveyor_line(self) -> bool:
        """Stop every belt, and report whether all of them stopped.

        Every belt is attempted even when one refuses, because a belt left
        running keeps carrying boxes into the cell. A refusal is logged rather
        than raised, so this is safe to call while another failure is unwinding.

        Returns:
            Whether every connected belt stopped.
        """
        stopped = True
        for belt in self.conveyors:
            if not belt.is_connected:
                continue
            try:
                belt.stop()
            except (RuntimeError, ConnectionError, OSError) as error:
                logger.error(
                    f"Could not stop {belt.name}, which may still be running: {error}"
                )
                stopped = False
        return stopped

    # ------------------------------------------------------------------ #
    # One pick-and-place cycle
    # ------------------------------------------------------------------ #

    def _advance_next_box(self) -> None:
        """Run the conveyor line until a box stands at the pick pose.

        The belts also wake the cargo under the configured cargo root as they
        start, so a box that came to rest while they were stopped is carried
        again. The line
        is stopped again however this ends, including when the cell has been
        asked to stop, in which case it returns without waiting for a box.

        Returns:
            None.

        Raises:
            RuntimeError: If a belt or the lightbeam is not connected or rejects
                a command, if the line cannot be stopped again, or if no box
                breaks the beam before the configured timeout.
        """
        deadline = time.monotonic() + self.simulation_config.beam_timeout_seconds
        beam_was_clear = False
        detected = False
        try:
            for belt in self.conveyors:
                belt.start(velocity=self.simulation_config.conveyor_velocity)

            while not self._stopping.is_set() and time.monotonic() < deadline:
                if not self.lightbeam.is_beam_broken():
                    # A box still standing in the beam from the previous cycle
                    # is not a new arrival, so the beam has to read clear first.
                    beam_was_clear = True
                elif beam_was_clear:
                    detected = True
                    break
                time.sleep(self.simulation_config.beam_poll_seconds)
        finally:
            # Reported rather than raised: raising from here would replace
            # whatever sent us into this block, and would abandon the belts
            # after the one that refused.
            line_stopped = self._stop_conveyor_line()

        # Only reached when the belts ran without error, so a line that will not
        # stop and a box that never arrived are this cycle's own failures.
        if not line_stopped:
            raise RuntimeError(
                "The conveyor line did not stop. Stop the belts in Isaac Sim "
                "before running the cell again."
            )
        if self._stopping.is_set():
            logger.warning("The cell was asked to stop, so no box was waited for.")
            return
        if not detected:
            raise RuntimeError(
                "No box reached the lightbeam within "
                f"{self.simulation_config.beam_timeout_seconds} s."
            )
        logger.info("A box is standing at the lightbeam.")

    def _pick(self) -> None:
        """Pick the next box off the conveyor and lift it clear.

        Returns:
            None.

        Raises:
            RuntimeError: If a Cartesian move fails, or the suction gripper
                reports no box after grasping.
            ValueError: If a TF frame is invalid.
        """
        print("Picking the box at the conveyor...")
        self._move_robot_to_frame("pre_pick", self.robot_config.cartesian_speed)
        self._move_robot_to_frame("pick_pose", self.robot_config.cartesian_place_speed)

        self.gripper.grasp()
        if not self.gripper.get_part_present():
            raise RuntimeError(
                "Suction did not detect a box. The robot was not lifted."
            )

        self._move_robot_to_frame("pre_pick", self.robot_config.cartesian_speed)

    def _place(self, x_index: int, y_index: int, z_index: int) -> None:
        """Place the held box in one pallet cell and retreat above it.

        Args:
            x_index: Pallet cell index along the pallet X axis.
            y_index: Pallet cell index along the pallet Y axis.
            z_index: Stack layer index.

        Returns:
            None.

        Raises:
            RuntimeError: If a Cartesian move fails, or the suction gripper
                still reports a box after releasing.
            ValueError: If a TF frame is invalid.
        """
        frame_suffix = f"{x_index}_{y_index}_{z_index}"

        # Cells other than the first are reached across the layer's own travel
        # height, so a carried box stays level instead of cutting a diagonal.
        if (x_index, y_index) != (0, 0):
            self._move_robot_to_frame(
                f"pre_place_0_0_{z_index}",
                self.robot_config.cartesian_carry_speed,
            )

        print(f"Placing at pallet frame {frame_suffix}...")
        self._move_robot_to_frame(
            f"pre_place_{frame_suffix}",
            self.robot_config.cartesian_carry_speed,
        )
        self._move_robot_to_frame(
            f"place_pose_{frame_suffix}",
            self.robot_config.cartesian_place_speed,
        )

        self.gripper.release()
        if self.gripper.get_part_present():
            raise RuntimeError("Suction still reports a box after release.")

        self._move_robot_to_frame(
            f"pre_place_{frame_suffix}",
            self.robot_config.cartesian_speed,
        )

    # ------------------------------------------------------------------ #
    # Motion and scene helpers
    # ------------------------------------------------------------------ #

    def _move_to_home(self) -> None:
        """Move the robot to its home joint pose.

        Returns:
            None.

        Raises:
            RuntimeError: If Synapse cannot execute the joint move.
        """
        self.robot.set_joint_positions(
            self.robot_config.home_joint_positions,
            speed=self.robot_config.joint_speed,
            acceleration=self.robot_config.joint_acceleration,
        )

    def _move_robot_to_frame(self, frame_name: str, speed: float) -> None:
        """Move the active robot TCP to a named TF frame.

        Every move runs at the configured Cartesian acceleration; only the speed
        differs between approach, contact, and carry motions.

        Args:
            frame_name: Name of the target frame in the cell transform tree.
            speed: Cartesian speed in metres per second.

        Returns:
            None.

        Raises:
            RuntimeError: If Synapse cannot generate or execute the Cartesian move.
            ValueError: If the TF frame or pose is invalid.
        """
        target_pose = self.tree.lookup_transform(
            "robot_base",
            frame_name,
            rot_type="deg",
        )
        self.robot.set_cartesian_pose(
            target_pose,
            speed=speed,
            acceleration=self.robot_config.cartesian_acceleration,
        )

    def _place_robot_on_mount(self) -> None:
        """Move the robot prim to the transform tree's ``robot_base`` frame.

        Returns:
            None.

        Raises:
            isaacsim_client.HTTPError: If a bridge request fails.
            ValueError: If the transform cannot be converted to a pose.
        """
        self.client.patch("/stage/simulation/timeline/stop", "{}")

        robot_base_pose_in_world = [
            float(value)
            for value in tfutils.transformation_matrix_to_pose(
                self.tree.lookup_transform("world", "robot_base"),
                rot_type="rotvec",
            )
        ]
        self.client.put(
            "/prims/poses",
            json.dumps(
                {
                    "prim_path": self.robot_config.prim_path,
                    "input_pose": {"pose": robot_base_pose_in_world},
                }
            ),
        )

        # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        """Palletize ``box_count`` boxes, one pick-and-place cycle each.

        In simulation the conveyors run on their own thread, so once a picked
        box is clear the next one travels to the lightbeam while the robot
        completes the current placement and returns home. A real cell runs its
        own conveyors, so the boxes simply arrive. Boxes fill the pallet cell by
        cell and layer by layer.

        Returns:
            None.

        Raises:
            RuntimeError: If a move fails, the suction gripper does not report
                the expected box, or no box reaches the lightbeam in time.
            ValueError: If a TF frame is invalid.
        """
        box_count = self.layout.box_count
        logger.info(f"Running the palletizing cycle for {box_count} boxes...")

        self._stopping.clear()
        with ThreadPoolExecutor(max_workers=1) as conveyor_executor:
            try:
                next_box = (
                    conveyor_executor.submit(self._advance_next_box)
                    if self.is_simulated
                    else None
                )

                for box_index in range(box_count):
                    z_index, layer_index = divmod(
                        box_index,
                        self.layout.place_numx * self.layout.place_numy,
                    )
                    x_index, y_index = divmod(layer_index, self.layout.place_numy)
                    logger.info(f"--- box {box_index + 1}/{box_count} ---")

                    if next_box is not None:
                        next_box.result()
                    self._pick()

                    # Once the held box is clear, move the next box to the
                    # lightbeam while the robot completes the current placement.
                    if self.is_simulated and box_index + 1 < box_count:
                        next_box = conveyor_executor.submit(self._advance_next_box)

                    self._place(x_index, y_index, z_index)
                    self._move_to_home()
            finally:
                # Leaving this block joins the conveyor thread, so release a wait
                # that is still polling rather than sitting out its timeout.
                self._stopping.set()

        input(f"Placed {box_count} boxes. Press Enter to disconnect...")

    def execute(self) -> None:
        """Set up the cell, palletize every box, and shut down.

        Installs the SIGINT and SIGTERM handlers and the exit hook that stop the
        conveyor line and release the hardware, so an interrupt, a kill or an
        unexpected exit still shuts the cell down. Has to be called from the
        main thread, because signal handlers can only be installed there.

        The cell is shut down whether the cycle finishes or fails. A failure is
        logged with its traceback and re-raised, so that a cell which stopped
        early always says why and never reports success.

        Returns:
            None.

        Raises:
            BaseException: Whatever stopped the cycle, after the cell has been
                shut down. Includes ``isaacsim_client.HTTPError`` if an Isaac
                Sim bridge request fails, ``RuntimeError`` if the hardware
                cannot be set up, a move fails or the suction gripper does not
                report the expected box, ``ValueError`` if a configured
                transform is invalid or ``box_count`` exceeds the available
                pallet frames, and ``KeyboardInterrupt`` if the cell was asked
                to stop.
        """
        # A crash inside a native dependency kills the process without printing
        # anything, which looks exactly like the cell finishing its cycle.
        faulthandler.enable()

        # A cell abandoned mid-cycle keeps its belts carrying boxes into it, so
        # Ctrl-C, a kill and an unexpected exit all have to reach shutdown().
        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        try:
            self._setup()
            self._run()
        # BaseException, not Exception: a KeyboardInterrupt from request_stop or
        # a SystemExit raised deeper down must not leave the cell stopping in
        # silence with a success exit code.
        except BaseException as ex:
            logger.exception(f"The palletizing cycle stopped early.{ex}")
            raise
        except Exception as ex:
            logger.exception(f"The palletizing cycle failed: {ex}")
            raise
        finally:
            self.shutdown()


def main() -> None:
    """Run the palletizing application in Isaac Sim.

    Returns:
        None.

    Raises:
        isaacsim_client.HTTPError: If an Isaac Sim bridge request fails.
        RuntimeError: If the robot or suction gripper cannot connect or attach.
        ValueError: If a configured transform is invalid.
    """
    robot = universal_robots.UniversalRobotsUR10E(name="ur10e")
    gripper = isaacsim.SuctionGripper()

    # Every value the cell runs on lives on one of these three configurations.
    # Adjust them here rather than editing the application.
    robot_config = RobotConfig()
    robot_config.cartesian_speed = 0.6
    robot_config.cartesian_place_speed = 0.3
    robot_config.cartesian_carry_speed = 0.4

    layout = PalletLayout()
    layout.pick_pose = [
        -2.1,
        -0.119918064,
        1.028834925,
        180.0,
        0.0,
        90.0,
    ]
    PalletizingApplication(
        robot,
        gripper,
        robot_config=robot_config,
        gripper_config=GripperConfig(),
        layout=layout,
        simulation_config=SimulationConfig(),
    ).execute()


if __name__ == "__main__":
    main()
