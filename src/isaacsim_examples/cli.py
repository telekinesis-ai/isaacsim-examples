"""Shared interactive CLI for Isaac Sim robot examples.

Provides a standard command loop (joints, IK, FK, camera, gripper, etc.)
that any example script can use after setting up a stage and connecting a
:class:`SimManipulator`. If a :class:`SimParallelGripper` is passed in,
gripper commands are enabled using the exact :class:`AbstractParallelGripper`
API — so swapping sim for a real gripper does not change call sites.
"""

import pathlib

import numpy as np
from loguru import logger

from .async_input import async_input, poll_input
from .stage import frame_robot, take_screenshot


HELP_TEXT_ARM = """
Commands:
  [j]oints      set joint positions in degrees (smooth motion)
  [ik]          move to cartesian pose via IK (smooth motion)
  [zero]        set all joints to zero
  [fk]          show current end-effector pose
  [info]        show robot details
  [s]creenshot  capture viewport to file
  [cam]era      set camera eye and target (ex,ey,ez,tx,ty,tz)
  [q]uit        exit"""

HELP_TEXT_GRIPPER = """
  [o]pen        open gripper (blocking)
  [c]lose       close gripper (blocking)
  [gm]          move gripper to position (prompts for pos/speed/force)
  [gs]          show gripper status"""


def _robot_label(sim_robot) -> str:
    """Derive a human-readable label from the robot's URDF path."""
    return pathlib.Path(str(sim_robot.urdf_path)).stem


def _process_gripper_command(cmd: str, sim_tool) -> bool:
    """Handle gripper commands. Returns ``True`` if the command was gripper-related."""
    if cmd in ("o", "open"):
        try:
            status = sim_tool.open()
            logger.info(f"Gripper open: {status}")
        except Exception as exc:
            logger.error(f"Gripper open failed: {exc}")
        return True

    if cmd in ("c", "close"):
        try:
            status = sim_tool.close()
            logger.info(f"Gripper close: {status}")
        except Exception as exc:
            logger.error(f"Gripper close failed: {exc}")
        return True

    if cmd == "gm":
        try:
            raw = input(
                f"Enter position [{sim_tool._default_position_unit}], "
                f"speed [{sim_tool._default_speed_unit}], "
                f"force [{sim_tool._default_force_unit}] "
                f"(comma-separated; blank speed/force = defaults): "
            )
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 1 or parts[0] == "":
                logger.error("Position is required.")
                return True
            position = float(parts[0])
            speed = float(parts[1]) if len(parts) > 1 and parts[1] else 100.0
            force = float(parts[2]) if len(parts) > 2 and parts[2] else 100.0
            status = sim_tool.move(position, speed=speed, force=force)
            logger.info(f"Gripper move: {status}")
        except Exception as exc:
            logger.error(f"Gripper move failed: {exc}")
        return True

    if cmd == "gs":
        print(f"  Connected:       {sim_tool.is_connected}")
        print(f"  Attached:        {sim_tool.articulation is not None}")
        print(f"  Joint(s):        {sim_tool.joint_prim_names}")
        print(f"  Position unit:   {sim_tool._default_position_unit}")
        print(f"  Speed unit:      {sim_tool._default_speed_unit}")
        print(f"  Force unit:      {sim_tool._default_force_unit}")
        print(f"  Stroke (mm):     {sim_tool._position_range_mm}")
        return True

    return False


def _process_command(cmd: str, sim_robot, sim_tool=None) -> str | None:
    """Process a single CLI command.

    Returns ``'quit'`` to exit the loop, or ``None`` to continue.
    """
    if cmd in ("q", "quit"):
        return "quit"

    if cmd in ("j", "joints"):
        try:
            raw = input(
                f"Enter {sim_robot.ndof} joint values in deg "
                f"(comma-separated): "
            )
            values = [float(v) for v in raw.split(",")]
            sim_robot.set_joint_positions(values)
            logger.info(f"Joint positions (deg): {sim_robot.get_joint_positions()}")
        except Exception as exc:
            logger.error(f"Failed to set joints: {exc}")

    elif cmd == "ik":
        try:
            raw = input("Enter target pose (x,y,z,rx,ry,rz) in m/deg: ")
            target_pose = [float(v) for v in raw.split(",")]
            if len(target_pose) != 6:
                logger.error("Expected 6 values: x,y,z,rx,ry,rz")
                return None
            sim_robot.set_cartesian_pose(target_pose)
            logger.info(f"Joint positions (deg): {sim_robot.get_joint_positions()}")
        except Exception as exc:
            logger.error(f"IK failed: {exc}", exc_info=True)

    elif cmd == "zero":
        sim_robot.set_joint_positions(np.zeros(sim_robot.ndof).tolist())
        logger.info("Joints set to zero")

    elif cmd == "fk":
        pose = sim_robot.get_cartesian_pose()
        logger.info(f"End-effector pose (m/deg): {pose}")
        # try:
        #     pose = sim_robot.get_cartesian_pose()
        #     logger.info(f"End-effector pose (m/deg): {pose}")
        # except Exception as exc:
        #     logger.error(f"FK failed: {exc}")

    elif cmd == "info":
        print(f"  Robot:        {_robot_label(sim_robot)}")
        print(f"  NDOF:         {sim_robot.ndof}")
        print(f"  Joint limits: {sim_robot.joint_limits}")
        print(f"  URDF:         {sim_robot.urdf_path}")
        print(f"  Active TCP:   {sim_robot.active_tcp}")

    elif cmd in ("s", "screenshot"):
        take_screenshot(sim_robot.simulation_app, _robot_label(sim_robot))

    elif cmd in ("cam", "camera"):
        try:
            raw = input("Enter eye and target (ex,ey,ez,tx,ty,tz): ")
            vals = [float(v) for v in raw.split(",")]
            if len(vals) != 6:
                logger.error("Expected 6 values: ex,ey,ez,tx,ty,tz")
                return None
            frame_robot(
                sim_robot.simulation_app,
                eye=np.array(vals[:3]),
                target=np.array(vals[3:]),
            )
        except Exception as exc:
            logger.error(f"Failed to set camera: {exc}")

    elif cmd in ("o", "open", "c", "close", "gm", "gs"):
        if sim_tool is None:
            print("No gripper attached. Pass sim_tool=... to interactive_loop.")
            return None
        _process_gripper_command(cmd, sim_tool)

    else:
        print("Unknown command. See the menu above.")

    return None


def interactive_loop(sim_robot, sim_tool=None):
    """Interactive command loop. Keeps the viewport responsive while waiting.

    Call this after connecting a :class:`SimManipulator` (and optionally
    attaching a gripper via :meth:`SimManipulator.attach_tool`). The loop
    pumps ``simulation_app.update()`` between keystrokes so the viewport
    stays live and PD controllers maintain joint targets.

    Args:
        sim_robot: A connected :class:`SimManipulator`.
        sim_tool: Optional gripper implementing the
            :class:`AbstractParallelGripper` contract (sim or real). When
            provided, gripper commands are enabled in the menu.
    """
    app = sim_robot.simulation_app
    label = _robot_label(sim_robot)

    help_text = HELP_TEXT_ARM
    if sim_tool is not None:
        help_text = HELP_TEXT_ARM + HELP_TEXT_GRIPPER

    while app.is_running():
        print(f"\n--- {label} ---")
        print(help_text)

        async_input("> ")

        while app.is_running():
            app.update()

            line = poll_input()
            if line is False:
                return
            if line is not None:
                cmd = line.strip().lower()
                result = _process_command(cmd, sim_robot, sim_tool=sim_tool)
                if result == "quit":
                    return
                break  # back to outer loop to re-print menu
