"""
Interactive IK example using SimManipulator.

This script creates a procedural stage (ground plane + lighting), loads
a robot via SimManipulator, and provides an interactive CLI to control
joints, solve IK/FK, and move the robot.
"""


# !!!!!! DONT CHANGE IMPORT ORDER (DLL version conflicts) !!!!!!!
from telekinesis.synapse.robots.manipulators import abb


from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})


from isaacsim_examples import (
    SimManipulator,
    setup_stage,
    frame_robot,
    render_frames,
    interactive_loop,
)

# 3rd party imports
import numpy as np
from loguru import logger


def main():
    robot = abb.AbbIRB120T358()
    logger.info(f"Robot: {type(robot).__name__} | ndof={robot.ndof}")

    setup_stage(simulation_app)

    sim_robot = SimManipulator(robot, simulation_app)
    sim_robot.connect()
    sim_robot.set_joint_positions(np.zeros(sim_robot.ndof).tolist())

    frame_robot(simulation_app)
    render_frames(simulation_app, 60)

    interactive_loop(sim_robot)

    sim_robot.disconnect()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error(f"An error occurred: {exc}")
    finally:
        simulation_app.close()
