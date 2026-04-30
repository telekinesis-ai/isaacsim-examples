"""
Simple robot sim/real example

Import order Backup:

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

"""
import time

# !!!!!! DONT CHANGE IMPORT ORDER (DLL version conflicts) !!!!!!!
from telekinesis.synapse.robots.manipulators import universal_robots



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


def get_robot(simulation=True, sim_app=None, robot_ip="192.168.1.2"):
    """
    Method that can be used to instatiate real or sim robot
    """
    if simulation and sim_app is None:
        raise ValueError("Simulation selected but simulation_app not given")

    robot = universal_robots.UniversalRobotsUR10E()
    if simulation:
        # setup isaac sim
        setup_stage(simulation_app)
        robot = SimManipulator(robot, sim_app)
        robot.connect(robot_ip)
        frame_robot(simulation_app)
        render_frames(simulation_app, 60)
    else:
        robot.connect(robot_ip)

    logger.success(f"Robot: {type(robot).__name__} | ndof={robot.ndof} in simulation: {simulation}")
    return robot


def run_program(robot):
    """
    Example of main program that should do something with the robot
    """
    # Stage 1 set robot to zero vector joint configuration
    robot.set_joint_positions(np.zeros(robot.ndof).tolist())
    logger.info("Sleeping for 2 seconds")
    time.sleep(2)

    # Stage 2 move to desired configuration
    q = np.asarray([0,30,30,30,30,30])
    robot.set_joint_positions(q)
    
    logger.info("Sleeping for 2 seconds")
    time.sleep(2)
    
    # Stage 3 go back to zero vector joint configuration
    robot.set_joint_positions(np.zeros(robot.ndof).tolist())
    logger.info("Sleeping for 2 seconds")
    time.sleep(2)

def main():
    """
    Prepare the robot, run the main loop (program) and then disconnect the robot
    """
    robot = get_robot(simulation=True, sim_app=simulation_app)

    run_program(robot=robot)

    robot.disconnect()


if __name__ == "__main__":

    main()

