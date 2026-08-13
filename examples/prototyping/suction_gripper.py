from isaacsim.robot.surface_gripper import _surface_gripper

GRIPPER_PATH = "/World/defitech_modelled_surface_gripper/SurfaceGripper"

# Get the Surface Gripper interface from the already-running Isaac Sim.
gripper = _surface_gripper.acquire_surface_gripper_interface()

# Optional:
# Mirror runtime status / gripped objects back into USD properties.
gripper.set_write_to_usd(True)


def close_gripper():
    print("Closing suction gripper...")
    gripper.close_gripper(GRIPPER_PATH)


def open_gripper():
    print("Opening suction gripper...")
    gripper.open_gripper(GRIPPER_PATH)


def print_status():
    status = gripper.get_gripper_status(GRIPPER_PATH)
    print("Gripper status:", status)


print_status()
close_gripper()
print("Done")