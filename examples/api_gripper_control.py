"""
Gripper-only example for the bridge: create -> close -> open.

The bridge is device-agnostic -- it has no notion of a "gripper", a fraction, or
open/close. A gripper is just an articulation whose single actuated joint you
drive. This example does the gripper-specific bits explicitly, over HTTP, so the
extension calls are visible:

  1. PUT  /articulations {prim_path, urdf_path?}            -> {articulation_id, ...}
  2. GET  /articulations/{id}/driver_joint                 -> {name, index}
       (USD schema walk on the bridge: the actuated joint, skipping mimic joints)
  3. PUT  /articulations/{id}/driven_joints {[driver_name]} -> narrow to that joint
  4. GET  /articulations/{id}/joint_limits                 -> {limits: [[open, closed]]}
  5. close/open: map a closed-ness fraction (1.0 / 0.0) to a joint angle and
       POST /articulations/{id}/joint_positions {positions:[rad]}
       -> blocks until the finger reaches the target or stalls; returns {done, reached, q}

Run:  python gripper_control.py
      python gripper_control.py --prim /World/rg6
      python gripper_control.py --prim /World/robotiq --urdf C:/path/to/robotiq.urdf

Requires the ``requests`` package (``pip install requests``).
"""

import argparse

import requests

try:
    import telekinesis_urdfs
except ImportError as e:
    raise ImportError(
        "telekinesis-urdfs is not installed. "
        "Install it from: https://github.com/telekinesis-ai/telekinesis-urdfs"
    ) from e

# Path to the grippers's URDF to import. Left blank on purpose -- set it to the URDF
# you want to load (an absolute path or a path the bridge process can read).
try:
    robot_description = telekinesis_urdfs.load("OnRobotRG6")
    if not robot_description.urdf_path.is_file():
        raise ValueError(f"No urdf found, i.e. {robot_description.urdf_path} is not a file")

except Exception as e:
    raise RuntimeError(
        f"Failed to load robot description for '{__class__.__name__}'. "
        "Ensure telekinesis-urdfs is installed: "
        "https://github.com/telekinesis-ai/telekinesis-urdfs"
    ) from e

URDF_PATH = str(robot_description.urdf_path)

HOST = "127.0.0.1"
PORT = 8766
GRIPPER_PRIM_PATH = "/World/rg6"

# A move blocks server-side until it finishes, so allow well over the bridge's
# own motion cap (~30 s) before the HTTP client gives up.
DEFAULT_TIMEOUT = 120.0


def _request(base, method, path, body=None):
    """Send one request and return the decoded JSON (None for an empty body)."""
    response = requests.request(method, base.rstrip("/") + path, json=body, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json() if response.content else None


def run_gripper(base, prim_path, urdf_path):
    info = _request(base, "PUT", "/articulations", {"prim_path": prim_path, "urdf_path": urdf_path})
    articulation_id = info["articulation_id"]
    print(f"created gripper: articulation_id={articulation_id} prim_path={info['prim_path']}")

    # Discover the actuated joint (the bridge walks the USD/PhysX schema) and narrow
    # the device to it -- after this the gripper drives exactly one joint.
    driver = _request(base, "GET", f"/articulations/{articulation_id}/driver_joint")
    _request(base, "PUT", f"/articulations/{articulation_id}/driven_joints", {"joint_names": [driver["name"]]})

    # With the device narrowed to the driver, joint_limits is a single pair.
    # Convention: lower = open, upper = closed.
    opened_rad, closed_rad = _request(base, "GET", f"/articulations/{articulation_id}/joint_limits")["limits"][0]
    print(f"  driver joint='{driver['name']}' open={opened_rad:.3f} closed={closed_rad:.3f} rad")

    # Map a closed-ness fraction to a joint angle (this is the only gripper-specific
    # math, done client-side): close = 1.0, open = 0.0.
    for label, fraction in (("close", 1.0), ("open", 0.0)):
        target_rad = opened_rad + fraction * (closed_rad - opened_rad)
        print(f"gripper {label} (fraction={fraction})")
        status = _request(
            base, "POST", f"/articulations/{articulation_id}/joint_positions", {"positions": [target_rad]}
        )
        print(f"  done (reached={status['reached']} q={status['q'][0]:.3f} rad)")


def main():
    parser = argparse.ArgumentParser(
        description="Gripper-only smoke test for the Isaac Sim bridge.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--prim",
        default=GRIPPER_PRIM_PATH,
        help="prim path of the gripper in the stage")
    parser.add_argument(
        "--urdf",
        default=URDF_PATH,
        help="optional URDF to import if the prim isn't in the stage")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"bridge: {base}  articulations: {_request(base, 'GET', '/articulations')}")
    run_gripper(base, args.prim, args.urdf)


if __name__ == "__main__":
    main()
