"""
Bridge example: load a robot from a URDF, then drive it.

Most examples assume the robot prim already exists in the stage. This one
exercises the *loading* path instead: when ``PUT /articulations`` is given a
``urdf_path`` and the ``prim_path`` is not yet in the stage, the bridge imports
the URDF at that prim path before binding it. After that it's an ordinary
articulation you drive with POST /articulations/{id}/joint_positions.

Flow (all over HTTP):
  1. PUT /articulations {prim_path, urdf_path} -> imports + binds
  2. POST /articulations/{id}/joint_positions {positions}  (radians)
     -> blocks until the move reaches the target or stalls; returns {done, reached, ...}

Run:  python robot_load_from_urdf.py
      python robot_load_from_urdf.py --prim /World/my_robot

Requires the ``requests`` package (``pip install requests``).
"""

import argparse
import numpy as np

import requests

try:
    import telekinesis_urdfs
except ImportError as e:
    raise ImportError(
        "telekinesis-urdfs is not installed. "
        "Install it from: https://github.com/telekinesis-ai/telekinesis-urdfs"
    ) from e

HOST = "127.0.0.1"
PORT = 8766
ROBOT_PRIM_PATH = "/World/ur10e"

# Path to the robot's URDF to import. Left blank on purpose -- set it to the URDF
# you want to load (an absolute path or a path the bridge process can read).
try:
    robot_description = telekinesis_urdfs.load("UniversalRobotsUR10e")
    if not robot_description.urdf_path.is_file():
        raise ValueError(f"No urdf found, i.e. {robot_description.urdf_path} is not a file")

except Exception as e:
    raise RuntimeError(
        f"Failed to load robot description for '{__class__.__name__}'. "
        "Ensure telekinesis-urdfs is installed: "
        "https://github.com/telekinesis-ai/telekinesis-urdfs"
    ) from e

URDF_PATH = str(robot_description.urdf_path)

# Joint target once loaded (degrees here for readability; radians on the wire).
TARGET_DEG = [-90.0, -90.0, 0.0, 0.0, 90.0, 0.0]

# A URDF import plus the blocking move can both take a while server-side, so give
# the HTTP client a generous timeout (well over the bridge's ~30 s motion cap).
DEFAULT_TIMEOUT = 120.0


def _request(base, method, path, body=None):
    """Send one request and return the decoded JSON (None for an empty body)."""
    response = requests.request(method, base.rstrip("/") + path, json=body, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json() if response.content else None


def run_robot(base, prim_path, urdf_path):
    # urdf_path tells the bridge to import the robot if prim_path isn't in the stage.
    info = _request(base, "PUT", "/articulations", {"prim_path": prim_path, "urdf_path": urdf_path})
    articulation_id = info["articulation_id"]
    print(f"loaded + created robot: articulation_id={articulation_id} prim_path={info['prim_path']}")
    print(f"  num_dof={info['num_dof']} dof_names={info['dof_names']}")

    print(f"move target (deg): {TARGET_DEG}")
    status = _request(
        base,
        "POST",
        f"/articulations/{articulation_id}/joint_positions",
        {"positions": np.deg2rad(TARGET_DEG).tolist()},
    )
    print(f"  done={status['done']} reached={status['reached']} (max_error={status['max_error']:.2e} rad)")


def main():
    parser = argparse.ArgumentParser(description="Load a robot from URDF via the Isaac Sim bridge.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--prim", default=ROBOT_PRIM_PATH, help="prim path to import the robot at")
    parser.add_argument("--urdf", default=URDF_PATH, help="path to the robot URDF to import")
    args = parser.parse_args()

    if args.urdf is ... or not args.urdf:
        parser.error("set URDF_PATH in this file or pass --urdf <path/to/robot.urdf>")

    base = f"http://{args.host}:{args.port}"
    print(f"bridge: {base}  articulations: {_request(base, 'GET', '/articulations')}")
    run_robot(base, args.prim, args.urdf)


if __name__ == "__main__":
    main()
