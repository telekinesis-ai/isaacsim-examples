"""
Standalone bridge example: create articulation -> move through joint targets.

The bridge is a single HTTP/JSON server on 127.0.0.1:8766
(see telekinesis_isaacsim_bridge). It is device-agnostic: it drives an
articulation's joints in radians and reports reached/stalled -- there is no
"robot" notion on the wire, so a robot is just an articulation whose joints you
set directly.

Flow (all over HTTP):
  1. PUT /articulations {prim_path, urdf_path?} -> {articulation_id, num_dof, ...}
  2. for each target: POST /articulations/{id}/joint_positions {positions}  (radians)
     -> blocks until the move reaches the target or stalls; returns {done, reached, ...}

The bridge runs each move to completion server-side (it steps physics on Isaac's
own loop), so the POST blocks until done -- no client-side polling.

Run:  python robot_set_joint_position.py
      python robot_set_joint_position.py --prim /World/ur10e

Requires the ``requests`` package (``pip install requests``).
"""
import time
import argparse
import numpy as np

import requests

HOST = "127.0.0.1"
PORT = 8766
ROBOT_PRIM_PATH = "/World/kuka_kr210"

# Joint targets (degrees here for readability; converted to radians on the wire).
TARGET_DEG = [-90.0, -90.0, 0.0, 0.0, 90.0, 0.0]
TARGET_DEG2 = [-20.0, 20.0, 0.0, 0.0, 80.0, 90.0]
TARGETS_DEG = [TARGET_DEG, TARGET_DEG2, TARGET_DEG]

# A move blocks server-side until it finishes, so allow well over the bridge's
# own motion cap (~30 s) before the HTTP client gives up.
DEFAULT_TIMEOUT = 120.0


def _request(base, method, path, body=None):
    """Send one request and return the decoded JSON (None for an empty body)."""
    response = requests.request(method, base.rstrip("/") + path, json=body, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json() if response.content else None


def run_robot(base, prim_path):
    info = _request(base, "PUT", "/articulations", {"prim_path": prim_path, "urdf_path": None})
    articulation_id = info["articulation_id"]
    print(f"created robot: articulation_id={articulation_id} prim_path={info['prim_path']}")
    print(f"  num_dof={info['num_dof']} dof_names={info['dof_names']}")

    for target_deg in TARGETS_DEG:
        print(f"move target (deg): {target_deg}")
        status = _request(
            base,
            "POST",
            f"/articulations/{articulation_id}/joint_positions",
            {"positions": np.deg2rad(target_deg).tolist()},
        )
        print(
            f"  done={status['done']} reached={status['reached']} (max_error={status['max_error']:.2e} rad)")
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(
        description="Robot joint-target example for the Isaac Sim bridge.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--prim",
        default=ROBOT_PRIM_PATH,
        help="prim path of the robot in the stage")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"bridge: {base}  articulations: {_request(base, 'GET', '/articulations')}")
    run_robot(base, args.prim)


if __name__ == "__main__":
    main()
