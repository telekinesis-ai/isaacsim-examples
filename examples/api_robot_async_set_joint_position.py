"""
Bridge example: the three ways to drive a robot through joint_positions.

POST /articulations/{id}/joint_positions takes an ``asynchronous`` flag. This
script shows all three resulting styles against one robot, with raw HTTP calls so
the difference is visible on the wire:

  1. asynchronous=True, fire-and-forget
       The bridge applies the action and returns immediately ({applied, target}).
       The script does NOT wait -- useful to queue a motion and move on (the move
       still completes in sim). We read the state right after to show it's moving.

  2. asynchronous=True, then wait on the client side for a tolerance
       The bridge returns immediately; the client polls GET joint_state until the
       joints are within a tolerance of the target. This puts the "am I done?"
       decision in the client.

  3. asynchronous=False, blocking
       The bridge runs the move to completion server-side (reaching the target or
       stalling) and returns the final status. Simplest: one call per move.

Run:  python robot_async.py
      python robot_async.py --prim /World/ur10e

Requires the ``requests`` package (``pip install requests``).
"""

import argparse
import numpy as np
import time

import requests

HOST = "127.0.0.1"
PORT = 8766
ROBOT_PRIM_PATH = "/World/kuka_kr210"

# Joint targets (degrees here for readability; converted to radians on the wire).
TARGET_A_DEG = [-90.0, -90.0, 0.0, 0.0, 90.0, 0.0]
TARGET_B_DEG = [90.0, -90.0, 0.0, 0.0, 90.0, 0.0]
TARGET_C_DEG = [0.0, -90.0, 0.0, 0.0, 90.0, 0.0]

# A blocking move runs to completion server-side (over the bridge's ~30 s motion
# cap), so give the HTTP client a generous timeout before it gives up.
DEFAULT_TIMEOUT = 120.0


def _request(base, method, path, body=None):
    """Send one request and return the decoded JSON (None for an empty body)."""
    response = requests.request(method, base.rstrip("/") + path, json=body, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json() if response.content else None


def set_joint_positions(base, articulation_id, positions, asynchronous):
    return _request(
        base,
        "POST",
        f"/articulations/{articulation_id}/joint_positions",
        {"positions": positions, "asynchronous": asynchronous},
    )


def wait_until_reached(
        base,
        articulation_id,
        target,
        tolerance_rad=5e-3,
        timeout_s=30.0,
        poll_s=0.05):
    """Poll joint_state until every joint is within ``tolerance_rad`` of target.

    The client-side "done" check for an asynchronous move: the bridge applied the
    action and returned immediately, so we watch the state ourselves. Sim keeps
    stepping (the timeline is playing), so polling converges.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        q = _request(base, "GET", f"/articulations/{articulation_id}/joint_state")["q"]
        max_error = max(abs(a - b) for a, b in zip(q, target))
        if max_error < tolerance_rad:
            return {"reached": True, "max_error": max_error}
        time.sleep(poll_s)
    q = _request(base, "GET", f"/articulations/{articulation_id}/joint_state")["q"]
    return {"reached": False, "max_error": max(abs(a - b) for a, b in zip(q, target))}


def run(base, prim_path):
    info = _request(base, "PUT", "/articulations", {"prim_path": prim_path, "urdf_path": None})
    articulation_id = info["articulation_id"]
    print(f"created robot: articulation_id={articulation_id} prim_path={info['prim_path']}")
    print(f"  num_dof={info['num_dof']} dof_names={info['dof_names']}")

    # 1) asynchronous, fire-and-forget -------------------------------------------
    print(f"\n[1] async fire-and-forget -> {TARGET_A_DEG}")
    target_a = np.deg2rad(TARGET_A_DEG).tolist()
    result = set_joint_positions(base, articulation_id, target_a, asynchronous=True)
    print(f"  returned immediately: {result}")
    time.sleep(2)
    q = _request(base, "GET", f"/articulations/{articulation_id}/joint_state")["q"]
    print(f"  state shortly after (still moving): q={[round(v, 3) for v in q]}")

    # 2) asynchronous, then wait client-side for a tolerance ---------------------
    print(f"\n[2] async + client-side wait -> {TARGET_B_DEG}")
    target_b = np.deg2rad(TARGET_B_DEG).tolist()
    set_joint_positions(base, articulation_id, target_b, asynchronous=True)
    status = wait_until_reached(base, articulation_id, target_b, tolerance_rad=5e-3)
    print(f"  client saw reached={status['reached']} (max_error={status['max_error']:.2e} rad)")

    # 3) blocking ----------------------------------------------------------------
    print(f"\n[3] blocking -> {TARGET_C_DEG}")
    target_c = np.deg2rad(TARGET_C_DEG).tolist()
    status = set_joint_positions(base, articulation_id, target_c, asynchronous=False)
    print(f"  server returned done={status['done']} reached={status['reached']} "
          f"(max_error={status['max_error']:.2e} rad)")


def main():
    parser = argparse.ArgumentParser(
        description="Async vs blocking robot moves for the Isaac Sim bridge.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--prim",
        default=ROBOT_PRIM_PATH,
        help="prim path of the robot in the stage")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    run(base, args.prim)


if __name__ == "__main__":
    main()
