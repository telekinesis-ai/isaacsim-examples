"""Runnable example client for the *implemented* bridge extension routes.

The bridge mirrors the Wandelbots Isaac Sim extension API (General / Stage /
Prims, alongside its own device routes), but most of those routes still answer
``501 Not Implemented``. This file is the small, runnable companion: it only
calls the endpoints the extension actually implements today -- the General
(``/status``, ``/version``), Stage (bar configuration import/export) and Prims
routes -- so you can run it end to end against the live bridge and see real
responses.

The bridge listens on 127.0.0.1:8766 (see telekinesis_isaacsim_bridge). With
Isaac Sim running and the extension enabled::

    python extension_client.py
    python extension_client.py --base_url http://127.0.0.1:8766

Add a function here each time another mirrored route graduates from a stub to a
real implementation.

Requires the ``requests`` package (``pip install requests``).
"""

import argparse

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8766"
DEFAULT_TIMEOUT = 30.0


def _request(base_url, method, path, params=None, body=None):
    """Send one request and return the decoded JSON, or None for an empty body."""
    response = requests.request(
        method, base_url.rstrip("/") + path, params=params, json=body, timeout=DEFAULT_TIMEOUT
    )
    response.raise_for_status()
    return response.json() if response.content else None


# -- General ----------------------------------------------------------------
def get_status(base_url):
    """GET /status -> {"status": "OK"} while the extension is running."""
    return _request(base_url, "GET", "/status")


def get_versions(base_url):
    """GET /version -> {extension_name: version} for installed extensions."""
    return _request(base_url, "GET", "/version")


# -- Stage ------------------------------------------------------------------
def get_active_scene(base_url):
    """GET /stage/scene -> URI of the open USD stage (string)."""
    return _request(base_url, "GET", "/stage/scene")


def open_scene(base_url, uri):
    """PUT /stage/scene -> open the USD stage at ``uri``."""
    return _request(base_url, "PUT", "/stage/scene", body={"uri": uri})


def list_stage_motion_groups(base_url):
    """GET /stage/motion-groups -> prim paths of articulation roots."""
    return _request(base_url, "GET", "/stage/motion-groups")


def get_stage_units(base_url):
    """GET /stage/units -> {"meters_per_unit": float}."""
    return _request(base_url, "GET", "/stage/units")


def update_stage_units(base_url, meters_per_unit):
    """PUT /stage/units -> set the stage scale."""
    return _request(base_url, "PUT", "/stage/units", body={"meters_per_unit": meters_per_unit})


def timeline_action(base_url, action):
    """PATCH /stage/simulation/timeline/{action} -> play / pause / stop."""
    return _request(base_url, "PATCH", f"/stage/simulation/timeline/{action}")


def simulation_state(base_url):
    """GET /stage/simulation -> {"timeline": "playing"|"paused"|"stopped"}."""
    return _request(base_url, "GET", "/stage/simulation")


# -- Prims ------------------------------------------------------------------
def get_pose(base_url, prim_path, coordinate_system="local", rotation_type="cartesian"):
    """GET /prims/poses -> {"pose": [...]} for one prim."""
    params = {
        "prim_path": prim_path,
        "coordinate_system": coordinate_system,
        "rotation_type": rotation_type,
    }
    return _request(base_url, "GET", "/prims/poses", params=params)


def update_pose(base_url, prim_path, pose):
    """PUT /prims/poses -> set a prim's world pose ([x,y,z,rx,ry,rz])."""
    body = {"prim_path": prim_path, "input_pose": {"pose": pose}}
    return _request(base_url, "PUT", "/prims/poses", body=body)


def get_relative_pose(base_url, prim_path_1, prim_path_2, mode="normal", rotation_type="cartesian"):
    """GET /prims/poses/relative -> pose of prim 2 in prim 1's frame."""
    params = {
        "prim_path_1": prim_path_1,
        "prim_path_2": prim_path_2,
        "mode": mode,
        "rotation_type": rotation_type,
    }
    return _request(base_url, "GET", "/prims/poses/relative", params=params)


def apply_relative_pose(base_url, prim_path, pose, object_first=False):
    """POST /prims/poses/relative -> nudge a prim by a relative pose."""
    body = {"prim_path": prim_path, "relative_pose": {"pose": pose}, "object_first": object_first}
    return _request(base_url, "POST", "/prims/poses/relative", body=body)


def list_default_poses(base_url):
    """GET /prims/poses/default -> {prim_path: {"pose": [...]}}."""
    return _request(base_url, "GET", "/prims/poses/default")


def assign_default_poses(base_url, prim_path):
    """PUT /prims/poses/default -> record a prim's current pose as its default."""
    return _request(base_url, "PUT", "/prims/poses/default", body=prim_path)


def clear_default_poses(base_url):
    """DELETE /prims/poses/default -> forget all stored default poses."""
    return _request(base_url, "DELETE", "/prims/poses/default")


def reset_to_default_poses(base_url, prim_path):
    """POST /prims/poses/default/reset -> restore a prim to its default pose."""
    return _request(base_url, "POST", "/prims/poses/default/reset", body=prim_path)


def set_prim_metadata(base_url, prim_path, category, type_):
    """PUT /prims/metadata -> store {category, type} on a prim."""
    body = {"prim_path": prim_path, "metadata": {"category": category, "type": type_}}
    return _request(base_url, "PUT", "/prims/metadata", body=body)


def remove_prim_metadata(base_url, prim_path):
    """DELETE /prims/metadata -> remove a prim's stored metadata."""
    return _request(base_url, "DELETE", "/prims/metadata", params={"prim_path": prim_path})


def set_prim_visibility(base_url, prim_path, visibility="show"):
    """PATCH /prims/visibility -> show or hide a prim."""
    body = {"prim_path": prim_path, "visibility": visibility}
    return _request(base_url, "PATCH", "/prims/visibility", body=body)


def set_joint_state(base_url, prim_path, enable):
    """PATCH /prims/physics/joints -> enable/disable a physics joint."""
    body = {"prim_path": prim_path, "enable": enable}
    return _request(base_url, "PATCH", "/prims/physics/joints", body=body)


def update_colliders(base_url, prim_path, enable):
    """PATCH /prims/physics/colliders/ -> enable/disable collision on a prim."""
    body = {"prim_path": prim_path, "enable": enable}
    return _request(base_url, "PATCH", "/prims/physics/colliders/", body=body)


def test_stage(base_url):
    """Exercise every implemented stage route in a meaningful read → mutate → restore order."""
    print("  active scene:      ", get_active_scene(base_url))
    print("  motion groups:     ", list_stage_motion_groups(base_url))

    original_units = get_stage_units(base_url)["meters_per_unit"]
    print(f"  units (original):  {original_units}")
    update_stage_units(base_url, 1.0)
    print("  units (set 1.0):   ", get_stage_units(base_url))
    update_stage_units(base_url, original_units)
    print("  units (restored):  ", get_stage_units(base_url))

    print("  sim (initial):     ", simulation_state(base_url))
    timeline_action(base_url, "play")
    print("  sim (after play):  ", simulation_state(base_url))
    timeline_action(base_url, "pause")
    print("  sim (after pause): ", simulation_state(base_url))
    timeline_action(base_url, "stop")
    print("  sim (after stop):  ", simulation_state(base_url))


def test_prims(base_url, prim_path):
    """Exercise every implemented prim route in a meaningful read → mutate → restore order."""
    # --- Pose lifecycle ---
    original_pose = get_pose(base_url, prim_path, "world")["pose"]
    print(f"  pose (original):       {original_pose}")

    assign_default_poses(base_url, prim_path)
    print("  default poses (saved): ", list_default_poses(base_url))

    nudged_pose = list(original_pose)
    nudged_pose[0] += 0.01
    update_pose(base_url, prim_path, nudged_pose)
    print("  pose (after move +X):  ", get_pose(base_url, prim_path, "world"))

    print("  relative pose (self):  ", get_relative_pose(base_url, prim_path, prim_path))

    apply_relative_pose(base_url, prim_path, [0.01, 0, 0, 0, 0, 0])
    print("  pose (after nudge):    ", get_pose(base_url, prim_path, "world"))

    # --- Restore ---
    reset_to_default_poses(base_url, prim_path)
    restored_pose = get_pose(base_url, prim_path, "world")["pose"]
    print(f"  pose (restored):       {restored_pose}")
    print(f"  matches original:      {restored_pose == original_pose}")

    clear_default_poses(base_url)
    print("  default poses (clear): ", list_default_poses(base_url))

    # --- Metadata ---
    set_prim_metadata(base_url, prim_path, "test_category", "test_type")
    print("  metadata (set):        ok")
    remove_prim_metadata(base_url, prim_path)
    print("  metadata (removed):    ok")

    # --- Visibility ---
    set_prim_visibility(base_url, prim_path, "hide")
    print("  visibility (hidden):   ok")
    set_prim_visibility(base_url, prim_path, "show")
    print("  visibility (shown):    ok")

    # --- Physics (may fail if prim has no joint or collider) ---
    try:
        set_joint_state(base_url, prim_path, enable=False)
        set_joint_state(base_url, prim_path, enable=True)
        print("  joint toggle:          ok")
    except Exception as exc:
        print(f"  joint toggle:          skipped ({exc})")

    try:
        update_colliders(base_url, prim_path, enable=False)
        update_colliders(base_url, prim_path, enable=True)
        print("  collider toggle:       ok")
    except Exception as exc:
        print(f"  collider toggle:       skipped ({exc})")


def main():
    parser = argparse.ArgumentParser(description="Call the implemented bridge extension routes.")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL,
                        help="Base URL of the running bridge")
    parser.add_argument("--prim", required=True, metavar="PRIM_PATH",
                        help="Prim path to use for the Prims test (e.g. /World/Cube)")
    args = parser.parse_args()

    print(f"bridge: {args.base_url}\n")

    print("=== General ===")
    print("  status:  ", get_status(args.base_url))
    print("  version: ", get_versions(args.base_url))

    print("\n=== Stage ===")
    test_stage(args.base_url)

    print("\n=== Prims ===")
    test_prims(args.base_url, args.prim)


if __name__ == "__main__":
    main()
