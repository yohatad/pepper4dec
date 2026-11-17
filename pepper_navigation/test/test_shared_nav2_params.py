#!/usr/bin/env python3
"""Guard the nav2 param sections that are meant to be identical across modes.

Of the five nav2 param files (amcl / fastloc / pointloc / rtabmap_loc, plus
the l2voxel_test bag variant), only 3 of 10 top-level nodes are true
duplicates -- and those are the drift risk: tuning a controller gain means
editing five files, with nothing to tell you if you edit four.

Extracting them into a shared base was considered and rejected: it would leave
7 nodes mode-specific at the cost of a launch-time yaml merge, a new code path
whose bugs would silently change nav2 behaviour. Checking is cheaper and
catches the same mistake.

    python3 test/test_shared_nav2_params.py     # or: colcon test
"""
import copy
import os
import sys

import pytest
import yaml

SHARED = ("behavior_server", "controller_server", "planner_server")

# Deliberate per-profile values inside a shared section: {dotted leaf under
# ros__parameters: files allowed to differ}. Each is dropped from the
# comparison in those files only, so the rest of the section stays guarded.
# Keep this short, and say why in the param file.
EXCEPTIONS = {
    "controller_server": {
        # fastloc alone at 1000 (fe0c122): 2000 used ~84% of a Jetson core and
        # missed the 50 ms loop; the others follow once it has more mileage.
        "FollowPath.batch_size": ("nav2_params_fastloc.yaml",),
    },
}
FILES = ("nav2_params_amcl.yaml", "nav2_params_fastloc.yaml",
         "nav2_params_rtabmap_loc.yaml", "nav2_params_pointloc.yaml",
         "nav2_params_kissicp.yaml", "bag_test/nav2_params_l2voxel_test.yaml")

# The robot's size, and the clearance kept around it, live inside sections
# that are otherwise mode-specific (the costmaps and collision monitor differ
# in frames and sources), so they are checked leaf by leaf. They drifted once
# already: amcl/rtabmap_loc kept robot_radius 0.25 after the loc profiles
# moved to 0.20.
GEOMETRY = (
    "local_costmap.local_costmap.ros__parameters.robot_radius",
    "local_costmap.local_costmap.ros__parameters.inflation_layer.inflation_radius",
    "local_costmap.local_costmap.ros__parameters.inflation_layer.cost_scaling_factor",
    "global_costmap.global_costmap.ros__parameters.robot_radius",
    "global_costmap.global_costmap.ros__parameters.inflation_layer.inflation_radius",
    "global_costmap.global_costmap.ros__parameters.inflation_layer.cost_scaling_factor",
    "collision_monitor.ros__parameters.PolygonStop.type",
    "collision_monitor.ros__parameters.PolygonStop.points",
    "collision_monitor.ros__parameters.PolygonSlow.type",
    "collision_monitor.ros__parameters.PolygonSlow.points",
    # Observation sources: a profile that quietly loses one is a profile with
    # a blind spot, which is exactly the kind of drift this guard is for.
    "collision_monitor.ros__parameters.observation_sources",
)

# bt_navigator differs per mode only in odom_topic, so the rest is checked leaf
# by leaf. bt_loop_duration drifted once already: the 500 Hz -> 20 Hz fix went
# into fastloc/pointloc only, leaving amcl/kissicp/rtabmap_loc ticking at
# 500 Hz.
BT_NAVIGATOR = tuple(
    "bt_navigator.ros__parameters." + k for k in (
        "global_frame", "robot_base_frame", "bt_loop_duration",
        "default_server_timeout", "wait_for_service_timeout",
        "default_nav_to_pose_bt_xml", "plugin_lib_names"))


def _load():
    """Load the nav2 param files, returning {filename: parsed yaml}."""
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
    loaded = {}
    for f in FILES:
        p = os.path.join(cfg, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f)
        loaded[f] = yaml.safe_load(open(p))
    return loaded


def _param_leaf(section, leaf, pop=False):
    """Read (or remove, with pop) a dotted leaf under ros__parameters."""
    *path, key = leaf.split(".")
    d = section.get("ros__parameters", {})
    for k in path:
        d = d.get(k, {})
    return d.pop(key, None) if pop else d.get(key)


def _exception_drift(node, loaded):
    """Each excepted leaf must still agree across the files NOT excepted."""
    out = []
    for leaf, files in EXCEPTIONS.get(node, {}).items():
        vals = {f: _param_leaf(d[node], leaf) for f, d in loaded.items()
                if node in d and f not in files}
        if len(set(map(repr, vals.values()))) > 1:
            out.append("%-20s %s differs outside its exception: %r" % (node, leaf, vals))
    return out


def _drift(node, loaded):
    """Return a list of human-readable drift reports for one shared node.

    An empty list means the node is byte-identical across all the param files.
    """
    have = {f: copy.deepcopy(d[node]) for f, d in loaded.items() if node in d}
    # Excepted leaves are compared by _exception_drift, not here.
    for section in have.values():
        for leaf in EXCEPTIONS.get(node, {}):
            _param_leaf(section, leaf, pop=True)
    missing = [f for f in FILES if f not in have]
    if missing:
        return ["%-20s absent from %s" % (node, ", ".join(missing))]

    ref_file, ref = next(iter(have.items()))
    for f, v in have.items():
        if v != ref:
            out = ["%-20s differs between %s and %s" % (node, ref_file, f)]
            for k in sorted(set(ref.get("ros__parameters", {})) |
                            set(v.get("ros__parameters", {}))):
                a = ref.get("ros__parameters", {}).get(k)
                b = v.get("ros__parameters", {}).get(k)
                if a != b:
                    out.append("       %-34s %r != %r" % (k, a, b))
            return out
    return []


@pytest.mark.parametrize("node", SHARED)
def test_shared_section_identical_across_param_files(node):
    """Each shared nav2 section must be identical in all the param files."""
    loaded = _load()
    drift = _drift(node, loaded) + _exception_drift(node, loaded)
    assert not drift, "\n".join(drift)


def _leaf(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return "<absent>"
        d = d[k]
    return d


def _leaf_drift(loaded, paths):
    """Return drift reports for the given leaves; empty when all agree."""
    out = []
    for path in paths:
        vals = {f: _leaf(d, path) for f, d in loaded.items()}
        if len(set(map(repr, vals.values()))) > 1:
            out.append("%s differs: %s" % (path, ", ".join(
                "%s=%r" % (f.replace("nav2_params_", "").replace(".yaml", ""), v)
                for f, v in vals.items())))
    return out


def test_robot_geometry_identical_across_param_files():
    """Robot radius, inflation and safety zones must match in every mode."""
    drift = _leaf_drift(_load(), GEOMETRY)
    assert not drift, "\n".join(drift)


def test_bt_navigator_identical_across_param_files():
    """Tick period, BT plugins and tree path must match in every mode."""
    drift = _leaf_drift(_load(), BT_NAVIGATOR)
    assert not drift, "\n".join(drift)


def main():
    try:
        loaded = _load()
    except FileNotFoundError as e:
        print("MISSING %s" % e.args[0])
        return 1

    failures = 0
    for node in SHARED:
        drift = _drift(node, loaded)
        if drift:
            for line in drift:
                print("FAIL %s" % line if line.strip() == line else line)
            failures += 1
        else:
            print("ok   %-20s identical across all %d files" % (node, len(FILES)))

    for label, paths in (("robot geometry", GEOMETRY),
                         ("bt_navigator", BT_NAVIGATOR)):
        drift = _leaf_drift(loaded, paths)
        for line in drift:
            print("FAIL %s" % line)
        if drift:
            failures += 1
        else:
            print("ok   %-20s identical across all %d files" % (label, len(FILES)))

    print("\n%s" % ("PASS" if not failures
                    else "%d shared section(s) have drifted" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
