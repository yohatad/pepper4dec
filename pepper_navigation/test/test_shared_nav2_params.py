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
import os
import sys

import pytest
import yaml

SHARED = ("behavior_server", "controller_server", "planner_server")
FILES = ("nav2_params_amcl.yaml", "nav2_params_fastloc.yaml",
         "nav2_params_rtabmap_loc.yaml", "nav2_params_pointloc.yaml",
         "nav2_params_kissicp.yaml", "nav2_params_l2voxel_test.yaml")

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
    "collision_monitor.ros__parameters.PolygonStop.radius",
    "collision_monitor.ros__parameters.PolygonSlow.radius",
)


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


def _drift(node, loaded):
    """Return a list of human-readable drift reports for one shared node.

    An empty list means the node is byte-identical across all the param files.
    """
    have = {f: d[node] for f, d in loaded.items() if node in d}
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
    drift = _drift(node, _load())
    assert not drift, "\n".join(drift)


def _leaf(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return "<absent>"
        d = d[k]
    return d


def _geometry_drift(loaded):
    """Return drift reports for the GEOMETRY leaves; empty when all agree."""
    out = []
    for path in GEOMETRY:
        vals = {f: _leaf(d, path) for f, d in loaded.items()}
        if len(set(map(repr, vals.values()))) > 1:
            out.append("%s differs: %s" % (path, ", ".join(
                "%s=%r" % (f.replace("nav2_params_", "").replace(".yaml", ""), v)
                for f, v in vals.items())))
    return out


def test_robot_geometry_identical_across_param_files():
    """Robot radius, inflation and safety zones must match in every mode."""
    drift = _geometry_drift(_load())
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

    geo = _geometry_drift(loaded)
    for line in geo:
        print("FAIL %s" % line)
    if geo:
        failures += 1
    else:
        print("ok   robot geometry       identical across all %d files" % len(FILES))

    print("\n%s" % ("PASS" if not failures
                    else "%d shared section(s) have drifted" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
