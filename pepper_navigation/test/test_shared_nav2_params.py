#!/usr/bin/env python3
"""Guard the nav2 param sections that are meant to be identical across modes.

There are four nav2 param files (amcl / fastlio_loc / rtabmap_loc / base). Most
of their content legitimately differs per mode -- only 3 of 10 top-level nodes
are true duplicates. Those three are the drift risk: tuning a controller gain
means editing four files, and nothing tells you if you edit three.

Extracting them into a shared base was considered and rejected: it would move 3
nodes and leave 7 mode-specific, at the cost of a launch-time yaml merge -- a new
code path whose bugs would silently change nav2 behaviour. Checking is cheaper
than merging and catches the same mistake.

    python3 test/test_shared_nav2_params.py     # or: colcon test
"""
import os
import sys

import pytest
import yaml

SHARED = ("behavior_server", "controller_server", "planner_server")
FILES = ("nav2_params_amcl.yaml", "nav2_params_fastlio_loc.yaml",
         "nav2_params_rtabmap_loc.yaml", "nav2_params.yaml")


def _load():
    """Load the four nav2 param files, returning {filename: parsed yaml}."""
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

    An empty list means the node is byte-identical across all four files.
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
    """Each shared nav2 section must be identical in all four param files."""
    drift = _drift(node, _load())
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

    print("\n%s" % ("PASS" if not failures
                    else "%d shared section(s) have drifted" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
