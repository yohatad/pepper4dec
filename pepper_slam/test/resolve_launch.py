#!/usr/bin/env python3
"""Resolve a launch file to the concrete nodes + parameter files it will start.

A verification net for refactoring launch files: snapshot before, snapshot after,
diff. Catches a parameter that stops reaching the estimator -- which
`--show-args loads OK` does not, and which otherwise surfaces only as bad
odometry some minutes into a run.
"""
import io
import json
import sys
from launch import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import SetLaunchConfiguration
from launch_ros.actions import Node


def describe(n, ctx):
    def perf(x):
        try:
            if x is None:
                return None
            if isinstance(x, (str, int, float, bool)):
                return x
            if isinstance(x, (list, tuple)):
                return "".join(str(perf(i)) for i in x)
            return x.perform(ctx)
        except Exception:
            return "<unresolved>"
    d = {"pkg": perf(getattr(n, "_Node__package", None)),
         "exe": perf(getattr(n, "_Node__node_executable", None)),
         "name": perf(getattr(n, "_Node__node_name", None)),
         "ns": perf(getattr(n, "_Node__expanded_node_namespace", None))}
    params = []
    for p in (getattr(n, "_Node__parameters", None) or []):
        try:
            params.append(perf(p) if not isinstance(p, dict)
                          else {str(perf(k)): str(perf(v))[:80] for k, v in p.items()})
        except Exception:
            params.append("<unresolved>")
    d["params"] = params
    remaps = []
    for r in (getattr(n, "_Node__remappings", None) or []):
        try:
            remaps.append("%s:=%s" % (perf(r[0]), perf(r[1])))
        except Exception:
            remaps.append("<unresolved>")
    d["remaps"] = remaps
    return d


def walk(entities, ctx, out, seen=0):
    if seen > 40:
        return
    for e in entities:
        if isinstance(e, Node):
            # Node.visit() would try to EXECUTE, so the condition is checked by
            # hand. Skipping this is how an earlier version reported both of
            # pepper_sensor_tf's mutually exclusive publishers as running.
            cond = getattr(e, "condition", None)
            try:
                enabled = True if cond is None else cond.evaluate(ctx)
            except Exception:
                enabled = True
            if enabled:
                out.append(describe(e, ctx))
            continue
        try:
            sub = e.visit(ctx)
        except Exception as ex:
            out.append({"ERROR": "%s: %s" % (type(e).__name__, str(ex)[:70])})
            continue
        if sub:
            walk(sub, ctx, out, seen + 1)


def snap(path, argv):
    ctx = LaunchContext(argv=argv)
    # LaunchContext(argv=...) does NOT set launch configurations; ros2 launch
    # parses 'name:=value' itself. Do the same or every argument is ignored.
    for a in argv:
        if ":=" in a:
            k, v = a.split(":=", 1)
            SetLaunchConfiguration(k, v).visit(ctx)
    ld = PythonLaunchDescriptionSource(path).get_launch_description(ctx)
    out = []
    walk(ld.entities, ctx, out)
    return out


if __name__ == "__main__":
    import os
    res = snap(sys.argv[1], sys.argv[2:])
    blob = json.dumps(res, indent=1, sort_keys=True, default=str)
    out = os.environ.get("SNAP_OUT")
    if out:                      # keep JSON off stdout: launch LogInfo lives there
        io.open(out, "w", encoding="utf-8").write(blob)
    else:
        print(blob)
