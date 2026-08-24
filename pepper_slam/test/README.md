# test/

## `resolve_launch.py`

Resolves a launch file to the concrete nodes, parameters and remappings it will
actually start, as JSON. Written as a safety net for refactoring launch files:
snapshot before, refactor, snapshot after, diff.

```bash
SNAP_OUT=after.json python3 test/resolve_launch.py \
    $(ros2 pkg prefix pepper_slam)/share/pepper_slam/launch/fastlio_odometry.launch.py
diff <(jq -S . test/launch_baselines/fl_odom.json) <(jq -S . after.json)
```

`SNAP_OUT` matters: launch `LogInfo` actions print to stdout, so JSON goes to a
file instead of being interleaved with them.

`ros2 launch ... --show-args` loading without error is NOT sufficient
verification for a launch refactor. It does not tell you whether a parameter
still reaches the estimator; a dropped one surfaces as bad odometry minutes into
a run, not as an error.

## `launch_baselines/`

Resolved snapshots of the six LIO entry points, captured 2026-08-24 with all
defaults. Regenerate deliberately, never casually -- their value is being a
record of behaviour known to be correct.

| baseline | launch file | nodes |
|---|---|---|
| `fl_odom` | `pepper_slam/fastlio_odometry.launch.py` | 4 |
| `pl_odom` | `pepper_slam/pointlio_odometry.launch.py` | 4 |
| `fl_lc` | `fastlio_lc_pgo/fastlio_lc_l2.launch.py` | 7 |
| `pl_lc` | `fastlio_lc_pgo/pointlio_lc_l2.launch.py` | 7 |
| `fl_loc` | `lio_localization/fastlio_localization_l2.launch.py` | 6 |
| `pl_loc` | `lio_localization/pointlio_localization_l2.launch.py` | 6 |

Counts are with default arguments and **conditions evaluated**. An earlier
version of `resolve_launch.py` did not check `IfCondition`, and so reported both
of `pepper_sensor_tf`'s mutually exclusive publishers as running -- inflating
every count by one or two. If a baseline node count ever looks too high, suspect
that first.

## `test_shared_nav2_params.py`

Guards the three nav2 param sections meant to be identical across the four mode
files. See its docstring for why checking beat extracting a shared base.
