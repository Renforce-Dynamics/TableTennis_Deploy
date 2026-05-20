# Track Motion Landing Current Guide

本文记录从 `Merge pull request #1 from Renforce-Dynamics/track_motion_deploy` 之后，到当前版本新增和保留的 track-motion landing 功能。当前主线是：

```text
track_motion_movable_base policy
  + external landing planner command injection
  + MuJoCo table-tennis scene
```

不要把 landing 主线切到 `track_motion_mjlab`，也不要默认切到 `g1_train_racket.xml`。`g1_train_racket.xml` 保留为训练模型参考；当前稳定 MuJoCo 场景仍使用 `g1_29dof_rev_1_0_racket.xml` 和 `right_racket_collision`。

## What Changed

After PR #1, the deployment branch added:

- `common/ctrlcomp.py`: planner command bridge fields on `StateAndCmd`, including `base_pos_target`, `rel_racket_target_pos_w`, `racket_target_vel_w`, `racket_target_time`, ball state, and planner diagnostics.
- `planner/`: lightweight ball trajectory and racket-target planners.
  - `hope_model_based_planner.py`: direct ball-pos/ball-vel model-based planner.
  - `hope_planner.py`: HOPE-style estimator + trajectory predictor + racket target planner.
- `common/landing_command.py`: converts ball/base state into track-motion command fields.
- `deploy_mujoco/deploy_mujoco_landing.py`: MuJoCo landing sim entry with ball, table, planner, and policy command injection.
- `deploy_mujoco/config/landing_planner.yaml`: planner, table, target landing, and command timing config.
- `deploy_real/ball_observer.py` and `deploy_real/deploy_real_landing.py`: sim2real scaffold with constant or UDP JSON ball observations.

## Current Recommended MuJoCo Command

Use the verified movable-base policy path:

```bash
uv run python deploy_mujoco/deploy_mujoco_landing.py \
  --start-policy track_motion_movable_base \
  --fixed-initial-ball \
  --ball-pos 3.5 -0.2 1.0 \
  --ball-vel -4.0 0.0 0.0 \
  --debug-every 1
```

The default `--planner-source mujoco` uses a short MuJoCo rollout to predict when the ball crosses `x_hit`. This keeps the simulated planner aligned with the actual XML/contact model.

Optional planner source:

```bash
uv run python deploy_mujoco/deploy_mujoco_landing.py \
  --start-policy track_motion_movable_base \
  --fixed-initial-ball \
  --planner-source model \
  --debug-every 1
```

Use `--planner-source model` only when you specifically want to test the portable model-based planner. For sim2sim debugging, `mujoco` is usually the better default.

## Runtime Keys

In the MuJoCo window:

| Key | Action |
| --- | --- |
| `r` | Reset simulation |
| `l` | Switch to `loco` |
| `t` | Switch to the `--start-policy` landing policy |
| `v` | Switch to `track_motion_movable_base` |
| `n` | Switch to `track_motion_mjlab` |
| `p` | Switch to `passive` |

Click the MuJoCo window once before using keys.

## Expected Debug Pattern

For the fixed ball command above, debug output should follow this rough sequence:

```text
too_early     t_hit=0.840  idle command
ok            t_hit decreases toward 0.020, hit ~= [0.4, -0.2, 0.857]
passed_hit    idle command after the ball crosses x_hit
```

If the ball already crossed `x_hit`, the planner writes an idle command instead of clearing command fields. This prevents `TrackMotionMovableBase` from falling back to its internal random command and continuing a large random swing after the ball is gone.

## Scene XML That Should Be Active

`g1_description/g1_table_tennis_scene.xml` should include:

```xml
<include file="g1_29dof_rev_1_0_racket.xml"/>
```

and the ball-racket contact pair should use:

```xml
geom2="right_racket_collision"
```

Do not use `g1_train_racket.xml` as the default deployment scene unless you are deliberately testing train-XML parity. If you do, contact names and dynamics parameters must be reviewed together.

## Policy Notes

- Recommended landing policy: `track_motion_movable_base`.
- `track_motion_mjlab` remains available for comparison, but it is not the default landing path.
- `track_motion_isaaclab` remains available as a separate policy for IsaacLab-exported models.
- Do not add `--force-default-pose` for the current recommended landing test. The landing script defaults to the same reset style as the previously working track-motion deployment path.

## Sim2Real Entry

The real landing scaffold is:

```bash
uv run --group real python deploy_real/deploy_real_landing.py \
  --policy track_motion_movable_base \
  --dry-run \
  --debug
```

Real ball sources:

```bash
# Constant debug ball
--ball-source constant --ball-pos 3.5 -0.2 1.0 --ball-vel -4.0 0.0 0.0

# UDP JSON stream
--ball-source udp --udp-host 0.0.0.0 --udp-port 15050
```

Expected UDP packet examples:

```json
{"pos": [3.5, -0.2, 1.0], "vel": [-4.0, 0.0, 0.0], "t": 123.4}
```

or:

```json
{"ball_pos_w": [3.5, -0.2, 1.0], "ball_vel_w": [-4.0, 0.0, 0.0], "timestamp": 123.4}
```

The real entry does not use MuJoCo rollout prediction. It uses `LandingCommandGenerator` with the configured model-based or HOPE planner.

## Quick Sanity Checks

```bash
python -m py_compile \
  common/landing_command.py \
  deploy_mujoco/deploy_mujoco_landing.py \
  deploy_real/deploy_real_landing.py \
  planner/common.py \
  planner/hope_model_based_planner.py
```

If behavior suddenly becomes poor, first check:

1. `g1_table_tennis_scene.xml` still includes `g1_29dof_rev_1_0_racket.xml`.
2. Ball-racket contact uses `right_racket_collision`.
3. `deploy_mujoco_landing.py` still defaults to `track_motion_movable_base`.
4. `landing_planner.yaml` has `activate_before_hit_s: 0.84` and `min_time_to_hit_s: 0.0`.
