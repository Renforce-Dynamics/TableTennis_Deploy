# RoboMimic Deploy

This branch uses `uv` for the Python environment. Do not use conda for this merged setup.

## Setup

```bash
uv venv --python 3.8
source .venv/bin/activate
uv sync
```

For MuJoCo simulation:

```bash
uv sync --group sim
```

For real robot deployment, install Unitree SDK into the same uv environment:

```bash
uv sync --group real
```

The `unitree-sdk2py` source is configured in `pyproject.toml` as `../unitree_sdk2_python`.

## Important: `deploy_mujoco_track_motion_movable_base.py`

If you are working on the current table-tennis / landing pipeline, this is the
main MuJoCo entry script to use:

```bash
python3 deploy_mujoco/deploy_mujoco_track_motion_movable_base.py \
  --start-policy track_motion_movable_base
```

It supports choosing the initial FSM state through `--start-policy`, for
example:

```bash
python3 deploy_mujoco/deploy_mujoco_track_motion_movable_base.py --start-policy passive
python3 deploy_mujoco/deploy_mujoco_track_motion_movable_base.py --start-policy loco
python3 deploy_mujoco/deploy_mujoco_track_motion_movable_base.py --start-policy track_motion_movable_base
python3 deploy_mujoco/deploy_mujoco_track_motion_movable_base.py --start-policy landing_assist_finetune
```

### How startup and switching work

There are two common ways to use this script:

1. Start directly in the target policy with `--start-policy ...`
2. Start from `passive` or `loco`, then switch at runtime

Recommended runtime flow when you want manual control over which landing policy
to enter:

```text
1. Start with --start-policy passive   or --start-policy loco
2. If needed, press L to enter loco
3. Press a number key to choose the target policy
4. Press T to enter the currently selected policy from loco
5. Press P at any time to return to passive
```

Current number-key mapping inside the MuJoCo window:

| Key | Selected policy |
| --- | --- |
| `1` | `table_tennis` |
| `2` | `table_tennis_distill` |
| `3` | `table_tennis_rev_racket` |
| `4` | `track_motion_movable_base` |
| `5` | `track_motion_mjlab` |
| `6` | `landing_assist_finetune` |

Current control keys:

| Key | Action |
| --- | --- |
| `L` | Switch to `loco` |
| `T` | Enter the currently selected table / landing policy from `loco` |
| `P` | Switch to `passive` |
| `R` | Reset the simulation |

Important behavior details:

- If you launch with `--start-policy track_motion_movable_base` or
  `--start-policy landing_assist_finetune`, the simulation starts directly in
  that policy.
- If you launch with `--start-policy passive`, you must first switch to
  `loco` before using `T` to enter a table / landing policy.
- If you are already inside a table / landing policy, the number keys only
  change the current selection; they do not force an immediate unsafe switch.
- Click the MuJoCo window once before using keyboard shortcuts, otherwise the
  viewer may not receive the key events.

## Mujoco

### Track Motion Landing (Current Recommended Path)

The current table-tennis landing work after PR #1 is documented in
[`README_TRACK_MOTION_LANDING.md`](README_TRACK_MOTION_LANDING.md).

Recommended MuJoCo command:

```bash
uv run python deploy_mujoco/deploy_mujoco_landing.py \
  --start-policy track_motion_movable_base \
  --fixed-initial-ball \
  --ball-pos 3.5 -0.2 1.0 \
  --ball-vel -4.0 0.0 0.0 \
  --debug-every 1
```

This path reuses the verified `track_motion_movable_base` inference stack and injects planner-generated `base_pos_target`, `rel_racket_target_pos_w`, `racket_target_vel_w`, and `racket_target_time` commands. In MuJoCo, the default planner source rolls out the current MuJoCo scene to keep the hit point synchronized with the XML/contact model.

### Table Tennis Without Joystick

Use `deploy_mujoco_no_joystick.py` to test the three table-tennis policies.

Start with the default table-tennis policy:

```bash
python3 deploy_mujoco/deploy_mujoco_no_joystick.py --start-policy loco --table-policy table_tennis
```

Start with the distilled student policy:

```bash
python3 deploy_mujoco/deploy_mujoco_no_joystick.py --start-policy loco --table-policy table_tennis_distill
```

Start with the rev-racket policy:

```bash
python3 deploy_mujoco/deploy_mujoco_no_joystick.py --start-policy loco --table-policy table_tennis_rev_racket
```

Runtime keyboard controls in the MuJoCo window:

| Key | Action |
| --- | --- |
| `L` | Enter `loco` |
| `1` | Select `table_tennis` |
| `2` | Select `table_tennis_distill` |
| `3` | Select `table_tennis_rev_racket` |
| `T` | Enter the selected table-tennis policy from `loco` |
| `P` | Enter `passive` |
| `F` | Enter `fixedpose` |
| `R` | Reset simulation |

Typical flow:

```text
1. Start deploy_mujoco_no_joystick.py.
2. Press L to enter loco.
3. Press 1, 2, or 3 to select the table-tennis model.
4. Press T to switch from loco to the selected table-tennis policy.
```

The three table-tennis policies share the same deployment logic:

| Policy name | Model | Notes |
| --- | --- | --- |
| `table_tennis` | `policy/table_tennis/model/policy.onnx` with external `.data` | Original end-to-end table-tennis model, includes `base_lin_vel` in the observation |
| `table_tennis_distill` | `policy/table_tennis_distill/model/student_policy.onnx` | Distilled student model, no `base_lin_vel` |
| `table_tennis_rev_racket` | `policy/table_tennis_rev_racket/model/policy.onnx` | Rev-racket model, no `base_lin_vel` |

### Track Motion Keyboard Deploy

Use `deploy_mujoco_keyboard.py` to test `loco` and track-motion policies:

```bash
python3 deploy_mujoco/deploy_mujoco_keyboard.py --start-policy loco
python3 deploy_mujoco/deploy_mujoco_keyboard.py --start-policy track_motion_mjlab
python3 deploy_mujoco/deploy_mujoco_keyboard.py --start-policy track_motion_movable_base
python3 deploy_mujoco/deploy_mujoco_keyboard.py --start-policy track_motion_isaaclab
```

Click the MuJoCo window once before pressing keys so it has keyboard focus.

Runtime keyboard controls:

| Key | Action |
| --- | --- |
| `L` | Enter `loco` |
| `N` | Enter `track_motion_mjlab`, the static/manual base-target mode |
| `V` | Enter `track_motion_movable_base`, the movable/random base-target mode |
| `T` | Enter `table_tennis` |
| `R` | Reset simulation |
| `H` | Print keyboard help |
| `0` | Reset manual velocity and base-target values |
| Arrow keys in `loco` | Adjust `vx` and `vy` |
| Arrow keys in `track_motion_mjlab` | Adjust `base_pos_target.x` and `base_pos_target.y` |

`track_motion_isaaclab` is available as a `--start-policy` value. The current keyboard mapping does not bind a runtime switch key for it, so start the script with `--start-policy track_motion_isaaclab` when you want to test that model.

Track-motion behavior:

| Policy name | How to enter | Behavior |
| --- | --- | --- |
| `loco` | `--start-policy loco` or press `L` | Arrow keys command walking velocity |
| `track_motion_mjlab` | `--start-policy track_motion_mjlab` or press `N` | Manual static base target; arrow keys modify `state_cmd.base_pos_target` |
| `track_motion_movable_base` | `--start-policy track_motion_movable_base` or press `V` | Uses the policy's internal movable/random base target |
| `track_motion_isaaclab` | `--start-policy track_motion_isaaclab` | IsaacLab-exported track-motion model |

Quick track-motion test flow:

```text
1. Start deploy_mujoco_keyboard.py with --start-policy loco.
2. Press L, then use arrow keys and confirm terminal prints loco vxy.
3. Press N, then use arrow keys and confirm terminal prints static_base_target.
4. Press V and confirm the movable-base policy runs with internal target logic.
5. Press 0 to reset manual command values, or R to reset the simulation.
```

### Registered Policy Names

Supported policy names are generated from `common/policy_registry.py`. The merged branch includes:

- `table_tennis`
- `table_tennis_distill`
- `table_tennis_rev_racket`
- `track_motion_isaaclab`
- `track_motion_mjlab`
- `track_motion_movable_base`
- `landing_assist_finetune`

## Architecture

`uv-version-2-test` remains the base architecture. `FSM/FSM.py` keeps a dynamic policy registry instead of hard-coded policy branches; new tasks are added through `common/policy_registry.py`.
