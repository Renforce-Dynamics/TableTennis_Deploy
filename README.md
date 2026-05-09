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

## Mujoco

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

## Architecture

`uv-version-2-test` remains the base architecture. `FSM/FSM.py` keeps a dynamic policy registry instead of hard-coded policy branches; new tasks are added through `common/policy_registry.py`.
