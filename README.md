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

Table tennis without joystick:

```bash
python3 deploy_mujoco/deploy_mujoco_no_joystick.py --start-policy passive
```

Track motion keyboard deploy:

```bash
python3 deploy_mujoco/deploy_mujoco_keyboard.py --start-policy loco
python3 deploy_mujoco/deploy_mujoco_keyboard.py --start-policy track_motion_mjlab
python3 deploy_mujoco/deploy_mujoco_keyboard.py --start-policy track_motion_movable_base
```

Supported policy names are generated from `common/policy_registry.py`, including:

- `table_tennis`
- `table_tennis_distill`
- `table_tennis_rev_racket`
- `track_motion_isaaclab`
- `track_motion_mjlab`
- `track_motion_movable_base`

## Architecture

`uv-version-2-test` remains the base architecture. `FSM/FSM.py` keeps a dynamic policy registry instead of hard-coded policy branches; new tasks are added through `common/policy_registry.py`.
