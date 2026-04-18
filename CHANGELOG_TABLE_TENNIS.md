# Table Tennis Integration Changelog

## Summary
Integrated the Table Tennis policy into the FSM framework with full joystick control support for both simulation and real robot deployment.

## Changes Made

### 1. Core FSM Integration

#### `common/utils.py`
- ✅ Added `FSMCommand.TABLE_TENNIS = 9` for table tennis command

#### `policy/loco_mode/LocoMode.py`
- ✅ Added table tennis transition in `checkChange()`:
  ```python
  elif(self.state_cmd.skill_cmd == FSMCommand.TABLE_TENNIS):
      return FSMStateName.SKILL_TABLE_TENNIS
  ```

### 2. Simulation Deployment

#### `deploy_mujoco/deploy_mujoco.py`
- ✅ Added `get_ball_pos()` helper function to extract ball position from simulation
- ✅ Added joystick control mapping: **L1 + B** → Table Tennis mode
- ✅ Added ball position tracking: `state_cmd.ball_pos = ball_pos.copy()`
- ✅ Ball position automatically updated every control cycle from simulation scene

**Joystick Mapping:**
```python
if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.L1):
    state_cmd.skill_cmd = FSMCommand.TABLE_TENNIS
```

### 3. Real Robot Deployment

#### `deploy_real/deploy_real.py`
- ✅ Added remote control mapping: **L1 + B** → Table Tennis mode
- ✅ Added fallback state fields required by table tennis policy:
  - `state_cmd.base_pos` - Default: `[0.0, 0.0, 0.76]`
  - `state_cmd.base_lin_vel` - Default: `[0.0, 0.0, 0.0]`
  - `state_cmd.ball_pos` - Default: `[3.5, -0.2, 1.0]` (TODO: integrate perception system)

**Remote Control Mapping:**
```python
if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.L1):
    self.state_cmd.skill_cmd = FSMCommand.TABLE_TENNIS
```

### 4. Documentation Updates

#### `README.md` (English)
- ✅ Added **TableTennis** to Policy Descriptions table
- ✅ Added **Section 3: Joystick Control Reference** with complete control mappings
- ✅ Updated operation instructions with L1+B table tennis trigger
- ✅ Added **Section 5: Table Tennis Specific Deployment** with deployment script parameters
- ✅ Added **Section 6.4: Table Tennis Policy Notes** explaining ball tracking and perception requirements
- ✅ Renumbered sections accordingly

#### `README_zh.md` (Chinese)
- ✅ Added **TableTennis** to 策略说明 table (乒乓球击球策略)
- ✅ Added **Section 3: 手柄控制参考** with complete control mappings
- ✅ Updated operation instructions with L1+B table tennis trigger
- ✅ Added **乒乓球专用部署** section with deployment script parameters
- ✅ Added **Section 6.4: 乒乓球策略说明** explaining ball tracking and perception requirements
- ✅ Renumbered sections accordingly

## Control Reference

### Simulation & Real Robot Controls

| Button Combination | Action                | From State | To State         |
|-------------------|-----------------------|------------|------------------|
| **L1 + B**        | Enter Table Tennis    | LocoMode   | TableTennis      |
| **R1 + A**        | Return to Walk        | TableTennis| SkillCooldown → LocoMode |
| **Start**         | Position Control      | TableTennis| FixedPose        |
| **L3/Select**     | Emergency Stop        | Any        | PassiveMode/Exit |

## Testing

✅ **test_table_tennis.py** - Verified policy loads successfully:
```bash
$ python test_table_tennis.py --steps 3 --seed 42
TableTennis policy initializing ...
policy_available: True
```

✅ **Syntax Check** - All modified files compile successfully:
```bash
$ python -m py_compile deploy_mujoco/deploy_mujoco.py deploy_real/deploy_real.py \
    common/utils.py policy/loco_mode/LocoMode.py
✓ All files compile successfully
```

## Architecture

```
┌─────────────────┐
│   User Input    │
│  (L1 + B Press) │
└────────┬────────┘
         │
         v
┌─────────────────────────┐
│  FSMCommand.TABLE_TENNIS│
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐      ┌──────────────────┐
│ LocoMode.checkChange()  │─────>│ FSMStateName.    │
│                         │      │ SKILL_TABLE_TENNIS│
└─────────────────────────┘      └────────┬─────────┘
                                          │
                                          v
                         ┌────────────────────────────┐
                         │  FSM.get_next_policy()     │
                         │  → self.table_tennis_policy│
                         └────────┬───────────────────┘
                                  │
                                  v
                    ┌─────────────────────────────┐
                    │  TableTennis Policy         │
                    │  - Reads ball_pos           │
                    │  - Generates joint commands │
                    │  - Updates policy_output    │
                    └─────────────────────────────┘
```

## Ball Tracking Status

### Simulation ✅
- Ball position automatically extracted from MuJoCo scene via `data.body("ball").xpos`
- Updates every control cycle
- Fallback to `[3.5, -0.2, 1.0]` if ball not found

### Real Robot ⚠️
- Currently uses **static fallback position**: `[3.5, -0.2, 1.0]`
- **TODO**: Integrate vision-based ball tracking system
- Requires real-time updates to `state_cmd.ball_pos`

## Future Work

1. **Perception Integration**
   - Integrate vision system for real-time ball tracking on real robot
   - Add ball velocity estimation for predictive control
   - Add confidence score handling for tracking failures

2. **Scene Enhancement**
   - Add table tennis table model to `scene.xml`
   - Add net and court markings
   - Improve ball physics properties

3. **Policy Improvements**
   - Add ball trajectory prediction
   - Implement contact event detection
   - Add multi-ball support for training scenarios

## Files Modified

```
common/utils.py                    # Added TABLE_TENNIS command
policy/loco_mode/LocoMode.py       # Added transition logic
deploy_mujoco/deploy_mujoco.py     # Added ball tracking + joystick
deploy_real/deploy_real.py         # Added remote control + fallback state
README.md                          # Complete English documentation
README_zh.md                       # Complete Chinese documentation
```

## Deployment Instructions

### Simulation
```bash
python deploy_mujoco/deploy_mujoco.py
# 1. Press Start → FixedPose
# 2. Press R1+A → LocoMode (press BACKSPACE to stand)
# 3. Press L1+B → TableTennis mode
```

### Real Robot (General)
```bash
python deploy_real/deploy_real.py
# Same controls as simulation
```

### Real Robot (Table Tennis Dedicated)
```bash
python deploy_real/deploy_real_table_tennis.py \
  --ball-pos 3.5 -0.2 1.0 \
  --base-height 0.76 \
  --max-delta 0.12 \
  --ramp-time 2.0 \
  [--dry-run] \
  [--debug]
```

---

**Status:** ✅ Ready for simulation testing | ⚠️ Real robot needs perception integration
