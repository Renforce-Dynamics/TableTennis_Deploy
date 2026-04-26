# deploy_mujoco_keyboard 与 deploy_real_track_motion 使用说明

本文档包含两部分：

- `python3 deploy_mujoco/deploy_mujoco_keyboard.py`（MuJoCo 键盘控制）
- `python3 deploy_real/deploy_real_track_motion.py`（真机遥控控制）

MuJoCo 部分重点覆盖以下三个任务/状态：

- `loco`
- `track_motion_mjlab`（文档中称为 `Track Motion Static`）
- `track_motion_movable_base`（文档中称为 `Track Motion Movable Base`）

## 1. 启动方式

在仓库根目录执行：

```bash
python3 deploy_mujoco/deploy_mujoco_keyboard.py
```

可选参数：

```bash
python3 deploy_mujoco/deploy_mujoco_keyboard.py --start-policy loco
python3 deploy_mujoco/deploy_mujoco_keyboard.py --start-policy track_motion_mjlab
python3 deploy_mujoco/deploy_mujoco_keyboard.py --start-policy track_motion_movable_base
```

支持的 `--start-policy`：

- `passive`
- `fixedpose`
- `loco`
- `dance`
- `kungfu`
- `kick`
- `kungfu2`
- `beyond_mimic`
- `table_tennis`
- `track_motion_isaaclab`
- `track_motion_mjlab`
- `track_motion_movable_base`

## 2. 状态切换按键（MuJoCo 窗口内）

- `L`: `loco`
- `N`: `track_motion_mjlab`（Track Motion Static）
- `V`: `track_motion_movable_base`（Track Motion Movable Base）
- `R`: reset
- `H`: 打印帮助
- `0`: 重置手动控制量（速度和静态 base target）

说明：键盘事件由 MuJoCo viewer 回调捕获，必须先点击 MuJoCo 窗口，确保焦点在仿真窗口上。

## 3. 三个任务的控制逻辑

### 3.1 `loco`

进入方式：按 `L`。

上下左右控制：

- `↑/↓`: 调整 `vx`（前后速度）
- `←/→`: 调整 `vy`（左右速度）

控制特点：

- 每次按键会在终端打印当前 `loco vxy`。
- 步长：`vx` 每次 `0.05 m/s`，`vy` 每次 `0.05 m/s`。
- 自动限幅（来自 `policy/loco_mode/config/LocoMode.yaml`）：
  - `vx ∈ [-0.4, 0.7]`
  - `vy ∈ [-0.4, 0.4]`

### 3.2 `track_motion_mjlab`（Track Motion Static）

进入方式：按 `N`。

行为定义：

- 进入 `N` 时，`base target` 自动重置为 `(0.0, 0.0)`。
- 该状态下允许手动控制 base target。

上下左右控制：

- `↑/↓`: 调整 `base_pos_target.x`
- `←/→`: 调整 `base_pos_target.y`

控制特点：

- 每次按键会在终端打印当前 `static_base_target=(x,y)`。
- 步长：`x` 每次 `0.02`，`y` 每次 `0.02`。
- 限幅：`x,y ∈ [-0.5, 0.5]`。
- 控制实现是向策略输入 `state_cmd.base_pos_target=[x,y]`。

### 3.3 `track_motion_movable_base`（Track Motion Movable Base）

进入方式：按 `V`。

行为定义：

- 该状态下不注入手动 `base_pos_target`。
- 保持策略内部随机命令（随机移动）逻辑。

上下左右控制：

- 在该状态下，方向键不会改 base target（终端会提示该键在当前策略被忽略）。

## 4. 快速测试流程

```text
1) 启动脚本
2) 按 L，测试 ↑ ↓ ← →，确认终端打印 loco vxy
3) 按 N，测试 ↑ ↓ ← →，确认终端打印 static_base_target(x,y)
4) 按 V，测试 ↑ ↓ ← →，确认终端提示被忽略（随机逻辑继续）
5) 按 0，确认手动控制量回到 0
```

## 5. 常见问题

### Q1: 按键偶尔没反应

- 确认焦点在 MuJoCo 仿真窗口。
- 切换输入法后建议再点击一次仿真窗口。

### Q2: 为什么 `V` 下方向键不控制 base

- 这是当前设计：`V` 用于“随机移动”策略验证，不叠加手动干预。
- 若要手动干预，请切到 `N`。

### Q3: 想改步长/限幅

- 步长在 `deploy_mujoco/deploy_mujoco_keyboard.py`：
  - `loco_step_x`, `loco_step_y`
  - `base_step_x`, `base_step_y`
- `loco` 限幅来自 `policy/loco_mode/config/LocoMode.yaml`。
- `track_motion_static` 的手动 base 限幅在脚本内为 `[-0.5, 0.5]`。

## 6. deploy_real_track_motion 使用说明（真机）

脚本文件：

- `deploy_real/deploy_real_track_motion.py`

### 6.1 启动方式

在仓库根目录执行：

```bash
python3 deploy_real/deploy_real_track_motion.py
```

说明：

- 使用 `deploy_real/config/real.yaml` 中的网络与 DDS topic 配置。
- 启动后默认直接进入 `loco`。

### 6.2 状态切换按键（遥控器）

- `A + R1`: `loco`（默认启动状态）
- `X + L1`: `track_motion_mjlab`（静态击球）
- `Y + L1`: `track_motion_movable_base`（随机 base y 移动击球）
- `START`: `fixedpose`
- `F1`: `passive`
- `SELECT`: 退出

### 6.3 三个任务的控制逻辑（真机）

#### 6.3.1 `loco`

- 左摇杆 `Y`：控制 `x` 向速度（前后）
- 左摇杆 `X`：控制 `y` 向速度（左右）
- 右摇杆 `X`：控制 `yaw` 角速度

#### 6.3.2 `track_motion_mjlab`（静态击球）

- 默认 `base target` 在原地（`x=0, y=0`）。
- 左摇杆 `X` 可手动调 `base_pos_target.y`。
- 手动 `y` 限幅：`[-0.5, 0.5]`。

实现方式：

- 向策略注入 `state_cmd.base_pos_target = [0.0, y]`。

#### 6.3.3 `track_motion_movable_base`（随机移动击球）

- 不注入手动 `base_pos_target`。
- 保持策略内部随机 `y` 方向移动逻辑。

### 6.4 快速测试流程（真机）

```text
1) 启动 deploy_real_track_motion.py，确认终端打印默认进入 LOCO
2) 摇左杆，确认 loco 可控 x/y
3) 按 X+L1 进入静态击球，摇左杆 X，观察终端 target_base_y 变化
4) 按 Y+L1 进入 movable base，观察机器人随机 y 移动击球
5) 按 A+R1 返回 loco；按 SELECT 退出
```
