# README

## 🛑 Track Motion 盲测排障说明 (Sim2Real 对照测试)

本项目新增了 **Blind 测试（盲测）** 专用的部署脚本。

在这种模式下（针对 `track_motion_mjlab` 策略），**代码强制屏蔽了外界反馈，将传递给网络的坐标数据强制固定**（`base_pos=[0.0, 0.0, 0.76]`, `base_lin_vel=[0.0, 0.0, 0.0]`）。此时神经网络仅依赖高质量的本体感受（关节电机位置/速度、IMU姿态/角速度）与网络内部生成的击球目标运行。两边获得的观测状态在逻辑上达到 100% 相同。

### 1. 仿真盲测 (MuJoCo)

在仿真中模拟真机完全无航位推算的表现。

```bash
python3 deploy_mujoco/deploy_mujoco_track_blind.py --start-policy track_motion_mjlab
```

*说明：在 MuJoCo 里面，当切入 `track_motion_mjlab` 后，传给控制器的 `base_pos` 和 `base_lin_vel` 会被强行截断和覆写。可以通过键盘调整目标，对照机器人表现。*

### 2. 真机盲测 (Real Hardware)

在真机上上机执行相同的截断测试。

```bash
python3 deploy_real/deploy_real_track_blind.py
```

*说明：脚本默认进入 `loco` 模式，请在准备好后按手柄 `X + L1` 切换到 `track_motion_mjlab` (静态击球) 模式。此时 Odom 数据会被彻底弃用，终端会循环打印 `(BLIND TEST: pos/vel hardcoded)`。*

**排查结论提示**：如果在仿真中（`deploy_mujoco_track_blind.py`）机器人不发散、不乱走，而真机（`deploy_real_track_blind.py`）表现却漂移摔倒，即可**完全排除观测传感器不对齐的干扰**，100% 确认问题来源于 **纯粹的物理动力学特征 Gap**（如：现实中机器人的运动延迟、地面摩擦系数、关节的 PD 增益镇不住现实的重量分配等）。

---

## 常规部署与使用（带 Odom / 完整环境反馈）

### 1. 激活 conda 环境

bash

运行

```
conda activate robomimic
```

2. 运行部署脚本

bash

运行

```
python3 deploy_mujoco/deploy_mujoco_no_joystick.py --start-policy track_motion_mjlab
```

## 修改内容

1. 新增`track_motion`状态
2. 修改对应 yaml 配置文件与 Python 代码
3. 修改 xml 文件，机器人形态统一为 IsaacLab 版本（移除手 + 球拍组合形态）

## IsaacLab / MJLab 导出 ONNX 使用说明

### 一、IsaacLab 配置（当前默认）

#### 1. 代码文件：`policy/track_motion_isaaclab/TrackMotion.py`

* 保留**IsaacLab** 关节名称配置
* 注释**MJLab** 关节名称配置

python

运行

```
# 下面的是isaaclab
self.train_joint_names = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# # 下面的是mjlab
# self.train_joint_names = [
#     "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
#     "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
#     "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
#     "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
#     "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
#     "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
#     "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
#     "right_wrist_yaw_joint",
# ]
```

#### 2. 配置文件：`policy/track_motion_mjlab/config/TrackMotionMjlab.yaml`

* 保留：`onnx_path: "policy/track_motion/model/policy_isaaclab.onnx"`
* 注释：`# onnx_path: "policy/track_motion/model/policy_mjlab.onnx"`
* 配置：`use_external_data: false`
* 注释：`# use_external_data: True`

---

### 二、MJLab 配置（与 IsaacLab 反向操作）

#### 目前mjlab的policy还在训练当中，应该22号能好，先用isaaclab的policy

#### 1. 代码文件：`policy/track_motion/TrackMotion.py`

* 注释**IsaacLab** 关节名称配置
* 保留**MJLab** 关节名称配置

#### 2. 配置文件：`policy/track_motion/config/TrackMotion.yaml`

* 注释：`# onnx_path: "policy/track_motion/model/policy_isaaclab.onnx"`
* 保留：`onnx_path: "policy/track_motion/model/policy_mjlab.onnx"`
* 注释：`# use_external_data: false`
* 配置：`use_external_data: True`

### 总结

1. 运行命令：`conda activate robomimic` →`python3 deploy_mujoco/deploy_mujoco_no_joystick.py --start-policy track_motion_mjlab`
2. 核心修改：新增`track_motion`、更新 yaml/xml/ 代码、统一机器人模型
3. ONNX 切换：IsaacLab 用`policy_isaaclab.onnx`，MJLab 用`policy.onnx`，关节名与配置同步反向修改

## deploy_mujoco_keyboard 与 deploy_real_track_motion 使用说明

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
