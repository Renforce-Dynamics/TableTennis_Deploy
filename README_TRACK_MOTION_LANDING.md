# Track Motion Landing 当前说明

这份文档说明当前仓库里和 `track motion + landing planner + MuJoCo` 相关的主线代码、启动方式和调试方法。

当前推荐主线是：

```text
command-conditioned track motion policy
  + external landing planner command injection
  + MuJoCo table-tennis scene
```

## 代码结构

当前有两个常用 landing 相关 policy 入口：

- `track_motion_movable_base`
- `landing_assist_finetune`

这两个入口现在**共用同一个 Python 状态实现类**：

- `policy.track_motion_movable_base.TrackMotionMovableBase`

它们的区别不在 deploy 控制逻辑，而在：

- 状态名不同
- 加载的配置文件不同
- 加载的 ONNX 模型不同

也就是说，现在仓库里只保留了一份 `movable-base landing` 的 deploy 实现，避免重复代码。

## 当前主线新增内容

相对早期纯 `track motion` 分支，当前 landing 主线主要新增了这些模块：

- `common/ctrlcomp.py`
  增加了 planner command bridge 字段，例如：
  `base_pos_target`、`rel_racket_target_pos_w`、`racket_target_vel_w`、`racket_target_time`、球状态与 planner 调试信息。
- `planner/`
  轻量级上层规划器实现：
  - `hope_model_based_planner.py`
  - `hope_planner.py`
- `common/landing_command.py`
  把球状态、机器人 base 状态转换成底层策略吃的 command 字段。
- `sim2sim/tennis_keyboard.py`
  landing 版 MuJoCo 入口，包含球、桌、网、planner、command 注入与调试显示。
- `sim2sim/tennis_keyboard.py`
  适合切换多个 table-tennis / landing policy 的 MuJoCo 入口。
- `configs/planner/landing_planner.yaml`
  上层 planner、球桌、落点目标、时间窗口、命令分布等参数。

## 推荐启动方式

### 1. 主 landing 调试入口

推荐直接用这个脚本：

```bash
python3 sim2sim/tennis_keyboard.py \
  --start-policy track_motion_movable_base
```

这个脚本支持通过 `--start-policy` 指定启动状态，例如：

```bash
python3 sim2sim/tennis_keyboard.py --start-policy passive
python3 sim2sim/tennis_keyboard.py --start-policy loco
python3 sim2sim/tennis_keyboard.py --start-policy track_motion_movable_base
python3 sim2sim/tennis_keyboard.py --start-policy landing_assist_finetune
```

当前默认：

- `--planner-source model`
- 使用 `configs/sim/g1_track_motion_movable_base.yaml`
- 使用 `configs/planner/landing_planner.yaml`

### 2. 直接跑 landing 单入口

如果你只想单独跑 landing 主线，也可以：

```bash
python3 sim2sim/tennis_keyboard.py \
  --start-policy track_motion_movable_base
```

### 3. 复现 LandingAssistFinetune 导出 ONNX

如果你要复现：

```bash
uv run python -m mjlab.scripts.play Mjlab-Table-Tennis-Unitree-G1-LandingAssistFinetune \
  --checkpoint-file good_chek/model_29999.pt \
  --viewer native --num-envs 1
```

对应当前仓库里的入口是：

```bash
python3 sim2sim/tennis_keyboard.py \
  --start-policy landing_assist_finetune
```

它加载的是：

- 配置：`policy/landing_assist_finetune/config/LandingAssistFinetune.yaml`
- 模型：`/home/xzx/Embodied_AI/TsingHua/mjhitter/good_chek/exported_uv_model_29999/policy.onnx`

## MuJoCo 运行时按键

### `deploy_tennis_keyboard.py`

在 MuJoCo 窗口中：

- `r`：重置仿真
- `l`：切回 `loco`
- `1`：选择 `table_tennis`
- `2`：选择 `table_tennis_distill`
- `3`：选择 `table_tennis_rev_racket`
- `4`：选择 `track_motion_movable_base`
- `5`：选择 `track_motion_mjlab`
- `6`：选择 `landing_assist_finetune`
- `t`：从 `loco` 进入当前选中的 table / landing policy
- `p`：切到 `passive`

说明：

- 可以直接通过 `--start-policy` 从 `passive`、`loco` 或某个具体 table / landing policy 启动。
- 如果从 `passive` 或其他非 `loco` 状态启动，按 `l` 可以先切回 `loco`。
- 回到 `loco` 之后，可以按数字键选择目标 policy，再按 `t` 从 `loco` 进入当前选中的状态。
- 数字键在 `loco` 状态下会直接进入对应 policy。
- 如果当前已经在某个 table / landing policy 中，数字键只会更新“下一次 `t` 进入时的目标 policy”。

### `deploy_tennis_keyboard.py`

在 MuJoCo 窗口中：

- `r`：重置
- `l`：切到 `loco`
- `t`：切回 `--start-policy` 指定的 landing policy
- `v`：切到 `track_motion_movable_base`
- `n`：切到 `track_motion_mjlab`
- `p`：切到 `passive`

## 当前可视化调试内容

当前 MuJoCo 里已经支持实时显示：

- 预测击球点
- 期望球拍位置
- 当前球拍位置
- 期望拍面方向
- 当前拍面方向
- 期望球拍速度方向
- 当前球拍速度方向

这部分主要在：

- `sim2sim/tennis_keyboard.py`
- `sim2sim/tennis_keyboard.py`

## 直给调试模式

如果你想强制测试“固定击球 `y`，并让 rot/vel 朝正前方”，可以直接加调试参数：

```bash
python3 sim2sim/tennis_keyboard.py \
  --start-policy track_motion_movable_base \
  --debug-fixed-hit-y -0.4 \
  --debug-front-speed 1.5 \
  --debug-front-pitch-deg 0.0
```

可选：

```bash
--debug-fixed-hit-x 0.4
--debug-fixed-hit-z 0.25
```

这会直接绕开正常 planner 混合过程，用固定的前向击球命令测试底层策略跟踪效果。

## 当前 landing planner 的关键参数

主配置文件：

- `configs/planner/landing_planner.yaml`

几个最常用的参数：

- `planner.x_hit`
  击球平面 `x`
- `table.x_min / table.x_max`
  球桌 `x` 范围
- `table.net_x`
  球网 `x`
- `command.forehand_y_range / backhand_y_range`
  正手、反手接球区间
- `command.pos_blend_* / vel_blend_*`
  planner 和原始 command 的融合强度
- `command.front_facing_lateral_scale`
  横向挥拍压缩比例
- `command.debug_fixed_front_strike`
  固定前向调试模式

## 当前机器人与击球平面参数

### deploy 当前使用

- 机器人世界初始 `x/y`
  默认沿用 XML `qpos0`
- XML 根位置
  `g1_description/g1_track_motion_movable_base.xml`
- planner 击球平面
  `configs/planner/landing_planner.yaml` 中的 `planner.x_hit`

当前配置里：

- base 所在世界 `x` 近似在 `0`
- `x_hit = 0.40`
- `table_x_min = 0.63`
- `net_x = 2.0`

### 和 mjhitter 的 LandingAssistFinetune 对应关系

当前 deploy 对齐的是：

- base 初始高度：`0.76`
- `x_hit = 0.40`
- `table_x_min = 0.63`
- `table_x_max = 3.37`
- `table_z_surface = 0.76`
- `net_x = 2.0`

## 更严格的重发球逻辑

当前比“球落地才重发”更严格：

- 击球后球到达球网附近就重发
- 球在桌面上持续滚动 / 停留一段时间就重发
- 超出 demo 区域仍然作为兜底重发条件

实现位置：

- `sim2sim/tennis_keyboard.py`
  `BallRespawnTracker`

## 已知注意事项

- 当前推荐 landing 主线不要默认切回 `track_motion_mjlab`。
- 当前推荐场景是：
  `g1_description/g1_track_motion_movable_base_scene.xml`
  配套 `g1_track_motion_movable_base.xml`
- 如果你切换模型、XML、碰撞几何或 planner 参数，要一起检查：
  - 拍面朝向
  - 击球平面 `x_hit`
  - base 初始位置
  - 球拍碰撞几何名
  - planner 显示箭头是否和实际拍面/拍速一致

## 快速自检

可以先做语法检查：

```bash
python3 -m py_compile \
  common/landing_command.py \
  common/policy_registry.py \
  sim2sim/tennis_keyboard.py \
  sim2sim/tennis_keyboard.py \
  policy/track_motion_movable_base/TrackMotionMovableBase.py
```

如果效果突然变差，优先检查：

1. `landing_planner.yaml` 的 `x_hit`、正反手区间、blend 参数是否被改过
2. 当前启动的是哪个 policy：`track_motion_movable_base` 还是 `landing_assist_finetune`
3. MuJoCo 场景 XML 是否仍然是 `g1_track_motion_movable_base_scene.xml`
4. 当前显示的期望拍面/当前拍面、期望拍速/当前拍速是否已经明显不一致
