最直接用法如下。

## 当前推荐：Track Motion Landing

详细说明见 [`README_TRACK_MOTION_LANDING.md`](README_TRACK_MOTION_LANDING.md)。

```bash
uv run python deploy_mujoco/deploy_mujoco_landing.py \
  --start-policy track_motion_movable_base \
  --fixed-initial-ball \
  --ball-pos 3.5 -0.2 1.0 \
  --ball-vel -4.0 0.0 0.0 \
  --debug-every 1
```

这条命令走当前主线：`track_motion_movable_base` + landing planner command injection。默认 `--planner-source mujoco`，会用 MuJoCo 自己向前滚动预测击球点，避免 planner 和 XML/contact 动力学不同步。

打开 viewer 后会显示 planner 可视化：紫色球是规划击球点，黄色箭头是期望出球参考方向，青色箭头是规划球拍速度方向。`--debug-every 1` 的终端输出里也会打印 `hit=`、`dir=`、`forehand=`，方便对照具体坐标。

注意：

- 当前场景应使用 `g1_29dof_rev_1_0_racket.xml`。
- 球拍 contact 应使用 `right_racket_collision`。
- 不建议默认加 `--force-default-pose`。
- 不建议把 landing 主线切到 `track_motion_mjlab`。

真机 scaffold / dry-run：

```bash
uv run --group real python deploy_real/deploy_real_landing.py \
  --policy track_motion_movable_base \
  --dry-run \
  --debug
```

**MuJoCo 里跑 distill**
```bash
cd /home/infinite/RoboMimic_Deploy
.venv/bin/python deploy_mujoco/deploy_mujoco_no_joystick.py --table-policy table_tennis_distill
```

带 debug 看 obs/action：
```bash
.venv/bin/python deploy_mujoco/deploy_mujoco_no_joystick.py --table-policy table_tennis_distill --debug-frames 20
```

键盘流程：
```text
启动后：PASSIVE
l：进入 LOCO
t：从 LOCO 切到乒乓任务
p：切回 PASSIVE
f：进入 fixed_pose
r：重置仿真
1/2/3：选择 table_tennis / table_tennis_distill / table_tennis_rev_racket
```

**真机上跑 distill，推荐用专用脚本**
```bash
cd /home/infinite/RoboMimic_Deploy
.venv/bin/python deploy_real/deploy_real_table_tennis.py --policy table_tennis_distill
```

先 dry-run，不发电机命令：
```bash
.venv/bin/python deploy_real/deploy_real_table_tennis.py --policy table_tennis_distill --dry-run --debug
```

如果你要手动指定球的位置：
```bash
.venv/bin/python deploy_real/deploy_real_table_tennis.py \
  --policy table_tennis_distill \
  --ball-pos 3.5 -0.2 1.0
```

真机安全参数也可以调：

```bash
.venv/bin/python deploy_real/deploy_real_table_tennis.py \
  --policy table_tennis_distill \
  --ramp-time 2.0 \
  --max-delta 0.12
```

`--ramp-time` 是启动时从当前关节姿态渐进到策略目标的时间。  
`--max-delta` 是每个控制周期目标关节最多离当前关节多远。越小越稳，越大越接近 MuJoCo，但风险也更高。

真机乒乓建议用 `deploy_real_table_tennis.py`，它有 `--dry-run`、`--debug`、`--ramp-time`、`--max-delta`，更适合真机试。
