最直接用法如下。

**MuJoCo 里跑 distill**
```bash
cd /home/infinite/RoboMimic_Deploy
.venv/bin/python deploy_mujoco/deploy_mujoco_no_joystick.py --start-policy table_tennis_distill
```

带 debug 看 obs/action：
```bash
.venv/bin/python deploy_mujoco/deploy_mujoco_no_joystick.py --start-policy table_tennis_distill --debug-frames 20
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

如果用通用 FSM 真机脚本：

```bash
.venv/bin/python deploy_real/deploy_real.py --start-policy table_tennis_distill
```

但我建议先用 `deploy_real_table_tennis.py`，它有 `--dry-run`、`--debug`、`--ramp-time`、`--max-delta`，更适合真机试。