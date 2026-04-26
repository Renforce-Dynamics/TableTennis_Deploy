# Table Tennis Deploy

## 仿真测试

conda activate robomimic

python deploy_mujoco/deploy_mujoco_no_joystick.py --start-policy table_tennis 

如需查看输出 --debug-frames (num)


已改好：`deploy_real_table_tennis.py` 现在不再启动后直接进乒乓任务，而是先进入 `passive_mode`，流程和 `deploy_real.py` 对齐。

使用方式：

```bash
uv run --group real deploy_real/deploy_real_table_tennis.py --policy table_tennis_distill
```

遥控流程：

```text
启动后：PASSIVE
START：进入 POS_RESET / fixed_pose
A + R1：进入 LOCO
B + R1：从 LOCO 切到乒乓任务
F1：切回 PASSIVE
SELECT：退出程序
```

`--policy` 只决定最后 `B+R1` 进哪个乒乓策略：

```bash
--policy table_tennis
--policy table_tennis_distill
```

我也加了保护：如果你没先进 `LOCO` 就按 `B+R1`，不会直接切任务，会提示：

```text
Enter loco first, then press B+R1 to start table tennis.
```

已验证：

```bash
uv run --group real python -m py_compile deploy_real/deploy_real_table_tennis.py
uv run --group real deploy_real/deploy_real_table_tennis.py --help
```

都通过。