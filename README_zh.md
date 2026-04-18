<div align="center">
  <h1 align="center">RoboMimic Deploy</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

<p align="center">
  🎮🚪 <strong>RoboMimic Deploy 是一个基于状态切换机制的机器人多策略部署框架，目前包含的策略适用于宇树G1机器人(29dof)</strong> 🚪🎮
</p>

## 写在前面

- **本部署框架仅适用于具有三自由度腰部的G1机器人，如果装有腰部固定件的话需要按照官网教程解锁，然后才能正常使用该部署框架。**

- **建议拆下手掌，舞蹈动作会存在干涉**

- **实际机器人部署中出现的问题，十有八九是策略适应性不足所致，大家不必过度怀疑硬件层面的缺陷。**

- **[视频教程](https://www.bilibili.com/video/BV1VTKHzSE6C/?vd_source=713b35f59bdf42930757aea07a44e7cb#reply114743994027967)**

## 安装配置

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 `uv` 管理项目环境。

### 1.1 创建项目环境

在仓库根目录运行：

```bash
uv venv --python 3.8
```

### 1.2 激活虚拟环境

```bash
source .venv/bin/activate
```

---

## 2. 安装依赖

### 2.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/ccrpRepo/RoboMimic_Deploy.git
```

### 2.2 安装 RoboMimic_Deploy 依赖

进入目录并安装：

```bash
cd RoboMimic_Deploy
uv sync
```

如果要运行 Mujoco 仿真，再安装仿真分组依赖：

```bash
uv sync --group sim
```

如果你需要 CUDA 版 PyTorch，而不是默认从 PyPI 安装的 wheel，可以在 `uv sync` 之后用 PyTorch 官方索引重新安装 torch 相关包。
这里的 `uv` 依赖使用了比原 README 更新的 NumPy 版本，因为当前支持 Python 3.8 的 `onnxruntime` wheel 需要 `numpy>=1.21.6`。

### 2.3 安装 unitree_sdk2_python

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
uv pip install -e .
```
---
## 运行代码

## 1. 运行Mujoco仿真代码
```bash
python deploy_mujoco/deploy_mujoco.py
```
---
## 2. Policy 说明
| 模式名称          | 描述                                                                 |
|------------------|----------------------------------------------------------------------|
| **PassiveMode**  | 阻尼保护模式                                                         |
| **FixedPose**    | 位控恢复至默认关节值                                                 |
| **LocoMode**     | 用于稳定行走的控制模式                                               |
| **Dance**        | 查尔斯顿舞蹈                                                         |
| **KungFu**       | 武术动作                                                             |
| **KungFu2**      | 训练失败的武术动作                                                   |
| **Kick**         | 拿来凑数的动作                                                       |
| **SkillCast**    | 下肢+腰部稳定站立，上肢位控至特定关节角，一般在执行Mimic策略前执行   |
| **SkillCooldown**| 下肢+腰部持续平衡，上肢恢复至默认关节角，一般在执行Mimic策略后执行    |

---
## 3. 仿真操作说明

1. 连接Xbox手柄

2. 运行仿真程序：
```bash
python deploy_mujoco/deploy_mujoco.py
```
3. Start键进入位控模式

4. 同时按住R1+A，进入LocoMode，并按下`BACKSPACE`在仿真中使机器人站立，之后能通过摇杆控制机器人行走

5. 同时按住R1+X，进入Dance，机器人开始跳查尔斯顿舞蹈，在该模式下，可以随时按下Select进入阻尼保护模式，也可以按住R1+A恢复行走模式（不推荐），或按Start进入位控模式（不推荐）

6. 终端会显示舞蹈的进度条，结束后可按下R1+A恢复至正常行走模式

7. 在LocoMode模式下，按R1+Y让机器人表演武术动作，**只推荐在仿真中使用**

8. 在LocoMode模式下，按L1+Y让机器人表演训练失败的武术动作，**只推荐在仿真中使用**

9. 在LocoMode模式下，按R1+B让机器人表演踢腿动作，**只推荐在仿真中使用**
---
## 4. 真机操作说明
1. 开机后将机器人吊起来，按L2+R2进入调试模式

2. 运行deploy_real程序：
```bash
python deploy_real/deploy_real.py
```
3. Start键进入位控模式

4. 后续操作与仿真中一致

---
## 注意事项
### 1. 框架兼容性说明
当前框架暂不支持在搭载Orin NX平台的G1机器人上直接部署。初步分析可能是由于`unitree_python_sdk`在Orin平台上的兼容性问题。针对机载Orin平台的部署需求，建议采用以下替代方案：

- 使用[unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)替代原Python SDK
- 基于ROS构建双节点架构：
  - **C++节点**：负责机器人与遥控器之间的数据收发
  - **Python节点**：专用于策略推理

### 2. Mimic策略可靠性警告
Mimic策略不保证100%成功率，特别是在湿滑/沙地等复杂地面上。若出现机器人失控情况：
- 按下`F1`键激活**阻尼保护模式**(PassiveMode)
- 按下`Select`键立即终止控制程序

### 3. 查尔斯顿舞蹈(R1+X) - 稳定策略说明
目前唯一在真机上验证稳定的策略：

⚠️ **重要注意事项**：
- **建议拆除手掌**：原始训练未考虑手掌碰撞（作者的G1初始无手掌）
- **起止稳定需求**：舞蹈开始/结束时可能需要短暂人工稳定
- **舞蹈后过渡**：虽然可以切换至**行走模式/位控模式/阻尼模式**，但建议：
  - 先切换至**位控模式**或**阻尼模式**
  - 过渡期间需提供人工稳定

### 4. 其他动作建议
其他所有动作目前均**不建议**在真机上部署。

### 5. 强烈建议
**务必**先在仿真环境中熟练操作，再尝试真机部署。

