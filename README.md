<div align="center">
  <h1 align="center">RoboMimic Deploy</h1>
  <p align="center">
    <span> 🌎English </span> | <a href="README_zh.md"> 🇨🇳中文 </a>
  </p>
</div>

<p align="center">
  <strong>​RoboMimic Deploy​​ is a multi-policy robot deployment framework based on a state-switching mechanism. Currently, the included policies are designed for the ​​Unitree G1 robot (29-DoF)​​.</strong> 
</p>

## Preface

- **​This deployment framework is only applicable to G1 robots with a 3-DOF waist. If a waist fixing bracket is installed, it must be unlocked according to the official tutorial before this framework can be used normally.​​**

- **It is recommended to remove the hands, as dance movements may cause interference.​**
  
- **When deploying real robots, if something goes wrong, it's probably the policy's fault—not your hardware. Don't waste time second-guessing your robot's physical setup.**

- **[video instruction](https://www.bilibili.com/video/BV1VTKHzSE6C/?vd_source=713b35f59bdf42930757aea07a44e7cb#reply114743994027967)**

## Installation and Configuration

## 1. Create a Virtual Environment

It is recommended to run training or deployment programs in a virtual environment. We suggest using Conda to create one.

### 1.1 Create a New Environment

Use the following command to create a virtual environment:
```bash
conda create -n robomimic python=3.8
```

### 1.2 Activate the Virtual Environment

```bash
conda activate robomimic
```

---

## 2. Install Dependencies

### 2.1 Install PyTorch
PyTorch is a neural network computation framework used for model training and inference. Install it with the following command:
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2.2 Install RoboMimic_Deploy

#### 2.2.1 Download
Clone the repository via git:

```bash
git clone https://github.com/ccrpRepo/RoboMimic_Deploy.git
```

#### 2.2.2 Install Components

Navigate to the directory and install:
```bash
cd RoboMimic_Deploy
pip install numpy==1.20.0
pip install onnx onnxruntime
```

#### 2.2.3 Install unitree_sdk2_python

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```
---
## Running the Code

## 1. Run Mujoco Simulation
```bash
python deploy_mujoco/deploy_mujoco.py
```

## 2. Policy Descriptions
| Mode Name          | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| **PassiveMode**    | Damping protection mode                                                     |
| **FixedPose**      | Position control reset to default joint values                              |
| **LocoMode**       | Stable walking control mode                                                 |
| **Dance**          | Charleston dance routine                                                    |
| **KungFu**         | Martial arts movement                                                       |
| **KungFu2**        | Failed martial arts training                                                |
| **Kick**           | Bad mimic policy                                                            |
| **TableTennis**    | Table tennis playing policy with ball tracking                              |
| **SkillCast**      | Lower body + waist stabilization with upper limbs positioned to specific joint angles (typically executed before Mimic strategy) |
| **SkillCooldown**  | Lower body + waist continuous balancing with upper limbs reset to default angles (typically executed after Mimic strategy) |


---
## 3. Joystick Control Reference

### Basic Controls
| Button Combination | Action                          | Description                           |
|-------------------|---------------------------------|---------------------------------------|
| **Select**        | Emergency Stop                  | Exit program immediately              |
| **L3**            | Enter PassiveMode               | Damping protection mode               |
| **Start**         | Enter FixedPose                 | Position control reset                |
| **R1 + A**        | Enter LocoMode                  | Stable walking mode                   |

### Skill Triggers (from LocoMode)
| Button Combination | Skill                          | Status                                |
|-------------------|---------------------------------|---------------------------------------|
| **R1 + X**        | Dance (Charleston)              | ✅ Stable on real robot              |
| **R1 + Y**        | KungFu                          | ⚠️ Simulation only                   |
| **R1 + B**        | Kick                            | ⚠️ Simulation only                   |
| **L1 + Y**        | KungFu2 (Failed)                | ⚠️ Simulation only                   |
| **L1 + B**        | **Table Tennis**                | 🎾 **New! Requires ball tracking**  |

---
## 4. Operation Instructions in Simulation
1. Connect an Xbox controller.
2. Run the simulation program:
```bash
python deploy_mujoco/deploy_mujoco.py
```
3. Press the ​​Start​​ button to enter position control mode.
4. Hold ​​R1 + A​​ to enter ​​LocoMode​​, then press BACKSPACE in the simulation to make the robot stand. Afterward, use the joystick to control walking.
5. Hold ​​R1 + X​​ to enter ​​Dance​​ mode—the robot will perform the Charleston. In this mode:
    - Press ​​Select​​ at any time to switch to damping protection mode.
    - Hold ​​R1 + A​​ to return to walking mode (not recommended).
    - Press ​​Start​​ to return to position control mode.

6. The terminal will display a progress bar for the dance. After completion, press ​​R1 + A​​ to return to normal walking mode.
7. In ​​LocoMode​​, pressing ​​R1 + Y​​ triggers a Martial arts movement —​ ​use only in simulation​​.
8. In ​​LocoMode​​, pressing ​​L1 + Y​​ triggers a Martial arts movement(Failed) —​ ​use only in simulation​​.
9. In ​​LocoMode​​, pressing ​​R1 + B​​ triggers a Kick movement(Failed) —​ ​use only in simulation​​.
10. In ​​LocoMode​​, pressing ​​L1 + B​​ triggers **Table Tennis** mode — the robot will track and play table tennis.
    - Ball position is automatically tracked from simulation
    - Press ​​R1 + A​​ to return to walking mode (transitions through SkillCooldown)
    - Press ​​Start​​ to return to position control mode
    - Press ​​L3​​ or ​​Select​​ to enter damping protection mode
---
## 5. Real Robot Operation Instructions

1. Power on the robot and suspend it (e.g., with a harness). and then hold L2+R2

2. Run the deploy_real program:
```bash
python deploy_real/deploy_real.py
```
3. Press the ​​Start​​ button to enter position control mode.
4. Subsequent operations are the same as in simulation.
5. For Table Tennis mode on real robot, press ​​L1 + B​​ from LocoMode.

### Table Tennis Specific Deployment

For dedicated table tennis deployment:
```bash
python deploy_real/deploy_real_table_tennis.py
```

#### Remote Controller Mapping (deploy_real_table_tennis.py)
| Button Combination | Action                          | Description                           |
|-------------------|---------------------------------|---------------------------------------|
| **F1**            | Enter PassiveMode               | Damping protection mode               |
| **Start**         | Enter FixedPose                 | Position control reset                |
| **B + L1**        | Enter Table Tennis Mode         | Activate table tennis policy          |
| **Select**        | Exit Program                    | Terminate and switch to damping       |

#### Important Notes
- **Starting State**: Robot starts in **PassiveMode** (safe damping state)
- **Recommended Workflow**:
  1. Start the program (robot in PassiveMode)
  2. Press **Start** to enter FixedPose (position reset)
  3. Press **B + L1** to activate Table Tennis mode
  4. Press **F1** to return to PassiveMode if needed
  5. Press **Select** to exit the program

**Note:** The current implementation uses **fallback ball position** values (`[3.5, -0.2, 1.0]`). For production deployment, integrate a perception system to update `state_cmd.ball_pos` in real-time.

---
## 6. Important Notes
### 1. Framework Compatibility Notice
The current framework does not natively support deployment on G1 robots equipped with Orin NX platforms. Preliminary analysis suggests compatibility issues with the `unitree_python_sdk` on Orin systems. For onboard Orin deployment, we recommend the following alternative solution:

- Replace with [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) (official C++ SDK)
- Implement a dual-node ROS architecture:
  - **C++ Node**: Handles data transmission between robot and controller
  - **Python Node**: Dedicated to policy inference

### 2. Mimic Policy Reliability Warning
The Mimic policy does not guarantee 100% success rate, particularly on slippery/sandy surfaces. In case of robot instability:
- Press `F1` to activate **PassiveMode** (damping protection)
- Press `Select` to immediately terminate the control program

### 3. Charleston Dance (R1+X) - Stable Policy Notes
Currently the only verified stable policy on physical robots:

⚠️ **Important Precautions**:
- **Palm Removal Recommended**: The original training didn't account for palm collisions (author's G1 lacked palms)
- **Initial/Final Stabilization**: Brief manual stabilization may be required when starting/ending the dance
- **Post-Dance Transition**: While switching to **Locomotion/PositionControl/PassiveMode** is possible, we recommend:
  - First transition to **PositionControl** or **PassiveMode**
  - Provide manual stabilization during transition

### 4. Table Tennis Policy Notes
The Table Tennis policy requires ball position tracking:
- **Simulation**: Ball position is automatically extracted from the scene
- **Real Robot**: Currently uses a **static fallback position**
  - Default ball position: `[3.5, -0.2, 1.0]` (3.5m in front, 0.2m to the left, 1.0m height)
  - **TODO**: Integrate vision-based ball tracking system for production use
- **Transitions**: When exiting table tennis mode, the robot transitions through `SkillCooldown` for stability

### 5. Other Movement Advisories
All other movements are currently **not recommended** for physical robot deployment.

### 6. Strong Recommendation
**Always** master operations in simulation before attempting physical robot deployment.
