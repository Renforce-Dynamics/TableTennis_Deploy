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

It is recommended to run training or deployment programs in a virtual environment. We suggest using `uv` to manage the project environment.

### 1.1 Create the Project Environment

Use the following command in the repository root:
```bash
uv venv --python 3.8
```

### 1.2 Activate the Virtual Environment

```bash
source .venv/bin/activate
```

---

## 2. Install Dependencies

### 2.1 Download
Clone the repository via git:

```bash
git clone https://github.com/ccrpRepo/RoboMimic_Deploy.git
```

### 2.2 Install RoboMimic_Deploy Dependencies

Navigate to the directory and install:
```bash
cd RoboMimic_Deploy
uv sync
```

If you want to run Mujoco simulation, install the simulation group as well:

```bash
uv sync --group sim
```

If you need CUDA PyTorch wheels instead of the default PyPI wheels, reinstall the torch packages with the PyTorch index after `uv sync`.
The `uv` setup uses a newer NumPy line than the original README because current `onnxruntime` builds for Python 3.8 require `numpy>=1.21.6`.

### 2.3 Install unitree_sdk2_python

The real robot scripts depend on Unitree's Python SDK, which lives outside this repository. After cloning it, install it into the same `uv` environment:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
uv pip install -e .
```
---
## Running the Code

## 1. Run Mujoco Simulation
```bash
python deploy_mujoco/deploy_mujoco.py
```

## 2. Policy Descriptions
| Mode Name        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **PassiveMode**  | Damping protection mode                                                     |
| **FixedPose**    | Position control reset to default joint values                              |
| **LocoMode**     | Stable walking control mode                                                 |
| **Dance**        | Charleston dance routine                                                    |
| **KungFu**       | Martial arts movement                                                       |
| **KungFu2**      | Failed martial arts training                                     |
| **Kick**         | Bad mimic policy                                     |
| **SkillCast**    | Lower body + waist stabilization with upper limbs positioned to specific joint angles (typically executed before Mimic strategy) |
| **SkillCooldown**| Lower body + waist continuous balancing with upper limbs reset to default angles (typically executed after Mimic strategy) |


---
## 3. Operation Instructions in Simulation
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
9. In ​​LocoMode​​, pressing ​​R1 + B​ triggers a Kick movement(Failed) —​ ​use only in simulation​​.
---
## 4. Real Robot Operation Instructions

1. Power on the robot and suspend it (e.g., with a harness). and then hold L2+R2

2. Run the deploy_real program:
```bash
python deploy_real/deploy_real.py
```
3. Press the ​​Start​​ button to enter position control mode.
4. Subsequent operations are the same as in simulation.

---
## Important Notes
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

### 4. Other Movement Advisories
All other movements are currently **not recommended** for physical robot deployment.

### 5. Strong Recommendation
**Always** master operations in simulation before attempting physical robot deployment.
