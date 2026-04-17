from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMCommand, progress_bar
import onnx
import onnxruntime
import torch
import os


class TableTennis(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_TABLE_TENNIS
        self.name_str = "skill_table_tennis"
        self.counter_step = 0
        self.ref_motion_phase = 0.0
        self.mj_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self.train_joint_names = [
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "waist_yaw_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "waist_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "waist_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_yaw_joint",
        ]
        self.mj_to_train = np.array(
            [self.mj_joint_names.index(name) for name in self.train_joint_names], dtype=np.int32
        )
        self.train_to_mj = np.array(
            [self.train_joint_names.index(name) for name in self.mj_joint_names], dtype=np.int32
        )

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "TableTennis.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
            self.onnx_data_path = self.onnx_path + ".data"
            # The local YAML is kept in Mujoco actuator order. Reorder once into
            # the training order expected by the policy.
            self.kps = np.array(config["kps"], dtype=np.float32)[self.mj_to_train]
            self.kds = np.array(config["kds"], dtype=np.float32)[self.mj_to_train]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)[self.mj_to_train]
            self.tau_limit = np.array(config["tau_limit"], dtype=np.float32)[self.mj_to_train]
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.obs_dim = config["obs_dim"]
            self.history_length = config["history_length"]
            self.motion_length = config["motion_length"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.lin_vel_scale = config.get("lin_vel_scale", 1.0)
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = np.array(config["action_scale"], dtype=np.float32)[self.mj_to_train]
            self.use_external_data = bool(config.get("use_external_data", True))

            self.obs = np.zeros(self.num_obs, dtype=np.float32)
            self.action = np.zeros(self.num_actions, dtype=np.float32)
            self.term_dims = {
                "base_lin_vel": 3,
                "base_ang_vel": 3,
                "projected_gravity": 3,
                "base_pos": 3,
                "base_quat": 4,
                "joint_pos": self.num_actions,
                "joint_vel": self.num_actions,
                "actions": self.num_actions,
                "ball_pos": 3,
            }
            self.term_order = [
                "base_lin_vel",
                "base_ang_vel",
                "projected_gravity",
                "base_pos",
                "base_quat",
                "joint_pos",
                "joint_vel",
                "actions",
                "ball_pos",
            ]
            self.term_history = {
                name: np.zeros((self.history_length, dim), dtype=np.float32)
                for name, dim in self.term_dims.items()
            }
            self.latest_obs_terms = {
                name: np.zeros(dim, dtype=np.float32) for name, dim in self.term_dims.items()
            }

        self._validate_config()
        self.policy_available = False
        self.init_error = ""
        try:
            self._load_policy()
            self.policy_available = True
            print("TableTennis policy initializing ...")
        except Exception as exc:
            self.init_error = str(exc)
            print(f"TableTennis policy unavailable: {self.init_error}")

    def _validate_config(self):
        if self.default_angles.shape[0] != self.num_actions:
            raise ValueError("TableTennis default_angles size must match num_actions.")
        if self.kps.shape[0] != self.num_actions or self.kds.shape[0] != self.num_actions:
            raise ValueError("TableTennis kps/kds size must match num_actions.")
        if self.action_scale.shape[0] not in (1, self.num_actions):
            raise ValueError("TableTennis action_scale must have length 1 or num_actions.")
        if self.obs_dim * self.history_length != self.num_obs:
            raise ValueError("TableTennis obs_dim * history_length must equal num_obs.")

    def _load_policy(self):
        if self.use_external_data and not os.path.exists(self.onnx_data_path):
            raise FileNotFoundError(
                f"Missing external ONNX data file: {self.onnx_data_path}. "
                "This model was exported with external tensor data, so both files are required."
            )

        # Load the graph once so config mismatches fail early.
        self.onnx_model = onnx.load(self.onnx_path, load_external_data=self.use_external_data)
        self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
        inputs = self.ort_session.get_inputs()
        outputs = self.ort_session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("TableTennis expects a single-input single-output ONNX policy.")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        if input_shape[-1] != self.num_obs:
            raise ValueError(
                f"TableTennis num_obs mismatch: config={self.num_obs}, onnx={input_shape[-1]}"
            )
        if output_shape[-1] != self.num_actions:
            raise ValueError(
                f"TableTennis num_actions mismatch: config={self.num_actions}, onnx={output_shape[-1]}"
            )

        for _ in range(5):
            obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).cpu().numpy().astype(np.float32)
            self.ort_session.run(None, {self.input_name: obs_tensor})[0]

    def _get_optional_state(self, name, size):
        value = getattr(self.state_cmd, name, None)
        if value is None:
            return np.zeros(size, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] != size:
            return np.zeros(size, dtype=np.float32)
        return value

    def _build_obs(self):
        qj = self.state_cmd.q.reshape(-1)[self.mj_to_train]
        dqj = self.state_cmd.dq.reshape(-1)[self.mj_to_train]
        gravity_orientation = self.state_cmd.gravity_ori.reshape(-1)
        ang_vel = self.state_cmd.ang_vel.reshape(-1) * self.ang_vel_scale
        base_lin_vel = self._get_optional_state("base_lin_vel", 3) * self.lin_vel_scale
        base_pos = self._get_optional_state("base_pos", 3)
        base_quat = self._get_optional_state("base_quat", 4)
        ball_pos = self._get_optional_state("ball_pos", 3)

        qj_obs = (qj - self.default_angles) * self.dof_pos_scale
        dqj_obs = dqj * self.dof_vel_scale
        obs_terms = {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": ang_vel,
            "projected_gravity": gravity_orientation,
            "base_pos": base_pos,
            "base_quat": base_quat,
            "joint_pos": qj_obs,
            "joint_vel": dqj_obs,
            "actions": self.action,
            "ball_pos": ball_pos,
        }
        self.latest_obs_terms = {name: value.copy() for name, value in obs_terms.items()}

        obs_chunks = []
        for name in self.term_order:
            value = np.asarray(obs_terms[name], dtype=np.float32).reshape(-1)
            if value.shape[0] != self.term_dims[name]:
                raise ValueError(
                    f"TableTennis term '{name}' mismatch: got {value.shape[0]}, expected {self.term_dims[name]}"
                )
            self.term_history[name] = np.roll(self.term_history[name], shift=-1, axis=0)
            self.term_history[name][-1] = value
            obs_chunks.append(self.term_history[name].reshape(-1))

        obs = np.concatenate(obs_chunks, axis=-1, dtype=np.float32)
        if obs.shape[0] != self.num_obs:
            raise ValueError(f"TableTennis obs mismatch: got {obs.shape[0]}, expected {self.num_obs}")
        return obs

    def enter(self):
        self.counter_step = 0
        self.ref_motion_phase = 0.0
        self.obs.fill(0.0)
        self.action.fill(0.0)
        for history in self.term_history.values():
            history.fill(0.0)
        for value in self.latest_obs_terms.values():
            value.fill(0.0)

    def run(self):
        if not self.policy_available:
            self.policy_output.actions = self.default_angles.astype(np.float32)
            self.policy_output.kps = self.kps.copy()
            self.policy_output.kds = self.kds.copy()
            return

        self.obs = self._build_obs()
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).cpu().numpy().astype(np.float32)
        self.action = np.squeeze(self.ort_session.run(None, {self.input_name: obs_tensor})[0]).astype(np.float32)

        target_dof_pos_train = self.action * self.action_scale + self.default_angles
        target_dof_pos = target_dof_pos_train[self.train_to_mj]
        kps = self.kps[self.train_to_mj]
        kds = self.kds[self.train_to_mj]

        self.policy_output.actions = target_dof_pos.astype(np.float32)
        self.policy_output.kps = kps.astype(np.float32)
        self.policy_output.kds = kds.astype(np.float32)

        self.counter_step += 1
        motion_time = self.counter_step * self.control_dt
        if self.motion_length > 0:
            self.ref_motion_phase = motion_time / self.motion_length
            print(
                progress_bar(min(motion_time, self.motion_length), self.motion_length),
                end="",
                flush=True,
            )

    def exit(self):
        self.counter_step = 0
        self.ref_motion_phase = 0.0
        self.obs.fill(0.0)
        self.action.fill(0.0)
        for history in self.term_history.values():
            history.fill(0.0)
        for value in self.latest_obs_terms.values():
            value.fill(0.0)
        print()

    def checkChange(self):
        if self.state_cmd.skill_cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        elif self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_TABLE_TENNIS
