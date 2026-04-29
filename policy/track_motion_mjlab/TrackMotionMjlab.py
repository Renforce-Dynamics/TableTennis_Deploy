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
from typing import Tuple


class TrackMotionMjlab(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_TRACK_MOTION_MJLAB
        self.name_str = "skill_track_motion_mjlab"
        self.counter_step = 0
        self.ref_motion_phase = 0.0

        # MuJoCo/runtime joint order
        self.mj_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
            "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
            "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
            "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        # RL training joint order (Isaac Gym / track_motion_mjlab)
        
        #下面的是isaaclab
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
        # #下面的是mjlab
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

        self.mj_to_train = np.array(
            [self.mj_joint_names.index(name) for name in self.train_joint_names], dtype=np.int32
        )
        self.train_to_mj = np.array(
            [self.train_joint_names.index(name) for name in self.mj_joint_names], dtype=np.int32
        )

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "TrackMotionMjlab.yaml")
        config = self._load_config(config_path)

        self.onnx_path = self._resolve_path(current_dir, config.get("onnx_path", "model/policy.onnx"))
        self.onnx_data_path = self.onnx_path + ".data"

        self.num_actions = int(config.get("num_actions", 29))
        self.num_obs = int(config.get("num_obs", 104))
        self.obs_dim = int(config.get("obs_dim", 104))
        self.history_length = int(config.get("history_length", 1))
        self.motion_length = float(config.get("motion_length", 0.0))

        self.ang_vel_scale = float(config.get("ang_vel_scale", 1.0))
        # self.lin_vel_scale = float(config.get("lin_vel_scale", 1.0))
        self.dof_pos_scale = float(config.get("dof_pos_scale", 1.0))
        self.dof_vel_scale = float(config.get("dof_vel_scale", 1.0))
        self.use_external_data = bool(config.get("use_external_data", True))
        self.obs_clip = float(config.get("obs_clip", 100.0))
        self.action_clip = float(config.get("action_clip", 5.0))

        # track_motion_mjlab command defaults
        self.base_target_pos = np.array(config.get("base_target_pos", [0.0, 0.0]), dtype=np.float32)
        self.racket_target_pos_w_default = np.array(
            config.get("racket_target_pos_w", [0.4, -0.45, 0.25]), dtype=np.float32
        )
        self.racket_target_vel_w_default = np.array(
            config.get("racket_target_vel_w", [1.5, 0.0, 0.2]), dtype=np.float32
        )
        self.hit_time_step = int(config.get("hit_time_step", 42))
        self.command_rng = np.random.default_rng(config.get("command_seed", None))
        self.forehand_racket_target_pose_range = self._load_racket_target_pose_range(
            config.get("forehand_racket_target_pose_range", {}),
            self.racket_target_pos_w_default,
            self.racket_target_vel_w_default,
        )
        self.backhand_racket_target_pose_range = self._load_racket_target_pose_range(
            config.get("backhand_racket_target_pose_range", {}),
            self.racket_target_pos_w_default,
            self.racket_target_vel_w_default,
        )
        base_target_pos_range_cfg = config.get("base_target_pos_range", {}) or {}
        if not isinstance(base_target_pos_range_cfg, dict):
            raise ValueError("TrackMotionMjlab base_target_pos_range must be a mapping.")
        self.base_target_pos_y_range = self._range_from_config(
            base_target_pos_range_cfg.get("pos_y", None),
            float(self.base_target_pos[1]),
        )
        motion_steps = int(round(self.motion_length / self.control_dt)) if self.motion_length > 0.0 else 0
        self.command_time_step_total = int(
            config.get(
                "command_time_step_total",
                max(self.hit_time_step + 1, motion_steps if motion_steps > 0 else self.hit_time_step + 1),
            )
        )
        self.command_time_step_total = max(self.command_time_step_total, self.hit_time_step + 1)
        self.command_step = 0
        self.current_is_forehand = True
        self.current_base_target_pos = self.base_target_pos.copy()
        self.current_rel_racket_target_pos_w = self.racket_target_pos_w_default.copy()
        self.current_racket_target_vel_w = self.racket_target_vel_w_default.copy()

        # Controller gains and action transforms (try yaml first, then onnx metadata)
        self.kps = self._array_from_config(config, "kps", self.num_actions)
        self.kds = self._array_from_config(config, "kds", self.num_actions)
        self.default_angles = self._array_from_config(config, "default_angles", self.num_actions)
        self.tau_limit = self._array_from_config(config, "tau_limit", self.num_actions, default_val=200.0)
        self.action_scale = self._array_from_config(config, "action_scale", self.num_actions, default_val=1.0)

        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)

        # Track_motion actor observation structure (104)
        self.term_dims = {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "forward_vec": 2,
            "rel_base_pos_target": 2,
            "rel_racket_target_pos_w": 3,
            "racket_target_time": 1,
            "racket_target_vel_w": 3,
            "joint_pos": self.num_actions,
            "joint_vel": self.num_actions,
            "actions": self.num_actions,
        }
        self.term_order = [
            "base_ang_vel",
            "projected_gravity",
            "forward_vec",
            "rel_base_pos_target",
            "rel_racket_target_pos_w",
            "racket_target_time",
            "racket_target_vel_w",
            "joint_pos",
            "joint_vel",
            "actions",
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
        self.policy_step = 0
        try:
            self._load_policy()
            self.policy_available = True
            print("TrackMotionMjlab policy initializing ...")
        except Exception as exc:
            self.init_error = str(exc)
            print(f"TrackMotionMjlab policy unavailable: {self.init_error}")

    def _load_config(self, config_path: str):
        if not os.path.exists(config_path):
            print(f"TrackMotionMjlab config not found: {config_path}, fallback to defaults.")
            return {}
        with open(config_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _resolve_path(self, current_dir: str, path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value
        candidate = os.path.join(current_dir, path_value)
        if os.path.exists(candidate):
            return candidate
        return os.path.join(PROJECT_ROOT, path_value)

    def _array_from_config(self, config, key: str, length: int, default_val: float = 0.0) -> np.ndarray:
        if key not in config:
            return np.full(length, default_val, dtype=np.float32)
        arr = np.array(config[key], dtype=np.float32).reshape(-1)
        if arr.shape[0] == 1 and length > 1:
            arr = np.full(length, float(arr[0]), dtype=np.float32)
        if arr.shape[0] != length:
            raise ValueError(f"TrackMotionMjlab {key} size must be 1 or {length}, got {arr.shape[0]}")
        return arr[self.mj_to_train]

    def _range_from_config(self, value, default: float) -> Tuple[float, float]:
        if value is None:
            return float(default), float(default)
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] == 1:
            low = high = float(arr[0])
        elif arr.shape[0] == 2:
            low, high = float(arr[0]), float(arr[1])
        else:
            raise ValueError(f"TrackMotionMjlab range expects 1 or 2 values, got {arr.shape[0]}")
        if low > high:
            low, high = high, low
        return low, high

    def _load_racket_target_pose_range(self, section, fallback_pos: np.ndarray, fallback_vel: np.ndarray):
        section = section or {}
        if not isinstance(section, dict):
            raise ValueError("TrackMotionMjlab racket target pose range config must be a mapping.")
        return {
            "pos_x": self._range_from_config(section.get("pos_x", None), float(fallback_pos[0])),
            "pos_y": self._range_from_config(section.get("pos_y", None), float(fallback_pos[1])),
            "pos_z": self._range_from_config(section.get("pos_z", None), float(fallback_pos[2])),
            "vel_x": self._range_from_config(section.get("vel_x", None), float(fallback_vel[0])),
            "vel_y": self._range_from_config(section.get("vel_y", None), float(fallback_vel[1])),
            "vel_z": self._range_from_config(section.get("vel_z", None), float(fallback_vel[2])),
        }

    def _sample_range_value(self, value_range: Tuple[float, float]) -> float:
        low, high = value_range
        if low == high:
            return low
        return float(self.command_rng.uniform(low, high))

    def _resample_internal_motion_command(self):
        self.current_is_forehand = bool(self.command_rng.uniform(0.0, 1.0) < 0.5)
        target_pose_range = (
            self.forehand_racket_target_pose_range
            if self.current_is_forehand
            else self.backhand_racket_target_pose_range
        )
        self.current_rel_racket_target_pos_w = np.array(
            [
                self._sample_range_value(target_pose_range["pos_x"]),
                self._sample_range_value(target_pose_range["pos_y"]),
                self._sample_range_value(target_pose_range["pos_z"]),
            ],
            dtype=np.float32,
        )
        self.current_racket_target_vel_w = np.array(
            [
                self._sample_range_value(target_pose_range["vel_x"]),
                self._sample_range_value(target_pose_range["vel_y"]),
                self._sample_range_value(target_pose_range["vel_z"]),
            ],
            dtype=np.float32,
        )
        self.current_base_target_pos = np.array(
            [
                float(self.base_target_pos[0]),
                self._sample_range_value(self.base_target_pos_y_range),
            ],
            dtype=np.float32,
        )
        self.command_step = 0

    def _maybe_resample_internal_motion_command(self):
        if self.command_step >= self.command_time_step_total:
            self._resample_internal_motion_command()

    def _get_state_if_valid(self, name: str, size: int):
        if not hasattr(self.state_cmd, name):
            return None
        value = getattr(self.state_cmd, name)
        if value is None:
            return None
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] != size:
            return None
        return value

    def _parse_csv_metadata(self, metadata: dict, key: str, cast_type=float):
        if key not in metadata:
            return None
        raw = metadata[key].strip()
        if not raw:
            return None
        if cast_type is str:
            return [x for x in raw.split(",") if x]
        return np.array([cast_type(x) for x in raw.split(",")], dtype=np.float32)

    def _fill_from_onnx_metadata_if_needed(self):
        try:
            model = onnx.load(self.onnx_path, load_external_data=self.use_external_data)
            metadata = {prop.key: prop.value for prop in model.metadata_props}
        except Exception:
            return

        joint_names = self._parse_csv_metadata(metadata, "joint_names", str)
        if joint_names and len(joint_names) == self.num_actions:
            self.train_joint_names = joint_names
            self.mj_to_train = np.array(
                [self.mj_joint_names.index(name) for name in self.train_joint_names], dtype=np.int32
            )
            self.train_to_mj = np.array(
                [self.train_joint_names.index(name) for name in self.mj_joint_names], dtype=np.int32
            )

        if np.allclose(self.default_angles, 0.0):
            meta_val = self._parse_csv_metadata(metadata, "default_joint_pos", float)
            if meta_val is not None and meta_val.shape[0] == self.num_actions:
                self.default_angles = meta_val.astype(np.float32)

        if np.allclose(self.kps, 0.0):
            meta_val = self._parse_csv_metadata(metadata, "joint_stiffness", float)
            if meta_val is not None and meta_val.shape[0] == self.num_actions:
                self.kps = meta_val.astype(np.float32)

        if np.allclose(self.kds, 0.0):
            meta_val = self._parse_csv_metadata(metadata, "joint_damping", float)
            if meta_val is not None and meta_val.shape[0] == self.num_actions:
                self.kds = meta_val.astype(np.float32)

        if np.allclose(self.action_scale, 1.0):
            meta_val = self._parse_csv_metadata(metadata, "action_scale", float)
            if meta_val is not None and meta_val.shape[0] == self.num_actions:
                self.action_scale = meta_val.astype(np.float32)

    def _validate_config(self):
        if self.default_angles.shape[0] != self.num_actions:
            raise ValueError("TrackMotionMjlab default_angles size must match num_actions.")
        if self.kps.shape[0] != self.num_actions or self.kds.shape[0] != self.num_actions:
            raise ValueError("TrackMotionMjlab kps/kds size must match num_actions.")
        if self.action_scale.shape[0] not in (1, self.num_actions):
            raise ValueError("TrackMotionMjlab action_scale must have length 1 or num_actions.")
        if self.obs_dim * self.history_length != self.num_obs:
            raise ValueError("TrackMotionMjlab obs_dim * history_length must equal num_obs.")
        if self.base_target_pos.shape[0] != 2:
            raise ValueError("TrackMotionMjlab base_target_pos must contain 2 values.")
        if self.racket_target_pos_w_default.shape[0] != 3:
            raise ValueError("TrackMotionMjlab racket_target_pos_w must contain 3 values.")
        if self.racket_target_vel_w_default.shape[0] != 3:
            raise ValueError("TrackMotionMjlab racket_target_vel_w must contain 3 values.")
        if self.command_time_step_total <= 0:
            raise ValueError("TrackMotionMjlab command_time_step_total must be positive.")

    def _load_policy(self):
        if self.use_external_data and not os.path.exists(self.onnx_data_path):
            print(f"TrackMotionMjlab external data file not found: {self.onnx_data_path}, continue without it.")

        self._fill_from_onnx_metadata_if_needed()

        self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
        inputs = self.ort_session.get_inputs()
        outputs = self.ort_session.get_outputs()
        if len(outputs) != 1:
            raise ValueError("TrackMotionMjlab expects single output ONNX policy.")

        self.input_names = [inp.name for inp in inputs]
        self.output_name = outputs[0].name

        # Validate obs/action shape.
        obs_input = None
        for inp in inputs:
            if inp.name == "obs":
                obs_input = inp
                break
        if obs_input is None:
            obs_input = inputs[0]
        obs_shape = obs_input.shape
        out_shape = outputs[0].shape

        if isinstance(obs_shape[-1], int) and obs_shape[-1] != self.num_obs:
            raise ValueError(f"TrackMotionMjlab num_obs mismatch: config={self.num_obs}, onnx={obs_shape[-1]}")
        if isinstance(out_shape[-1], int) and out_shape[-1] != self.num_actions:
            raise ValueError(f"TrackMotionMjlab num_actions mismatch: config={self.num_actions}, onnx={out_shape[-1]}")

        # Warm-up
        for _ in range(5):
            obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).cpu().numpy().astype(np.float32)
            ort_inputs = {}
            for name in self.input_names:
                if name == "obs":
                    ort_inputs[name] = obs_tensor
                elif name == "time_step":
                    ort_inputs[name] = np.array([[float(self.policy_step)]], dtype=np.float32)
                else:
                    ort_inputs[name] = obs_tensor
            self.ort_session.run(None, ort_inputs)[0]

    def _get_optional_state(self, name: str, size: int, default=None):
        value = getattr(self.state_cmd, name, default)
        if value is None:
            return np.zeros(size, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] != size:
            return np.zeros(size, dtype=np.float32)
        return value

    def _yaw_forward_vec(self, base_quat: np.ndarray) -> np.ndarray:
        # base_quat expected as [w, x, y, z]
        qw, qx, qy, qz = base_quat
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)

    def _build_obs(self):
        qj = self.state_cmd.q.reshape(-1)[self.mj_to_train]
        dqj = self.state_cmd.dq.reshape(-1)[self.mj_to_train]

        gravity_orientation = self.state_cmd.gravity_ori.reshape(-1)
        ang_vel = self.state_cmd.ang_vel.reshape(-1) * self.ang_vel_scale

        base_pos = self._get_optional_state("base_pos", 3)
        base_quat = self._get_optional_state("base_quat", 4, default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        state_base_target = self._get_state_if_valid("base_pos_target", 2)
        state_rel_racket_target_pos_w = self._get_state_if_valid("rel_racket_target_pos_w", 3)
        state_racket_target_vel_w = self._get_state_if_valid("racket_target_vel_w", 3)
        state_racket_target_time = self._get_state_if_valid("racket_target_time", 1)

        if state_rel_racket_target_pos_w is None or state_racket_target_vel_w is None:
            self._maybe_resample_internal_motion_command()

        base_target = state_base_target if state_base_target is not None else self.current_base_target_pos
        rel_base_pos_target = np.array([base_target[0] - base_pos[0], base_target[1] - base_pos[1]], dtype=np.float32)
        rel_racket_target_pos_w = (
            state_rel_racket_target_pos_w
            if state_rel_racket_target_pos_w is not None
            else self.current_rel_racket_target_pos_w
        )
        racket_target_vel_w = (
            state_racket_target_vel_w if state_racket_target_vel_w is not None else self.current_racket_target_vel_w
        )

        # Keep the same behavior as mjhitter: once command_step passes hit_time_step,
        # observation time resets to max hit horizon instead of clamping at zero.
        if state_racket_target_time is not None:
            racket_target_time = state_racket_target_time
        else:
            remain = (self.hit_time_step - self.command_step) * self.control_dt
            max_time = self.hit_time_step * self.control_dt
            racket_target_time = np.array([remain if remain >= 0.0 else max_time], dtype=np.float32)

        qj_obs = (qj - self.default_angles) * self.dof_pos_scale
        dqj_obs = dqj * self.dof_vel_scale

        obs_terms = {
            "base_ang_vel": ang_vel,
            "projected_gravity": gravity_orientation,
            "forward_vec": self._yaw_forward_vec(base_quat),
            "rel_base_pos_target": rel_base_pos_target,
            "rel_racket_target_pos_w": rel_racket_target_pos_w,
            "racket_target_time": racket_target_time,
            "racket_target_vel_w": racket_target_vel_w,
            "joint_pos": qj_obs,
            "joint_vel": dqj_obs,
            "actions": self.action,
        }
        self.latest_obs_terms = {name: value.copy() for name, value in obs_terms.items()}

        obs_chunks = []
        for name in self.term_order:
            value = np.asarray(obs_terms[name], dtype=np.float32).reshape(-1)
            if value.shape[0] != self.term_dims[name]:
                raise ValueError(
                    f"TrackMotionMjlab obs term '{name}' mismatch: {value.shape[0]} != {self.term_dims[name]}"
                )
            self.term_history[name] = np.roll(self.term_history[name], shift=-1, axis=0)
            self.term_history[name][-1] = value
            obs_chunks.append(self.term_history[name].reshape(-1))

        obs = np.concatenate(obs_chunks, axis=-1, dtype=np.float32)
        if self.obs_clip > 0.0:
            obs = np.clip(obs, -self.obs_clip, self.obs_clip)
        if obs.shape[0] != self.num_obs:
            raise ValueError(f"TrackMotionMjlab obs mismatch: got {obs.shape[0]}, expected {self.num_obs}")
        return obs

    def enter(self):
        self.counter_step = 0
        self.ref_motion_phase = 0.0
        self.policy_step = 0
        self.command_step = 0
        self._resample_internal_motion_command()
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

        ort_inputs = {}
        for name in self.input_names:
            if name == "obs":
                ort_inputs[name] = obs_tensor
            elif name == "time_step":
                ort_inputs[name] = np.array([[float(self.policy_step)]], dtype=np.float32)
            else:
                ort_inputs[name] = obs_tensor

        self.action = np.squeeze(self.ort_session.run(None, ort_inputs)[0]).astype(np.float32)
        if self.action_clip > 0.0:
            self.action = np.clip(self.action, -self.action_clip, self.action_clip)

        target_dof_pos_train = self.action * self.action_scale + self.default_angles

        target_dof_pos = target_dof_pos_train[self.train_to_mj]
        kps = self.kps[self.train_to_mj]
        kds = self.kds[self.train_to_mj]

        self.policy_output.actions = target_dof_pos.astype(np.float32)
        self.policy_output.kps = kps.astype(np.float32)
        self.policy_output.kds = kds.astype(np.float32)

        self.counter_step += 1
        self.policy_step += 1
        self.command_step += 1
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
        self.policy_step = 0
        self.command_step = 0
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
            return FSMStateName.SKILL_TRACK_MOTION_MJLAB
