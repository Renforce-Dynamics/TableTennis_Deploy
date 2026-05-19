from common.path_config import PROJECT_ROOT

import numpy as np
from common.utils import FSMCommand


class StateAndCmd:
    def __init__(self, num_joints):
        # robot state
        self.num_joints = num_joints
        self.q = np.zeros(num_joints, dtype=np.float32)
        self.dq = np.zeros(num_joints, dtype=np.float32)
        self.ddq = np.zeros(num_joints, dtype=np.float32)
        self.tau_est = np.zeros(num_joints, dtype=np.float32)
        self.base_pos = np.zeros(3, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.ball_pos = np.zeros(3, dtype=np.float32)
        self.ball_vel = np.zeros(3, dtype=np.float32)
        self.gravity_ori = np.array([0., 0., 1.], dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)

        # Upper-planner command bridge for command-conditioned track-motion / landing policies.
        # These fields are intentionally optional: when left as None, the track-motion policy
        # falls back to its internal command sampler.
        self.base_pos_target = None              # np.ndarray shape=(2,), world-frame target base x/y.
        self.rel_racket_target_pos_w = None      # np.ndarray shape=(3,), target racket position relative to base, world axes.
        self.racket_target_vel_w = None          # np.ndarray shape=(3,), target racket velocity in world axes.
        self.racket_target_time = None           # np.ndarray shape=(1,), seconds until planned hit.

        # Optional planner diagnostics. They are not consumed by the policy directly.
        self.predicted_hit_ball_pos_w = None
        self.predicted_hit_ball_vel_w = None
        self.target_landing_pos_w = None
        self.desired_ball_dir_w = None
        self.is_forehand = True
        self.planner_valid = False
        # joy cmd
        self.vel_cmd = np.zeros(3)
        self.skill_cmd = FSMCommand.INVALID
        # skill change cmd
        # self.skill_set = FSMCommand.SKILL_1

class PolicyOutput:
    def __init__(self, num_joints):
        # actions
        self.actions = np.zeros(num_joints, dtype=np.float32)
        self.kps = np.zeros(num_joints, dtype=np.float32)
        self.kds = np.zeros(num_joints, dtype=np.float32)
        self.tau_limit = np.zeros(num_joints, dtype=np.float32)
        