"""TrackMotion variant trained inside the IsaacLab stack.

Differs from the mjlab/movable_base siblings only in the training joint
permutation (IsaacLab interleaves left/right legs and the waist) and the
yaml/onnx assets. Obs construction and ONNX inference are inherited.
"""

from common.path_config import PROJECT_ROOT  # noqa: F401  (kept so import order matches sibling policies)
from common.utils import FSMStateName

from policy.track_motion_movable_base.TrackMotionMovableBase import TrackMotionMovableBase


class TrackMotionIsaaclab(TrackMotionMovableBase):
    # IsaacLab joint order interleaves left/right legs and the waist.
    TRAIN_JOINT_NAMES = [
        "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
        "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
        "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
        "left_knee_joint", "right_knee_joint",
        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        "left_shoulder_roll_joint", "right_shoulder_roll_joint",
        "left_ankle_roll_joint", "right_ankle_roll_joint",
        "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
        "left_elbow_joint", "right_elbow_joint",
        "left_wrist_roll_joint", "right_wrist_roll_joint",
        "left_wrist_pitch_joint", "right_wrist_pitch_joint",
        "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    ]
    DEFAULT_STATE_NAME = FSMStateName.SKILL_TRACK_MOTION_ISAACLAB
    DEFAULT_STATE_NAME_STR = "skill_track_motion_isaaclab"
    DEFAULT_CONFIG_FILENAME = "TrackMotionIsaaclab.yaml"
    LOG_PREFIX = "TrackMotionIsaaclab"
