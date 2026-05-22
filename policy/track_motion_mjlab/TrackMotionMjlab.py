"""TrackMotion variant trained inside the mjlab stack.

Same obs/action interface as ``TrackMotionMovableBase`` — only the FSM identity
and the yaml/onnx assets it loads from differ. Everything else (joint order,
obs construction, ONNX inference, PD output) is inherited unchanged.
"""

from common.path_config import PROJECT_ROOT  # noqa: F401  (kept so import order matches sibling policies)
from common.utils import FSMStateName

from policy.track_motion_movable_base.TrackMotionMovableBase import TrackMotionMovableBase


class TrackMotionMjlab(TrackMotionMovableBase):
    DEFAULT_STATE_NAME = FSMStateName.SKILL_TRACK_MOTION_MJLAB
    DEFAULT_STATE_NAME_STR = "skill_track_motion_mjlab"
    DEFAULT_CONFIG_FILENAME = "TrackMotionMjlab.yaml"
    LOG_PREFIX = "TrackMotionMjlab"
