from dataclasses import dataclass, field
from importlib import import_module

from common.path_config import PROJECT_ROOT
from common.utils import FSMStateName


@dataclass(frozen=True)
class ExtraPolicySpec:
    key: str
    state: object
    class_path: str
    attr_name: str
    init_kwargs: dict = field(default_factory=dict)


BASE_POLICY_STATES = {
    "passive": FSMStateName.PASSIVE,
    "fixedpose": FSMStateName.FIXEDPOSE,
    "loco": FSMStateName.LOCOMODE,
    "dance": FSMStateName.SKILL_Dance,
    "kungfu": FSMStateName.SKILL_KungFu,
    "kick": FSMStateName.SKILL_KICK,
    "kungfu2": FSMStateName.SKILL_KungFu2,
    "beyond_mimic": FSMStateName.SKILL_BEYOND_MIMIC,
}


EXTRA_POLICY_SPECS = (
    ExtraPolicySpec(
        "table_tennis",
        FSMStateName.SKILL_TABLE_TENNIS,
        "policy.table_tennis.TableTennis",
        "TableTennis",
    ),
    ExtraPolicySpec(
        "table_tennis_distill",
        FSMStateName.SKILL_TABLE_TENNIS_DISTILL,
        "policy.table_tennis_distill.TableTennisDistill",
        "TableTennisDistill",
    ),
    ExtraPolicySpec(
        "table_tennis_rev_racket",
        FSMStateName.SKILL_TABLE_TENNIS_REV_RACKET,
        "policy.table_tennis_rev_racket.TableTennisRevRacket",
        "TableTennisRevRacket",
    ),
    ExtraPolicySpec(
        "track_motion_mjlab",
        FSMStateName.SKILL_TRACK_MOTION_MJLAB,
        "policy.track_motion_mjlab.TrackMotionMjlab",
        "TrackMotionMjlab",
    ),
    ExtraPolicySpec(
        "track_motion_isaaclab",
        FSMStateName.SKILL_TRACK_MOTION_ISAACLAB,
        "policy.track_motion_isaaclab.TrackMotionIsaaclab",
        "TrackMotionIsaaclab",
    ),
    ExtraPolicySpec(
        "track_motion_movable_base",
        FSMStateName.SKILL_TRACK_MOTION_MOVABLE_BASE,
        "policy.track_motion_movable_base.TrackMotionMovableBase",
        "TrackMotionMovableBase",
    ),
    ExtraPolicySpec(
        "landing_assist_finetune",
        FSMStateName.SKILL_LANDING_ASSIST_FINETUNE,
        "policy.track_motion_movable_base.TrackMotionMovableBase",
        "TrackMotionMovableBase",
        init_kwargs={
            "state_name": FSMStateName.SKILL_LANDING_ASSIST_FINETUNE,
            "state_name_str": "skill_landing_assist_finetune",
            "config_path": f"{PROJECT_ROOT}/policy/landing_assist_finetune/config/LandingAssistFinetune.yaml",
        },
    ),
)


def get_policy_state(policy_name: str):
    if policy_name in BASE_POLICY_STATES:
        return BASE_POLICY_STATES[policy_name]
    for spec in EXTRA_POLICY_SPECS:
        if spec.key == policy_name:
            return spec.state
    raise KeyError(f"Unknown policy: {policy_name}")


def get_policy_choices(include_base=True, include_extra=True):
    choices = []
    if include_base:
        choices.extend(BASE_POLICY_STATES.keys())
    if include_extra:
        choices.extend(spec.key for spec in EXTRA_POLICY_SPECS)
    return choices


def load_extra_policy_class(spec: ExtraPolicySpec):
    module = import_module(spec.class_path)
    return getattr(module, spec.attr_name)


# Policies that consume the planner→policy LandingCommand bridge in the
# table-tennis sim2sim entrypoints (tennis_keyboard / _joystick).
LANDING_POLICY_STATES = (
    FSMStateName.SKILL_TRACK_MOTION_MOVABLE_BASE,
    FSMStateName.SKILL_LANDING_ASSIST_FINETUNE,
)


def is_landing_policy(policy_name) -> bool:
    """Accept either a string key (e.g. 'track_motion_mjlab') or an FSMStateName."""
    if isinstance(policy_name, str):
        try:
            policy_name = get_policy_state(policy_name)
        except KeyError:
            return False
    return policy_name in LANDING_POLICY_STATES
