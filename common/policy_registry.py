from dataclasses import dataclass
from importlib import import_module

from common.utils import FSMStateName


@dataclass(frozen=True)
class ExtraPolicySpec:
    key: str
    class_path: str
    attr_name: str


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
    ExtraPolicySpec("table_tennis", "policy.table_tennis.TableTennis", "TableTennis"),
    ExtraPolicySpec(
        "table_tennis_distill",
        "policy.table_tennis_distill.TableTennisDistill",
        "TableTennisDistill",
    ),
    ExtraPolicySpec(
        "table_tennis_rev_racket",
        "policy.table_tennis_rev_racket.TableTennisRevRacket",
        "TableTennisRevRacket",
    ),
)


def get_policy_state(policy_name: str):
    if policy_name in BASE_POLICY_STATES:
        return BASE_POLICY_STATES[policy_name]
    if any(spec.key == policy_name for spec in EXTRA_POLICY_SPECS):
        return policy_name
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
