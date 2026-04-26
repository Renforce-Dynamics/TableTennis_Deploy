import os

from common.utils import FSMStateName
from policy.table_tennis.TableTennis import TableTennis


class TableTennisDistill(TableTennis):
    policy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "table_tennis_distill.py"))
    config_filename = "TableTennisDistill.yaml"
    fsm_state_name = FSMStateName.SKILL_TABLE_TENNIS_DISTILL
    policy_name_str = "skill_table_tennis_distill"
    include_base_lin_vel = False
