import os

from common.utils import FSMStateName
from policy.table_tennis.TableTennis import TableTennis


class TableTennisRevRacket(TableTennis):
    policy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    config_filename = "TableTennisRevRacket.yaml"
    fsm_state_name = FSMStateName.SKILL_TABLE_TENNIS_REV_RACKET
    policy_name_str = "skill_table_tennis_rev_racket"
    include_base_lin_vel = False
    prime_history_on_first_obs = True
