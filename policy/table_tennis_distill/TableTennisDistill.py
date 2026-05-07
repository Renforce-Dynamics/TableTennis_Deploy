import os

from policy.table_tennis.TableTennis import TableTennis


class TableTennisDistill(TableTennis):
    policy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    config_filename = "TableTennisDistill.yaml"
    fsm_state_name = "table_tennis_distill"
    policy_name_str = "skill_table_tennis_distill"
    include_base_lin_vel = False
