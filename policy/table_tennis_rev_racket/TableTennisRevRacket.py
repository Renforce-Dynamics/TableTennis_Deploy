import os

from policy.table_tennis.TableTennis import TableTennis


class TableTennisRevRacket(TableTennis):
    policy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    config_filename = "TableTennisRevRacket.yaml"
    fsm_state_name = "table_tennis_rev_racket"
    policy_name_str = "skill_table_tennis_rev_racket"
    include_base_lin_vel = False
    prime_history_on_first_obs = True
