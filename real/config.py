import os

import yaml

from common.path_config import PROJECT_ROOT


class Config:
    def __init__(self) -> None:
        real_yaml_path = os.path.join(PROJECT_ROOT, "configs", "real", "real.yaml")
        with open(real_yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.net = config["net"]
            self.num_joints = config["num_joints"]
            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]
            self.control_dt = config["control_dt"]
            self.error_over_time = config["error_over_time"]
