import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT
from common.ctrlcomp import *
from FSM.FSM import *
from typing import Union
import argparse
import numpy as np
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, init_cmd_hg, MotorMode
from common.rotation_helper import get_gravity_orientation_real
from common.remote_controller import RemoteController, KeyMap
from config import Config


def clamp_targets(target_q, current_q, max_delta):
    return np.clip(target_q, current_q - max_delta, current_q + max_delta)


def get_policy_state(policy_name: str):
    policy_map = {
        "table_tennis": FSMStateName.SKILL_TABLE_TENNIS,
        "table_tennis_distill": FSMStateName.SKILL_TABLE_TENNIS_DISTILL,
    }
    return policy_map[policy_name]


class TableTennisController:
    def __init__(self, config: Config, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.remote_controller = RemoteController()
        self.num_joints = config.num_joints
        self.control_dt = config.control_dt

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)
        self.wait_for_low_state()

        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)

        self.state_cmd = StateAndCmd(self.num_joints)
        self.policy_output = PolicyOutput(self.num_joints)
        self.fsm_controller = FSM(self.state_cmd, self.policy_output)
        self.fsm_controller.cur_policy.enter()
        print("current policy is ", self.fsm_controller.cur_policy.name_str)

        self.start_time = time.time()
        self.running = True
        self.counter_over_time = 0
        self.last_policy_hint_time = 0.0

    def switch_to_policy(self, policy_name: FSMStateName):
        if self.fsm_controller.cur_policy.name == policy_name:
            return
        self.fsm_controller.cur_policy.exit()
        self.fsm_controller.get_next_policy(policy_name)
        self.fsm_controller.cur_policy.enter()
        self.fsm_controller.FSMmode = FSMMode.NORMAL
        self.start_time = time.time()
        print("Switched to ", self.fsm_controller.cur_policy.name_str)

    def low_state_handler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def build_state(self):
        for i in range(self.num_joints):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        gravity_orientation = get_gravity_orientation_real(quat)

        self.state_cmd.q = self.qj.copy()
        self.state_cmd.dq = self.dqj.copy()
        self.state_cmd.gravity_ori = gravity_orientation.copy()
        self.state_cmd.ang_vel = ang_vel.copy()
        self.state_cmd.base_quat = quat.copy()

        # Real robot deployment usually lacks a trusted global root state.
        # Keep these aligned with the Mujoco deployment assumptions for now.
        self.state_cmd.base_pos = np.array([0.0, 0.0, self.args.base_height], dtype=np.float32)
        self.state_cmd.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.state_cmd.ball_pos = np.array(self.args.ball_pos, dtype=np.float32)
        self.state_cmd.vel_cmd[:] = 0.0 # ？

    def handle_remote_commands(self):
        if self.remote_controller.is_button_pressed(KeyMap.F1):
            self.state_cmd.skill_cmd = FSMCommand.PASSIVE
        if self.remote_controller.is_button_pressed(KeyMap.start):
            self.state_cmd.skill_cmd = FSMCommand.POS_RESET
        if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.R1):
            self.state_cmd.skill_cmd = FSMCommand.LOCO
        if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.R1):
            if self.fsm_controller.cur_policy.name == FSMStateName.LOCOMODE:
                self.switch_to_policy(get_policy_state(self.args.policy))
            elif time.time() - self.last_policy_hint_time > 1.0:
                print("Enter loco first, then press B+R1 to start table tennis.")
                self.last_policy_hint_time = time.time()
            self.state_cmd.skill_cmd = FSMCommand.INVALID

    def compute_targets(self):
        self.fsm_controller.run()
        target_q = self.policy_output.actions.copy()

        # Ramp in over the first seconds so we don't snap straight into policy output.
        elapsed = time.time() - self.start_time
        alpha = min(elapsed / self.args.ramp_time, 1.0) if self.args.ramp_time > 0 else 1.0
        target_q = self.qj * (1.0 - alpha) + target_q * alpha
        target_q = clamp_targets(target_q, self.qj, self.args.max_delta)
        return target_q

    def run(self):
        try:
            loop_start_time = time.time()
            self.build_state()
            self.handle_remote_commands()

            target_q = self.compute_targets()
            kps = self.policy_output.kps.copy()
            kds = self.policy_output.kds.copy()

            if self.args.debug:
                policy = self.fsm_controller.cur_policy
                raw_action = getattr(policy, "action", np.zeros(1, dtype=np.float32))
                print(
                    "{} target range [{:.3f}, {:.3f}] raw action [{:.3f}, {:.3f}] ball_pos {}".format(
                        policy.name_str,
                        float(np.min(target_q)),
                        float(np.max(target_q)),
                        float(np.min(raw_action)),
                        float(np.max(raw_action)),
                        self.state_cmd.ball_pos.tolist(),
                    )
                )

            if not self.args.dry_run:
                for i in range(self.num_joints):
                    self.low_cmd.motor_cmd[i].q = target_q[i]
                    self.low_cmd.motor_cmd[i].qd = 0.0
                    self.low_cmd.motor_cmd[i].kp = kps[i]
                    self.low_cmd.motor_cmd[i].kd = kds[i]
                    self.low_cmd.motor_cmd[i].tau = 0.0
                self.send_cmd(self.low_cmd)

            loop_end_time = time.time()
            delta_time = loop_end_time - loop_start_time
            if delta_time < self.control_dt:
                time.sleep(self.control_dt - delta_time)
                self.counter_over_time = 0
            else:
                print("control loop over time.")
                self.counter_over_time += 1
        except ValueError as e:
            print(str(e))


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy table tennis policy on the real robot.")
    parser.add_argument(
        "--policy",
        default="table_tennis",
        choices=["table_tennis", "table_tennis_distill"],
        help="Table tennis policy to deploy. Names match deploy_mujoco_no_joystick.py --start-policy.",
    )
    parser.add_argument(
        "--ball-pos",
        type=float,
        nargs=3,
        default=[3.5, -0.2, 1.0],
        help="Fallback constant ball position when no perception is connected.",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=0.76,
        help="Fallback base height used when no global state estimator is connected.",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=0.12,
        help="Maximum per-joint position delta from current state each control step.",
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=2.0,
        help="Seconds used to blend from current joint positions into policy targets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full state/policy loop without sending motor commands.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print target range and ball position each cycle.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config()
    ChannelFactoryInitialize(0, config.net)

    controller = TableTennisController(config, args)

    try:
        while controller.running:
            controller.run()
            if controller.remote_controller.is_button_pressed(KeyMap.select):
                break
    except KeyboardInterrupt:
        pass

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
