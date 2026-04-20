import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT
from common.ctrlcomp import *
from FSM.FSM import *
from typing import Union
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


class TableTennisController:
    def __init__(self, config: Config):
        self.config = config
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
        self.FSM_controller = FSM(self.state_cmd, self.policy_output)

        self.running = True
        self.counter_over_time = 0

        print("=" * 50)
        print("Table Tennis Controller Initialized")
        print("Controls:")
        print("  F1            - PASSIVE mode")
        print("  START         - FIXEDPOSE mode")
        print("  B + L1        - TABLE TENNIS mode")
        print("  SELECT        - Exit program")
        print("=" * 50)

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

    def run(self):
        try:
            loop_start_time = time.time()

            # Remote control commands
            if self.remote_controller.is_button_pressed(KeyMap.F1):
                self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            if self.remote_controller.is_button_pressed(KeyMap.start):
                self.state_cmd.skill_cmd = FSMCommand.POS_RESET
            if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.L1):
                self.state_cmd.skill_cmd = FSMCommand.TABLE_TENNIS

            self.state_cmd.vel_cmd[0] = self.remote_controller.ly
            self.state_cmd.vel_cmd[1] = self.remote_controller.lx * -1
            self.state_cmd.vel_cmd[2] = self.remote_controller.rx * -1

            for i in range(self.num_joints):
                self.qj[i] = self.low_state.motor_state[i].q
                self.dqj[i] = self.low_state.motor_state[i].dq

            quat = self.low_state.imu_state.quaternion
            ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
            gravity_orientation = get_gravity_orientation_real(quat)

            self.state_cmd.q = self.qj.copy()
            self.state_cmd.dq = self.dqj.copy()
            self.state_cmd.gravity_ori = gravity_orientation.copy()
            self.state_cmd.ang_vel = ang_vel.copy()
            self.state_cmd.base_quat = quat

            # Set default values for table tennis policy
            # TODO: Replace with actual perception system
            self.state_cmd.base_pos = np.array([0.0, 0.0, 0.76], dtype=np.float32)
            self.state_cmd.base_lin_vel = np.zeros(3, dtype=np.float32)
            self.state_cmd.ball_pos = np.array([3.5, -0.2, 1.0], dtype=np.float32)

            self.FSM_controller.run()
            policy_output_action = self.policy_output.actions.copy()
            kps = self.policy_output.kps.copy()
            kds = self.policy_output.kds.copy()

            # Build low cmd
            for i in range(self.num_joints):
                self.low_cmd.motor_cmd[i].q = policy_output_action[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = kps[i]
                self.low_cmd.motor_cmd[i].kd = kds[i]
                self.low_cmd.motor_cmd[i].tau = 0

            # Send the command
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


if __name__ == "__main__":
    config = Config()
    ChannelFactoryInitialize(0, config.net)

    controller = TableTennisController(config)

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.is_button_pressed(KeyMap.select):
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
