import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.ctrlcomp import StateAndCmd, PolicyOutput
from FSM.FSM import FSM, FSMStateName
from common.utils import FSMCommand
from typing import Union
import numpy as np
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.nav_msg.msg.dds_ import Odometry_
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, init_cmd_hg, MotorMode
from common.rotation_helper import get_gravity_orientation_real
from common.remote_controller import RemoteController, KeyMap
from config import Config


class TrackMotionController:
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
        
        self.odom_subscriber = ChannelSubscriber("rt/lf/odom", Odometry_)
        self.odom_subscriber.Init(self.odom_handler, 10)
        self.received_odom = False

        self.wait_for_low_state()

        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)

        self.state_cmd = StateAndCmd(self.num_joints)
        self.policy_output = PolicyOutput(self.num_joints)
        self.FSM_controller = FSM(self.state_cmd, self.policy_output)

        self.counter_over_time = 0

        # Manual base target control for static track motion mode.
        self.track_static_base_x = 0.0
        self.track_static_base_y = 0.0
        self.track_static_base_y_range = (-0.5, 0.5)
        self.track_static_y_speed = 0.8  # m/s equivalent update rate from joystick axis.
        self.track_static_stick_deadzone = 0.12

        self._last_mode_name = None
        self._last_track_y_print_time = 0.0

        self._set_default_world_state()
        self._force_initial_loco_mode()
        self._print_controls()

    def _print_controls(self):
        print("=" * 64)
        print("Track Motion Real Controller Initialized")
        print("Mode Switching:")
        print("  A + R1        - LOCO mode (default at startup)")
        print("  X + L1        - TRACK MOTION MJLAB (static base, manual target Y)")
        print("  Y + L1        - TRACK MOTION MOVABLE BASE (random base Y)")
        print("  START         - FIXEDPOSE mode")
        print("  F1            - PASSIVE mode")
        print("  SELECT        - Exit program")
        print("Control:")
        print("  Left stick X/Y in LOCO: control base y/x velocity")
        print("  Left stick X in TRACK MOTION MJLAB: control target base y")
        print("=" * 64)

    def _force_initial_loco_mode(self):
        self.state_cmd.skill_cmd = FSMCommand.INVALID
        self.FSM_controller.get_next_policy(FSMStateName.LOCOMODE)
        self.FSM_controller.cur_policy.enter()
        print("[Init] Start in LOCO mode.")

    def low_state_handler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def odom_handler(self, msg: Odometry_):
        self.state_cmd.base_pos[0] = msg.pose.pose.position.x
        self.state_cmd.base_pos[1] = msg.pose.pose.position.y
        self.state_cmd.base_pos[2] = msg.pose.pose.position.z
        
        self.state_cmd.base_lin_vel[0] = msg.twist.twist.linear.x
        self.state_cmd.base_lin_vel[1] = msg.twist.twist.linear.y
        self.state_cmd.base_lin_vel[2] = msg.twist.twist.linear.z
        self.received_odom = True

    def send_cmd(self, cmd: Union[LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def _set_default_world_state(self):
        # If we have not received odom yet, we fall back to a baseline
        if not self.received_odom:
            self.state_cmd.base_pos = np.array([0.0, 0.0, 0.76], dtype=np.float32)
            self.state_cmd.base_lin_vel = np.zeros(3, dtype=np.float32)
        
        # Track motion policies do not use ball_pos, but the base StateAndCmd struct might have it.
        if hasattr(self.state_cmd, "ball_pos"):
            self.state_cmd.ball_pos = np.zeros(3, dtype=np.float32)

    def switch_to_policy(self, policy_name: FSMStateName):
        if self.FSM_controller.cur_policy.name == policy_name:
            return
        self.FSM_controller.cur_policy.exit()
        self.FSM_controller.get_next_policy(policy_name)
        self.FSM_controller.cur_policy.enter()

    def _handle_remote_commands(self):
        if self.remote_controller.is_button_pressed(KeyMap.F1):
            self.switch_to_policy(FSMStateName.PASSIVE)
        if self.remote_controller.is_button_pressed(KeyMap.start):
            self.switch_to_policy(FSMStateName.FIXEDPOSE)
        if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.R1):
            self.switch_to_policy(FSMStateName.LOCOMODE)
        if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.L1):
            self.switch_to_policy(FSMStateName.SKILL_TRACK_MOTION_MJLAB)
        if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.L1):
            self.switch_to_policy(FSMStateName.SKILL_TRACK_MOTION_MOVABLE_BASE)

    def _update_base_target_override(self):
        current_mode = self.FSM_controller.cur_policy.name

        in_or_to_track_static = (
            current_mode == FSMStateName.SKILL_TRACK_MOTION_MJLAB
        )
        in_or_to_track_movable = (
            current_mode == FSMStateName.SKILL_TRACK_MOTION_MOVABLE_BASE
        )

        if in_or_to_track_static:
            stick_x = -float(self.remote_controller.lx)
            if abs(stick_x) < self.track_static_stick_deadzone:
                stick_x = 0.0
            self.track_static_base_y += stick_x * self.track_static_y_speed * self.control_dt
            self.track_static_base_y = float(
                np.clip(self.track_static_base_y, self.track_static_base_y_range[0], self.track_static_base_y_range[1])
            )
            self.state_cmd.base_pos_target = np.array(
                [self.track_static_base_x, self.track_static_base_y], dtype=np.float32
            )
            return

        if in_or_to_track_movable:
            # Let policy internal sampler randomize base target y.
            self.state_cmd.base_pos_target = None
            return

        self.state_cmd.base_pos_target = None

    def _update_robot_state_cmd(self):
        # LOCO command uses normalized joystick values [-1, 1].
        self.state_cmd.vel_cmd[0] = float(self.remote_controller.ly)
        self.state_cmd.vel_cmd[1] = float(-self.remote_controller.lx)
        self.state_cmd.vel_cmd[2] = float(-self.remote_controller.rx)

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

        self._set_default_world_state()

        if self.FSM_controller.cur_policy.name == FSMStateName.SKILL_TRACK_MOTION_MJLAB:
            self.state_cmd.base_pos = np.array([0.0, 0.0, 0.76], dtype=np.float32)
            self.state_cmd.base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _print_runtime_status(self):
        current_mode = self.FSM_controller.cur_policy.name
        if current_mode != self._last_mode_name:
            self._last_mode_name = current_mode
            print(f"[Mode] {self.FSM_controller.cur_policy.name_str}")

        if current_mode == FSMStateName.SKILL_TRACK_MOTION_MJLAB:
            now = time.time()
            if now - self._last_track_y_print_time > 0.5:
                self._last_track_y_print_time = now
                print(f"[TrackMotionMjlab] target_base_y={self.track_static_base_y:+.3f} (BLIND TEST: pos/vel hardcoded)")

    def run(self):
        try:
            loop_start_time = time.time()

            self._handle_remote_commands()
            self._update_base_target_override()
            self._update_robot_state_cmd()

            self.FSM_controller.run()

            policy_output_action = self.policy_output.actions.copy()
            kps = self.policy_output.kps.copy()
            kds = self.policy_output.kds.copy()

            for i in range(self.num_joints):
                self.low_cmd.motor_cmd[i].q = policy_output_action[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = kps[i]
                self.low_cmd.motor_cmd[i].kd = kds[i]
                self.low_cmd.motor_cmd[i].tau = 0

            self.send_cmd(self.low_cmd)
            self._print_runtime_status()

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

    controller = TrackMotionController(config)

    while True:
        try:
            controller.run()
            if controller.remote_controller.is_button_pressed(KeyMap.select):
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
