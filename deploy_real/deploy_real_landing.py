import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.ctrlcomp import PolicyOutput, StateAndCmd
from common.landing_command import LandingCommandGenerator, apply_landing_command_to_state
from common.policy_registry import get_policy_choices, get_policy_state
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation_real
from common.utils import FSMCommand, FSMStateName
from FSM.FSM import FSM, FSMMode
from typing import Union
import argparse
import numpy as np
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, init_cmd_hg, MotorMode
from config import Config
from deploy_real.ball_observer import ConstantBallObserver, UdpJsonBallObserver


LANDING_POLICY_DEFAULT = "track_motion_movable_base"
LANDING_POLICY_STATES = (
    FSMStateName.SKILL_TRACK_MOTION_MOVABLE_BASE,
    FSMStateName.SKILL_TRACK_MOTION_MJLAB,
)


def is_landing_policy(policy_state):
    return policy_state in LANDING_POLICY_STATES


def clamp_targets(target_q, current_q, max_delta):
    return np.clip(target_q, current_q - max_delta, current_q + max_delta)


class RealLandingController:
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

        self.landing_generator = LandingCommandGenerator(args.planner_config)
        self.ball_observer = self._make_ball_observer()
        self.start_time = time.time()
        self.running = True
        self.counter_over_time = 0
        self.last_policy_hint_time = 0.0
        self.last_debug_time = 0.0

        print("=" * 64)
        print("Real Landing Controller Initialized")
        print("Controls:")
        print("  F1       - PASSIVE mode")
        print("  START    - FIXEDPOSE mode")
        print("  A + R1   - LOCO mode")
        print("  B + R1   - selected landing policy")
        print("  SELECT   - Exit program")
        print("=" * 64)

    def _make_ball_observer(self):
        if self.args.ball_source == "udp":
            return UdpJsonBallObserver(self.args.udp_host, self.args.udp_port, timeout_s=0.0)
        return ConstantBallObserver(self.args.ball_pos, self.args.ball_vel)

    def switch_to_policy(self, policy_name):
        target_state = get_policy_state(policy_name) if isinstance(policy_name, str) else policy_name
        if self.fsm_controller.cur_policy.name == target_state or self.fsm_controller.cur_policy.name == policy_name:
            return
        self.fsm_controller.cur_policy.exit()
        self.fsm_controller.get_next_policy(target_state)
        self.fsm_controller.cur_policy.enter()
        self.fsm_controller.FSMmode = FSMMode.NORMAL
        self.start_time = time.time()
        if is_landing_policy(target_state):
            self.landing_generator.reset()
        print("Switched to", self.fsm_controller.cur_policy.name_str)

    def low_state_handler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdHG]):
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

        # Real robot deployment normally has no trusted global base estimator yet.
        # Keep this frame consistent with MuJoCo landing deployment until a state estimator is connected.
        self.state_cmd.base_pos = np.array([0.0, 0.0, self.args.base_height], dtype=np.float32)
        self.state_cmd.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.state_cmd.vel_cmd[:] = 0.0

        ball_obs = self.ball_observer.update()
        if ball_obs.valid:
            self.state_cmd.ball_pos = ball_obs.pos_w.copy()
            self.state_cmd.ball_vel = ball_obs.vel_w.copy()
        else:
            self.state_cmd.ball_pos = np.array(self.args.ball_pos, dtype=np.float32)
            self.state_cmd.ball_vel = np.array(self.args.ball_vel, dtype=np.float32)
        return ball_obs

    def update_landing_command(self):
        if not is_landing_policy(self.fsm_controller.cur_policy.name):
            self.state_cmd.base_pos_target = None
            self.state_cmd.rel_racket_target_pos_w = None
            self.state_cmd.racket_target_vel_w = None
            self.state_cmd.racket_target_time = None
            self.state_cmd.planner_valid = False
            return None

        cmd = self.landing_generator.update(
            ball_pos_w=self.state_cmd.ball_pos,
            ball_vel_w=self.state_cmd.ball_vel,
            robot_base_pos_w=self.state_cmd.base_pos,
            robot_base_quat_wxyz=self.state_cmd.base_quat,
            dt=self.control_dt,
            episode_time_s=time.time() - self.start_time,
        )
        apply_landing_command_to_state(self.state_cmd, cmd)
        return cmd

    def handle_remote_commands(self):
        if self.remote_controller.is_button_pressed(KeyMap.F1):
            self.state_cmd.skill_cmd = FSMCommand.PASSIVE
        if self.remote_controller.is_button_pressed(KeyMap.start):
            self.state_cmd.skill_cmd = FSMCommand.POS_RESET
        if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.R1):
            self.state_cmd.skill_cmd = FSMCommand.LOCO
        if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.R1):
            if self.fsm_controller.cur_policy.name == FSMStateName.LOCOMODE:
                self.switch_to_policy(self.args.policy)
            elif time.time() - self.last_policy_hint_time > 1.0:
                print("Enter loco first, then press B+R1 to start landing policy.")
                self.last_policy_hint_time = time.time()
            self.state_cmd.skill_cmd = FSMCommand.INVALID

    def compute_targets(self):
        self.fsm_controller.run()
        target_q = self.policy_output.actions.copy()
        elapsed = time.time() - self.start_time
        alpha = min(elapsed / self.args.ramp_time, 1.0) if self.args.ramp_time > 0 else 1.0
        target_q = self.qj * (1.0 - alpha) + target_q * alpha
        target_q = clamp_targets(target_q, self.qj, self.args.max_delta)
        return target_q

    def print_debug(self, ball_obs, landing_cmd, target_q):
        if not self.args.debug:
            return
        now = time.time()
        if now - self.last_debug_time < 0.25:
            return
        self.last_debug_time = now
        policy = self.fsm_controller.cur_policy
        if landing_cmd is None:
            print(
                f"[{policy.name_str}] ball_valid={ball_obs.valid} ball={np.round(self.state_cmd.ball_pos, 3).tolist()} "
                f"target_q=[{float(np.min(target_q)):.3f}, {float(np.max(target_q)):.3f}]"
            )
            return
        print(
            "[landing-real] valid={} reason={} t_hit={:.3f} ball={} vel={} racket_pos={} racket_vel={}".format(
                landing_cmd.valid,
                landing_cmd.reason,
                float(landing_cmd.racket_target_time[0]),
                np.round(self.state_cmd.ball_pos, 3).tolist(),
                np.round(self.state_cmd.ball_vel, 3).tolist(),
                np.round(landing_cmd.rel_racket_target_pos_w, 3).tolist(),
                np.round(landing_cmd.racket_target_vel_w, 3).tolist(),
            )
        )

    def run(self):
        try:
            loop_start_time = time.time()
            ball_obs = self.build_state()
            self.handle_remote_commands()
            landing_cmd = self.update_landing_command()
            target_q = self.compute_targets()
            kps = self.policy_output.kps.copy()
            kds = self.policy_output.kds.copy()

            self.print_debug(ball_obs, landing_cmd, target_q)
            if not self.args.dry_run:
                for i in range(self.num_joints):
                    self.low_cmd.motor_cmd[i].q = target_q[i]
                    self.low_cmd.motor_cmd[i].qd = 0.0
                    self.low_cmd.motor_cmd[i].kp = kps[i]
                    self.low_cmd.motor_cmd[i].kd = kds[i]
                    self.low_cmd.motor_cmd[i].tau = 0.0
                self.send_cmd(self.low_cmd)

            delta_time = time.time() - loop_start_time
            if delta_time < self.control_dt:
                time.sleep(self.control_dt - delta_time)
                self.counter_over_time = 0
            else:
                print("control loop over time.")
                self.counter_over_time += 1
        except ValueError as e:
            print(str(e))


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy landing planner + track-motion policy on the real robot.")
    parser.add_argument("--policy", default=LANDING_POLICY_DEFAULT, choices=get_policy_choices(include_base=False))
    parser.add_argument("--planner-config", default="deploy_mujoco/config/landing_planner.yaml")
    parser.add_argument("--ball-source", choices=["constant", "udp"], default="constant")
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=15050)
    parser.add_argument("--ball-pos", type=float, nargs=3, default=[3.5, -0.2, 1.0])
    parser.add_argument("--ball-vel", type=float, nargs=3, default=[-4.0, 0.0, 0.0])
    parser.add_argument("--base-height", type=float, default=0.793)
    parser.add_argument("--max-delta", type=float, default=0.12)
    parser.add_argument("--ramp-time", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config()
    ChannelFactoryInitialize(0, config.net)
    controller = RealLandingController(config, args)

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
