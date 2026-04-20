import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import argparse
import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_policy_state(policy_name: str):
    policy_map = {
        "passive": FSMStateName.PASSIVE,
        "fixedpose": FSMStateName.FIXEDPOSE,
        "loco": FSMStateName.LOCOMODE,
        "dance": FSMStateName.SKILL_Dance,
        "kungfu": FSMStateName.SKILL_KungFu,
        "kick": FSMStateName.SKILL_KICK,
        "kungfu2": FSMStateName.SKILL_KungFu2,
        "beyond_mimic": FSMStateName.SKILL_BEYOND_MIMIC,
        "table_tennis": FSMStateName.SKILL_TABLE_TENNIS,
    }
    return policy_map[policy_name]


def load_default_joint_pos():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "policy", "table_tennis", "config", "TableTennis.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return np.array(config["default_angles"], dtype=np.float32)


def get_robot_state_slices(model):
    first_robot_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint")
    if first_robot_joint_id == -1:
        raise ValueError("Could not locate the first robot joint in Mujoco model.")

    qpos_start = model.jnt_qposadr[first_robot_joint_id]
    qvel_start = model.jnt_dofadr[first_robot_joint_id]
    return slice(qpos_start, qpos_start + model.nu), slice(qvel_start, qvel_start + model.nu)


def initialize_ball_state(model, data):
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    if ball_joint_id == -1:
        return

    qpos_adr = model.jnt_qposadr[ball_joint_id]
    qvel_adr = model.jnt_dofadr[ball_joint_id]
    data.qpos[qpos_adr:qpos_adr + 7] = np.array([3.5, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qvel[qvel_adr:qvel_adr + 6] = np.array([-4.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def get_ball_pos(data):
    return np.array(data.body("ball").xpos, dtype=np.float32)


def quat_rotate_inverse(quat_wxyz, vec_xyz):
    qw, qx, qy, qz = quat_wxyz
    qvec = np.array([qx, qy, qz], dtype=np.float32)
    vec = np.asarray(vec_xyz, dtype=np.float32)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec - 2.0 * (qw * uv + uuv)


def apply_initial_configuration(model, data, start_policy, robot_qpos_slice):
    if start_policy == "table_tennis":
        data.qpos[2] = 0.76
        data.qpos[robot_qpos_slice] = load_default_joint_pos()
        data.qvel[:] = 0.0
        initialize_ball_state(model, data)
    mujoco.mj_forward(model, data)


def reset_simulation(model, data, start_policy, robot_qpos_slice, num_joints):
    mujoco.mj_resetData(model, data)
    apply_initial_configuration(model, data, start_policy, robot_qpos_slice)
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0

    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    fsm_controller = FSM(state_cmd, policy_output)

    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0

    initial_policy = get_policy_state(start_policy)
    if initial_policy != FSMStateName.PASSIVE:
        fsm_controller.get_next_policy(initial_policy)
        fsm_controller.cur_policy.enter()
        print(f"Initialized to policy: {fsm_controller.cur_policy.name_str}")
    else:
        fsm_controller.cur_policy.enter()
        print("Initialized to policy: PASSIVE")

    return state_cmd, policy_output, fsm_controller, policy_output_action, kps, kds, sim_counter


def switch_policy(fsm_controller, new_policy_state):
    """Switch to a new policy state"""
    if fsm_controller.cur_policy.name == new_policy_state:
        print(f"Already in {fsm_controller.cur_policy.name_str}")
        return

    fsm_controller.get_next_policy(new_policy_state)
    fsm_controller.cur_policy.enter()
    print(f"Switched to policy: {fsm_controller.cur_policy.name_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mujoco deployment with keyboard state switching.")
    parser.add_argument(
        "--start-policy",
        default="passive",
        choices=[
            "passive",
            "fixedpose",
            "loco",
            "dance",
            "kungfu",
            "kick",
            "kungfu2",
            "beyond_mimic",
            "table_tennis",
        ],
        help="Initial FSM policy when the simulation starts.",
    )
    parser.add_argument(
        "--debug-frames",
        type=int,
        default=0,
        help="Print key observation/action statistics for the first N control frames.",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    with open(mujoco_yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = os.path.join(PROJECT_ROOT, config["xml_path"])
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    num_joints = m.nu
    robot_qpos_slice, robot_qvel_slice = get_robot_state_slices(m)

    reset_requested = [False]
    policy_switch_requested = [None]

    # Keyboard mapping for policy switching
    KEY_MAP = {
        'p': ('passive', FSMStateName.PASSIVE),
        'f': ('fixedpose', FSMStateName.FIXEDPOSE),
        'l': ('loco', FSMStateName.LOCOMODE),
        'd': ('dance', FSMStateName.SKILL_Dance),
        'k': ('kungfu', FSMStateName.SKILL_KungFu),
        'c': ('kick', FSMStateName.SKILL_KICK),
        '2': ('kungfu2', FSMStateName.SKILL_KungFu2),
        'b': ('beyond_mimic', FSMStateName.SKILL_BEYOND_MIMIC),
        't': ('table_tennis', FSMStateName.SKILL_TABLE_TENNIS),
    }

    def key_callback(keycode):
        try:
            key = chr(keycode).lower()
        except ValueError:
            return

        if key == "r":
            reset_requested[0] = True
        elif key in KEY_MAP:
            policy_name, policy_state = KEY_MAP[key]
            policy_switch_requested[0] = (policy_name, policy_state)
        elif key == 'h':
            print("\n=== Keyboard Controls ===")
            print("R: Reset simulation")
            print("P: Passive mode")
            print("F: Fixed pose")
            print("L: Locomotion mode")
            print("D: Dance")
            print("K: Kung Fu")
            print("C: Kick")
            print("2: Kung Fu 2")
            print("B: Beyond Mimic")
            print("T: Table Tennis")
            print("H: Show this help")
            print("========================\n")

    (
        state_cmd,
        policy_output,
        FSM_controller,
        policy_output_action,
        kps,
        kds,
        sim_counter,
    ) = reset_simulation(m, d, args.start_policy, robot_qpos_slice, num_joints)

    # Print keyboard controls at startup
    print("\n=== Keyboard Controls ===")
    print("Press 'H' for help at any time")
    print("========================\n")

    running = True
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running() and running:
            step_start = time.time()
            try:
                # Handle reset request
                if reset_requested[0]:
                    with viewer.lock():
                        (
                            state_cmd,
                            policy_output,
                            FSM_controller,
                            policy_output_action,
                            kps,
                            kds,
                            sim_counter,
                        ) = reset_simulation(m, d, args.start_policy, robot_qpos_slice, num_joints)
                    reset_requested[0] = False

                # Handle policy switch request
                if policy_switch_requested[0] is not None:
                    policy_name, policy_state = policy_switch_requested[0]
                    with viewer.lock():
                        switch_policy(FSM_controller, policy_state)
                    policy_switch_requested[0] = None

                state_cmd.vel_cmd[:] = 0.0

                tau = pd_control(
                    policy_output_action,
                    d.qpos[robot_qpos_slice],
                    kps,
                    np.zeros_like(kps),
                    d.qvel[robot_qvel_slice],
                    kds,
                )
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                sim_counter += 1

                if sim_counter % control_decimation == 0:
                    qj = d.qpos[robot_qpos_slice]
                    dqj = d.qvel[robot_qvel_slice]
                    base_pos = d.qpos[0:3]
                    quat = d.qpos[3:7]
                    base_lin_vel = quat_rotate_inverse(quat, d.qvel[0:3])
                    omega = quat_rotate_inverse(quat, d.qvel[3:6])
                    gravity_orientation = get_gravity_orientation(quat)

                    state_cmd.q = qj.copy()
                    state_cmd.dq = dqj.copy()
                    state_cmd.base_pos = base_pos.copy()
                    state_cmd.base_lin_vel = base_lin_vel.copy()
                    state_cmd.ball_pos = get_ball_pos(d)
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.base_quat = quat.copy()
                    state_cmd.ang_vel = omega.copy()

                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()

                    if args.debug_frames > 0 and FSM_controller.cur_policy.name == FSMStateName.SKILL_TABLE_TENNIS:
                        policy = FSM_controller.cur_policy
                        print("\n[debug] frame", args.debug_frames)
                        for term_name in policy.term_order:
                            term = policy.latest_obs_terms[term_name]
                            print(
                                f"  {term_name}: shape={term.shape} min={float(np.min(term)):.4f} max={float(np.max(term)):.4f}"
                            )
                        print(
                            "  raw_action: min={:.4f} max={:.4f}".format(
                                float(np.min(policy.action)), float(np.max(policy.action))
                            )
                        )
                        print(
                            "  target_q: min={:.4f} max={:.4f}".format(
                                float(np.min(policy_output_action)), float(np.max(policy_output_action))
                            )
                        )
                        args.debug_frames -= 1
            except ValueError as e:
                print(str(e))

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
