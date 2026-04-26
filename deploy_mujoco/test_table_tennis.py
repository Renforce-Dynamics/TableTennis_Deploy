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
        "table_tennis_distill": FSMStateName.SKILL_TABLE_TENNIS_DISTILL,
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


def initialize_ball_state(model, data, ball_pos, ball_vel):
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    if ball_joint_id == -1:
        return

    qpos_adr = model.jnt_qposadr[ball_joint_id]
    qvel_adr = model.jnt_dofadr[ball_joint_id]
    data.qpos[qpos_adr:qpos_adr + 7] = np.array(
        [ball_pos[0], ball_pos[1], ball_pos[2], 1.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    data.qvel[qvel_adr:qvel_adr + 6] = np.array(
        [ball_vel[0], ball_vel[1], ball_vel[2], 0.0, 0.0, 0.0], dtype=np.float64
    )


def get_ball_pos(model, data):
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    if ball_joint_id == -1:
        return np.array(data.body("ball").xpos, dtype=np.float32)

    qpos_adr = model.jnt_qposadr[ball_joint_id]
    return np.array(data.qpos[qpos_adr:qpos_adr + 3], dtype=np.float32)


def ball_is_outside_demo_area(model, data):
    ball_pos = get_ball_pos(model, data)
    return bool(
        ball_pos[0] > 3.6
        or (abs(ball_pos[1]) > 0.8 and ball_pos[0] > 2.0)
    )


def sample_ball_reset_state(rng, default_ball_pos, default_ball_vel):
    ball_pos = default_ball_pos.copy()
    ball_vel = default_ball_vel.copy()
    # Match g1-main reset_root_state_uniform: add y in [-0.5, 0.0] to the
    # ball default pose instead of sampling an absolute world y.
    ball_pos[1] += rng.uniform(-0.5, 0.0)
    return ball_pos, ball_vel


def quat_rotate_inverse(quat_wxyz, vec_xyz):
    qw, qx, qy, qz = quat_wxyz
    qvec = np.array([qx, qy, qz], dtype=np.float32)
    vec = np.asarray(vec_xyz, dtype=np.float32)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec - 2.0 * (qw * uv + uuv)


def apply_initial_configuration(model, data, start_policy, robot_qpos_slice, ball_pos, ball_vel):
    if start_policy in ("table_tennis", "table_tennis_distill"):
        data.qpos[2] = 0.76
        data.qpos[robot_qpos_slice] = load_default_joint_pos()
        data.qvel[:] = 0.0
        initialize_ball_state(model, data, ball_pos, ball_vel)
    mujoco.mj_forward(model, data)


def reset_simulation(model, data, start_policy, robot_qpos_slice, num_joints, ball_pos, ball_vel):
    mujoco.mj_resetData(model, data)
    apply_initial_configuration(model, data, start_policy, robot_qpos_slice, ball_pos, ball_vel)
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
        print("current policy is ", fsm_controller.cur_policy.name_str)
    else:
        fsm_controller.cur_policy.enter()

    print("Simulation reset to initial state.")
    return state_cmd, policy_output, fsm_controller, policy_output_action, kps, kds, sim_counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mujoco deployment without requiring a joystick.")
    parser.add_argument(
        "--start-policy",
        default="table_tennis",
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
            "table_tennis_distill",
        ],
        help="Initial FSM policy when the simulation starts.",
    )
    parser.add_argument(
        "--debug-frames",
        type=int,
        default=0,
        help="Print key observation/action statistics for the first N control frames.",
    )
    parser.add_argument(
        "--ball-pos",
        type=float,
        nargs=3,
        default=[3.5, -0.2, 1.0],
        help="Base ball position in world coordinates. Resets add the g1-main y range [-0.5, 0.0].",
    )
    parser.add_argument(
        "--ball-vel",
        type=float,
        nargs=3,
        default=[-4.0, 0.0, 0.0],
        help="Initial ball linear velocity in world coordinates.",
    )
    parser.add_argument(
        "--fixed-initial-ball",
        action="store_true",
        help="Use --ball-pos exactly for the first full reset instead of sampling the g1-main reset range.",
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
    rng = np.random.default_rng()
    default_ball_pos = np.array(args.ball_pos, dtype=np.float32)
    default_ball_vel = np.array(args.ball_vel, dtype=np.float32)
    if args.fixed_initial_ball:
        initial_ball_pos = default_ball_pos.copy()
        initial_ball_vel = default_ball_vel.copy()
    else:
        initial_ball_pos, initial_ball_vel = sample_ball_reset_state(
            rng, default_ball_pos, default_ball_vel
        )

    reset_requested = [False]

    def key_callback(keycode):
        try:
            key = chr(keycode).lower()
        except ValueError:
            return
        if key == "r":
            reset_requested[0] = True

    (
        state_cmd,
        policy_output,
        FSM_controller,
        policy_output_action,
        kps,
        kds,
        sim_counter,
    ) = reset_simulation(
        m,
        d,
        args.start_policy,
        robot_qpos_slice,
        num_joints,
        initial_ball_pos,
        initial_ball_vel,
    )

    running = True
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running() and running:
            step_start = time.time()
            try:
                if reset_requested[0]:
                    if args.fixed_initial_ball:
                        reset_ball_pos = default_ball_pos.copy()
                        reset_ball_vel = default_ball_vel.copy()
                    else:
                        reset_ball_pos, reset_ball_vel = sample_ball_reset_state(
                            rng, default_ball_pos, default_ball_vel
                        )
                    with viewer.lock():
                        (
                            state_cmd,
                            policy_output,
                            FSM_controller,
                            policy_output_action,
                            kps,
                            kds,
                            sim_counter,
                        ) = reset_simulation(
                            m,
                            d,
                            args.start_policy,
                            robot_qpos_slice,
                            num_joints,
                            reset_ball_pos,
                            reset_ball_vel,
                        )
                    reset_requested[0] = False

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

                if args.start_policy in ("table_tennis", "table_tennis_distill") and ball_is_outside_demo_area(m, d):
                    reset_ball_pos, reset_ball_vel = sample_ball_reset_state(
                        rng, default_ball_pos, default_ball_vel
                    )
                    initialize_ball_state(
                        m,
                        d,
                        reset_ball_pos,
                        reset_ball_vel,
                    )
                    mujoco.mj_forward(m, d)

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
                    state_cmd.ball_pos = get_ball_pos(m, d)
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.base_quat = quat.copy()
                    state_cmd.ang_vel = omega.copy()

                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()

                    if args.debug_frames > 0 and FSM_controller.cur_policy.name in (
                        FSMStateName.SKILL_TABLE_TENNIS,
                        FSMStateName.SKILL_TABLE_TENNIS_DISTILL,
                    ):
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
