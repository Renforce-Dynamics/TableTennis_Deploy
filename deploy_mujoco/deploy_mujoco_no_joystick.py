import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import argparse
import os
import time

os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("GLFW_PLATFORM", "x11")

import mujoco.viewer
import mujoco
import numpy as np
import yaml
from common.ctrlcomp import *
from common.landing_command import LandingCommandGenerator, apply_landing_command_to_state
from FSM.FSM import *
from common.policy_registry import EXTRA_POLICY_SPECS, get_policy_choices, get_policy_state
from common.utils import get_gravity_orientation
try:
    from deploy_mujoco.deploy_mujoco_landing import (
        has_contact,
        has_racket_ball_contact,
        is_landing_policy,
        print_debug as print_landing_debug,
        update_landing_command,
    )
except ModuleNotFoundError:
    from deploy_mujoco_landing import (
        has_contact,
        has_racket_ball_contact,
        is_landing_policy,
        print_debug as print_landing_debug,
        update_landing_command,
    )


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


TABLE_POLICY_NAMES = tuple(spec.key for spec in EXTRA_POLICY_SPECS)
TABLE_POLICY_STATES = tuple(spec.state for spec in EXTRA_POLICY_SPECS)
TABLE_POLICY_BY_KEY = {
    "1": "table_tennis",
    "2": "table_tennis_distill",
    "3": "table_tennis_rev_racket",
    "4": "track_motion_movable_base",
    "5": "track_motion_mjlab",
}


def is_table_policy_name(policy_name):
    return policy_name in TABLE_POLICY_NAMES or policy_name in TABLE_POLICY_STATES


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


def get_ball_vel(model, data):
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    if ball_joint_id == -1:
        return np.zeros(3, dtype=np.float32)

    qvel_adr = model.jnt_dofadr[ball_joint_id]
    return np.array(data.qvel[qvel_adr:qvel_adr + 3], dtype=np.float32)


def ball_is_outside_demo_area(model, data):
    ball_pos = get_ball_pos(model, data)
    return bool(
        ball_pos[0] > 3.6
        or (abs(ball_pos[1]) > 0.8 and ball_pos[0] > 2.0)
        or (abs(ball_pos[2]) < 0.4)
    )


def sample_ball_reset_state(rng, default_ball_pos, default_ball_vel):
    ball_pos = default_ball_pos.copy()
    ball_vel = default_ball_vel.copy()
    # Match g1-main reset_root_state_uniform: add y in [-0.5, 0.0] to the
    # ball default pose instead of sampling an absolute world y.
    ball_pos[1] += rng.uniform(-0.5, 0.0)
    return ball_pos, ball_vel


def get_reset_ball_state(args, rng, default_ball_pos, default_ball_vel):
    if args.fixed_initial_ball:
        return default_ball_pos.copy(), default_ball_vel.copy()
    return sample_ball_reset_state(rng, default_ball_pos, default_ball_vel)


def quat_rotate_inverse(quat_wxyz, vec_xyz):
    qw, qx, qy, qz = quat_wxyz
    qvec = np.array([qx, qy, qz], dtype=np.float32)
    vec = np.asarray(vec_xyz, dtype=np.float32)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec - 2.0 * (qw * uv + uuv)


def populate_state_cmd(model, data, state_cmd, robot_qpos_slice, robot_qvel_slice, policy_name=None):
    qj = data.qpos[robot_qpos_slice]
    dqj = data.qvel[robot_qvel_slice]
    base_pos = data.qpos[0:3]
    quat = data.qpos[3:7]
    base_lin_vel = quat_rotate_inverse(quat, data.qvel[0:3])
    omega = quat_rotate_inverse(quat, data.qvel[3:6])
    gravity_orientation = get_gravity_orientation(quat)

    state_cmd.q = qj.copy()
    state_cmd.dq = dqj.copy()
    state_cmd.base_pos = base_pos.copy()
    state_cmd.base_lin_vel = base_lin_vel.copy()
    state_cmd.ball_pos = get_ball_pos(model, data)
    state_cmd.ball_vel = get_ball_vel(model, data)
    state_cmd.gravity_ori = gravity_orientation.copy()
    state_cmd.base_quat = quat.copy()
    state_cmd.ang_vel = omega.copy()


def clear_landing_command(state_cmd):
    state_cmd.base_pos_target = None
    state_cmd.rel_racket_target_pos_w = None
    state_cmd.racket_target_vel_w = None
    state_cmd.racket_target_time = None
    state_cmd.planner_valid = False


def maybe_update_landing_command(
    model,
    data,
    state_cmd,
    fsm_controller,
    landing_generator,
    control_dt,
    episode_time_s,
    use_mujoco_predictor,
):
    if not is_landing_policy(fsm_controller.cur_policy.name):
        clear_landing_command(state_cmd)
        return None

    landing_cmd = update_landing_command(
        model,
        data,
        state_cmd,
        landing_generator,
        control_dt=control_dt,
        episode_time_s=episode_time_s,
        use_mujoco_predictor=use_mujoco_predictor,
    )
    apply_landing_command_to_state(state_cmd, landing_cmd)
    return landing_cmd


def apply_initial_configuration(model, data, start_policy, robot_qpos_slice, base_height, ball_pos, ball_vel):
    data.qpos[2] = base_height
    data.qpos[robot_qpos_slice] = load_default_joint_pos()
    data.qvel[:] = 0.0
    initialize_ball_state(model, data, ball_pos, ball_vel)
    mujoco.mj_forward(model, data)


def reset_simulation(
    model,
    data,
    start_policy,
    robot_qpos_slice,
    robot_qvel_slice,
    num_joints,
    base_height,
    ball_pos,
    ball_vel,
    landing_generator,
    use_mujoco_predictor,
):
    mujoco.mj_resetData(model, data)
    apply_initial_configuration(model, data, start_policy, robot_qpos_slice, base_height, ball_pos, ball_vel)
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    landing_generator.reset()

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

    populate_state_cmd(model, data, state_cmd, robot_qpos_slice, robot_qvel_slice, fsm_controller.cur_policy.name)
    maybe_update_landing_command(
        model,
        data,
        state_cmd,
        fsm_controller,
        landing_generator,
        control_dt=0.0,
        episode_time_s=0.0,
        use_mujoco_predictor=use_mujoco_predictor,
    )
    fsm_controller.run()
    policy_output_action = policy_output.actions.copy()
    kps = policy_output.kps.copy()
    kds = policy_output.kds.copy()

    print("Simulation reset to initial state.")
    return state_cmd, policy_output, fsm_controller, policy_output_action, kps, kds, sim_counter


def switch_policy(fsm_controller, policy_name):
    if fsm_controller.cur_policy.name == policy_name:
        return True

    previous_policy = fsm_controller.cur_policy
    fsm_controller.cur_policy.exit()
    fsm_controller.get_next_policy(policy_name)
    if fsm_controller.cur_policy is previous_policy and previous_policy.name != policy_name:
        print("Policy is unavailable:", policy_name)
        return False

    fsm_controller.cur_policy.enter()
    fsm_controller.FSMmode = FSMMode.NORMAL
    print("Switched to ", fsm_controller.cur_policy.name_str)
    return True


def reset_ball_for_table_policy(model, data, args, rng, default_ball_pos, default_ball_vel, landing_generator):
    ball_pos, ball_vel = get_reset_ball_state(args, rng, default_ball_pos, default_ball_vel)
    initialize_ball_state(model, data, ball_pos, ball_vel)
    landing_generator.reset()
    mujoco.mj_forward(model, data)


def enter_table_policy(fsm_controller, policy_name, model, data, args, rng, default_ball_pos, default_ball_vel, landing_generator):
    reset_ball_for_table_policy(model, data, args, rng, default_ball_pos, default_ball_vel, landing_generator)
    switch_policy(fsm_controller, policy_name)


def handle_keyboard_command(
    key,
    fsm_controller,
    model,
    data,
    args,
    rng,
    default_ball_pos,
    default_ball_vel,
    selected_table_policy,
    landing_generator,
):
    if key is None:
        return selected_table_policy
    if key == "p":
        switch_policy(fsm_controller, FSMStateName.PASSIVE)
    elif key == "f":
        switch_policy(fsm_controller, FSMStateName.FIXEDPOSE)
    elif key == "l":
        if is_table_policy_name(fsm_controller.cur_policy.name):
            fsm_controller.state_cmd.skill_cmd = FSMCommand.LOCO
            print("Requested loco through skill_cooldown.")
        else:
            switch_policy(fsm_controller, FSMStateName.LOCOMODE)
    elif key in TABLE_POLICY_BY_KEY:
        selected_table_policy = TABLE_POLICY_BY_KEY[key]
        print("Selected table tennis policy:", selected_table_policy)
        if fsm_controller.cur_policy.name == FSMStateName.LOCOMODE:
            enter_table_policy(
                fsm_controller,
                selected_table_policy,
                model,
                data,
                args,
                rng,
                default_ball_pos,
                default_ball_vel,
                landing_generator,
            )
        elif is_table_policy_name(fsm_controller.cur_policy.name):
            print("Currently in a table policy; press 'l' to return to loco, then press the number to enter safely.")
    elif key == "t":
        if fsm_controller.cur_policy.name != FSMStateName.LOCOMODE and not is_table_policy_name(fsm_controller.cur_policy.name):
            print("Enter loco first: press 'l', or press 1/2/3/4/5 from loco to enter a table policy.")
            return selected_table_policy
        enter_table_policy(
            fsm_controller,
            selected_table_policy,
            model,
            data,
            args,
            rng,
            default_ball_pos,
            default_ball_vel,
            landing_generator,
        )
    return selected_table_policy


def print_keyboard_help(selected_table_policy):
    print(
        "Keyboard controls: p=passive, f=fixedpose, l=loco, "
        "1=table_tennis, 2=distill, 3=rev_racket, 4=track_motion_movable_base, 5=track_motion_mjlab, "
        "t=re-enter selected table policy, r=reset"
    )
    print("Selected table tennis policy:", selected_table_policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mujoco deployment without requiring a joystick.")
    parser.add_argument(
        "--start-policy",
        default="passive",
        choices=get_policy_choices(),
        help="Initial FSM policy when the simulation starts.",
    )
    parser.add_argument(
        "--table-policy",
        default="table_tennis",
        choices=get_policy_choices(include_base=False),
        help="Policy entered by pressing 't' after loco. Track-motion policies receive planner commands from ball observation.",
    )
    parser.add_argument("--planner-config", default="deploy_mujoco/config/landing_planner.yaml")
    parser.add_argument(
        "--planner-source",
        choices=["mujoco", "model"],
        default="mujoco",
        help="Planner source used for track-motion table policies.",
    )
    parser.add_argument("--landing-debug-every", type=int, default=0, help="Print landing planner status every N control ticks.")
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
    parser.add_argument(
        "--base-height",
        type=float,
        default=0.76,
        help="Initial robot base height. The default 0.76 matches g1-main's G1_RACKET_CFG init_state.pos.",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "g1_tennis.yaml")
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
    landing_generator = LandingCommandGenerator(args.planner_config)
    control_dt = float(simulation_dt) * int(control_decimation)
    initial_ball_pos, initial_ball_vel = get_reset_ball_state(
        args, rng, default_ball_pos, default_ball_vel
    )

    reset_requested = [False]
    keyboard_command = [None]
    selected_table_policy = [args.table_policy]

    def key_callback(keycode):
        try:
            key = chr(keycode).lower()
        except ValueError:
            return
        if key == "r":
            reset_requested[0] = True
        elif key in ("p", "f", "l", "t", "1", "2", "3", "4", "5"):
            keyboard_command[0] = key

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
        robot_qvel_slice,
        num_joints,
        args.base_height,
        initial_ball_pos,
        initial_ball_vel,
        landing_generator,
        args.planner_source == "mujoco",
    )

    print_keyboard_help(selected_table_policy[0])

    running = True
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running() and running:
            step_start = time.time()
            try:
                if reset_requested[0]:
                    reset_ball_pos, reset_ball_vel = get_reset_ball_state(
                        args, rng, default_ball_pos, default_ball_vel
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
                            robot_qvel_slice,
                            num_joints,
                            args.base_height,
                            reset_ball_pos,
                            reset_ball_vel,
                            landing_generator,
                            args.planner_source == "mujoco",
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
                if np.any(policy_output.tau_limit > 0.0):
                    tau = np.clip(tau, -policy_output.tau_limit, policy_output.tau_limit)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                sim_counter += 1

                if is_table_policy_name(FSM_controller.cur_policy.name) and ball_is_outside_demo_area(m, d):
                    reset_ball_pos, reset_ball_vel = get_reset_ball_state(
                        args, rng, default_ball_pos, default_ball_vel
                    )
                    initialize_ball_state(
                        m,
                        d,
                        reset_ball_pos,
                        reset_ball_vel,
                    )
                    landing_generator.reset()
                    mujoco.mj_forward(m, d)

                if sim_counter % control_decimation == 0:
                    populate_state_cmd(m, d, state_cmd, robot_qpos_slice, robot_qvel_slice, FSM_controller.cur_policy.name)
                    selected_table_policy[0] = handle_keyboard_command(
                        keyboard_command[0],
                        FSM_controller,
                        m,
                        d,
                        args,
                        rng,
                        default_ball_pos,
                        default_ball_vel,
                        selected_table_policy[0],
                        landing_generator,
                    )
                    keyboard_command[0] = None
                    populate_state_cmd(m, d, state_cmd, robot_qpos_slice, robot_qvel_slice, FSM_controller.cur_policy.name)

                    landing_cmd = maybe_update_landing_command(
                        m,
                        d,
                        state_cmd,
                        FSM_controller,
                        landing_generator,
                        control_dt=control_dt,
                        episode_time_s=sim_counter * simulation_dt,
                        use_mujoco_predictor=args.planner_source == "mujoco",
                    )

                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()

                    if args.debug_frames > 0 and is_table_policy_name(FSM_controller.cur_policy.name):
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

                    if landing_cmd is not None and args.landing_debug_every > 0:
                        control_tick = sim_counter // control_decimation
                        contact_rb = has_racket_ball_contact(m, d)
                        contact_table = has_contact(m, d, "ball_geom", "table_top")
                        if control_tick % args.landing_debug_every == 0 or contact_rb:
                            print_landing_debug(state_cmd, landing_cmd, contact_rb, contact_table, m, d)
            except ValueError as e:
                print(str(e))

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
