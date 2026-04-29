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
        "track_motion_mjlab": FSMStateName.SKILL_TRACK_MOTION_MJLAB,
        "track_motion_movable_base": FSMStateName.SKILL_TRACK_MOTION_MOVABLE_BASE,
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
    # Support both a dynamic ball (freejoint) and a static visual marker body.
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    if ball_joint_id != -1:
        qpos_adr = model.jnt_qposadr[ball_joint_id]
        qvel_adr = model.jnt_dofadr[ball_joint_id]
        data.qpos[qpos_adr:qpos_adr + 7] = np.array([8, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        data.qvel[qvel_adr:qvel_adr + 6] = np.array([-4.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return

    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if ball_body_id != -1:
        model.body_pos[ball_body_id] = np.array([0.5, -0.5, 1.0], dtype=np.float64)
        mujoco.mj_forward(model, data)


def get_ball_pos(data):
    try:
        return np.array(data.body("ball").xpos, dtype=np.float32)
    except Exception:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def set_ball_marker_position(model, data, ball_body_id, pos_xyz):
    if ball_body_id == -1:
        return
    model.body_pos[ball_body_id] = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
    mujoco.mj_forward(model, data)


def get_track_target_for_visualization(state_cmd, current_policy):
    # Priority: external command injection from state_cmd, fallback to policy internal random command.
    if hasattr(state_cmd, "rel_racket_target_pos_w"):
        val = np.asarray(getattr(state_cmd, "rel_racket_target_pos_w"), dtype=np.float32).reshape(-1)
        if val.shape[0] == 3:
            return val.copy()

    if hasattr(current_policy, "current_rel_racket_target_pos_w"):
        val = np.asarray(current_policy.current_rel_racket_target_pos_w, dtype=np.float32).reshape(-1)
        if val.shape[0] == 3:
            return val.copy()

    return None


def quat_rotate_inverse(quat_wxyz, vec_xyz):
    qw, qx, qy, qz = quat_wxyz
    qvec = np.array([qx, qy, qz], dtype=np.float32)
    vec = np.asarray(vec_xyz, dtype=np.float32)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec - 2.0 * (qw * uv + uuv)


def clamp(value, low, high):
    return max(low, min(high, value))


def velocity_to_axis(value, value_range):
    low, high = value_range
    if high <= low:
        return 0.0
    return (2.0 * (value - low) / (high - low)) - 1.0


GLFW_KEY_LEFT = 263
GLFW_KEY_RIGHT = 262
GLFW_KEY_DOWN = 264
GLFW_KEY_UP = 265


def print_keyboard_help():
    print("\n=== Keyboard Controls (Mujoco Window) ===")
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
    print("M: Track Motion Isaaclab")
    print("N: Track Motion Static (manual base target x/y)")
    print("V: Track Motion Movable Base (random base target)")
    print("Arrow keys:")
    print("  In loco -> adjust vx/vy")
    print("  In track_motion_static (N) -> adjust base target x/y")
    print("0: reset manual control (vx/vy and static base target x/y)")
    print("H: Show this help")
    print("=========================================\n")


def apply_initial_configuration(model, data, start_policy, robot_qpos_slice):
    # Always initialize ball state for testing
    initialize_ball_state(model, data)

    if start_policy == "table_tennis":
        data.qpos[2] = 0.76
        data.qpos[robot_qpos_slice] = load_default_joint_pos()
        data.qvel[:] = 0.0
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

    # Populate state_cmd from the actual simulation state before entering the first policy.
    state_cmd.q = d.qpos[robot_qpos_slice].copy()
    state_cmd.dq = d.qvel[robot_qvel_slice].copy()

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
        default="loco",
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
            "track_motion_isaaclab",
            "track_motion_mjlab",
            "track_motion_movable_base",
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

    # Load locomotion command range (m/s) for manual keyboard adjustment.
    loco_cfg_path = os.path.join(PROJECT_ROOT, "policy", "loco_mode", "config", "LocoMode.yaml")
    with open(loco_cfg_path, "r") as f:
        loco_cfg = yaml.load(f, Loader=yaml.FullLoader)
    loco_range_x = (
        float(loco_cfg["cmd_range"]["lin_vel_x"][0]),
        float(loco_cfg["cmd_range"]["lin_vel_x"][1]),
    )
    loco_range_y = (
        float(loco_cfg["cmd_range"]["lin_vel_y"][0]),
        float(loco_cfg["cmd_range"]["lin_vel_y"][1]),
    )
    movable_base_x_range = (-0.5, 0.5)
    movable_base_y_range = (-0.5, 0.5)
    loco_step_x = 0.05
    loco_step_y = 0.05
    base_step_x = 0.02
    base_step_y = 0.02

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    ball_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    num_joints = m.nu
    robot_qpos_slice, robot_qvel_slice = get_robot_state_slices(m)

    reset_requested = [False]
    policy_switch_requested = [None]
    loco_cmd_vxy = [0.0, 0.0]
    static_base_target_xy = [0.0, 0.0]

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
        'n': ('track_motion_mjlab', FSMStateName.SKILL_TRACK_MOTION_MJLAB),
        'v': ('track_motion_movable_base', FSMStateName.SKILL_TRACK_MOTION_MOVABLE_BASE),
    }
    ARROW_KEY_NAMES = {
        GLFW_KEY_UP: "UP",
        GLFW_KEY_DOWN: "DOWN",
        GLFW_KEY_LEFT: "LEFT",
        GLFW_KEY_RIGHT: "RIGHT",
    }

    def print_manual_status():
        print(
            "[manual status] "
            f"loco_vx={loco_cmd_vxy[0]:.3f} m/s "
            f"loco_vy={loco_cmd_vxy[1]:.3f} m/s "
            f"static_base_target=({static_base_target_xy[0]:.3f}, {static_base_target_xy[1]:.3f})"
        )

    def on_arrow_key(keycode):
        cur_policy = FSM_controller.cur_policy.name
        arrow = ARROW_KEY_NAMES[keycode]

        if cur_policy == FSMStateName.LOCOMODE:
            if keycode == GLFW_KEY_UP:
                loco_cmd_vxy[0] += loco_step_x
            elif keycode == GLFW_KEY_DOWN:
                loco_cmd_vxy[0] -= loco_step_x
            elif keycode == GLFW_KEY_LEFT:
                loco_cmd_vxy[1] += loco_step_y
            elif keycode == GLFW_KEY_RIGHT:
                loco_cmd_vxy[1] -= loco_step_y
            loco_cmd_vxy[0] = clamp(loco_cmd_vxy[0], loco_range_x[0], loco_range_x[1])
            loco_cmd_vxy[1] = clamp(loco_cmd_vxy[1], loco_range_y[0], loco_range_y[1])
            print(
                f"[key] {arrow} -> loco vxy=({loco_cmd_vxy[0]:.3f}, {loco_cmd_vxy[1]:.3f}) m/s"
            )
            return

        if cur_policy == FSMStateName.SKILL_TRACK_MOTION_MJLAB:
            if keycode == GLFW_KEY_UP:
                static_base_target_xy[0] += base_step_x
            elif keycode == GLFW_KEY_DOWN:
                static_base_target_xy[0] -= base_step_x
            elif keycode == GLFW_KEY_LEFT:
                static_base_target_xy[1] += base_step_y
            elif keycode == GLFW_KEY_RIGHT:
                static_base_target_xy[1] -= base_step_y
            static_base_target_xy[0] = clamp(
                static_base_target_xy[0], movable_base_x_range[0], movable_base_x_range[1]
            )
            static_base_target_xy[1] = clamp(
                static_base_target_xy[1], movable_base_y_range[0], movable_base_y_range[1]
            )
            print(
                "[key] {} -> static_base_target=({:.3f}, {:.3f})".format(
                    arrow, static_base_target_xy[0], static_base_target_xy[1]
                )
            )
            return

        print(f"[key] {arrow} ignored in policy: {FSM_controller.cur_policy.name_str}")

    def key_callback(keycode):
        if keycode in ARROW_KEY_NAMES:
            on_arrow_key(keycode)
            return

        if keycode < 0 or keycode > 255:
            print(f"[key] code={keycode} -> no binding")
            return

        key = chr(keycode).lower()
        if key == "r":
            reset_requested[0] = True
            print("[key] R -> reset simulation")
        elif key in KEY_MAP:
            policy_name, policy_state = KEY_MAP[key]
            policy_switch_requested[0] = (policy_name, policy_state)
            print(f"[key] {key.upper()} -> switch policy: {policy_name}")
        elif key == "0":
            loco_cmd_vxy[0] = 0.0
            loco_cmd_vxy[1] = 0.0
            static_base_target_xy[0] = 0.0
            static_base_target_xy[1] = 0.0
            print("[key] 0 -> reset manual control values")
            print_manual_status()
        elif key == 'h':
            print_keyboard_help()
        else:
            print(f"[key] {key} -> no binding")

    (
        state_cmd,
        policy_output,
        FSM_controller,
        policy_output_action,
        kps,
        kds,
        sim_counter,
    ) = reset_simulation(m, d, args.start_policy, robot_qpos_slice, num_joints)

    print_keyboard_help()
    print_manual_status()

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
                    print_manual_status()

                # Handle policy switch request
                if policy_switch_requested[0] is not None:
                    policy_name, policy_state = policy_switch_requested[0]
                    with viewer.lock():
                        switch_policy(FSM_controller, policy_state)
                    if policy_state == FSMStateName.SKILL_TRACK_MOTION_MJLAB:
                        static_base_target_xy[0] = 0.0
                        static_base_target_xy[1] = 0.0
                        print("[policy] track_motion_static -> reset base target to (0.000, 0.000)")
                    policy_switch_requested[0] = None

                # Convert physical velocity command to loco joystick domain [-1, 1].
                vx = clamp(loco_cmd_vxy[0], loco_range_x[0], loco_range_x[1])
                vy = clamp(loco_cmd_vxy[1], loco_range_y[0], loco_range_y[1])
                state_cmd.vel_cmd[0] = velocity_to_axis(vx, loco_range_x)
                state_cmd.vel_cmd[1] = velocity_to_axis(vy, loco_range_y)
                state_cmd.vel_cmd[2] = 0.0

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

                    if FSM_controller.cur_policy.name == FSMStateName.SKILL_TRACK_MOTION_MJLAB:
                        state_cmd.base_pos_target = np.array(static_base_target_xy, dtype=np.float32)
                    else:
                        # For movable-base policy, keep target None so policy uses internal random command.
                        state_cmd.base_pos_target = None

                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()

                    # Keep the visual ball aligned with the intended hitting target in track-motion states.
                    if FSM_controller.cur_policy.name in (
                        FSMStateName.SKILL_TRACK_MOTION_MJLAB,
                        FSMStateName.SKILL_TRACK_MOTION_MOVABLE_BASE,
                    ):
                        rel_target = get_track_target_for_visualization(state_cmd, FSM_controller.cur_policy)
                        if rel_target is not None:
                            ball_display_pos = base_pos.copy() + rel_target
                            ball_display_pos[2] = max(0.02, float(ball_display_pos[2]))
                            set_ball_marker_position(m, d, ball_body_id, ball_display_pos)
                            state_cmd.ball_pos = ball_display_pos.astype(np.float32)
                    else:
                        state_cmd.ball_pos = get_ball_pos(d)

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
