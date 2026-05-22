import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import argparse
import os
import time

os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("GLFW_PLATFORM", "x11")

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from common.landing_command import LandingCommandGenerator
from common.policy_registry import (
    EXTRA_POLICY_SPECS,
    get_policy_choices,
    get_policy_state,
    is_landing_policy,
)
from common.utils import FSMCommand, FSMStateName

from deploy_mujoco.deploy_mujoco_landing import (
    BallRespawnTracker,
    add_debug_strike_args,
    add_planner_args,
    add_serve_args,
    apply_debug_strike_overrides,
    apply_pd_and_step,
    ball_is_outside_demo_area,
    get_robot_state_slices,
    has_contact,
    has_racket_ball_contact,
    initialize_ball_state,
    populate_state_cmd,
    print_debug,
    reset_simulation,
    run_control_tick,
    switch_policy,
)


TABLE_POLICY_BY_KEY = {
    "1": "table_tennis",
    "2": "table_tennis_distill",
    "3": "table_tennis_rev_racket",
    "4": "track_motion_movable_base",
    "5": "track_motion_mjlab",
    "6": "landing_assist_finetune",
}
TRACK_MOTION_MOVABLE_BASE_STATE = get_policy_state("track_motion_movable_base")
TABLE_POLICY_NAMES = tuple(spec.key for spec in EXTRA_POLICY_SPECS)
TABLE_POLICY_STATES = tuple(spec.state for spec in EXTRA_POLICY_SPECS)


def is_table_policy_name(policy_name):
    return policy_name in TABLE_POLICY_NAMES or policy_name in TABLE_POLICY_STATES


def parse_args():
    parser = argparse.ArgumentParser(description="Run new-XML table-tennis policies in MuJoCo.")
    parser.add_argument("--start-policy", default="loco", choices=get_policy_choices())
    parser.add_argument(
        "--table-policy",
        default="track_motion_movable_base",
        choices=get_policy_choices(include_base=False),
        help="Policy entered by pressing 't' after loco. Number keys switch this selection.",
    )
    parser.add_argument("--mujoco-config", default="deploy_mujoco/config/g1_track_motion_movable_base.yaml")
    add_planner_args(parser)
    add_serve_args(parser)
    parser.add_argument(
        "--serve-x-range",
        type=float,
        nargs=2,
        default=None,
        help="Optional random serve ball x range. Defaults to fixed --ball-pos x, matching no_joystick timing.",
    )
    parser.add_argument(
        "--serve-y-range",
        type=float,
        nargs=2,
        default=[-0.7625, 0.7625],
        help="Random serve ball y range in world coordinates. Defaults to the full table width.",
    )
    parser.add_argument(
        "--serve-z-range",
        type=float,
        nargs=2,
        default=None,
        help="Optional random serve ball z range. Defaults to fixed --ball-pos z.",
    )
    add_debug_strike_args(parser)
    parser.add_argument("--no-draw-planner", action="store_false", dest="draw_planner")
    parser.set_defaults(draw_planner=True)
    parser.add_argument("--planner-arrow-length", type=float, default=0.35)
    return parser.parse_args()


def load_mujoco_config(config_path):
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    xml_path = Path(cfg["xml_path"])
    if not xml_path.is_absolute():
        xml_path = Path(PROJECT_ROOT) / xml_path
    return str(xml_path), float(cfg["simulation_dt"]), int(cfg["control_decimation"])


def sample_serve_state(args, rng, default_ball_pos, default_ball_vel):
    if args.fixed_initial_ball:
        return default_ball_pos.copy(), default_ball_vel.copy()

    ball_pos = default_ball_pos.copy()
    ball_vel = default_ball_vel.copy()
    y_low, y_high = sorted(float(v) for v in args.serve_y_range)
    if args.serve_x_range is not None:
        x_low, x_high = sorted(float(v) for v in args.serve_x_range)
        ball_pos[0] = rng.uniform(x_low, x_high)
    ball_pos[1] = rng.uniform(y_low, y_high)
    if args.serve_z_range is not None:
        z_low, z_high = sorted(float(v) for v in args.serve_z_range)
        ball_pos[2] = rng.uniform(z_low, z_high)
    return ball_pos, ball_vel


def reset_ball_only(
    model,
    data,
    args,
    rng,
    default_ball_pos,
    default_ball_vel,
    landing_generator,
    respawn_tracker=None,
):
    ball_pos, ball_vel = sample_serve_state(args, rng, default_ball_pos, default_ball_vel)
    initialize_ball_state(model, data, ball_pos, ball_vel)
    landing_generator.reset()
    if respawn_tracker is not None:
        respawn_tracker.reset()
    mujoco.mj_forward(model, data)


def enter_table_policy(
    fsm_controller,
    policy_name,
    model,
    data,
    args,
    rng,
    default_ball_pos,
    default_ball_vel,
    landing_generator,
    respawn_tracker=None,
):
    reset_ball_only(
        model,
        data,
        args,
        rng,
        default_ball_pos,
        default_ball_vel,
        landing_generator,
        respawn_tracker,
    )
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
    respawn_tracker,
):
    if key is None:
        return selected_table_policy
    if key == "p":
        switch_policy(fsm_controller, "passive")
    elif key == "l":
        if is_table_policy_name(fsm_controller.cur_policy.name):
            fsm_controller.state_cmd.skill_cmd = FSMCommand.LOCO
            print("Requested loco through skill_cooldown.")
        else:
            switch_policy(fsm_controller, "loco")
    elif key in TABLE_POLICY_BY_KEY:
        selected_table_policy = TABLE_POLICY_BY_KEY[key]
        print("Selected table policy:", selected_table_policy)
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
                respawn_tracker,
            )
        elif is_table_policy_name(fsm_controller.cur_policy.name):
            print("Currently in a table policy; press 'l' to return to loco, then press the number to enter safely.")
    elif key == "t":
        if fsm_controller.cur_policy.name != FSMStateName.LOCOMODE and not is_table_policy_name(fsm_controller.cur_policy.name):
            print("Enter loco first: press 'l', or press 1/2/3/4/5/6 from loco to enter a table policy.")
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
            respawn_tracker,
        )
    return selected_table_policy


def print_keyboard_help(selected_table_policy):
    print(
        "Keyboard controls: p=passive, l=loco, "
        "1=table_tennis, 2=distill, 3=rev_racket, 4=track_motion_movable_base, 5=track_motion_mjlab, 6=landing_assist_finetune, "
        "t=re-enter selected table policy, r=reset"
    )
    print("Selected table policy:", selected_table_policy)


def add_sphere_marker(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_arrow_marker(scene, start, direction, length, rgba):
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-8 or scene.ngeom >= scene.maxgeom:
        return
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = start + direction / norm * float(length)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.018,
        start,
        end,
    )
    geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
    scene.ngeom += 1


def _get_first_geom_id(model, candidates):
    for name in candidates:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            return gid
    return -1


def _get_racket_body_id(model):
    for name in ("right_racket", "right_wrist_yaw_link", "right_hand"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            return bid
    return -1


def draw_planner_markers(viewer, model, data, fsm_controller, state_cmd, landing_cmd, args):
    viewer.user_scn.ngeom = 0
    if (
        not args.draw_planner
        or landing_cmd is None
        or not landing_cmd.valid
        or fsm_controller.cur_policy.name != TRACK_MOTION_MOVABLE_BASE_STATE
    ):
        return

    hit_pos = np.asarray(landing_cmd.predicted_hit_ball_pos_w, dtype=np.float64)
    base_pos = np.asarray(getattr(state_cmd, "base_pos", np.zeros(3)), dtype=np.float64).reshape(3)
    rel_target = getattr(state_cmd, "rel_racket_target_pos_w", None)
    expected_racket_pos = None
    if rel_target is not None:
        rel_target = np.asarray(rel_target, dtype=np.float64).reshape(3)
        expected_racket_pos = base_pos + rel_target

    geom_id = _get_first_geom_id(model, ("right_hand_collision", "right_racket_collision"))
    actual_racket_pos = None
    actual_normal_w = None
    if geom_id != -1:
        actual_racket_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3)
        xmat = data.geom_xmat[geom_id].reshape(3, 3)
        local_axis = getattr(state_cmd, "racket_face_axis_local", None)
        if local_axis is None:
            local_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            local_axis = np.asarray(local_axis, dtype=np.float64).reshape(3)
        actual_normal_w = xmat @ local_axis

    body_id = _get_racket_body_id(model)
    actual_vel_w = None
    if body_id != -1:
        vel6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel6, 0)
        actual_vel_w = vel6[3:6]

    add_sphere_marker(viewer.user_scn, hit_pos, 0.045, [0.0, 1.0, 0.15, 0.85])
    add_arrow_marker(
        viewer.user_scn,
        hit_pos,
        landing_cmd.desired_ball_dir_w,
        args.planner_arrow_length,
        [0.0, 1.0, 0.15, 0.9],
    )
    if expected_racket_pos is not None:
        add_sphere_marker(viewer.user_scn, expected_racket_pos, 0.030, [1.0, 0.25, 0.85, 0.95])
        add_arrow_marker(
            viewer.user_scn,
            expected_racket_pos,
            landing_cmd.racket_target_vel_w,
            args.planner_arrow_length,
            [1.0, 0.1, 0.05, 0.9],
        )
    desired_normal = getattr(landing_cmd, "desired_racket_normal_w", None)
    if desired_normal is not None and expected_racket_pos is not None:
        add_arrow_marker(
            viewer.user_scn,
            expected_racket_pos,
            desired_normal,
            args.planner_arrow_length,
            [0.2, 0.45, 1.0, 0.9],
        )

    # Actual racket face normal in world frame (cyan) and actual racket
    # linear velocity (yellow), both anchored at the current racket position.
    if actual_racket_pos is not None:
        add_sphere_marker(viewer.user_scn, actual_racket_pos, 0.024, [1.0, 0.75, 0.1, 0.95])
    if actual_racket_pos is not None and actual_normal_w is not None:
        add_arrow_marker(
            viewer.user_scn,
            actual_racket_pos,
            actual_normal_w,
            args.planner_arrow_length,
            [0.0, 1.0, 1.0, 0.95],
        )
    if actual_racket_pos is not None and actual_vel_w is not None:
        add_arrow_marker(
            viewer.user_scn,
            actual_racket_pos,
            actual_vel_w,
            args.planner_arrow_length,
            [1.0, 1.0, 0.0, 0.95],
        )


if __name__ == "__main__":
    args = parse_args()
    xml_path, simulation_dt, control_decimation = load_mujoco_config(args.mujoco_config)
    control_dt = simulation_dt * control_decimation

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt
    num_joints = model.nu
    robot_qpos_slice, robot_qvel_slice = get_robot_state_slices(model)

    rng = np.random.default_rng()
    default_ball_pos = np.array(args.ball_pos, dtype=np.float32)
    default_ball_vel = np.array(args.ball_vel, dtype=np.float32)
    landing_generator = LandingCommandGenerator(args.planner_config)
    apply_debug_strike_overrides(landing_generator, args)
    respawn_tracker = BallRespawnTracker.from_landing_generator(landing_generator)
    initial_ball_pos, initial_ball_vel = sample_serve_state(args, rng, default_ball_pos, default_ball_vel)

    (
        state_cmd,
        policy_output,
        fsm_controller,
        policy_output_action,
        kps,
        kds,
        sim_counter,
    ) = reset_simulation(
        model,
        data,
        args.start_policy,
        robot_qpos_slice,
        robot_qvel_slice,
        num_joints,
        args.base_height,
        initial_ball_pos,
        initial_ball_vel,
        landing_generator,
        force_default_pose=args.force_default_pose,
        use_mujoco_predictor=args.planner_source == "mujoco",
    )
    respawn_tracker.reset()

    reset_requested = [False]
    keyboard_command = [None]
    selected_table_policy = [args.table_policy]
    latest_landing_cmd = [None]

    def key_callback(keycode):
        try:
            key = chr(keycode).lower()
        except ValueError:
            return
        if key == "r":
            reset_requested[0] = True
        elif key in ("p", "l", "t", "1", "2", "3", "4", "5", "6"):
            keyboard_command[0] = key

    print("XML:", xml_path)
    print_keyboard_help(selected_table_policy[0])

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            rally_end_reason = respawn_tracker.update(model, data, simulation_dt)
            if rally_end_reason is not None:
                print(f"[landing] respawn: {rally_end_reason}")

            if reset_requested[0]:
                ball_pos, ball_vel = sample_serve_state(args, rng, default_ball_pos, default_ball_vel)
                with viewer.lock():
                    (
                        state_cmd,
                        policy_output,
                        fsm_controller,
                        policy_output_action,
                        kps,
                        kds,
                        sim_counter,
                    ) = reset_simulation(
                        model,
                        data,
                        args.start_policy,
                        robot_qpos_slice,
                        robot_qvel_slice,
                        num_joints,
                        args.base_height,
                        ball_pos,
                        ball_vel,
                        landing_generator,
                        force_default_pose=args.force_default_pose,
                        use_mujoco_predictor=args.planner_source == "mujoco",
                    )
                respawn_tracker.reset()
                rally_end_reason = None
                reset_requested[0] = False

            if rally_end_reason is not None or ball_is_outside_demo_area(model, data):
                with viewer.lock():
                    reset_ball_only(
                        model,
                        data,
                        args,
                        rng,
                        default_ball_pos,
                        default_ball_vel,
                        landing_generator,
                        respawn_tracker,
                    )

            apply_pd_and_step(
                model, data,
                policy_output_action, kps, kds, policy_output.tau_limit,
                robot_qpos_slice, robot_qvel_slice,
            )
            sim_counter += 1

            if sim_counter % control_decimation == 0:
                # Keyboard runs at control rate so command-driven policy switches
                # take effect before run_control_tick re-queries the FSM state.
                populate_state_cmd(model, data, state_cmd, robot_qpos_slice, robot_qvel_slice)
                selected_table_policy[0] = handle_keyboard_command(
                    keyboard_command[0],
                    fsm_controller,
                    model,
                    data,
                    args,
                    rng,
                    default_ball_pos,
                    default_ball_vel,
                    selected_table_policy[0],
                    landing_generator,
                    respawn_tracker,
                )
                keyboard_command[0] = None

                policy_output_action, kps, kds, landing_cmd = run_control_tick(
                    model, data, state_cmd, policy_output, fsm_controller, landing_generator,
                    robot_qpos_slice, robot_qvel_slice,
                    control_dt=control_dt,
                    episode_time_s=sim_counter * simulation_dt,
                    use_mujoco_predictor=args.planner_source == "mujoco",
                )
                latest_landing_cmd[0] = landing_cmd

                contact_rb = has_racket_ball_contact(model, data)
                contact_table = has_contact(model, data, "ball_geom", "table_top")
                if landing_cmd is not None and args.debug_every > 0:
                    control_tick = sim_counter // control_decimation
                    if control_tick % args.debug_every == 0 or contact_rb:
                        print_debug(state_cmd, landing_cmd, contact_rb, contact_table, model, data)

            draw_planner_markers(viewer, model, data, fsm_controller, state_cmd, latest_landing_cmd[0], args)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
