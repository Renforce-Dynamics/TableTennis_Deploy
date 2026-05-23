"""Joystick variant of tennis_keyboard.

Stub: shares all simulation scaffolding (tennis XML scene, planner, ball
lifecycle, respawn tracker, planner debug viz) with tennis_keyboard.
The keyboard command-handling block is replaced by a JoyStick poll that maps
buttons → FSMCommand skill_cmd.

Button map (mirrors blind_joystick + adds the 3 newer table policies
via D-pad):
  SELECT             -> exit
  L3                 -> PASSIVE
  START              -> full reset (robot + ball + respawn tracker)
  D-pad UP           -> re-serve ball (rally reset, keep policy)
  A + R1             -> LOCO
  X + R1 / Y + R1 / B + R1   -> SKILL_1 / SKILL_2 / SKILL_3
  B + L1             -> TABLE_TENNIS
  A + L1             -> TRACK_MOTION_ISAACLAB
  X + L1             -> TRACK_MOTION_MJLAB
  Y + L1             -> TRACK_MOTION_MOVABLE_BASE
  D-pad LEFT         -> TABLE_TENNIS_DISTILL
  D-pad RIGHT        -> TABLE_TENNIS_REV_RACKET
  D-pad DOWN         -> LANDING_ASSIST_FINETUNE
  axes 1/0/3         -> vel_cmd x/y/yaw (matches blind_joystick sign)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

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

from common.joystick import JoyStick, JoystickButton
from common.landing_command import LandingCommandGenerator
from common.policy_registry import get_policy_choices
from common.utils import FSMCommand

from sim2sim.tennis_keyboard import (
    load_mujoco_config,
    reset_ball_only,
    sample_serve_state,
)
from sim2sim.common import (
    BallRespawnTracker,
    add_debug_strike_args,
    add_planner_args,
    add_serve_args,
    apply_debug_strike_overrides,
    apply_pd_and_step,
    ball_is_outside_demo_area,
    draw_planner_markers,
    get_robot_state_slices,
    has_contact,
    has_racket_ball_contact,
    populate_state_cmd,
    print_debug,
    reset_simulation,
    run_control_tick,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run table-tennis policies in MuJoCo with a joystick.")
    parser.add_argument("--start-policy", default="loco", choices=get_policy_choices())
    parser.add_argument("--mujoco-config", default="configs/sim/g1_track_motion_movable_base.yaml")
    add_planner_args(parser)
    add_serve_args(parser)
    parser.add_argument("--serve-x-range", type=float, nargs=2, default=None)
    parser.add_argument("--serve-y-range", type=float, nargs=2, default=[-0.7625, 0.7625])
    parser.add_argument("--serve-z-range", type=float, nargs=2, default=None)
    add_debug_strike_args(parser)
    parser.add_argument("--no-draw-planner", action="store_false", dest="draw_planner")
    parser.set_defaults(draw_planner=True)
    parser.add_argument("--planner-arrow-length", type=float, default=0.35)
    return parser.parse_args()


def poll_joystick_commands(joystick, state_cmd):
    """Translate joystick state into FSMCommand on state_cmd.skill_cmd.

    Returns one of {None, "exit", "reset_full", "reset_ball"} for top-level
    control signals that the outer loop must act on.
    """
    if joystick.is_button_pressed(JoystickButton.SELECT):
        return "exit"

    joystick.update()

    if joystick.is_button_released(JoystickButton.START):
        return "reset_full"
    if joystick.is_button_released(JoystickButton.UP):
        return "reset_ball"

    if joystick.is_button_released(JoystickButton.L3):
        state_cmd.skill_cmd = FSMCommand.PASSIVE

    r1 = joystick.is_button_pressed(JoystickButton.R1)
    l1 = joystick.is_button_pressed(JoystickButton.L1)
    if r1:
        if joystick.is_button_released(JoystickButton.A):
            state_cmd.skill_cmd = FSMCommand.LOCO
        if joystick.is_button_released(JoystickButton.X):
            state_cmd.skill_cmd = FSMCommand.SKILL_1
        if joystick.is_button_released(JoystickButton.Y):
            state_cmd.skill_cmd = FSMCommand.SKILL_2
        if joystick.is_button_released(JoystickButton.B):
            state_cmd.skill_cmd = FSMCommand.SKILL_3
    if l1:
        if joystick.is_button_released(JoystickButton.B):
            state_cmd.skill_cmd = FSMCommand.TABLE_TENNIS
        if joystick.is_button_released(JoystickButton.A):
            state_cmd.skill_cmd = FSMCommand.TRACK_MOTION_ISAACLAB
        if joystick.is_button_released(JoystickButton.X):
            state_cmd.skill_cmd = FSMCommand.TRACK_MOTION_MJLAB
        if joystick.is_button_released(JoystickButton.Y):
            state_cmd.skill_cmd = FSMCommand.TRACK_MOTION_MOVABLE_BASE

    if joystick.is_button_released(JoystickButton.LEFT):
        state_cmd.skill_cmd = FSMCommand.TABLE_TENNIS_DISTILL
    if joystick.is_button_released(JoystickButton.RIGHT):
        state_cmd.skill_cmd = FSMCommand.TABLE_TENNIS_REV_RACKET
    if joystick.is_button_released(JoystickButton.DOWN):
        state_cmd.skill_cmd = FSMCommand.LANDING_ASSIST_FINETUNE

    state_cmd.vel_cmd[0] = -joystick.get_axis_value(1)
    state_cmd.vel_cmd[1] = -joystick.get_axis_value(0)
    state_cmd.vel_cmd[2] = -joystick.get_axis_value(3)
    return None


def print_joystick_help():
    print("Joystick controls:")
    print("  SELECT=exit, L3=PASSIVE, START=full reset, D-pad UP=re-serve ball")
    print("  A+R1=LOCO, X/Y/B+R1=SKILL_1/2/3")
    print("  B+L1=TABLE_TENNIS, A+L1=TRACK_MOTION_ISAACLAB,")
    print("  X+L1=TRACK_MOTION_MJLAB, Y+L1=TRACK_MOTION_MOVABLE_BASE")
    print("  D-pad LEFT/RIGHT/DOWN = DISTILL / REV_RACKET / LANDING_ASSIST_FINETUNE")


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

    joystick = JoyStick()
    latest_landing_cmd = None

    print("XML:", xml_path)
    print_joystick_help()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        running = True
        while viewer.is_running() and running:
            step_start = time.time()
            try:
                signal = poll_joystick_commands(joystick, state_cmd)
            except ValueError as e:
                print(str(e))
                signal = None

            if signal == "exit":
                running = False
                continue

            rally_end_reason = respawn_tracker.update(model, data, simulation_dt)
            if rally_end_reason is not None:
                print(f"[landing] respawn: {rally_end_reason}")

            if signal == "reset_full":
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
                        model, data,
                        args.start_policy,
                        robot_qpos_slice, robot_qvel_slice,
                        num_joints, args.base_height,
                        ball_pos, ball_vel,
                        landing_generator,
                        force_default_pose=args.force_default_pose,
                        use_mujoco_predictor=args.planner_source == "mujoco",
                    )
                respawn_tracker.reset()
                rally_end_reason = None
            elif signal == "reset_ball" or rally_end_reason is not None or ball_is_outside_demo_area(model, data):
                with viewer.lock():
                    reset_ball_only(
                        model, data, args, rng,
                        default_ball_pos, default_ball_vel,
                        landing_generator, respawn_tracker,
                    )

            apply_pd_and_step(
                model, data,
                policy_output_action, kps, kds, policy_output.tau_limit,
                robot_qpos_slice, robot_qvel_slice,
            )
            sim_counter += 1

            if sim_counter % control_decimation == 0:
                populate_state_cmd(model, data, state_cmd, robot_qpos_slice, robot_qvel_slice)
                policy_output_action, kps, kds, landing_cmd = run_control_tick(
                    model, data, state_cmd, policy_output, fsm_controller, landing_generator,
                    robot_qpos_slice, robot_qvel_slice,
                    control_dt=control_dt,
                    episode_time_s=sim_counter * simulation_dt,
                    use_mujoco_predictor=args.planner_source == "mujoco",
                )
                latest_landing_cmd = landing_cmd

                contact_rb = has_racket_ball_contact(model, data)
                contact_table = has_contact(model, data, "ball_geom", "table_top")
                if landing_cmd is not None and args.debug_every > 0:
                    control_tick = sim_counter // control_decimation
                    if control_tick % args.debug_every == 0 or contact_rb:
                        print_debug(state_cmd, landing_cmd, contact_rb, contact_table, model, data)

            draw_planner_markers(
                viewer, model, data, fsm_controller, state_cmd, latest_landing_cmd,
                enabled=args.draw_planner,
                arrow_length=args.planner_arrow_length,
            )
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
