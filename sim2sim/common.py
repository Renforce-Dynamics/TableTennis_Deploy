from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

"""Shared sim2sim helpers used by tennis_keyboard/joystick entrypoints.

Library only — no ``__main__``. Holds the planner/landing bookkeeping,
ball lifecycle, PD step, control-tick branch, and the shared argparse
builders (``add_serve_args`` / ``add_planner_args`` / ``add_debug_strike_args``).
Blind variants (no ball, no planner) bypass this module.
"""

from common.path_config import PROJECT_ROOT  # noqa: F401  (consumers may re-export)

import os

os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("GLFW_PLATFORM", "x11")

import mujoco
import numpy as np
import yaml

from common.ctrlcomp import PolicyOutput, StateAndCmd
from common.landing_command import apply_landing_command_to_state
from common.policy_registry import get_policy_state, is_landing_policy
from common.utils import get_gravity_orientation
from FSM.FSM import FSM, FSMMode


TABLE_POLICY_DEFAULT = "track_motion_movable_base"
RACKET_GEOM_CANDIDATES = ("right_racket_collision", "right_hand_collision")


class BallRespawnTracker:
    """Track stricter rally-end conditions for deployment re-serves.

    We keep the original out-of-bounds reset as a final fallback, but end the
    rally earlier in two common "finished" cases:
      1) after the robot has contacted the ball, the outgoing ball reaches the net;
      2) the ball stays in sustained table contact and effectively rolls/rests.
    """

    def __init__(
        self,
        *,
        net_x: float = 2.0,
        table_x_min: float = 0.63,
        table_x_max: float = 3.37,
        table_y_min: float = -0.7625,
        table_y_max: float = 0.7625,
        table_z_surface: float = 0.76,
        ball_radius: float = 0.02,
        net_reset_margin: float = 0.02,
        rolling_contact_time_s: float = 0.18,
        rolling_height_tol: float = 0.04,
        rolling_max_abs_vz: float = 0.20,
    ):
        self.net_x = float(net_x)
        self.table_x_min = float(table_x_min)
        self.table_x_max = float(table_x_max)
        self.table_y_min = float(table_y_min)
        self.table_y_max = float(table_y_max)
        self.table_z_surface = float(table_z_surface)
        self.ball_radius = float(ball_radius)
        self.net_reset_margin = float(net_reset_margin)
        self.rolling_contact_time_s = float(rolling_contact_time_s)
        self.rolling_height_tol = float(rolling_height_tol)
        self.rolling_max_abs_vz = float(rolling_max_abs_vz)
        self.reset()

    @classmethod
    def from_landing_generator(cls, landing_generator):
        return cls(
            net_x=float(landing_generator.table.net_x),
            table_x_min=float(landing_generator.table.x_min),
            table_x_max=float(landing_generator.table.x_max),
            table_y_min=float(landing_generator.table.y_min),
            table_y_max=float(landing_generator.table.y_max),
            table_z_surface=float(landing_generator.table.z_surface),
            ball_radius=float(landing_generator.physics.radius),
        )

    def reset(self):
        self._seen_racket_ball_contact = False
        self._table_contact_elapsed_s = 0.0

    def update(self, model, data, dt: float) -> str | None:
        contact_rb = has_racket_ball_contact(model, data)
        contact_table = has_contact(model, data, "ball_geom", "table_top")

        if contact_rb:
            self._seen_racket_ball_contact = True

        ball_pos = np.asarray(get_ball_pos(model, data), dtype=np.float64).reshape(3)
        ball_vel = np.asarray(get_ball_vel(model, data), dtype=np.float64).reshape(3)

        # End the rally once the returned ball reaches the net plane region.
        if self._seen_racket_ball_contact and ball_pos[0] >= (self.net_x - self.net_reset_margin):
            self._table_contact_elapsed_s = 0.0
            return "ball_reached_net_after_hit"

        table_height = self.table_z_surface + self.ball_radius
        within_table_xy = (
            self.table_x_min - 0.05 <= ball_pos[0] <= self.table_x_max + 0.05
            and self.table_y_min - 0.05 <= ball_pos[1] <= self.table_y_max + 0.05
        )
        sustained_table_contact = (
            contact_table
            and within_table_xy
            and abs(ball_pos[2] - table_height) <= self.rolling_height_tol
            and abs(ball_vel[2]) <= self.rolling_max_abs_vz
        )
        if sustained_table_contact:
            self._table_contact_elapsed_s += float(dt)
        else:
            self._table_contact_elapsed_s = 0.0

        if self._table_contact_elapsed_s >= self.rolling_contact_time_s:
            return "ball_rolling_on_table"

        return None


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def load_default_joint_pos(policy_name="track_motion_movable_base"):
    config_by_policy = {
        "track_motion_movable_base": ("track_motion_movable_base", "TrackMotionMovableBase.yaml"),
        "track_motion_mjlab": ("track_motion_mjlab", "TrackMotionMjlab.yaml"),
        "landing_assist_finetune": ("landing_assist_finetune", "LandingAssistFinetune.yaml"),
    }
    policy_dir, filename = config_by_policy.get(policy_name, config_by_policy["track_motion_movable_base"])
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "policy",
        policy_dir,
        "config",
        filename,
    )
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
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


def sample_ball_reset_state(rng, default_ball_pos, default_ball_vel, fixed_initial_ball):
    if fixed_initial_ball:
        return default_ball_pos.copy(), default_ball_vel.copy()
    ball_pos = default_ball_pos.copy()
    ball_vel = default_ball_vel.copy()
    ball_pos[1] += rng.uniform(-0.7625, 0.7625)
    return ball_pos, ball_vel


def ball_is_outside_demo_area(model, data):
    ball_pos = get_ball_pos(model, data)
    return bool(
        ball_pos[0] > 3.8
        or ball_pos[0] < -0.8
        or abs(ball_pos[1]) > 1.5
        or ball_pos[2] < 0.20
        or ball_pos[2] > 2.2
    )


def quat_rotate_inverse(quat_wxyz, vec_xyz):
    qw, qx, qy, qz = quat_wxyz
    qvec = np.array([qx, qy, qz], dtype=np.float32)
    vec = np.asarray(vec_xyz, dtype=np.float32)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec - 2.0 * (qw * uv + uuv)


def yaw_rotate_inverse(quat_wxyz, vec_xyz):
    qw, qx, qy, qz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    x, y, z = np.asarray(vec_xyz, dtype=np.float64).reshape(3)
    return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float64)


def populate_state_cmd(model, data, state_cmd, robot_qpos_slice, robot_qvel_slice):
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
    state_cmd.base_quat = quat.copy()
    state_cmd.ang_vel = omega.copy()
    state_cmd.gravity_ori = gravity_orientation.copy()
    state_cmd.ball_pos = get_ball_pos(model, data)
    state_cmd.ball_vel = get_ball_vel(model, data)


def apply_initial_configuration(
    model,
    data,
    robot_qpos_slice,
    base_height,
    ball_pos,
    ball_vel,
    *,
    policy_name="track_motion_movable_base",
    force_default_pose=False,
):
    """Initialize exactly like the proven track-motion deployment path by default.

    sim2sim/blind_keyboard.py leaves the robot at
    the XML qpos0 for track-motion policies and only initializes the ball/marker.
    Forcing default_angles and a different base height here changes the startup
    distribution and can cause leg split/fall even with the same ONNX.
    """
    if force_default_pose:
        data.qpos[2] = base_height
        data.qpos[robot_qpos_slice] = load_default_joint_pos(policy_name)
        data.qvel[:] = 0.0
    initialize_ball_state(model, data, ball_pos, ball_vel)
    mujoco.mj_forward(model, data)


def switch_policy(fsm_controller, policy_name):
    target_state = get_policy_state(policy_name) if isinstance(policy_name, str) else policy_name
    if fsm_controller.cur_policy.name == target_state or fsm_controller.cur_policy.name == policy_name:
        return True
    previous_policy = fsm_controller.cur_policy
    previous_policy.exit()
    fsm_controller.get_next_policy(target_state)
    if fsm_controller.cur_policy is previous_policy and previous_policy.name != target_state:
        print("Policy is unavailable:", policy_name)
        return False
    fsm_controller.cur_policy.enter()
    fsm_controller.FSMmode = FSMMode.NORMAL
    print("Switched to", fsm_controller.cur_policy.name_str)
    return True


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
    force_default_pose=False,
    use_mujoco_predictor=False,
):
    mujoco.mj_resetData(model, data)
    apply_initial_configuration(
        model,
        data,
        robot_qpos_slice,
        base_height,
        ball_pos,
        ball_vel,
        policy_name=start_policy,
        force_default_pose=force_default_pose,
    )
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

    fsm_controller.get_next_policy(get_policy_state(start_policy))
    fsm_controller.cur_policy.enter()
    populate_state_cmd(model, data, state_cmd, robot_qpos_slice, robot_qvel_slice)

    # Prime the command-conditioned policy with the same external command bridge
    # before the first ONNX inference. Without this, the first action uses the
    # policy's internal random command and then abruptly switches to planner
    # command one control tick later.
    if is_landing_policy(fsm_controller.cur_policy.name):
        landing_cmd = update_landing_command(
            model,
            data,
            state_cmd,
            landing_generator,
            control_dt=0.0,
            episode_time_s=0.0,
            use_mujoco_predictor=use_mujoco_predictor,
        )
        apply_landing_command_to_state(state_cmd, landing_cmd)

    fsm_controller.run()
    policy_output_action = policy_output.actions.copy()
    kps = policy_output.kps.copy()
    kds = policy_output.kds.copy()
    print("Simulation reset. Current policy:", fsm_controller.cur_policy.name_str)
    return state_cmd, policy_output, fsm_controller, policy_output_action, kps, kds, sim_counter


def geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def has_contact(model, data, geom_name_a, geom_name_b):
    gid_a = geom_id(model, geom_name_a)
    gid_b = geom_id(model, geom_name_b)
    if gid_a == -1 or gid_b == -1:
        return False
    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 == gid_a and con.geom2 == gid_b) or (con.geom1 == gid_b and con.geom2 == gid_a):
            return True
    return False


def has_racket_ball_contact(model, data):
    for racket_geom in RACKET_GEOM_CANDIDATES:
        if has_contact(model, data, "ball_geom", racket_geom):
            return True
    return False


def predict_ball_hit_mujoco(model, data, x_hit, max_predict_time):
    """Roll out the current MuJoCo scene to predict the ball crossing x_hit."""
    ball_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    if ball_joint_id == -1:
        return None

    pred = mujoco.MjData(model)
    pred.qpos[:] = data.qpos
    pred.qvel[:] = data.qvel
    pred.ctrl[:] = data.ctrl
    pred.qfrc_applied[:] = data.qfrc_applied
    pred.xfrc_applied[:] = data.xfrc_applied
    mujoco.mj_forward(model, pred)

    last_pos = get_ball_pos(model, pred).astype(np.float64)
    last_vel = get_ball_vel(model, pred).astype(np.float64)
    if last_vel[0] >= -1.0e-6 or last_pos[0] <= x_hit:
        return None

    max_steps = max(1, int(float(max_predict_time) / float(model.opt.timestep)))
    for step in range(1, max_steps + 1):
        mujoco.mj_step(model, pred)
        pos = get_ball_pos(model, pred).astype(np.float64)
        vel = get_ball_vel(model, pred).astype(np.float64)
        if last_pos[0] > x_hit and pos[0] <= x_hit and vel[0] < 0.0:
            dx = float(pos[0] - last_pos[0])
            frac = float((x_hit - last_pos[0]) / dx) if abs(dx) > 1.0e-9 else 1.0
            frac = float(np.clip(frac, 0.0, 1.0))
            hit_pos = last_pos + frac * (pos - last_pos)
            hit_vel = last_vel + frac * (vel - last_vel)
            hit_pos[0] = x_hit
            time_to_hit = (float(step) - 1.0 + frac) * float(model.opt.timestep)
            return hit_pos.astype(np.float32), hit_vel.astype(np.float32), time_to_hit
        last_pos = pos
        last_vel = vel
    return None


def update_landing_command(
    model,
    data,
    state_cmd,
    landing_generator,
    control_dt,
    episode_time_s,
    *,
    use_mujoco_predictor=False,
):
    if use_mujoco_predictor:
        prediction = predict_ball_hit_mujoco(
            model,
            data,
            landing_generator.cfg.x_hit,
            landing_generator.cfg.max_predict_time,
        )
        if prediction is not None:
            hit_pos, hit_vel, time_to_hit = prediction
            return landing_generator.update_from_prediction(
                predicted_hit_ball_pos_w=hit_pos,
                predicted_hit_ball_vel_w=hit_vel,
                time_to_hit_s=time_to_hit,
                ball_pos_w=state_cmd.ball_pos,
                ball_vel_w=state_cmd.ball_vel,
                robot_base_pos_w=state_cmd.base_pos,
                robot_base_quat_wxyz=state_cmd.base_quat,
                dt=control_dt,
                episode_time_s=episode_time_s,
            )

    return landing_generator.update(
        ball_pos_w=state_cmd.ball_pos,
        ball_vel_w=state_cmd.ball_vel,
        robot_base_pos_w=state_cmd.base_pos,
        robot_base_quat_wxyz=state_cmd.base_quat,
        dt=control_dt,
        episode_time_s=episode_time_s,
    )


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        return None
    return vec / norm


def _racket_rot_debug(model, data, state_cmd, landing_cmd):
    target = _unit(landing_cmd.racket_target_vel_w)
    if target is None:
        return None

    geom_id = -1
    geom_name = ""
    for candidate in RACKET_GEOM_CANDIDATES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, candidate)
        if geom_id != -1:
            geom_name = candidate
            break
    if geom_id == -1:
        return None

    xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    local_z_w = _unit(xmat[:, 2])
    if local_z_w is None:
        return None

    face_axis = getattr(state_cmd, "racket_face_axis_local", None)
    face_axis_name = getattr(state_cmd, "racket_face_axis_name", None)
    if face_axis is None:
        face_axis = np.array([0.0, 0.0, 1.0 if getattr(state_cmd, "is_forehand", True) else -1.0])
        face_axis_name = "z" if getattr(state_cmd, "is_forehand", True) else "-z"
    face_axis = np.asarray(face_axis, dtype=np.float64).reshape(3)
    selected_normal = _unit(xmat @ face_axis)
    if selected_normal is None:
        return None

    dot_pos_z = float(np.dot(local_z_w, target))
    dot_neg_z = float(np.dot(-local_z_w, target))
    dot_selected = float(np.dot(selected_normal, target))
    vel_w = np.asarray(landing_cmd.racket_target_vel_w, dtype=np.float64).reshape(3)
    normal_w = np.asarray(getattr(landing_cmd, "desired_racket_normal_w", vel_w), dtype=np.float64).reshape(3)
    normal_target = _unit(normal_w)
    dot_normal = float(np.dot(selected_normal, normal_target)) if normal_target is not None else 0.0
    vel_yaw_b = yaw_rotate_inverse(getattr(state_cmd, "base_quat", np.array([1.0, 0.0, 0.0, 0.0])), vel_w)
    normal_yaw_b = yaw_rotate_inverse(getattr(state_cmd, "base_quat", np.array([1.0, 0.0, 0.0, 0.0])), normal_w)
    dot_for_angle = float(np.clip(dot_selected, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot_for_angle)))
    return (
        " geom={} stroke={} face_axis={} dot(+Z,target)={:.3f} "
        "dot(-Z,target)={:.3f} dot(selected,target)={:.3f} angle_deg={:.1f} "
        "dot(selected,normal)={:.3f} vel_yaw_b={} normal_yaw_b={}"
    ).format(
        geom_name,
        "forehand" if getattr(state_cmd, "is_forehand", True) else "backhand",
        face_axis_name,
        dot_pos_z,
        dot_neg_z,
        dot_selected,
        angle_deg,
        dot_normal,
        np.round(vel_yaw_b, 3).tolist(),
        np.round(normal_yaw_b, 3).tolist(),
    )


def print_debug(state_cmd, landing_cmd, contact_ball_racket, contact_ball_table, model=None, data=None):
    rot_debug = ""
    if model is not None and data is not None:
        rot_msg = _racket_rot_debug(model, data, state_cmd, landing_cmd)
        if rot_msg is not None:
            rot_debug = " rot:{}".format(rot_msg)
    print(
        "[landing] valid={} reason={} t_hit={:.3f} base_tgt={} racket_pos={} racket_vel={} "
        "racket_normal={} ball={} vel={} hit={} contact_rb={} contact_table={}{}".format(
            landing_cmd.valid,
            landing_cmd.reason,
            float(landing_cmd.racket_target_time[0]),
            np.round(landing_cmd.base_pos_target, 3).tolist(),
            np.round(landing_cmd.rel_racket_target_pos_w, 3).tolist(),
            np.round(landing_cmd.racket_target_vel_w, 3).tolist(),
            np.round(getattr(landing_cmd, "desired_racket_normal_w", np.zeros(3)), 3).tolist(),
            np.round(state_cmd.ball_pos, 3).tolist(),
            np.round(state_cmd.ball_vel, 3).tolist(),
            np.round(landing_cmd.predicted_hit_ball_pos_w, 3).tolist(),
            contact_ball_racket,
            contact_ball_table,
            rot_debug,
        )
    )


def add_serve_args(parser, *, default_ball_pos=(3.5, -0.2, 1.0), default_ball_vel=(-4.0, 0.0, 0.0)):
    """Initial ball pose + base height. Shared across all sim2sim entrypoints."""
    parser.add_argument("--ball-pos", type=float, nargs=3, default=list(default_ball_pos))
    parser.add_argument("--ball-vel", type=float, nargs=3, default=list(default_ball_vel))
    parser.add_argument("--fixed-initial-ball", action="store_true")
    parser.add_argument(
        "--base-height",
        type=float,
        default=0.76,
        help="Initial robot base height; 0.76 matches the mjhitter training init_state.",
    )
    parser.add_argument(
        "--force-default-pose",
        action="store_true",
        help="Force default_angles/base_height at reset. Off by default to match the proven track-motion deploy path.",
    )


def add_planner_args(parser):
    """Landing-planner config + source + verbose printing."""
    parser.add_argument("--planner-config", default="configs/planner/landing_planner.yaml")
    parser.add_argument(
        "--planner-source",
        choices=["mujoco", "model"],
        default="model",
        help="Use the model-based planner from LandingAssistFinetune by default; choose mujoco only for rollout-based comparison.",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=10,
        help="Print planner status every N control ticks; 0 disables.",
    )


def add_debug_strike_args(parser):
    """Optional debug overrides that pin the commanded strike target."""
    parser.add_argument(
        "--debug-fixed-hit-y",
        type=float,
        default=None,
        help="Debug override: force the commanded hit relative-y to this value, e.g. -0.4.",
    )
    parser.add_argument(
        "--debug-front-speed",
        type=float,
        default=1.5,
        help="Debug override speed magnitude for the front-facing strike command.",
    )
    parser.add_argument(
        "--debug-front-pitch-deg",
        type=float,
        default=0.0,
        help="Debug override pitch angle in degrees for the front-facing strike command.",
    )
    parser.add_argument(
        "--debug-fixed-hit-x",
        type=float,
        default=None,
        help="Optional debug override for commanded relative hit x.",
    )
    parser.add_argument(
        "--debug-fixed-hit-z",
        type=float,
        default=None,
        help="Optional debug override for commanded relative hit z.",
    )


def apply_debug_strike_overrides(landing_generator, args):
    """If --debug-fixed-hit-y is supplied, lock the planner to a front strike."""
    if args.debug_fixed_hit_y is None:
        return
    landing_generator.configure_debug_fixed_front_strike(
        enabled=True,
        rel_hit_y=args.debug_fixed_hit_y,
        front_speed=args.debug_front_speed,
        front_pitch_deg=args.debug_front_pitch_deg,
        rel_hit_x=args.debug_fixed_hit_x,
        rel_hit_z=args.debug_fixed_hit_z,
    )
    print(
        "[landing] debug fixed front strike enabled:",
        f"rel_hit_y={args.debug_fixed_hit_y}",
        f"speed={args.debug_front_speed}",
        f"pitch_deg={args.debug_front_pitch_deg}",
    )


def clear_state_cmd_planner_fields(state_cmd):
    """Wipe planner-only fields so non-landing policies don't read stale targets."""
    state_cmd.base_pos_target = None
    state_cmd.rel_racket_target_pos_w = None
    state_cmd.racket_target_vel_w = None
    state_cmd.racket_target_time = None
    state_cmd.planner_valid = False


def apply_pd_and_step(model, data, target_q, kps, kds, tau_limit, robot_qpos_slice, robot_qvel_slice):
    """Compute PD torque, clip to tau_limit, write ctrl, advance one mj_step."""
    tau = pd_control(
        target_q,
        data.qpos[robot_qpos_slice],
        kps,
        np.zeros_like(kps),
        data.qvel[robot_qvel_slice],
        kds,
    )
    if np.any(tau_limit > 0.0):
        tau = np.clip(tau, -tau_limit, tau_limit)
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)


def run_control_tick(
    model,
    data,
    state_cmd,
    policy_output,
    fsm_controller,
    landing_generator,
    robot_qpos_slice,
    robot_qvel_slice,
    *,
    control_dt,
    episode_time_s,
    use_mujoco_predictor,
):
    """One control-rate update: refresh obs, push planner cmd, run policy.

    Returns ``(policy_action, kps, kds, landing_cmd_or_None)``.
    """
    populate_state_cmd(model, data, state_cmd, robot_qpos_slice, robot_qvel_slice)
    state_cmd.vel_cmd[:] = 0.0

    landing_cmd = None
    if is_landing_policy(fsm_controller.cur_policy.name):
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
    else:
        clear_state_cmd_planner_fields(state_cmd)

    fsm_controller.run()
    return (
        policy_output.actions.copy(),
        policy_output.kps.copy(),
        policy_output.kds.copy(),
        landing_cmd,
    )


# ---------------------------------------------------------------------------
# Planner debug-marker drawing (shared by both tennis entrypoints)
# ---------------------------------------------------------------------------


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


def draw_planner_markers(
    viewer,
    model,
    data,
    fsm_controller,
    state_cmd,
    landing_cmd,
    *,
    enabled: bool = True,
    arrow_length: float = 0.35,
):
    """Render planner-side debug markers in the viewer's user_scn.

    Only active for FSM states that consume the LandingCommand bridge
    (``is_landing_policy``). Tennis entrypoints call this each viewer tick.
    """
    viewer.user_scn.ngeom = 0
    if (
        not enabled
        or landing_cmd is None
        or not landing_cmd.valid
        or not is_landing_policy(fsm_controller.cur_policy.name)
    ):
        return

    hit_pos = np.asarray(landing_cmd.predicted_hit_ball_pos_w, dtype=np.float64)
    base_pos = np.asarray(getattr(state_cmd, "base_pos", np.zeros(3)), dtype=np.float64).reshape(3)
    rel_target = getattr(state_cmd, "rel_racket_target_pos_w", None)
    expected_racket_pos = None
    if rel_target is not None:
        rel_target = np.asarray(rel_target, dtype=np.float64).reshape(3)
        expected_racket_pos = base_pos + rel_target

    geom_id = _get_first_geom_id(model, RACKET_GEOM_CANDIDATES)
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
        arrow_length,
        [0.0, 1.0, 0.15, 0.9],
    )
    if expected_racket_pos is not None:
        add_sphere_marker(viewer.user_scn, expected_racket_pos, 0.030, [1.0, 0.25, 0.85, 0.95])
        add_arrow_marker(
            viewer.user_scn,
            expected_racket_pos,
            landing_cmd.racket_target_vel_w,
            arrow_length,
            [1.0, 0.1, 0.05, 0.9],
        )
    desired_normal = getattr(landing_cmd, "desired_racket_normal_w", None)
    if desired_normal is not None and expected_racket_pos is not None:
        add_arrow_marker(
            viewer.user_scn,
            expected_racket_pos,
            desired_normal,
            arrow_length,
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
            arrow_length,
            [0.0, 1.0, 1.0, 0.95],
        )
    if actual_racket_pos is not None and actual_vel_w is not None:
        add_arrow_marker(
            viewer.user_scn,
            actual_racket_pos,
            actual_vel_w,
            arrow_length,
            [1.0, 1.0, 0.0, 0.95],
        )
