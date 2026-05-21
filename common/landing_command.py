from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.path_config import PROJECT_ROOT
from planner.common import BallPhysics, PlannerConfig, TableParams
from planner.hope_model_based_planner import HopeModelBasedPlanner
from planner.hope_planner import HOPEPlanner


@dataclass
class LandingCommandOutput:
    valid: bool
    base_pos_target: np.ndarray
    rel_racket_target_pos_w: np.ndarray
    racket_target_vel_w: np.ndarray
    racket_target_time: np.ndarray
    predicted_hit_ball_pos_w: np.ndarray
    predicted_hit_ball_vel_w: np.ndarray
    target_landing_pos_w: np.ndarray
    desired_ball_dir_w: np.ndarray
    desired_racket_normal_w: np.ndarray
    is_forehand: bool
    racket_face_axis_name: str
    racket_face_axis_local: np.ndarray
    reason: str = ""


class LandingCommandGenerator:
    """Numpy single-robot landing command generator for deployment.

    It converts ball state into the 4 command fields consumed by the
    command-conditioned track-motion policies:

      - base_pos_target, shape [2]
      - rel_racket_target_pos_w, shape [3]
      - racket_target_vel_w, shape [3]
      - racket_target_time, shape [1]

    The implementation deliberately stays independent of mjlab CommandTerm so the
    same class can be used in MuJoCo sim2sim and real-robot deployment.
    """

    def __init__(self, config_path: str | Path | None = None, config: dict[str, Any] | None = None):
        if config is None:
            config = self._load_config(config_path)
        self.raw_config = config

        planner_cfg = config.get("planner", {}) or {}
        table_cfg = config.get("table", {}) or {}
        command_cfg = config.get("command", {}) or {}
        target_cfg = config.get("landing_target", {}) or {}

        self.forehand_y_range = self._range(command_cfg.get("forehand_y_range", [-0.60, -0.35]))
        self.backhand_y_range = self._range(command_cfg.get("backhand_y_range", [-0.10, 0.30]))
        self.forehand_pos_offset = self._vec3(command_cfg.get("forehand_pos_offset", [0.02, -0.01, 0.0]))
        self.backhand_pos_offset = self._vec3(command_cfg.get("backhand_pos_offset", [0.02, 0.01, 0.0]))
        self.base_out_of_range_gain = float(command_cfg.get("base_out_of_range_gain", 2.0))
        self.stroke_side_in_base_frame = bool(command_cfg.get("stroke_side_in_base_frame", False))

        self.planner_type = str(planner_cfg.get("type", "model_based")).lower()
        self.table = TableParams(
            x_min=float(table_cfg.get("x_min", 0.63)),
            x_max=float(table_cfg.get("x_max", 3.37)),
            y_min=float(table_cfg.get("y_min", -0.7625)),
            y_max=float(table_cfg.get("y_max", 0.7625)),
            z_surface=float(table_cfg.get("z_surface", 0.76)),
            net_x=float(table_cfg.get("net_x", 2.0)),
            net_height=float(table_cfg.get("net_height", 0.1525)),
            net_overhang=float(table_cfg.get("net_overhang", 0.15)),
        )
        self.physics = BallPhysics(
            k=float(planner_cfg.get("drag_k", 0.0)),
            c_h=float(table_cfg.get("restitution_h", 0.75)),
            c_v=float(table_cfg.get("restitution_v", 0.85)),
            g=(0.0, 0.0, float(planner_cfg.get("gravity_z", -9.81))),
            radius=float(table_cfg.get("ball_radius", 0.02)),
        )
        self.cfg = PlannerConfig(
            dt_integrate=float(planner_cfg.get("dt_integrate", 0.001)),
            max_predict_time=float(planner_cfg.get("max_predict_time", 2.0)),
            x_hit=float(planner_cfg.get("x_hit", 0.40)),
            delta_t_flight=float(planner_cfg.get("delta_t_flight", 0.50)),
            c_r=float(planner_cfg.get("racket_restitution", 0.88)),
            fit_window=int(planner_cfg.get("fit_window", 31)),
            poly_order=int(planner_cfg.get("poly_order", 2)),
            bounce_z_tol=float(planner_cfg.get("bounce_z_tol", 0.02)),
            max_table_bounces=int(planner_cfg.get("max_table_bounces", 8)),
            fh_rel_y_min=float(self.forehand_y_range[0]),
            fh_rel_y_max=float(self.forehand_y_range[1]),
            bh_rel_y_min=float(self.backhand_y_range[0]),
            bh_rel_y_max=float(self.backhand_y_range[1]),
            base_out_of_range_gain=float(self.base_out_of_range_gain),
        )
        self.model_planner = HopeModelBasedPlanner(self.table, self.physics, self.cfg)
        self.hope_planner = HOPEPlanner(self.physics, self.cfg, self.table)

        self.target_land_xy = np.array(
            [float(target_cfg.get("pos_x", 2.68)), float(target_cfg.get("pos_y", 0.0))],
            dtype=np.float64,
        )
        self.activate_before_hit_s = float(command_cfg.get("activate_before_hit_s", 1.50))
        self.min_time_to_hit_s = float(command_cfg.get("min_time_to_hit_s", 0.03))
        self.max_time_to_hit_s = float(command_cfg.get("max_time_to_hit_s", 2.0))
        self.command_hold_s = float(command_cfg.get("command_hold_s", 0.10))
        self.advance_hit_time_s = float(command_cfg.get("advance_hit_time_s", 0.0))
        self.post_hit_x_margin = float(command_cfg.get("post_hit_x_margin", 0.0))
        self.incoming_vx_threshold = float(command_cfg.get("incoming_vx_threshold", -0.05))
        self.inactive_time_s = float(command_cfg.get("inactive_time_s", self.activate_before_hit_s))
        self.inactive_rel_racket_target_pos_w = self._vec3(
            command_cfg.get("inactive_rel_racket_target_pos_w", [0.40, -0.10, 0.25])
        )
        self.inactive_racket_target_vel_w = self._vec3(
            command_cfg.get("inactive_racket_target_vel_w", [0.0, 0.0, 0.0])
        )
        self._no_hold_invalid_reasons = {"passed_hit", "time_to_hit_expired"}

        self.axis_map = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
            "-z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        self.forehand_axis = str(command_cfg.get("forehand_axis", "z"))
        self.backhand_axis = str(command_cfg.get("backhand_axis", "-z"))
        for axis_name in (self.forehand_axis, self.backhand_axis):
            if axis_name not in self.axis_map:
                raise ValueError(f"unsupported racket face axis '{axis_name}'")
        self.command_rng = np.random.default_rng(command_cfg.get("command_seed", None))
        self.forehand_racket_target_pose_range = self._load_racket_target_pose_range(
            command_cfg.get("forehand_racket_target_pose_range", {}),
            fallback_pos=np.array([0.40, -0.45, 0.25], dtype=np.float64),
            fallback_vel=np.array([1.50, 0.0, 0.25], dtype=np.float64),
            fallback_pos_y=self.forehand_y_range,
        )
        self.backhand_racket_target_pose_range = self._load_racket_target_pose_range(
            command_cfg.get("backhand_racket_target_pose_range", {}),
            fallback_pos=np.array([0.40, 0.10, 0.25], dtype=np.float64),
            fallback_vel=np.array([1.50, 0.0, 0.25], dtype=np.float64),
            fallback_pos_y=self.backhand_y_range,
        )
        self.pos_blend_min = float(command_cfg.get("pos_blend_min", command_cfg.get("pos_blend_alpha", 0.20)))
        self.pos_blend_max = float(command_cfg.get("pos_blend_max", command_cfg.get("pos_blend_alpha", 0.85)))
        self.vel_blend_min = float(command_cfg.get("vel_blend_min", command_cfg.get("vel_blend_alpha", 0.10)))
        self.vel_blend_max = float(command_cfg.get("vel_blend_max", command_cfg.get("vel_blend_alpha", 0.50)))
        directional_cfg = command_cfg.get("directional_velocity_sampling", {}) or {}
        self.directional_velocity_sampling_enabled = bool(
            directional_cfg.get("enabled", True)
        )
        self.directional_pitch_deg_range = self._range(
            directional_cfg.get("pitch_deg_range", [-30.0, 45.0])
        )
        self.directional_yaw_deg_range = self._range(
            directional_cfg.get("yaw_deg_range", [0.0, 0.0])
        )
        self.directional_speed_scale_range = self._range(
            directional_cfg.get("speed_scale_range", [0.85, 1.20])
        )
        self.directional_min_speed = float(directional_cfg.get("min_speed", 0.8))
        self.directional_max_speed = float(directional_cfg.get("max_speed", 2.6))

        self.base_target_x = float(command_cfg.get("base_target_x", 0.0))
        self.base_target_y_min = float(command_cfg.get("base_target_y_min", -1.2))
        self.base_target_y_max = float(command_cfg.get("base_target_y_max", 1.2))
        self.base_target_max_delta = float(command_cfg.get("base_target_max_delta", 0.06))

        self.racket_speed_min = float(command_cfg.get("racket_speed_min", 0.0))
        self.racket_speed_max = float(command_cfg.get("racket_speed_max", 3.0))
        self.base_blend_alpha = float(command_cfg.get("base_blend_alpha", 0.35))
        self.align_racket_velocity_to_normal = bool(command_cfg.get("align_racket_velocity_to_normal", False))
        self.front_facing_lateral_scale = float(command_cfg.get("front_facing_lateral_scale", 0.25))
        debug_front_cfg = command_cfg.get("debug_fixed_front_strike", {}) or {}
        self.debug_fixed_front_strike_enabled = bool(debug_front_cfg.get("enabled", False))
        self.debug_fixed_rel_hit_y = float(debug_front_cfg.get("rel_hit_y", -0.4))
        self.debug_fixed_front_speed = float(debug_front_cfg.get("front_speed", 1.5))
        self.debug_fixed_front_pitch_deg = float(debug_front_cfg.get("front_pitch_deg", 0.0))
        self.debug_fixed_rel_hit_x = self._optional_float(debug_front_cfg.get("rel_hit_x", None))
        self.debug_fixed_rel_hit_z = self._optional_float(debug_front_cfg.get("rel_hit_z", None))

        self._last_output: LandingCommandOutput | None = None
        self._last_valid_t: float | None = None
        self._last_base_pos_target: np.ndarray | None = None
        self._reference_is_forehand: bool | None = None
        self._reference_rel_racket_target_pos_w = np.zeros(3, dtype=np.float64)
        self._reference_racket_target_vel_w = np.zeros(3, dtype=np.float64)
        self._sim_time_s = 0.0

    @staticmethod
    def _load_config(config_path: str | Path | None) -> dict[str, Any]:
        if config_path is None:
            config_path = Path(PROJECT_ROOT) / "deploy_mujoco" / "config" / "landing_planner.yaml"
        path = Path(config_path)
        if not path.is_absolute():
            path = Path(PROJECT_ROOT) / path
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _range(value: Any) -> tuple[float, float]:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] == 1:
            return float(arr[0]), float(arr[0])
        if arr.shape[0] != 2:
            raise ValueError(f"range expects one or two values, got {arr.shape[0]}")
        low, high = float(arr[0]), float(arr[1])
        return (low, high) if low <= high else (high, low)

    @staticmethod
    def _vec3(value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"expected 3 values, got {arr.shape[0]}")
        return arr

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    def _range_from_config(self, section: dict[str, Any], key: str, fallback: tuple[float, float] | float) -> tuple[float, float]:
        return self._range(section.get(key, fallback))

    def _load_racket_target_pose_range(
        self,
        section: Any,
        fallback_pos: np.ndarray,
        fallback_vel: np.ndarray,
        fallback_pos_y: tuple[float, float],
    ) -> dict[str, tuple[float, float]]:
        section = section or {}
        if not isinstance(section, dict):
            raise ValueError("racket target pose range config must be a mapping")
        return {
            "pos_x": self._range_from_config(section, "pos_x", (float(fallback_pos[0]), float(fallback_pos[0]))),
            "pos_y": self._range_from_config(section, "pos_y", fallback_pos_y),
            "pos_z": self._range_from_config(section, "pos_z", (0.0, 0.5)),
            "vel_x": self._range_from_config(section, "vel_x", (1.0, 2.0)),
            "vel_y": self._range_from_config(section, "vel_y", (-0.7, 0.7)),
            "vel_z": self._range_from_config(section, "vel_z", (-0.2, 0.7)),
        }

    def _sample_range_value(self, value_range: tuple[float, float]) -> float:
        low, high = value_range
        if low == high:
            return float(low)
        return float(self.command_rng.uniform(low, high))

    def _resample_reference_command(
        self,
        is_forehand: bool,
        robot_base_quat_wxyz: np.ndarray | None = None,
    ) -> None:
        target_range = (
            self.forehand_racket_target_pose_range
            if is_forehand
            else self.backhand_racket_target_pose_range
        )
        self._reference_is_forehand = bool(is_forehand)
        self._reference_rel_racket_target_pos_w = np.array(
            [
                self._sample_range_value(target_range["pos_x"]),
                self._sample_range_value(target_range["pos_y"]),
                self._sample_range_value(target_range["pos_z"]),
            ],
            dtype=np.float64,
        )
        self._reference_racket_target_vel_w = np.array(
            [
                self._sample_range_value(target_range["vel_x"]),
                self._sample_range_value(target_range["vel_y"]),
                self._sample_range_value(target_range["vel_z"]),
            ],
            dtype=np.float64,
        )
        self._reference_racket_target_vel_w = self._sample_reference_velocity_direction(
            self._reference_racket_target_vel_w,
            robot_base_quat_wxyz,
        )

    def reset(self) -> None:
        self.hope_planner.reset()
        self._last_output = None
        self._last_valid_t = None
        self._last_base_pos_target = None
        self._reference_is_forehand = None
        self._sim_time_s = 0.0

    def configure_debug_fixed_front_strike(
        self,
        *,
        enabled: bool,
        rel_hit_y: float | None = None,
        front_speed: float | None = None,
        front_pitch_deg: float | None = None,
        rel_hit_x: float | None = None,
        rel_hit_z: float | None = None,
    ) -> None:
        self.debug_fixed_front_strike_enabled = bool(enabled)
        if rel_hit_y is not None:
            self.debug_fixed_rel_hit_y = float(rel_hit_y)
        if front_speed is not None:
            self.debug_fixed_front_speed = float(front_speed)
        if front_pitch_deg is not None:
            self.debug_fixed_front_pitch_deg = float(front_pitch_deg)
        if rel_hit_x is not None:
            self.debug_fixed_rel_hit_x = float(rel_hit_x)
        if rel_hit_z is not None:
            self.debug_fixed_rel_hit_z = float(rel_hit_z)

    def update(
        self,
        ball_pos_w: np.ndarray,
        ball_vel_w: np.ndarray,
        robot_base_pos_w: np.ndarray,
        robot_base_quat_wxyz: np.ndarray | None = None,
        dt: float = 0.02,
        episode_time_s: float | None = None,
    ) -> LandingCommandOutput:
        if episode_time_s is None:
            self._sim_time_s += float(dt)
            episode_time_s = self._sim_time_s
        else:
            self._sim_time_s = float(episode_time_s)

        ball_pos_w = np.asarray(ball_pos_w, dtype=np.float64).reshape(3)
        ball_vel_w = np.asarray(ball_vel_w, dtype=np.float64).reshape(3)
        robot_base_pos_w = np.asarray(robot_base_pos_w, dtype=np.float64).reshape(3)
        robot_base_quat_wxyz = (
            None
            if robot_base_quat_wxyz is None
            else np.asarray(robot_base_quat_wxyz, dtype=np.float64).reshape(4)
        )

        output = self._plan(
            ball_pos_w,
            ball_vel_w,
            robot_base_pos_w,
            robot_base_quat_wxyz,
            float(episode_time_s),
        )
        return self._finalize_output(output, float(episode_time_s), float(dt))

    def update_from_prediction(
        self,
        predicted_hit_ball_pos_w: np.ndarray,
        predicted_hit_ball_vel_w: np.ndarray,
        time_to_hit_s: float,
        ball_pos_w: np.ndarray,
        ball_vel_w: np.ndarray,
        robot_base_pos_w: np.ndarray,
        robot_base_quat_wxyz: np.ndarray | None = None,
        dt: float = 0.02,
        episode_time_s: float | None = None,
    ) -> LandingCommandOutput:
        if episode_time_s is None:
            self._sim_time_s += float(dt)
            episode_time_s = self._sim_time_s
        else:
            self._sim_time_s = float(episode_time_s)

        ball_pos_w = np.asarray(ball_pos_w, dtype=np.float64).reshape(3)
        ball_vel_w = np.asarray(ball_vel_w, dtype=np.float64).reshape(3)
        robot_base_pos_w = np.asarray(robot_base_pos_w, dtype=np.float64).reshape(3)
        robot_base_quat_wxyz = (
            None
            if robot_base_quat_wxyz is None
            else np.asarray(robot_base_quat_wxyz, dtype=np.float64).reshape(4)
        )

        if (
            ball_vel_w[0] < self.incoming_vx_threshold
            and ball_pos_w[0] <= float(self.cfg.x_hit) - self.post_hit_x_margin
        ):
            output = self._invalid("passed_hit", robot_base_pos_w)
        else:
            p_hit = np.asarray(predicted_hit_ball_pos_w, dtype=np.float64).reshape(3)
            v_hit = np.asarray(predicted_hit_ball_vel_w, dtype=np.float64).reshape(3)
            target_land = np.array(
                [self.target_land_xy[0], self.target_land_xy[1], self.table.z_surface + self.physics.radius],
                dtype=np.float64,
            )
            v_racket = self._compute_racket_velocity(p_hit, v_hit, target_land)
            desired_racket_normal = self._compute_racket_normal(v_racket)
            p_hit, v_hit, time_to_hit = self._advance_strike_if_needed(
                p_hit,
                v_hit,
                float(time_to_hit_s),
            )
            output = self._build_command_from_strike(
                p_hit=p_hit,
                v_hit=v_hit,
                v_racket=v_racket,
                time_to_hit=time_to_hit,
                target_land=target_land,
                robot_base_pos_w=robot_base_pos_w,
                robot_base_quat_wxyz=robot_base_quat_wxyz,
                desired_racket_normal_w=desired_racket_normal,
            )
        return self._finalize_output(output, float(episode_time_s), float(dt))

    def _finalize_output(
        self,
        output: LandingCommandOutput,
        episode_time_s: float,
        dt: float,
    ) -> LandingCommandOutput:
        if output.valid:
            self._last_output = output
            self._last_valid_t = episode_time_s
            return output

        if output.reason in self._no_hold_invalid_reasons:
            return output

        if (
            self._last_output is not None
            and self._last_valid_t is not None
            and episode_time_s - self._last_valid_t <= self.command_hold_s
        ):
            held = self._last_output
            return LandingCommandOutput(
                valid=True,
                base_pos_target=held.base_pos_target.copy(),
                rel_racket_target_pos_w=held.rel_racket_target_pos_w.copy(),
                racket_target_vel_w=held.racket_target_vel_w.copy(),
                racket_target_time=np.array(
                    [max(float(held.racket_target_time[0]) - dt, self.min_time_to_hit_s)],
                    dtype=np.float32,
                ),
                predicted_hit_ball_pos_w=held.predicted_hit_ball_pos_w.copy(),
                predicted_hit_ball_vel_w=held.predicted_hit_ball_vel_w.copy(),
                target_landing_pos_w=held.target_landing_pos_w.copy(),
                desired_ball_dir_w=held.desired_ball_dir_w.copy(),
                desired_racket_normal_w=held.desired_racket_normal_w.copy(),
                is_forehand=held.is_forehand,
                racket_face_axis_name=held.racket_face_axis_name,
                racket_face_axis_local=held.racket_face_axis_local.copy(),
                reason="held_previous_command",
            )

        return output

    def _plan(
        self,
        ball_pos_w: np.ndarray,
        ball_vel_w: np.ndarray,
        robot_base_pos_w: np.ndarray,
        robot_base_quat_wxyz: np.ndarray | None,
        episode_time_s: float,
    ) -> LandingCommandOutput:
        if (
            ball_vel_w[0] < self.incoming_vx_threshold
            and ball_pos_w[0] <= float(self.cfg.x_hit) - self.post_hit_x_margin
        ):
            return self._invalid("passed_hit", robot_base_pos_w)

        preferred_is_forehand = None
        preferred_base_pos_target = None
        desired_racket_normal = None
        if self.planner_type == "hope":
            hope_cmd = self.hope_planner.update(
                episode_time_s,
                ball_pos_w,
                target_land_xy=self.target_land_xy,
                p_base=robot_base_pos_w,
                base_quat=robot_base_quat_wxyz,
            )
            if hope_cmd is None or not hope_cmd.valid:
                return self._invalid("hope_not_ready", robot_base_pos_w)
            p_hit = hope_cmd.p_intercept.astype(np.float64)
            v_hit = self.hope_planner.latest_strike.v_ball.astype(np.float64) if self.hope_planner.latest_strike else ball_vel_w
            v_racket = hope_cmd.v_racket.astype(np.float64)
            time_to_hit = float(hope_cmd.t_strike) - float(episode_time_s)
            target_land = hope_cmd.target_land.astype(np.float64)
            desired_racket_normal = hope_cmd.n_racket.astype(np.float64)
            preferred_is_forehand = bool(getattr(hope_cmd, "is_forehand", True))
            preferred_base_pos_target = np.array(
                [
                    self.base_target_x,
                    float(getattr(hope_cmd, "p_base_target", np.array([0.0, robot_base_pos_w[1]], dtype=np.float64))[1]),
                ],
                dtype=np.float64,
            )
        else:
            plan = self.model_planner.plan(ball_pos_w, ball_vel_w, self.target_land_xy)
            if not plan.strike.valid:
                return self._invalid("model_based_invalid_strike", robot_base_pos_w)
            p_hit = plan.strike.p_ball.astype(np.float64)
            v_hit = plan.strike.v_ball.astype(np.float64)
            v_racket = plan.v_racket.astype(np.float64)
            time_to_hit = float(plan.strike.time_to_strike_s)
            target_land = np.array(
                [self.target_land_xy[0], self.target_land_xy[1], self.table.z_surface + self.physics.radius],
                dtype=np.float64,
            )
            desired_racket_normal = self._compute_racket_normal(v_racket)

        p_hit, v_hit, time_to_hit = self._advance_strike_if_needed(p_hit, v_hit, time_to_hit)
        return self._build_command_from_strike(
            p_hit=p_hit,
            v_hit=v_hit,
            v_racket=v_racket,
            time_to_hit=time_to_hit,
            target_land=target_land,
            robot_base_pos_w=robot_base_pos_w,
            robot_base_quat_wxyz=robot_base_quat_wxyz,
            preferred_is_forehand=preferred_is_forehand,
            preferred_base_pos_target=preferred_base_pos_target,
            desired_racket_normal_w=desired_racket_normal,
        )

    def _build_command_from_strike(
        self,
        p_hit: np.ndarray,
        v_hit: np.ndarray,
        v_racket: np.ndarray,
        time_to_hit: float,
        target_land: np.ndarray,
        robot_base_pos_w: np.ndarray,
        robot_base_quat_wxyz: np.ndarray | None = None,
        preferred_is_forehand: bool | None = None,
        preferred_base_pos_target: np.ndarray | None = None,
        desired_racket_normal_w: np.ndarray | None = None,
    ) -> LandingCommandOutput:
        if not np.all(np.isfinite(p_hit)) or not np.all(np.isfinite(v_racket)):
            return self._invalid("nan_or_inf", robot_base_pos_w)
        if time_to_hit < self.min_time_to_hit_s:
            return self._invalid("time_to_hit_expired", robot_base_pos_w)
        if time_to_hit > self.max_time_to_hit_s:
            return self._invalid("time_to_hit_out_of_range", robot_base_pos_w)
        if time_to_hit > self.activate_before_hit_s:
            return self._invalid("too_early", robot_base_pos_w)

        is_forehand, base_pos_target = self._select_stroke_and_base_target(
            p_hit,
            robot_base_pos_w,
            robot_base_quat_wxyz,
            preferred_is_forehand=preferred_is_forehand,
            preferred_base_pos_target=preferred_base_pos_target,
        )
        if self._reference_is_forehand is None or self._reference_is_forehand != is_forehand:
            self._resample_reference_command(is_forehand, robot_base_quat_wxyz)
        pos_offset = self.forehand_pos_offset if is_forehand else self.backhand_pos_offset

        if self._last_base_pos_target is not None:
            delta = np.clip(
                base_pos_target - self._last_base_pos_target,
                -self.base_target_max_delta,
                self.base_target_max_delta,
            )
            base_pos_target = self._last_base_pos_target + delta
            base_pos_target = self._blend(self._last_base_pos_target, base_pos_target, self.base_blend_alpha)
        self._last_base_pos_target = base_pos_target.copy()

        predicted_target = p_hit - robot_base_pos_w + pos_offset
        predicted_target = self._project_racket_position_to_stroke_range(predicted_target, is_forehand)
        pos_blend, vel_blend = self._planner_blend(time_to_hit)
        if self.debug_fixed_front_strike_enabled:
            is_forehand = self.debug_fixed_rel_hit_y <= float(self.forehand_y_range[1])
            base_pos_target = self._compute_debug_base_target(
                p_hit,
                robot_base_pos_w,
                robot_base_quat_wxyz,
                self.debug_fixed_rel_hit_y,
            )
            self._last_base_pos_target = base_pos_target.copy()
            rel_racket_target = predicted_target.copy()
            rel_racket_target[1] = self.debug_fixed_rel_hit_y
            if self.debug_fixed_rel_hit_x is not None:
                rel_racket_target[0] = self.debug_fixed_rel_hit_x
            if self.debug_fixed_rel_hit_z is not None:
                rel_racket_target[2] = self.debug_fixed_rel_hit_z
            rel_racket_target = self._project_racket_position_to_stroke_range(
                rel_racket_target,
                is_forehand,
            )
            v_racket = self._forward_velocity_world(
                robot_base_quat_wxyz,
                speed=self.debug_fixed_front_speed,
                pitch_deg=self.debug_fixed_front_pitch_deg,
            )
            desired_racket_normal = self._normalize_or_default(
                v_racket,
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
        else:
            rel_racket_target = self._blend(self._reference_rel_racket_target_pos_w, predicted_target, pos_blend)
            planner_racket_vel = np.asarray(v_racket, dtype=np.float64)
            desired_racket_normal = (
                self._compute_racket_normal(planner_racket_vel)
                if desired_racket_normal_w is None
                else self._normalize_or_default(
                    desired_racket_normal_w,
                    np.array([1.0, 0.0, 0.0], dtype=np.float64),
                )
            )
            v_racket = self._blend(self._reference_racket_target_vel_w, planner_racket_vel, vel_blend)
            v_racket = self._bias_velocity_to_front(v_racket, robot_base_quat_wxyz)
            vel_before_projection = v_racket.copy()
            rel_racket_target, v_racket = self._project_racket_command_to_stroke_range(
                rel_racket_target,
                v_racket,
                is_forehand,
            )
            speed = float(np.linalg.norm(vel_before_projection))
            if speed > 1.0e-8:
                direction = vel_before_projection / speed
                v_racket = direction * min(speed, self._stroke_max_speed(is_forehand))
            if self.align_racket_velocity_to_normal:
                v_racket = self._align_velocity_direction(v_racket, desired_racket_normal)
            # LandingAssistFinetune's rotation reward aligns the selected racket-face
            # normal with the final commanded racket velocity direction.
            desired_racket_normal = self._normalize_or_default(v_racket, desired_racket_normal)
        time_arr = np.array([time_to_hit], dtype=np.float64)

        desired_dir = self._compute_outgoing_ball_velocity(p_hit, target_land)
        norm = float(np.linalg.norm(desired_dir))
        if norm > 1.0e-8:
            desired_dir = desired_dir / norm
        else:
            desired_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        axis_name, axis_local = self.selected_racket_axis(is_forehand)
        return LandingCommandOutput(
            valid=True,
            base_pos_target=base_pos_target.astype(np.float32),
            rel_racket_target_pos_w=rel_racket_target.astype(np.float32),
            racket_target_vel_w=v_racket.astype(np.float32),
            racket_target_time=time_arr.astype(np.float32),
            predicted_hit_ball_pos_w=p_hit.astype(np.float32),
            predicted_hit_ball_vel_w=v_hit.astype(np.float32),
            target_landing_pos_w=target_land.astype(np.float32),
            desired_ball_dir_w=desired_dir.astype(np.float32),
            desired_racket_normal_w=desired_racket_normal.astype(np.float32),
            is_forehand=is_forehand,
            racket_face_axis_name=axis_name,
            racket_face_axis_local=axis_local,
            reason="ok",
        )

    def _compute_racket_velocity(
        self,
        p_hit: np.ndarray,
        v_hit: np.ndarray,
        target_land: np.ndarray,
    ) -> np.ndarray:
        dt = max(float(self.cfg.delta_t_flight), 1.0e-3)
        g = np.array(self.physics.g, dtype=np.float64)
        v_outgoing = (target_land - p_hit) / dt - 0.5 * g * dt
        delta_v = v_outgoing - v_hit
        delta_v_norm = float(np.linalg.norm(delta_v))
        if delta_v_norm < 1.0e-8:
            return np.zeros(3, dtype=np.float64)
        n_racket = delta_v / delta_v_norm
        c_r = float(self.cfg.c_r)
        v_o_n = float(np.dot(v_outgoing, n_racket))
        v_i_n = float(np.dot(v_hit, n_racket))
        v_r_n = (v_o_n + c_r * v_i_n) / max(1.0 + c_r, 1.0e-6)
        return v_r_n * n_racket

    def _compute_racket_normal(self, v_racket: np.ndarray) -> np.ndarray:
        normal = self._unit_or_none(v_racket)
        if normal is None:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return normal

    def _align_velocity_direction(self, velocity: np.ndarray, normal: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(velocity))
        if speed < 1.0e-8:
            return np.zeros(3, dtype=np.float64)
        normal_unit = self._unit_or_none(normal)
        if normal_unit is None:
            return velocity
        return normal_unit * speed

    def _compute_outgoing_ball_velocity(self, p_hit: np.ndarray, target_land: np.ndarray) -> np.ndarray:
        dt = max(float(self.cfg.delta_t_flight), 1.0e-3)
        g = np.array(self.physics.g, dtype=np.float64)
        return (target_land - p_hit) / dt - 0.5 * g * dt

    def _forward_velocity_world(
        self,
        robot_base_quat_wxyz: np.ndarray | None,
        *,
        speed: float,
        pitch_deg: float,
    ) -> np.ndarray:
        pitch = np.deg2rad(float(pitch_deg))
        local_dir = np.array([np.cos(pitch), 0.0, np.sin(pitch)], dtype=np.float64)
        if robot_base_quat_wxyz is None:
            world_dir = local_dir
        else:
            yaw = self._yaw_from_quat(robot_base_quat_wxyz)
            c = np.cos(yaw)
            s = np.sin(yaw)
            world_dir = np.array(
                [
                    c * local_dir[0] - s * local_dir[1],
                    s * local_dir[0] + c * local_dir[1],
                    local_dir[2],
                ],
                dtype=np.float64,
            )
        world_dir = self._normalize_or_default(world_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        return world_dir * float(speed)

    def _compute_debug_base_target(
        self,
        p_hit: np.ndarray,
        robot_base_pos_w: np.ndarray,
        robot_base_quat_wxyz: np.ndarray | None,
        target_rel_y: float,
    ) -> np.ndarray:
        del robot_base_quat_wxyz
        base_y_target = float(p_hit[1] - target_rel_y)
        base_y_target = float(np.clip(base_y_target, self.base_target_y_min, self.base_target_y_max))
        return np.array([self.base_target_x, base_y_target], dtype=np.float64)

    def _bias_velocity_to_front(
        self,
        velocity_w: np.ndarray,
        robot_base_quat_wxyz: np.ndarray | None,
    ) -> np.ndarray:
        velocity_w = np.asarray(velocity_w, dtype=np.float64).reshape(3)
        scale = float(np.clip(self.front_facing_lateral_scale, 0.0, 1.0))
        if scale >= 0.999 or robot_base_quat_wxyz is None:
            return velocity_w

        yaw = self._yaw_from_quat(robot_base_quat_wxyz)
        c = np.cos(yaw)
        s = np.sin(yaw)
        local_x = c * velocity_w[0] + s * velocity_w[1]
        local_y = -s * velocity_w[0] + c * velocity_w[1]
        local_z = velocity_w[2]
        local_y *= scale
        world_x = c * local_x - s * local_y
        world_y = s * local_x + c * local_y
        return np.array([world_x, world_y, local_z], dtype=np.float64)

    def _sample_reference_velocity_direction(
        self,
        base_velocity: np.ndarray,
        robot_base_quat_wxyz: np.ndarray | None,
    ) -> np.ndarray:
        velocity = np.asarray(base_velocity, dtype=np.float64).reshape(3)
        if not self.directional_velocity_sampling_enabled:
            return velocity

        base_speed = float(np.linalg.norm(velocity))
        speed_scale = self._sample_range_value(self.directional_speed_scale_range)
        speed = np.clip(
            base_speed * speed_scale,
            self.directional_min_speed,
            self.directional_max_speed,
        )
        pitch = np.deg2rad(self._sample_range_value(self.directional_pitch_deg_range))
        yaw = np.deg2rad(self._sample_range_value(self.directional_yaw_deg_range))
        cos_pitch = np.cos(pitch)
        local_dir = np.array(
            [cos_pitch * np.cos(yaw), cos_pitch * np.sin(yaw), np.sin(pitch)],
            dtype=np.float64,
        )
        local_dir = self._normalize_or_default(local_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if robot_base_quat_wxyz is None:
            world_dir = local_dir
        else:
            base_yaw = self._yaw_from_quat(robot_base_quat_wxyz)
            c = np.cos(base_yaw)
            s = np.sin(base_yaw)
            world_dir = np.array(
                [
                    c * local_dir[0] - s * local_dir[1],
                    s * local_dir[0] + c * local_dir[1],
                    local_dir[2],
                ],
                dtype=np.float64,
            )
        world_dir = self._normalize_or_default(world_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        return world_dir * speed

    def _advance_strike_if_needed(
        self,
        p_hit: np.ndarray,
        v_hit: np.ndarray,
        time_to_hit: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        advance = min(max(self.advance_hit_time_s, 0.0), max(float(time_to_hit) - self.min_time_to_hit_s, 0.0))
        if advance <= 0.0:
            return p_hit, v_hit, time_to_hit
        g = np.array(self.physics.g, dtype=np.float64)
        p_advanced = p_hit - v_hit * advance + 0.5 * g * advance * advance
        v_advanced = v_hit - g * advance
        return p_advanced, v_advanced, float(time_to_hit) - advance

    @staticmethod
    def _yaw_from_quat(quat_wxyz: np.ndarray) -> float:
        qw, qx, qy, qz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    def _relative_hit_y(
        self,
        p_hit: np.ndarray,
        robot_base_pos_w: np.ndarray,
        robot_base_quat_wxyz: np.ndarray | None,
    ) -> float:
        delta = np.asarray(p_hit, dtype=np.float64).reshape(3) - np.asarray(robot_base_pos_w, dtype=np.float64).reshape(3)
        if not self.stroke_side_in_base_frame or robot_base_quat_wxyz is None:
            return float(delta[1])
        yaw = self._yaw_from_quat(robot_base_quat_wxyz)
        return float(-np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1])

    def _select_stroke_and_base_target(
        self,
        p_hit: np.ndarray,
        robot_base_pos_w: np.ndarray,
        robot_base_quat_wxyz: np.ndarray | None,
        *,
        preferred_is_forehand: bool | None = None,
        preferred_base_pos_target: np.ndarray | None = None,
    ) -> tuple[bool, np.ndarray]:
        rel_hit_y = self._relative_hit_y(p_hit, robot_base_pos_w, robot_base_quat_wxyz)
        fh_min, fh_max = self.forehand_y_range
        bh_min, bh_max = self.backhand_y_range

        in_forehand = fh_min <= rel_hit_y <= fh_max
        in_backhand = bh_min <= rel_hit_y <= bh_max
        in_gap_prefers_backhand = fh_max < rel_hit_y < bh_min

        if preferred_is_forehand is None:
            if rel_hit_y < fh_min:
                is_forehand = True
            elif in_forehand:
                is_forehand = True
            elif in_gap_prefers_backhand:
                is_forehand = False
            elif in_backhand:
                is_forehand = False
            else:
                is_forehand = False
        else:
            is_forehand = bool(preferred_is_forehand)

        if preferred_base_pos_target is not None:
            base_pos_target = np.asarray(preferred_base_pos_target, dtype=np.float64).reshape(2).copy()
            base_pos_target[0] = float(self.base_target_x)
            base_pos_target[1] = float(np.clip(base_pos_target[1], self.base_target_y_min, self.base_target_y_max))
            return is_forehand, base_pos_target

        if is_forehand:
            if rel_hit_y < fh_min:
                desired_rel_y = fh_min
            elif rel_hit_y > fh_max:
                desired_rel_y = fh_max
            else:
                desired_rel_y = rel_hit_y
        else:
            if rel_hit_y < bh_min:
                desired_rel_y = bh_min
            elif rel_hit_y > bh_max:
                desired_rel_y = bh_max
            else:
                desired_rel_y = rel_hit_y

        already_in_selected_range = (
            in_forehand if is_forehand else in_backhand
        )
        delta_y = self.base_out_of_range_gain * (rel_hit_y - desired_rel_y)
        base_y_target = float(
            robot_base_pos_w[1]
            if already_in_selected_range
            else robot_base_pos_w[1] + delta_y
        )
        base_y_target = float(np.clip(base_y_target, self.base_target_y_min, self.base_target_y_max))
        return is_forehand, np.array([self.base_target_x, base_y_target], dtype=np.float64)

    def selected_racket_axis(self, is_forehand: bool) -> tuple[str, np.ndarray]:
        axis_name = self.forehand_axis if is_forehand else self.backhand_axis
        return axis_name, self.axis_map[axis_name].copy()

    def _planner_blend(self, time_to_hit: float) -> tuple[float, float]:
        horizon = max(self.activate_before_hit_s - self.min_time_to_hit_s, 1.0e-6)
        progress = np.clip((self.activate_before_hit_s - float(time_to_hit)) / horizon, 0.0, 1.0)
        pos_blend = self.pos_blend_min + (self.pos_blend_max - self.pos_blend_min) * progress
        vel_blend = self.vel_blend_min + (self.vel_blend_max - self.vel_blend_min) * progress
        return float(np.clip(pos_blend, 0.0, 1.0)), float(np.clip(vel_blend, 0.0, 1.0))

    def _project_racket_position_to_stroke_range(self, pos: np.ndarray, is_forehand: bool) -> np.ndarray:
        target_range = (
            self.forehand_racket_target_pose_range
            if is_forehand
            else self.backhand_racket_target_pose_range
        )
        out = np.asarray(pos, dtype=np.float64).copy()
        out[0] = np.clip(out[0], *target_range["pos_x"])
        out[1] = np.clip(out[1], *target_range["pos_y"])
        out[2] = np.clip(out[2], *target_range["pos_z"])
        return out

    def _project_racket_command_to_stroke_range(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        is_forehand: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_range = (
            self.forehand_racket_target_pose_range
            if is_forehand
            else self.backhand_racket_target_pose_range
        )
        pos = self._project_racket_position_to_stroke_range(pos, is_forehand)
        vel = np.asarray(vel, dtype=np.float64).copy()
        speed = float(np.linalg.norm(vel))
        if speed > 1.0e-8:
            max_speed = float(
                np.linalg.norm(
                    [
                        max(abs(target_range["vel_x"][0]), abs(target_range["vel_x"][1])),
                        max(abs(target_range["vel_y"][0]), abs(target_range["vel_y"][1])),
                        max(abs(target_range["vel_z"][0]), abs(target_range["vel_z"][1])),
                    ]
                )
            )
            min_speed = float(min(abs(target_range["vel_x"][0]), abs(target_range["vel_x"][1])))
            target_speed = float(np.clip(speed, min_speed, max_speed))
            target_speed = float(np.clip(target_speed, self.racket_speed_min, self.racket_speed_max))
            vel = vel / speed * target_speed
        return pos, vel

    def _stroke_max_speed(self, is_forehand: bool) -> float:
        target_range = (
            self.forehand_racket_target_pose_range
            if is_forehand
            else self.backhand_racket_target_pose_range
        )
        return float(
            np.linalg.norm(
                [
                    max(abs(target_range["vel_x"][0]), abs(target_range["vel_x"][1])),
                    max(abs(target_range["vel_y"][0]), abs(target_range["vel_y"][1])),
                    max(abs(target_range["vel_z"][0]), abs(target_range["vel_z"][1])),
                ]
            )
        )

    def _limit_speed(self, v: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(v))
        if speed < 1.0e-8:
            return np.zeros(3, dtype=np.float64)
        target_speed = float(np.clip(speed, self.racket_speed_min, self.racket_speed_max))
        return v / speed * target_speed

    @staticmethod
    def _unit_or_none(vec: np.ndarray) -> np.ndarray | None:
        vec = np.asarray(vec, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(vec))
        if norm < 1.0e-8:
            return None
        return vec / norm

    @staticmethod
    def _normalize_or_default(vec: np.ndarray, default: np.ndarray) -> np.ndarray:
        unit = LandingCommandGenerator._unit_or_none(vec)
        if unit is None:
            return np.asarray(default, dtype=np.float64).reshape(3)
        return unit

    @staticmethod
    def _blend(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return old * (1.0 - alpha) + new * alpha

    def _invalid(self, reason: str, robot_base_pos_w: np.ndarray) -> LandingCommandOutput:
        target_land = np.array(
            [self.target_land_xy[0], self.target_land_xy[1], self.table.z_surface + self.physics.radius],
            dtype=np.float32,
        )
        axis_name, axis_local = self.selected_racket_axis(True)
        return LandingCommandOutput(
            valid=False,
            base_pos_target=np.array([self.base_target_x, robot_base_pos_w[1]], dtype=np.float32),
            rel_racket_target_pos_w=self.inactive_rel_racket_target_pos_w.astype(np.float32),
            racket_target_vel_w=self.inactive_racket_target_vel_w.astype(np.float32),
            racket_target_time=np.array(
                [np.clip(self.inactive_time_s, self.min_time_to_hit_s, self.max_time_to_hit_s)],
                dtype=np.float32,
            ),
            predicted_hit_ball_pos_w=np.zeros(3, dtype=np.float32),
            predicted_hit_ball_vel_w=np.zeros(3, dtype=np.float32),
            target_landing_pos_w=target_land,
            desired_ball_dir_w=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            desired_racket_normal_w=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            is_forehand=True,
            racket_face_axis_name=axis_name,
            racket_face_axis_local=axis_local,
            reason=reason,
        )


def apply_landing_command_to_state(state_cmd, cmd: LandingCommandOutput) -> None:
    """Write a LandingCommandOutput into StateAndCmd using track-motion command fields."""
    state_cmd.planner_valid = bool(cmd.valid)
    state_cmd.predicted_hit_ball_pos_w = cmd.predicted_hit_ball_pos_w.copy()
    state_cmd.predicted_hit_ball_vel_w = cmd.predicted_hit_ball_vel_w.copy()
    state_cmd.target_landing_pos_w = cmd.target_landing_pos_w.copy()
    state_cmd.desired_ball_dir_w = cmd.desired_ball_dir_w.copy()
    state_cmd.desired_racket_normal_w = cmd.desired_racket_normal_w.copy()
    state_cmd.is_forehand = bool(cmd.is_forehand)
    state_cmd.racket_face_axis_name = cmd.racket_face_axis_name
    state_cmd.racket_face_axis_local = cmd.racket_face_axis_local.copy()

    state_cmd.base_pos_target = cmd.base_pos_target.copy()
    state_cmd.rel_racket_target_pos_w = cmd.rel_racket_target_pos_w.copy()
    state_cmd.racket_target_vel_w = cmd.racket_target_vel_w.copy()
    state_cmd.racket_target_time = cmd.racket_target_time.copy()
