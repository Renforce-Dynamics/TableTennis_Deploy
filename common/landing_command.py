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
    is_forehand: bool
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

        self.forehand_y_range = self._range(command_cfg.get("forehand_y_range", [-0.60, -0.30]))
        self.backhand_y_range = self._range(command_cfg.get("backhand_y_range", [-0.10, 0.30]))
        self.forehand_pos_offset = self._vec3(command_cfg.get("forehand_pos_offset", [0.02, -0.01, 0.0]))
        self.backhand_pos_offset = self._vec3(command_cfg.get("backhand_pos_offset", [0.02, 0.01, 0.0]))

        self.base_target_x = float(command_cfg.get("base_target_x", 0.0))
        self.base_target_y_min = float(command_cfg.get("base_target_y_min", -1.2))
        self.base_target_y_max = float(command_cfg.get("base_target_y_max", 1.2))
        self.base_target_max_delta = float(command_cfg.get("base_target_max_delta", 0.06))

        self.racket_pos_min = self._vec3(command_cfg.get("racket_pos_min", [0.15, -0.80, -0.10]))
        self.racket_pos_max = self._vec3(command_cfg.get("racket_pos_max", [0.70, 0.55, 0.65]))
        self.racket_speed_min = float(command_cfg.get("racket_speed_min", 0.0))
        self.racket_speed_max = float(command_cfg.get("racket_speed_max", 3.0))

        self.pos_blend_alpha = float(command_cfg.get("pos_blend_alpha", 0.45))
        self.vel_blend_alpha = float(command_cfg.get("vel_blend_alpha", 0.35))
        self.base_blend_alpha = float(command_cfg.get("base_blend_alpha", 0.35))

        self._last_output: LandingCommandOutput | None = None
        self._last_valid_t: float | None = None
        self._last_base_pos_target: np.ndarray | None = None
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

    def reset(self) -> None:
        self.hope_planner.reset()
        self._last_output = None
        self._last_valid_t = None
        self._last_base_pos_target = None
        self._sim_time_s = 0.0

    def update(
        self,
        ball_pos_w: np.ndarray,
        ball_vel_w: np.ndarray,
        robot_base_pos_w: np.ndarray,
        robot_base_quat_wxyz: np.ndarray | None = None,
        dt: float = 0.02,
        episode_time_s: float | None = None,
    ) -> LandingCommandOutput:
        del robot_base_quat_wxyz  # reserved for future body-frame command variants

        if episode_time_s is None:
            self._sim_time_s += float(dt)
            episode_time_s = self._sim_time_s
        else:
            self._sim_time_s = float(episode_time_s)

        ball_pos_w = np.asarray(ball_pos_w, dtype=np.float64).reshape(3)
        ball_vel_w = np.asarray(ball_vel_w, dtype=np.float64).reshape(3)
        robot_base_pos_w = np.asarray(robot_base_pos_w, dtype=np.float64).reshape(3)

        output = self._plan(ball_pos_w, ball_vel_w, robot_base_pos_w, float(episode_time_s))
        return self._finalize_output(output, float(episode_time_s), float(dt))

    def update_from_prediction(
        self,
        predicted_hit_ball_pos_w: np.ndarray,
        predicted_hit_ball_vel_w: np.ndarray,
        time_to_hit_s: float,
        ball_pos_w: np.ndarray,
        ball_vel_w: np.ndarray,
        robot_base_pos_w: np.ndarray,
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
            output = self._build_command_from_strike(
                p_hit=p_hit,
                v_hit=v_hit,
                v_racket=v_racket,
                time_to_hit=float(time_to_hit_s),
                target_land=target_land,
                robot_base_pos_w=robot_base_pos_w,
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
                is_forehand=held.is_forehand,
                reason="held_previous_command",
            )

        return output

    def _plan(
        self,
        ball_pos_w: np.ndarray,
        ball_vel_w: np.ndarray,
        robot_base_pos_w: np.ndarray,
        episode_time_s: float,
    ) -> LandingCommandOutput:
        if (
            ball_vel_w[0] < self.incoming_vx_threshold
            and ball_pos_w[0] <= float(self.cfg.x_hit) - self.post_hit_x_margin
        ):
            return self._invalid("passed_hit", robot_base_pos_w)

        if self.planner_type == "hope":
            hope_cmd = self.hope_planner.update(episode_time_s, ball_pos_w, self.target_land_xy)
            if hope_cmd is None or not hope_cmd.valid:
                return self._invalid("hope_not_ready", robot_base_pos_w)
            p_hit = hope_cmd.p_intercept.astype(np.float64)
            v_hit = self.hope_planner.latest_strike.v_ball.astype(np.float64) if self.hope_planner.latest_strike else ball_vel_w
            v_racket = hope_cmd.v_racket.astype(np.float64)
            time_to_hit = float(hope_cmd.t_strike) - float(episode_time_s)
            target_land = hope_cmd.target_land.astype(np.float64)
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

        return self._build_command_from_strike(
            p_hit=p_hit,
            v_hit=v_hit,
            v_racket=v_racket,
            time_to_hit=time_to_hit,
            target_land=target_land,
            robot_base_pos_w=robot_base_pos_w,
        )

    def _build_command_from_strike(
        self,
        p_hit: np.ndarray,
        v_hit: np.ndarray,
        v_racket: np.ndarray,
        time_to_hit: float,
        target_land: np.ndarray,
        robot_base_pos_w: np.ndarray,
    ) -> LandingCommandOutput:
        if not np.all(np.isfinite(p_hit)) or not np.all(np.isfinite(v_racket)):
            return self._invalid("nan_or_inf", robot_base_pos_w)
        if time_to_hit < self.min_time_to_hit_s:
            return self._invalid("time_to_hit_expired", robot_base_pos_w)
        if time_to_hit > self.max_time_to_hit_s:
            return self._invalid("time_to_hit_out_of_range", robot_base_pos_w)
        if time_to_hit > self.activate_before_hit_s:
            return self._invalid("too_early", robot_base_pos_w)

        rel_hit_y = float(p_hit[1] - robot_base_pos_w[1])
        is_forehand = self._choose_forehand(rel_hit_y)
        y_range = self.forehand_y_range if is_forehand else self.backhand_y_range
        pos_offset = self.forehand_pos_offset if is_forehand else self.backhand_pos_offset

        desired_rel_y = float(np.clip(rel_hit_y, y_range[0], y_range[1]))
        base_y_target = float(p_hit[1] - desired_rel_y)
        base_y_target = float(np.clip(base_y_target, self.base_target_y_min, self.base_target_y_max))

        base_pos_target = np.array([self.base_target_x, base_y_target], dtype=np.float64)
        if self._last_base_pos_target is not None:
            delta = np.clip(
                base_pos_target - self._last_base_pos_target,
                -self.base_target_max_delta,
                self.base_target_max_delta,
            )
            base_pos_target = self._last_base_pos_target + delta
            base_pos_target = self._blend(self._last_base_pos_target, base_pos_target, self.base_blend_alpha)
        self._last_base_pos_target = base_pos_target.copy()

        rel_racket_target = p_hit - robot_base_pos_w + pos_offset
        rel_racket_target = np.clip(rel_racket_target, self.racket_pos_min, self.racket_pos_max)

        v_racket = self._limit_speed(v_racket)
        time_arr = np.array([time_to_hit], dtype=np.float64)

        if self._last_output is not None:
            rel_racket_target = self._blend(
                self._last_output.rel_racket_target_pos_w.astype(np.float64),
                rel_racket_target,
                self.pos_blend_alpha,
            )
            v_racket = self._blend(
                self._last_output.racket_target_vel_w.astype(np.float64),
                v_racket,
                self.vel_blend_alpha,
            )
            time_arr[0] = np.clip(time_arr[0], self.min_time_to_hit_s, self.max_time_to_hit_s)

        desired_dir = target_land - p_hit
        norm = float(np.linalg.norm(desired_dir))
        if norm > 1.0e-8:
            desired_dir = desired_dir / norm
        else:
            desired_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

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
            is_forehand=is_forehand,
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

    def _choose_forehand(self, rel_hit_y: float) -> bool:
        forehand_center = 0.5 * (self.forehand_y_range[0] + self.forehand_y_range[1])
        backhand_center = 0.5 * (self.backhand_y_range[0] + self.backhand_y_range[1])
        return abs(rel_hit_y - forehand_center) <= abs(rel_hit_y - backhand_center)

    def _limit_speed(self, v: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(v))
        if speed < 1.0e-8:
            return np.zeros(3, dtype=np.float64)
        target_speed = float(np.clip(speed, self.racket_speed_min, self.racket_speed_max))
        return v / speed * target_speed

    @staticmethod
    def _blend(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return old * (1.0 - alpha) + new * alpha

    def _invalid(self, reason: str, robot_base_pos_w: np.ndarray) -> LandingCommandOutput:
        target_land = np.array(
            [self.target_land_xy[0], self.target_land_xy[1], self.table.z_surface + self.physics.radius],
            dtype=np.float32,
        )
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
            is_forehand=True,
            reason=reason,
        )


def apply_landing_command_to_state(state_cmd, cmd: LandingCommandOutput) -> None:
    """Write a LandingCommandOutput into StateAndCmd using track-motion command fields."""
    state_cmd.planner_valid = bool(cmd.valid)
    state_cmd.predicted_hit_ball_pos_w = cmd.predicted_hit_ball_pos_w.copy()
    state_cmd.predicted_hit_ball_vel_w = cmd.predicted_hit_ball_vel_w.copy()
    state_cmd.target_landing_pos_w = cmd.target_landing_pos_w.copy()
    state_cmd.desired_ball_dir_w = cmd.desired_ball_dir_w.copy()
    state_cmd.is_forehand = bool(cmd.is_forehand)

    state_cmd.base_pos_target = cmd.base_pos_target.copy()
    state_cmd.rel_racket_target_pos_w = cmd.rel_racket_target_pos_w.copy()
    state_cmd.racket_target_vel_w = cmd.racket_target_vel_w.copy()
    state_cmd.racket_target_time = cmd.racket_target_time.copy()
