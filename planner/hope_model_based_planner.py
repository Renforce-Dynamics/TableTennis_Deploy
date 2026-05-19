from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import BallPhysics, PlannerConfig, TableParams  # noqa: F401 - re-export


@dataclass
class StrikeState:
  valid: bool
  p_ball: np.ndarray
  v_ball: np.ndarray
  time_to_strike_s: float
  landing_xy: np.ndarray | None


@dataclass
class PlanResult:
  strike: StrikeState
  v_racket: np.ndarray


class HopeModelBasedPlanner:
  """【链路: 上层规划器】分析型轨迹预测 + 碰撞模型求解球拍速度.

  被 MotionCommand._apply_ball_tracking_assist() 每步调用.

  输入: 当前帧球位置 p0 (局部坐标), 球速度 v0 (世界坐标), 目标落点 target_land_xy
  输出: PlanResult { strike (击球点/时间/球速), v_racket (期望球拍速度) }

  三阶段计算:
    1) _compute_strike: 分段常加速度轨迹预测
       Phase 1 — 解二次方程求台面反弹时间, 应用恢复系数
       Phase 2 — 从反弹点到击球平面 x=x_hit 的飞行
    2) 期望出球速度: v_out = (p_land - p_strike)/dt - 0.5*g*dt
    3) 碰撞模型: 沿法向一维恢复方程求解球拍法向速度
       v_r_n = (v_o_n + c_r * v_i_n) / (1 + c_r)
  """

  def __init__(self, table: TableParams, physics: BallPhysics, cfg: PlannerConfig):
    self.table = table
    self.physics = physics
    self.cfg = cfg
    self._g = np.array(physics.g, dtype=np.float64)
    # Pre-computed table contact height (ball centre at table surface).
    self._zt = self.table.z_surface + self.physics.radius

  def plan(
    self,
    p0: np.ndarray,
    v0: np.ndarray,
    target_land_xy: np.ndarray,
  ) -> PlanResult:
    p0 = np.asarray(p0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    target_land_xy = np.asarray(target_land_xy, dtype=np.float64)

    strike = self._compute_strike(p0, v0)
    if not strike.valid:
      return PlanResult(strike=strike, v_racket=np.zeros(3, dtype=np.float64))

    target_land = np.array(
      [target_land_xy[0], target_land_xy[1], self._zt],
      dtype=np.float64,
    )
    # Desired outgoing ball velocity from ballistic flight:
    # p_land = p_strike + v_out * dt + 0.5*g*dt^2
    # => v_out = (p_land - p_strike)/dt - 0.5*g*dt
    dt = max(float(self.cfg.delta_t_flight), 1.0e-3)
    v_outgoing = (target_land - strike.p_ball) / dt - 0.5 * self._g * dt
    v_incoming = strike.v_ball
    delta_v = v_outgoing - v_incoming
    delta_v_norm = float(np.linalg.norm(delta_v))
    if delta_v_norm < 1.0e-8:
      vin_norm = float(np.linalg.norm(v_incoming))
      if vin_norm > 1.0e-8:
        n_racket = -v_incoming / vin_norm
      else:
        n_racket = np.array([1.0, 0.0, 0.0], dtype=np.float64)
      v_racket = np.zeros(3, dtype=np.float64)
      return PlanResult(strike=strike, v_racket=v_racket)

    # Racket normal points along the required impulse direction.
    n_racket = delta_v / delta_v_norm
    c_r = float(self.cfg.c_r)
    v_o_n = float(np.dot(v_outgoing, n_racket))
    v_i_n = float(np.dot(v_incoming, n_racket))
    # From restitution along normal:
    # v_o_n - v_r_n = -c_r * (v_i_n - v_r_n)
    # => v_r_n = (v_o_n + c_r * v_i_n) / (1 + c_r)
    v_r_n = (v_o_n + c_r * v_i_n) / max(1.0 + c_r, 1.0e-6)
    v_racket = v_r_n * n_racket
    return PlanResult(strike=strike, v_racket=v_racket)

  def _compute_strike(self, p0: np.ndarray, v0: np.ndarray) -> StrikeState:
    invalid = StrikeState(
      valid=False,
      p_ball=np.copy(p0),
      v_ball=np.copy(v0),
      time_to_strike_s=0.0,
      landing_xy=None,
    )

    # General direction check:
    # Do not assume the ball must move along -x.
    # A valid ball only needs to reach x_hit in positive time.
    t_direct = self._time_to_x(p0[0], v0[0], self.cfg.x_hit)
    if t_direct is None or t_direct < 0.0 or t_direct > self.cfg.max_predict_time:
      return invalid

    # ── Phase 1: possible pre-strike table bounce ───────────────────
    t_bounce = self._time_to_z(p0[2], v0[2], self._zt)

    p_bounce: np.ndarray | None = None
    v_bounce: np.ndarray | None = None

    # If the ball is already very close to the table and moving upward,
    # it is probably just after a bounce. Avoid predicting a fake immediate
    # second bounce.
    near_table_and_rising = p0[2] < self._zt + 0.1 and v0[2] > 0.0

    # The bounce is useful only if it happens before the direct strike time.
    if (
      t_bounce is not None
      and t_bounce > 1.0e-6
      and t_bounce < t_direct
      and not near_table_and_rising
    ):
      p_candidate = p0 + v0 * t_bounce + 0.5 * self._g * (t_bounce**2)

      # Only accept the bounce if the contact point is inside the table.
      if self._on_table(p_candidate):
        v_pre = v0 + self._g * t_bounce
        v_post = np.array(
          [
            self.physics.c_h * v_pre[0],
            self.physics.c_h * v_pre[1],
            -self.physics.c_v * v_pre[2],
          ],
          dtype=np.float64,
        )

        # After bounce, the ball must still be able to reach x_hit.
        t_post = self._time_to_x(p_candidate[0], v_post[0], self.cfg.x_hit)
        if t_post is not None and t_post >= 0.0:
          t_total_candidate = t_bounce + t_post
          if t_total_candidate <= self.cfg.max_predict_time:
            p_bounce = p_candidate
            v_bounce = v_post

    if p_bounce is not None and v_bounce is not None:
      # ── Phase 2: post-bounce → strike ─────────────────────────────
      t_post = self._time_to_x(p_bounce[0], v_bounce[0], self.cfg.x_hit)
      if t_post is None or t_post < 0.0:
        return invalid

      t_total = t_bounce + t_post
      if t_total > self.cfg.max_predict_time:
        return invalid

      p_strike = p_bounce + v_bounce * t_post + 0.5 * self._g * (t_post**2)
      v_strike = v_bounce + self._g * t_post
    else:
      # ── No bounce: direct flight → strike ─────────────────────────
      t_total = t_direct
      p_strike = p0 + v0 * t_total + 0.5 * self._g * (t_total**2)
      v_strike = v0 + self._g * t_total

    landing_xy = self._predict_landing_xy(p_strike, v_strike)

    return StrikeState(
      valid=True,
      p_ball=p_strike,
      v_ball=v_strike,
      time_to_strike_s=float(t_total),
      landing_xy=landing_xy,
    )
  
  def _time_to_x(self, x0: float, vx: float, x_hit: float) -> float | None:
    if abs(vx) < 1.0e-8:
      return None
    return (x_hit - x0) / vx

  def _time_to_z(self, z0: float, vz: float, z_target: float) -> float | None:
    """Time for the ball centre to reach z_target under constant gravity.

    Returns the smallest positive root, or None if no positive real root exists.
    """
    gz = float(self._g[2])

    if abs(gz) < 1.0e-8:
      if abs(vz) < 1.0e-8:
        return None
      t = (z_target - z0) / vz
      return t if t > 1.0e-6 else None

    a = 0.5 * gz
    b = float(vz)
    c = float(z0 - z_target)

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
      return None

    sqrt_disc = float(np.sqrt(disc))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    ts = [t for t in (t1, t2) if t > 1.0e-6]
    return min(ts) if ts else None

  def _time_to_table(self, p0: np.ndarray, v0: np.ndarray) -> float | None:
    """Backward-compatible wrapper for old getting-BiFight call sites."""
    return self._time_to_z(float(p0[2]), float(v0[2]), self._zt)

  def _on_table(self, p: np.ndarray) -> bool:
    return (
      self.table.x_min <= float(p[0]) <= self.table.x_max
      and self.table.y_min <= float(p[1]) <= self.table.y_max
    )

  def _predict_landing_xy(
    self, p: np.ndarray, v: np.ndarray
  ) -> np.ndarray | None:
    z0 = float(p[2])
    vz = float(v[2])
    gz = float(self._g[2])

    # 0.5*gz*t^2 + vz*t + (z0 - self._zt) = 0
    a = 0.5 * gz
    b = vz
    c = z0 - self._zt
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
      return None
    sqrt_disc = float(np.sqrt(disc))
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    ts = [t for t in roots if t > 1.0e-6]
    if not ts:
      return None
    t = min(ts)
    landing = p + v * t + 0.5 * self._g * (t**2)
    return np.array([landing[0], landing[1]], dtype=np.float64)
