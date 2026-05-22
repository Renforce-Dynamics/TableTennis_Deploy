"""HOPE 三层规划器：状态估计 → 轨迹预测 → 球拍目标规划。

Pipeline:
  Stage 1 — BallStateEstimator: 滑动窗口二阶多项式拟合，从位置流估计速度，
             内建弹跳检测以防止多项式跨越速度不连续点拟合。
  Stage 2 — BallTrajectoryPredictor: 前向欧拉积分（含二次空气阻力和台面弹跳），
             寻找球穿越击球平面 x=x_hit 的瞬间。
  Stage 3 — RacketTargetPlanner: 碰撞模型求解球拍速度/法向 + 过网间隙验证，
             自动调整飞行时间以保证回球过网。

与旧 HopeModelBasedPlanner 的区别：
  - 多帧拟合估计速度（而非直接用 MuJoCo 瞬时速度）
  - 前向数值积分 + 空气阻力（而非解析分段常加速度）
  - 弹跳检测清空缓冲区（防止跨弹跳拟合）
  - 过网验证 + 自动飞行时间调整
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .common import BallPhysics, PlannerConfig, TableParams


# ============================================================
# Stage 1 — 球状态估计
# ============================================================


class BallStateEstimator:
  """从位置测量流中估计球的平滑位置和速度。

  维护最近位置测量的滑动窗口，执行最小二乘多项式拟合以提取
  平滑的位置和速度估计。每次检测到台面弹跳时清空缓冲区，
  防止多项式跨越速度不连续点进行拟合。

  使用方法::

      est = BallStateEstimator(config, table_z_surface=0.76)
      for each sim step:
          est.push(t, p_ball)
          if est.bounce_detected:
              ...  # 弹跳发生，缓冲区已自动清空
          if est.ready:
              p_smooth, v_smooth, t_ref = est.estimate()
  """

  def __init__(
    self,
    config: PlannerConfig,
    table_z_surface: float = 0.76,
  ):
    self.config = config
    self._table_z = table_z_surface
    self.t_buffer: list[float] = []
    self.p_buffer: list[np.ndarray] = []

    # 弹跳检测：三样本 z 高度环形缓冲区
    self._z_hist: list[float | None] = [None, None, None]
    self._bounce_detected: bool = False

  def reset(self) -> None:
    """清空估计缓冲区（弹跳检测时或 episode 重置时调用）。"""
    self.t_buffer.clear()
    self.p_buffer.clear()
    self._z_hist = [None, None, None]
    self._bounce_detected = False

  def push(self, t: float, p: np.ndarray) -> None:
    """添加新的位置测量值。"""
    # 更新 z 历史环形缓冲区
    self._z_hist[0] = self._z_hist[1]
    self._z_hist[1] = self._z_hist[2]
    self._z_hist[2] = float(p[2])

    # 弹跳检测：三样本模式 (above → below → above)
    self._bounce_detected = False
    z_pp, z_p, z_c = self._z_hist
    tol = float(getattr(self.config, "bounce_z_tol", 0.02))
    z_surf = self._table_z
    if z_pp is not None and z_p is not None and z_c is not None:
      above = lambda z: z > z_surf + tol
      below = lambda z: z <= z_surf + tol
      if above(z_pp) and below(z_p) and above(z_c):
        self._bounce_detected = True
        self.reset()

    self.t_buffer.append(t)
    self.p_buffer.append(p.copy().astype(np.float64))

    window = int(getattr(self.config, "fit_window", 31))
    if len(self.t_buffer) > window:
      self.t_buffer.pop(0)
      self.p_buffer.pop(0)

  @property
  def bounce_detected(self) -> bool:
    return self._bounce_detected

  @property
  def ready(self) -> bool:
    return len(self.t_buffer) >= 6

  def estimate(self) -> tuple[np.ndarray, np.ndarray, float]:
    """计算最新时间戳下的平滑位置和速度。

    Returns:
      (p_est, v_est, t_ref): 平滑位置 [3], 平滑速度 [3], 参考时间
    """
    if not self.ready:
      raise RuntimeError(
        f"需要 >= 6 个样本，当前有 {len(self.t_buffer)} 个"
      )

    t_arr = np.array(self.t_buffer, dtype=np.float64)
    p_arr = np.array(self.p_buffer, dtype=np.float64)

    # 时间归一化以改善数值条件
    t_ref = t_arr[-1]
    t_norm = t_arr - t_ref

    p_est = np.zeros(3, dtype=np.float64)
    v_est = np.zeros(3, dtype=np.float64)

    poly_order = int(getattr(self.config, "poly_order", 2))
    for axis in range(3):
      coeffs = np.polyfit(t_norm, p_arr[:, axis], deg=poly_order)
      # np.polyfit 返回 [a2, a1, a0] (降序), t_norm=0 对应最新样本
      p_est[axis] = coeffs[-1]  # a0 at t_norm = 0
      v_est[axis] = coeffs[-2]  # a1 at t_norm = 0

    return p_est, v_est, t_ref


# ============================================================
# Stage 2 — 轨迹预测
# ============================================================


@dataclass
class StrikeTarget:
  """第 2 阶段的输出：预测击球平面上的球状态。"""

  p_ball: np.ndarray  # 预测的击球时球位置 [x, y, z]
  v_ball: np.ndarray  # 预测的击球时球速度 [vx, vy, vz]
  t_strike: float  # 绝对击球时间
  num_bounces: int  # 击球前的弹跳次数
  valid: bool  # 如果找到了有效的平面穿越则为 True


class BallTrajectoryPredictor:
  """前向积分球轨迹并寻找击球平面的穿越点。

  使用显式欧拉法以 1 kHz 步长积分飞行动力学（含二次空气阻力 +
  重力），在 z 穿越台面高度时应用对角恢复矩阵进行弹跳处理。
  """

  def __init__(
    self,
    physics: BallPhysics,
    config: PlannerConfig,
    table: TableParams,
  ):
    self.physics = physics
    self.config = config
    self.table = table
    self._g = np.array(physics.g, dtype=np.float64)
    self._z_surface = float(table.z_surface)

  def _is_on_table(self, p: np.ndarray) -> bool:
    """检查球是否能接触到球桌表面（含半径容差的边缘接触）。"""
    r = self.physics.radius
    return (
      self.table.x_min - r <= p[0] <= self.table.x_max + r
      and self.table.y_min - r <= p[1] <= self.table.y_max + r
    )

  def _flight_acceleration(self, v: np.ndarray) -> np.ndarray:
    """自由飞行期间的球加速度: a = -k|v|v + g"""
    speed = float(np.linalg.norm(v))
    k = float(self.physics.k)
    return -k * speed * v + self._g

  def _apply_bounce(self, v: np.ndarray) -> np.ndarray:
    """应用球桌弹跳恢复: v+ = diag(C_h, C_h, -C_v) @ v-"""
    return np.array(
      [
        self.physics.c_h * v[0],
        self.physics.c_h * v[1],
        -self.physics.c_v * v[2],
      ],
      dtype=np.float64,
    )

  def predict(
    self,
    p0: np.ndarray,
    v0: np.ndarray,
    t0: float,
  ) -> StrikeTarget:
    """前向积分并寻找击球平面的穿越点。

    Args:
      p0: 初始球位置 [x, y, z] (世界坐标)
      v0: 初始球速度 [vx, vy, vz] (世界坐标)
      t0: 初始绝对时间

    Returns:
      StrikeTarget: 预测的击球状态
    """
    dt = float(self.config.dt_integrate)
    max_steps = int(self.config.max_predict_time / dt)
    x_hit = float(self.config.x_hit)
    z_surf = self._z_surface

    p = p0.copy().astype(np.float64)
    v = v0.copy().astype(np.float64)
    t = float(t0)
    bounces = 0
    bounce_this_step = False

    for _step in range(max_steps):
      p_prev_x = float(p[0])

      # --- 欧拉积分步 ---
      a = self._flight_acceleration(v)
      v_new = v + a * dt
      p_new = p + v * dt + 0.5 * a * dt * dt
      t += dt
      bounce_this_step = False

      # --- 弹跳检测 ---
      if p_new[2] < z_surf and v_new[2] < 0.0:
        if self._is_on_table(p_new):
          # 子步插值寻找精确的弹跳时间
          dz = float(p[2] - p_new[2])
          frac = float(p[2] - z_surf) / dz if abs(dz) > 1e-9 else 0.5
          frac = np.clip(frac, 0.0, 1.0)

          p_bounce = p + frac * (p_new - p)
          p_bounce[2] = z_surf
          v_at_bounce = v + a * (frac * dt)

          v_post = self._apply_bounce(v_at_bounce)

          # 带有二阶修正的弹跳后继续积分
          remaining_dt = (1.0 - frac) * dt
          a_post = self._flight_acceleration(v_post)
          p_new = (
            p_bounce
            + v_post * remaining_dt
            + 0.5 * a_post * remaining_dt * remaining_dt
          )
          v_new = v_post + a_post * remaining_dt
          bounces += 1
          bounce_this_step = True
        else:
          p_new[2] = max(p_new[2], z_surf)

      # --- 击球平面穿越检测 ---
      if p_prev_x > x_hit and p_new[0] <= x_hit and v_new[0] < 0:
        if bounce_this_step:
          # 使用弹跳后的弧线进行插值
          dx_arc = float(p_new[0] - p_bounce[0])
          if abs(dx_arc) > 1e-9:
            frac_cross = float(p_bounce[0] - x_hit) / float(
              p_bounce[0] - p_new[0]
            )
          else:
            frac_cross = 0.5
          frac_cross = np.clip(frac_cross, 0.0, 1.0)
          p_cross = p_bounce + frac_cross * (p_new - p_bounce)
          v_cross = v_post + frac_cross * (v_new - v_post)
          t_cross = (t - remaining_dt) + frac_cross * remaining_dt
        else:
          dx_step = float(p[0] - p_new[0])
          if abs(dx_step) > 1e-9:
            frac_cross = float(p[0] - x_hit) / dx_step
          else:
            frac_cross = 0.5
          frac_cross = np.clip(frac_cross, 0.0, 1.0)
          p_cross = p + frac_cross * (p_new - p)
          v_cross = v + frac_cross * (v_new - v)
          t_cross = t - dt + frac_cross * dt

        p_cross[0] = x_hit

        return StrikeTarget(
          p_ball=p_cross,
          v_ball=v_cross,
          t_strike=float(t_cross),
          num_bounces=bounces,
          valid=True,
        )

      p = p_new
      v = v_new

    return StrikeTarget(
      p_ball=p,
      v_ball=v,
      t_strike=float(t),
      num_bounces=bounces,
      valid=False,
    )


# ============================================================
# Stage 3 — 球拍目标规划
# ============================================================


@dataclass
class RacketCommand:
  """第 3 阶段的输出：击球时的期望球拍状态 + 正反手/base 目标。

  这是规划器输出给全身控制器的指令。
  对应论文中 planner 的完整输出:
    p̂_base, p̂_racket, v̂_racket, t_strike, stroke_type
  """

  p_intercept: np.ndarray  # 拦截时球拍中心的期望位置 (等于预测的击球点位置)
  v_racket: np.ndarray  # 期望球拍速度向量 [vx, vy, vz]
  n_racket: np.ndarray  # 期望拍面法向 (单位向量)
  t_strike: float  # 预测的击球时间
  v_ball_outgoing: np.ndarray  # 预期的出球速度
  target_land: np.ndarray  # 目标落点 (3D 世界坐标)
  clears_net: bool  # 如果回球轨迹能过网则为 True
  bypasses_net_posts: bool  # 如果球从网架 Y 轴范围外绕过则为 True
  valid: bool  # 如果所有计算均成功则为 True
  is_forehand: bool = True  # 是否为正手击球
  p_base_target: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))  # 目标 base XY (世界坐标)
  stroke_confident: bool = True  # 击球点是否明确落在正手或反手区间内


class RacketTargetPlanner:
  """计算期望球拍速度和朝向以实现有效回球。

  包含过网间隙验证：回球轨迹必须在 x=net_x 处越过 net_height，
  且球在球网处的 Y 坐标落在球网横向范围内。
  如果初始飞行时间无法过网，自动尝试替代飞行时间。
  """

  def __init__(
    self,
    physics: BallPhysics,
    config: PlannerConfig,
    table: TableParams,
  ):
    self.physics = physics
    self.config = config
    self.table = table
    self._g = np.array(physics.g, dtype=np.float64)

  def _compute_outgoing_velocity(
    self,
    p_strike: np.ndarray,
    p_land: np.ndarray,
    delta_t: float,
  ) -> np.ndarray:
    """从弹道飞行反算出球速度: v_o = (p_land - p)/dt + 0.5*g*dt"""
    return (p_land - p_strike) / delta_t + 0.5 * self._g * delta_t

  def _compute_racket_velocity(
    self,
    v_incoming: np.ndarray,
    v_outgoing: np.ndarray,
    C_r: float,
  ) -> tuple[np.ndarray, np.ndarray]:
    """从碰撞模型计算期望的球拍速度和拍面法向。

    沿法向的一维恢复方程:
      v_o_n - v_r_n = -C_r * (v_i_n - v_r_n)
      => v_r_n = (v_o_n + C_r * v_i_n) / (1 + C_r)
    """
    delta_v = v_outgoing - v_incoming
    delta_v_norm = float(np.linalg.norm(delta_v))

    if delta_v_norm < 1e-6:
      vin_norm = float(np.linalg.norm(v_incoming))
      if vin_norm > 1e-6:
        n = -v_incoming / vin_norm
      else:
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
      return np.zeros(3, dtype=np.float64), n

    u_hat = delta_v / delta_v_norm
    v_o_n = float(np.dot(v_outgoing, u_hat))
    v_i_n = float(np.dot(v_incoming, u_hat))
    v_r_n = (v_o_n + C_r * v_i_n) / (1.0 + C_r)

    return v_r_n * u_hat, u_hat

  def _check_net_clearance(
    self,
    p_strike: np.ndarray,
    v_outgoing: np.ndarray,
    margin: float = 0.03,
  ) -> tuple[bool, bool]:
    """检查回球是否能过网。

    Args:
      p_strike: 击球点位置
      v_outgoing: 出球速度
      margin: 过网上方余量 (m)

    Returns:
      (clears_net, bypasses_posts):
        clears_net: 球在 x=net_x 处高于 net_height + margin
        bypasses_posts: 球从网架 Y 范围外绕过
    """
    x_net = float(self.table.net_x)
    z_net = float(self.table.z_surface + self.table.net_height)
    y_half = float(self.table.y_max)  # 半桌宽

    dx = x_net - p_strike[0]
    if v_outgoing[0] <= 0:
      return False, False

    t_net = dx / v_outgoing[0]
    if t_net < 0:
      return False, False

    z_at_net = float(
      p_strike[2]
      + v_outgoing[2] * t_net
      + 0.5 * self._g[2] * t_net * t_net
    )
    y_at_net = float(p_strike[1] + v_outgoing[1] * t_net)

    y_net_min = -y_half - self.table.net_overhang
    y_net_max = y_half + self.table.net_overhang

    bypasses_posts = (y_at_net < y_net_min) or (y_at_net > y_net_max)
    if bypasses_posts:
      return False, True

    return z_at_net > (z_net + margin), False

  def plan(
    self,
    strike: StrikeTarget,
    target_land_xy: np.ndarray,
  ) -> RacketCommand:
    """为有效回击计算球拍目标状态。

    Args:
      strike: 第 2 阶段输出的预测击球状态
      target_land_xy: 目标落点 XY [2] (局部坐标)

    Returns:
      RacketCommand: 期望球拍状态
    """
    invalid = RacketCommand(
      p_intercept=strike.p_ball.copy(),
      v_racket=np.zeros(3, dtype=np.float64),
      n_racket=np.array([1.0, 0.0, 0.0], dtype=np.float64),
      t_strike=strike.t_strike,
      v_ball_outgoing=np.zeros(3, dtype=np.float64),
      target_land=np.zeros(3, dtype=np.float64),
      clears_net=False,
      bypasses_net_posts=False,
      valid=False,
    )

    if not strike.valid:
      return invalid

    p_strike = strike.p_ball.copy()
    v_incoming = strike.v_ball.copy()
    zt = float(self.table.z_surface + self.physics.radius)
    p_land = np.array(
      [float(target_land_xy[0]), float(target_land_xy[1]), zt],
      dtype=np.float64,
    )
    C_r = float(self.config.c_r)
    delta_t = float(self.config.delta_t_flight)

    v_outgoing = self._compute_outgoing_velocity(p_strike, p_land, delta_t)
    v_racket, n_racket = self._compute_racket_velocity(
      v_incoming, v_outgoing, C_r
    )
    clears, bypasses = self._check_net_clearance(p_strike, v_outgoing)

    # 如果无法过网，自动尝试替代飞行时间
    if not clears and not bypasses:
      for dt_adj in [0.4, 0.6, 0.35, 0.7, 0.3]:
        v_out_adj = self._compute_outgoing_velocity(
          p_strike, p_land, dt_adj
        )
        clears_adj, bypasses_adj = self._check_net_clearance(
          p_strike, v_out_adj
        )
        if clears_adj:
          v_outgoing = v_out_adj
          v_racket, n_racket = self._compute_racket_velocity(
            v_incoming, v_outgoing, C_r
          )
          clears, bypasses = True, bypasses_adj
          break

    return RacketCommand(
      p_intercept=p_strike,
      v_racket=v_racket,
      n_racket=n_racket,
      t_strike=strike.t_strike,
      v_ball_outgoing=v_outgoing,
      target_land=p_land,
      clears_net=clears,
      bypasses_net_posts=bypasses,
      valid=True,
    )


# ============================================================
# 顶层规划器 — 组合 Stage 1-3
# ============================================================


class HOPEPlanner:
  """组合阶段 1-3 + 正反手/base 目标的顶层规划器。

  按照仿真帧率使用每个球位置调用 .update()。
  通过 .racket_command 检索最新期望球拍状态。

  使用方法::

      planner = HOPEPlanner(physics, config, table)
      for each sim step:
          cmd = planner.update(
              episode_time,
              ball_pos_w,
              target_land_xy=...,
              p_base=robot_base_pos,
              base_quat=robot_base_quat,
          )
          if cmd is not None and cmd.valid:
              ...  # 使用 cmd.p_intercept, cmd.v_racket, cmd.t_strike,
                   # cmd.is_forehand, cmd.p_base_target 等
  """

  def __init__(
    self,
    physics: BallPhysics | None = None,
    config: PlannerConfig | None = None,
    table: TableParams | None = None,
  ):
    self.physics = physics or BallPhysics(
      k=0.0, c_h=0.75, c_v=0.85,
      g=(0.0, 0.0, -9.81), radius=0.02,
    )
    self.config = config or PlannerConfig(
      dt_integrate=0.001, max_predict_time=2.0,
      x_hit=0.40, delta_t_flight=0.5, c_r=0.88,
    )
    self.table = table or TableParams(
      x_min=0.63, x_max=3.37,
      y_min=-0.7625, y_max=0.7625,
      z_surface=0.76, net_x=2.0,
      net_height=0.1525, net_overhang=0.15,
    )

    self.estimator = BallStateEstimator(
      self.config, table_z_surface=self.table.z_surface
    )
    self.predictor = BallTrajectoryPredictor(
      self.physics, self.config, self.table
    )
    self.target_planner = RacketTargetPlanner(
      self.physics, self.config, self.table
    )

    self._latest_command: RacketCommand | None = None
    self._latest_strike: StrikeTarget | None = None
    self._target_land_xy: np.ndarray = np.zeros(2, dtype=np.float64)

  def reset(self) -> None:
    """重置规划器状态（episode 开始时调用）。"""
    self.estimator.reset()
    self._latest_command = None
    self._latest_strike = None

  @staticmethod
  def _yaw_from_quat(quat: np.ndarray) -> float:
    """从四元数 [w, x, y, z] 提取 yaw 角 (绕 Z 轴旋转)."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

  def _world_to_local_y(
    self,
    p_world: np.ndarray,
    p_base: np.ndarray,
    base_yaw: float,
  ) -> float:
    """将世界坐标点转换到机器人局部坐标系的 Y 分量."""
    dx = float(p_world[0] - p_base[0])
    dy = float(p_world[1] - p_base[1])
    return float(-np.sin(base_yaw) * dx + np.cos(base_yaw) * dy)

  def _choose_stroke(self, local_y: float) -> tuple[bool, bool]:
    """根据击球点在机器人局部 Y 坐标选择正手/反手."""
    fh_min = float(self.config.fh_rel_y_min)
    fh_max = float(self.config.fh_rel_y_max)
    bh_min = float(self.config.bh_rel_y_min)
    bh_max = float(self.config.bh_rel_y_max)

    in_fh = fh_min <= local_y <= fh_max
    in_bh = bh_min <= local_y <= bh_max
    gap_prefers_bh = fh_max < local_y < bh_min

    if local_y < fh_min:
      return True, True
    if in_fh:
      return True, True
    if gap_prefers_bh:
      return False, True
    if in_bh:
      return False, True
    return False, True

  def _compute_base_target(
    self,
    p_base_xy: np.ndarray,
    base_yaw: float,
    p_racket_xy: np.ndarray,
    is_forehand: bool,
  ) -> np.ndarray:
    """计算目标 base XY 使预测击球点落入选定击球方式的可达区域."""
    local_y = self._world_to_local_y(p_racket_xy, p_base_xy, base_yaw)

    if is_forehand:
      y_min = float(self.config.fh_rel_y_min)
      y_max = float(self.config.fh_rel_y_max)
    else:
      y_min = float(self.config.bh_rel_y_min)
      y_max = float(self.config.bh_rel_y_max)

    clamped_y = float(np.clip(local_y, y_min, y_max))
    excess = local_y - clamped_y
    if abs(excess) < 1.0e-6:
      return p_base_xy.copy()

    gain = float(self.config.base_out_of_range_gain)
    delta_x = float(-np.sin(base_yaw) * excess * gain)
    delta_y = float(np.cos(base_yaw) * excess * gain)
    return np.array(
      [float(p_base_xy[0]) + delta_x, float(p_base_xy[1]) + delta_y],
      dtype=np.float64,
    )

  def update(
    self,
    t: float,
    p_ball: np.ndarray,
    target_land_xy: np.ndarray | None = None,
    p_base: np.ndarray | None = None,
    base_quat: np.ndarray | None = None,
  ) -> RacketCommand | None:
    """处理新的球位置测量值。

    Args:
      t: 当前 episode 绝对时间 (秒)
      p_ball: 球位置 [x, y, z] (世界坐标)
      target_land_xy: 目标落点 XY [2] (局部坐标), 为 None 则使用上次的值
      p_base: 机器人 base 位置 [x, y, z] (世界坐标), 用于正反手/base 计算
      base_quat: 机器人 base 四元数 [w, x, y, z], 用于正反手/base 计算

    Returns:
      RacketCommand 或 None (缓冲区未就绪或球未向击球平面移动)
    """
    if target_land_xy is not None:
      self._target_land_xy = np.asarray(target_land_xy, dtype=np.float64)

    self.estimator.push(t, p_ball)

    if not self.estimator.ready:
      return None

    p_est, v_est, t_est = self.estimator.estimate()

    # 仅当球向击球平面移动时才进行预测 (v_x < 0)
    if v_est[0] >= 0:
      self._latest_command = None
      return None

    strike = self.predictor.predict(p_est, v_est, t_est)
    self._latest_strike = strike

    command = self.target_planner.plan(strike, self._target_land_xy)
    if command.valid and p_base is not None and base_quat is not None:
      p_base_arr = np.asarray(p_base, dtype=np.float64)
      base_quat_arr = np.asarray(base_quat, dtype=np.float64)
      base_yaw = self._yaw_from_quat(base_quat_arr)
      local_y = self._world_to_local_y(command.p_intercept, p_base_arr, base_yaw)
      is_forehand, confident = self._choose_stroke(local_y)
      command.is_forehand = is_forehand
      command.stroke_confident = confident
      command.p_base_target = self._compute_base_target(
        p_base_arr[:2].copy(),
        base_yaw,
        command.p_intercept[:2].copy(),
        is_forehand,
      )
    self._latest_command = command
    return command

  @property
  def racket_command(self) -> RacketCommand | None:
    return self._latest_command

  @property
  def latest_strike(self) -> StrikeTarget | None:
    return self._latest_strike

  @property
  def time_to_strike(self) -> float | None:
    """剩余击球时间 (秒)，需要在外部用当前时间减去。"""
    if self._latest_command is None or not self._latest_command.valid:
      return None
    return self._latest_command.t_strike
