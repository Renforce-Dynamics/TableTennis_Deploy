"""两个规划器共享的基础数据结构。

HopeModelBasedPlanner 和 HOPEPlanner 都使用这些 dataclass。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TableParams:
  """球桌几何参数 (局部坐标系)."""
  x_min: float       # 桌面前缘 X
  x_max: float       # 桌面后缘 X
  y_min: float       # 桌面左缘 Y
  y_max: float       # 桌面右缘 Y
  z_surface: float   # 桌面高度
  net_x: float       # 球网 X 位置
  net_height: float  # 球网高度 (从桌面起算)
  net_overhang: float  # 球网 Y 方向超出桌面的距离


@dataclass
class BallPhysics:
  """球的物理参数."""
  k: float                          # 空气阻力系数 (二次: a = -k|v|v)
  c_h: float                        # 台面水平恢复系数
  c_v: float                        # 台面垂直恢复系数
  g: tuple[float, float, float]     # 重力加速度 (m/s^2)
  radius: float                     # 球半径 (m)


@dataclass
class PlannerConfig:
  """规划器通用配置."""
  dt_integrate: float       # 积分步长 (s)
  max_predict_time: float   # 最大预测时间 (s)
  x_hit: float              # 击球平面 X 坐标
  delta_t_flight: float     # 假设击球→落点的飞行时间 (s)
  c_r: float                # 球拍恢复系数
  # HOPE planner 专用字段 (带默认值向后兼容)
  fit_window: int = 31          # 多项式拟合窗口大小 (样本数)
  poly_order: int = 2           # 多项式阶数
  bounce_z_tol: float = 0.02    # 弹跳检测 Z 容差 (m)
