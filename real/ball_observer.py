from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time

import numpy as np


@dataclass
class BallObservation:
    valid: bool
    pos_w: np.ndarray
    vel_w: np.ndarray
    timestamp_s: float
    source: str
    reason: str = ""


class FiniteDifferenceVelocityEstimator:
    def __init__(self, max_dt_s: float = 0.20, alpha: float = 0.40):
        self.max_dt_s = float(max_dt_s)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._last_t: float | None = None
        self._last_pos: np.ndarray | None = None
        self._vel = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self._last_t = None
        self._last_pos = None
        self._vel[:] = 0.0

    def update(self, pos_w: np.ndarray, timestamp_s: float) -> np.ndarray:
        pos_w = np.asarray(pos_w, dtype=np.float32).reshape(3)
        timestamp_s = float(timestamp_s)
        if self._last_t is None or self._last_pos is None:
            self._last_t = timestamp_s
            self._last_pos = pos_w.copy()
            self._vel[:] = 0.0
            return self._vel.copy()

        dt = timestamp_s - self._last_t
        if dt <= 1.0e-6 or dt > self.max_dt_s:
            self._last_t = timestamp_s
            self._last_pos = pos_w.copy()
            self._vel[:] = 0.0
            return self._vel.copy()

        raw_vel = (pos_w - self._last_pos) / dt
        self._vel = (1.0 - self.alpha) * self._vel + self.alpha * raw_vel.astype(np.float32)
        self._last_t = timestamp_s
        self._last_pos = pos_w.copy()
        return self._vel.copy()


class ConstantBallObserver:
    def __init__(self, ball_pos_w, ball_vel_w):
        self.ball_pos_w = np.asarray(ball_pos_w, dtype=np.float32).reshape(3)
        self.ball_vel_w = np.asarray(ball_vel_w, dtype=np.float32).reshape(3)

    def update(self) -> BallObservation:
        return BallObservation(
            valid=True,
            pos_w=self.ball_pos_w.copy(),
            vel_w=self.ball_vel_w.copy(),
            timestamp_s=time.time(),
            source="constant",
        )


class UdpJsonBallObserver:
    """Receive ball observations from a UDP JSON stream.

    Expected packet examples:
      {"pos": [x, y, z], "vel": [vx, vy, vz], "t": 123.4}
      {"ball_pos_w": [x, y, z], "timestamp": 123.4}

    If velocity is omitted, it is estimated by finite difference.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 15050, timeout_s: float = 0.0):
        self.host = host
        self.port = int(port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(float(timeout_s))
        self.vel_estimator = FiniteDifferenceVelocityEstimator()
        self._last_obs: BallObservation | None = None

    def close(self) -> None:
        self.sock.close()

    def update(self) -> BallObservation:
        latest_payload = None
        while True:
            try:
                payload, _addr = self.sock.recvfrom(4096)
                latest_payload = payload
            except (BlockingIOError, socket.timeout):
                break

        if latest_payload is None:
            if self._last_obs is not None:
                return self._last_obs
            return BallObservation(
                valid=False,
                pos_w=np.zeros(3, dtype=np.float32),
                vel_w=np.zeros(3, dtype=np.float32),
                timestamp_s=time.time(),
                source="udp_json",
                reason="no_packet",
            )

        try:
            msg = json.loads(latest_payload.decode("utf-8"))
            pos = msg.get("pos", msg.get("ball_pos_w", None))
            if pos is None:
                raise ValueError("missing pos / ball_pos_w")
            pos_w = np.asarray(pos, dtype=np.float32).reshape(3)
            timestamp_s = float(msg.get("t", msg.get("timestamp", time.time())))
            if "vel" in msg:
                vel_w = np.asarray(msg["vel"], dtype=np.float32).reshape(3)
            elif "ball_vel_w" in msg:
                vel_w = np.asarray(msg["ball_vel_w"], dtype=np.float32).reshape(3)
            else:
                vel_w = self.vel_estimator.update(pos_w, timestamp_s)
            obs = BallObservation(True, pos_w, vel_w, timestamp_s, "udp_json")
            self._last_obs = obs
            return obs
        except Exception as exc:
            return BallObservation(
                valid=False,
                pos_w=np.zeros(3, dtype=np.float32),
                vel_w=np.zeros(3, dtype=np.float32),
                timestamp_s=time.time(),
                source="udp_json",
                reason=str(exc),
            )
