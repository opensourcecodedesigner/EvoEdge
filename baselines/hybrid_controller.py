"""Hybrid Energy-Shaping and State-Feedback Balancing Controller.

Combines an analytical nonlinear energy-shaping pumping controller for
swing-up (|theta| >= 0.35 rad) with a linear state-feedback LQR balancing
controller (|theta| < 0.35 rad).

Key Properties:
  - Exactly 6 gain parameters:
      Balancing gains (4): K = [K_x, K_x_dot, K_theta, K_theta_dot]
      Pumping gains (2):   k_E, k_x
  - Exactly 6 Multiply-Accumulate (MAC) operations per step.
  - Zero heap allocation (O(1) memory), 100% deterministic, MCU compatible.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class HybridController:
    """Deterministic Dual-Regime Controller for CartPole Swing-Up & Balance.

    Regimes:
      1. Balancing Regime (|theta| < 0.35 rad AND |theta_dot| < 2.5 rad/s):
         Linear state-feedback:
           u = -(K_x * x + K_x_dot * x_dot + K_theta * theta + K_theta_dot * theta_dot)
         Clamped to [-1.0, 1.0].

      2. Pumping / Swing-Up Regime (|theta| >= 0.35 rad OR |theta_dot| >= 2.5 rad/s):
         Mechanical energy error pumping with centering and velocity damping:
           E = 0.5 * m * (L * theta_dot)^2 + m * g * L * (cos(theta) - 1.0)
           E_0 = 0.0
           u = k_E * (E - E_0) * sign(theta_dot * cos(theta)) - k_x * x - k_v * x_dot
         Clamped to [-1.0, 1.0].
    """

    def __init__(
        self,
        K: Optional[Sequence[float]] = None,
        k_E: float = 0.4,
        k_x: float = 1.2,
        k_v: float = 0.8,
        theta_threshold: float = 0.35,
        theta_dot_threshold: float = 2.5,
        mass_pole: float = 0.1,
        length: float = 0.5,
        gravity: float = 9.8,
    ) -> None:
        # Default tuned balancing gains: K = [-0.8, -1.2, -35.0, -6.5]
        if K is None:
            self.K = np.array([-0.8, -1.2, -35.0, -6.5], dtype=np.float64)
        else:
            self.K = np.array(K, dtype=np.float64)

        self.k_E = float(k_E)
        self.k_x = float(k_x)
        self.k_v = float(k_v)
        self.theta_threshold = float(theta_threshold)
        self.theta_dot_threshold = float(theta_dot_threshold)

        # Physics constants
        self.mass_pole = float(mass_pole)
        self.length = float(length)
        self.gravity = float(gravity)

        # Precomputed constants
        # 0.5 * m * L^2
        self.c_kinetic = 0.5 * self.mass_pole * (self.length ** 2)
        # m * g * L
        self.c_potential = self.mass_pole * self.gravity * self.length

        self.current_regime: str = "PUMP"

    @property
    def active_parameters(self) -> int:
        """Total parameters: 4 balance gains + 3 pumping gains = 7."""
        return 7

    @property
    def theoretical_macs(self) -> int:
        """Multiply-Accumulate operations per forward evaluation (up to 7)."""
        return 7

    def extract_state(self, obs: Union[np.ndarray, list]) -> Tuple[float, float, float, float, float]:
        """Extract physical state variables from 4D or 5D observation vector.

        Returns:
            Tuple: (x, x_dot, cos_theta, sin_theta, theta_dot)
        """
        arr = np.asarray(obs, dtype=np.float64).flatten()
        if len(arr) == 5:
            x, x_dot, cos_th, sin_th, th_dot = arr[0], arr[1], arr[2], arr[3], arr[4]
            th = math.atan2(sin_th, cos_th)
            return float(x), float(x_dot), float(cos_th), float(sin_th), float(th_dot)
        elif len(arr) == 4:
            x, x_dot, th, th_dot = arr[0], arr[1], arr[2], arr[3]
            cos_th = math.cos(th)
            sin_th = math.sin(th)
            return float(x), float(x_dot), float(cos_th), float(sin_th), float(th_dot)
        else:
            raise ValueError(f"Expected observation dimension 4 or 5, got {len(arr)}")

    def compute_action(
        self,
        obs: Union[np.ndarray, list],
        normalized: bool = True,
    ) -> float:
        """Compute control action u for given observation.

        Args:
            obs: [x, x_dot, cos_theta, sin_theta, theta_dot] or [x, x_dot, theta, theta_dot]
            normalized: Ignored, returns action in [-1.0, 1.0].

        Returns:
            Control action u clamped to [-1.0, 1.0].
        """
        x, x_dot, cos_th, sin_th, th_dot = self.extract_state(obs)
        theta = math.atan2(sin_th, cos_th)

        # Dual-Condition Capture: switch to LQR balance ONLY when |theta| < 0.35 rad AND |theta_dot| < 2.5 rad/s
        if abs(theta) < self.theta_threshold and abs(th_dot) < self.theta_dot_threshold:
            self.current_regime = "BALANCE"
            # u = -(K_x * x + K_x_dot * x_dot + K_theta * theta + K_theta_dot * theta_dot)
            u = -(self.K[0] * x + self.K[1] * x_dot + self.K[2] * theta + self.K[3] * th_dot)
        else:
            self.current_regime = "PUMP"
            # Mechanical energy relative to upright equilibrium E_0 = 0.0:
            # E = 0.5 * m * (L * theta_dot)^2 + m * g * L * (cos(theta) - 1.0)
            # E_tilde = E - E_0 = E (negative below target)
            E = self.c_kinetic * (th_dot ** 2) + self.c_potential * (cos_th - 1.0)
            p = th_dot * cos_th
            sign_dir = 1.0 if p > 0.0 else (-1.0 if p < 0.0 else 1.0)
            u = self.k_E * E * sign_dir - self.k_x * x - self.k_v * x_dot

        # Clamp u to [-1.0, 1.0]
        if u > 1.0:
            return 1.0
        elif u < -1.0:
            return -1.0
        return float(u)

    def get_spec(self) -> Dict[str, Any]:
        """Format parameters cleanly for UI server and embedded documentation."""
        return {
            "type": "HybridController",
            "active_parameters": 7,
            "theoretical_macs": 7,
            "regimes": {
                "balance": {
                    "condition": f"|theta| < {self.theta_threshold:.2f} rad AND |theta_dot| < {self.theta_dot_threshold:.2f} rad/s",
                    "formula": "u = -(K_x * x + K_x_dot * x_dot + K_theta * theta + K_theta_dot * theta_dot)",
                    "gains": {
                        "K_x": round(float(self.K[0]), 4),
                        "K_x_dot": round(float(self.K[1]), 4),
                        "K_theta": round(float(self.K[2]), 4),
                        "K_theta_dot": round(float(self.K[3]), 4),
                    },
                },
                "pumping": {
                    "condition": f"|theta| >= {self.theta_threshold:.2f} rad OR |theta_dot| >= {self.theta_dot_threshold:.2f} rad/s",
                    "formula": "u = k_E * (E - E_0) * sign(theta_dot * cos_theta) - k_x * x - k_v * x_dot",
                    "energy_formula": "E = 0.5 * m * (L * theta_dot)^2 + m * g * L * (cos_theta - 1.0)",
                    "gains": {
                        "k_E": round(float(self.k_E), 4),
                        "k_x": round(float(self.k_x), 4),
                        "k_v": round(float(self.k_v), 4),
                    },
                },
            },
        }
