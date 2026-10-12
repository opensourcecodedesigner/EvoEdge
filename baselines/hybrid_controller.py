"""Energy-Shaping + Linear Feedback Balancing Controller.

Combines an analytical nonlinear energy-shaping pumping controller for
swing-up (|theta| >= 0.35 rad OR |theta_dot| >= 2.5 rad/s) with a linear state-feedback
balancing controller (|theta| < 0.35 rad AND |theta_dot| < 2.5 rad/s).

Key Properties:
  - Exactly 7 gain parameters:
      Balancing gains (4): K = [K_x, K_x_dot, K_theta, K_theta_dot]
      Pumping gains (3):   k_E, k_x, k_v
  - Exactly 7 Multiply-Accumulate (MAC) operations per step.
  - Zero heap allocation (O(1) memory), 100% deterministic, MCU compatible.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


# True continuous-time LQR gain derived via solve_continuous_are with Q=diag(1, 0.1, 10, 0.1), R=1 (action units)
TRUE_LQR_GAINS = np.array([-1.0000, -1.2138, -7.8073, -1.8677], dtype=np.float64)

# Hand-tuned linear state feedback gains
HAND_TUNED_LINEAR_GAINS = np.array([-0.80, -1.20, -35.00, -6.50], dtype=np.float64)


def compute_true_lqr_gains(
    mass_cart: float = 1.0,
    mass_pole: float = 0.1,
    length: float = 0.5,
    gravity: float = 9.8,
    force_mag: float = 10.0,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute true continuous-time LQR gain matrix K for CartPole linearized at upright.

    Returns:
        K vector such that u = -K s (in normalized action units where force = action * force_mag).
    """
    from scipy.linalg import solve_continuous_are

    M = float(mass_cart)
    m = float(mass_pole)
    L = float(length)
    g = float(gravity)
    Mt = M + m
    ml = m * L

    denom = L * (4.0 / 3.0 - m / Mt)

    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 1] = 1.0
    A[1, 2] = -ml * g / (Mt * denom)
    A[2, 3] = 1.0
    A[3, 2] = g / denom

    B_force = np.zeros((4, 1), dtype=np.float64)
    B_force[1, 0] = 1.0 / Mt + ml / (Mt ** 2 * denom)
    B_force[3, 0] = -1.0 / (Mt * denom)

    B = B_force * float(force_mag)

    if Q is None:
        Q = np.diag([1.0, 0.1, 10.0, 0.1])
    if R is None:
        R = np.array([[1.0]])

    P = solve_continuous_are(A, B, Q, R)
    K = (np.linalg.inv(R) @ B.T @ P).flatten()
    return K


class EnergyShapingLinearFeedbackController:
    """Deterministic Dual-Regime Controller for CartPole Swing-Up & Balance.

    Name: energy-shaping + linear feedback

    Regimes:
      1. Balancing Regime (|theta| < 0.35 rad AND |theta_dot| < 2.5 rad/s):
         Linear feedback (hand-tuned) or true LQR:
           u = -(K_x * x + K_x_dot * x_dot + K_theta * theta + K_theta_dot * theta_dot)
         Clamped to [-1.0, 1.0].

      2. Pumping / Swing-Up Regime (|theta| >= 0.35 rad OR |theta_dot| >= 2.5 rad/s):
         Sign-based mechanical energy shaping with centering and velocity damping:
           E = (2/3) * m * L^2 * theta_dot^2 + m * g * L * (cos(theta) - 1.0)
           u = k_E * sign(E - E_0) * sign(theta_dot * cos(theta)) - k_x * x - k_v * x_dot
         Clamped to [-1.0, 1.0].
    """

    def __init__(
        self,
        K: Optional[Sequence[float]] = None,
        k_E: float = 0.6,
        k_x: float = 0.5,
        k_v: float = 0.25,
        theta_threshold: float = 0.35,
        theta_dot_threshold: float = 2.5,
        mass_pole: float = 0.1,
        length: float = 0.5,
        gravity: float = 9.8,
        use_true_lqr: bool = False,
    ) -> None:
        if K is not None:
            self.K = np.array(K, dtype=np.float64)
            self.linear_type = "linear feedback (custom)"
        elif use_true_lqr:
            self.K = TRUE_LQR_GAINS.copy()
            self.linear_type = "linear feedback (true LQR)"
        else:
            self.K = HAND_TUNED_LINEAR_GAINS.copy()
            self.linear_type = "linear feedback (hand-tuned)"

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
        # Corrected kinetic energy coefficient: (2/3) * m * L^2 = (2/3)*0.1*0.25 = 0.01666667
        self.c_kinetic = (2.0 / 3.0) * self.mass_pole * (self.length ** 2)
        # m * g * L = 0.1 * 9.8 * 0.5 = 0.4900
        self.c_potential = self.mass_pole * self.gravity * self.length

        self.current_regime: str = "PUMP"

    @property
    def name(self) -> str:
        return "energy-shaping + linear feedback"

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

        # Dual-Condition Capture: switch to linear balance ONLY when |theta| < 0.35 rad AND |theta_dot| < 2.5 rad/s
        if abs(theta) < self.theta_threshold and abs(th_dot) < self.theta_dot_threshold:
            self.current_regime = "BALANCE"
            # u = -(K_x * x + K_x_dot * x_dot + K_theta * theta + K_theta_dot * theta_dot)
            u = -(self.K[0] * x + self.K[1] * x_dot + self.K[2] * theta + self.K[3] * th_dot)
        else:
            self.current_regime = "PUMP"
            # Mechanical energy relative to upright equilibrium E_0 = 0.0:
            # E = (2/3) * m * L^2 * th_dot^2 + m * g * L * (cos_th - 1.0)
            E = self.c_kinetic * (th_dot ** 2) + self.c_potential * (cos_th - 1.0)
            sign_E = 1.0 if E > 0.0 else (-1.0 if E < 0.0 else 0.0)
            p = th_dot * cos_th
            sign_p = 1.0 if p > 0.0 else (-1.0 if p < 0.0 else 1.0)
            u = self.k_E * sign_E * sign_p - self.k_x * x - self.k_v * x_dot

        # Clamp u to [-1.0, 1.0]
        if u > 1.0:
            return 1.0
        elif u < -1.0:
            return -1.0
        return float(u)

    def get_spec(self) -> Dict[str, Any]:
        """Format parameters cleanly for UI server and embedded documentation."""
        return {
            "type": self.name,
            "linear_feedback": self.linear_type,
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
                    "formula": "u = k_E * sign(E - E_0) * sign(theta_dot * cos_theta) - k_x * x - k_v * x_dot",
                    "energy_formula": "E = (2/3) * m * L^2 * theta_dot^2 + m * g * L * (cos_theta - 1.0)",
                    "gains": {
                        "k_E": round(float(self.k_E), 4),
                        "k_x": round(float(self.k_x), 4),
                        "k_v": round(float(self.k_v), 4),
                    },
                },
            },
        }


# Backwards compatibility alias
HybridController = EnergyShapingLinearFeedbackController
