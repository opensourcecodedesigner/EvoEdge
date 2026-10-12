"""Linear Quadratic Regulator (LQR) baseline for CartPole.

Formulates the linearized state-space dynamics around the upright equilibrium
(x = [x, x_dot, theta, theta_dot]^T = 0) using exact physical constants from
CartPoleSwingUpEnv, solves the Algebraic Riccati Equation (ARE / DARE), and
provides a closed-loop state-feedback controller.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np

try:
    from scipy.linalg import solve_continuous_are, solve_discrete_are
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── System Linearization ─────────────────────────────────────────────────────


def get_cartpole_linearized_system(
    mass_cart: float = 1.0,
    mass_pole: float = 0.1,
    length: float = 0.5,
    gravity: float = 9.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute continuous-time linearized state-space matrices (A, B).

    The system is linearized around the upright equilibrium:
        x = [x, x_dot, theta, theta_dot]^T = [0, 0, 0, 0]^T
        u = 0 (force)

    Dynamics from CartPoleSwingUpEnv:
        temp = (u + m*L*theta_dot^2 * sin(theta)) / (M + m)
        theta_acc = (g*sin(theta) - cos(theta)*temp) / (L * (4/3 - m*cos(theta)^2 / (M + m)))
        x_acc = temp - (m*L*theta_acc*cos(theta)) / (M + m)

    Linearized around theta = 0, theta_dot = 0, u = 0:
        total_mass = M + m
        denom = L * (4/3 - m / total_mass)
        d(theta_acc)/d(theta) = g / denom
        d(theta_acc)/d(u)     = -1 / (total_mass * denom)
        d(x_acc)/d(theta)     = - (m * L * g) / (total_mass * denom)
        d(x_acc)/d(u)         = (1 / total_mass) * (1 + (m * L) / (total_mass * denom))

    Returns:
        Tuple (A, B) where A is (4, 4) and B is (4, 1).
    """
    total_mass = mass_cart + mass_pole
    polemass_length = mass_pole * length
    denom = length * (4.0 / 3.0 - mass_pole / total_mass)

    a_12 = - (polemass_length * gravity) / (total_mass * denom)
    a_32 = gravity / denom

    b_10 = (1.0 / total_mass) * (1.0 + polemass_length / (total_mass * denom))
    b_30 = - 1.0 / (total_mass * denom)

    A = np.array([
        [0.0, 1.0,  0.0, 0.0],
        [0.0, 0.0, a_12, 0.0],
        [0.0, 0.0,  0.0, 1.0],
        [0.0, 0.0, a_32, 0.0],
    ], dtype=np.float64)

    B = np.array([
        [0.0],
        [b_10],
        [0.0],
        [b_30],
    ], dtype=np.float64)

    return A, B


def discretize_linear_system(
    A: np.ndarray,
    B: np.ndarray,
    dt: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize continuous system (A, B) using zero-order hold approximation.

    A_d = exp(A * dt) ≈ I + A*dt + 0.5*A^2*dt^2
    B_d = ∫_0^dt exp(A*tau) dtau * B ≈ (I*dt + 0.5*A*dt^2)*B
    """
    n = A.shape[0]
    I = np.eye(n, dtype=np.float64)
    A_dt = A * dt
    A2_dt2 = (A @ A) * (0.5 * dt * dt)

    Ad = I + A_dt + A2_dt2
    Bd = (I * dt + 0.5 * A * (dt * dt)) @ B
    return Ad, Bd


# ── Algebraic Riccati Equation Solvers ───────────────────────────────────────


def solve_dare_iterative(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve Discrete Algebraic Riccati Equation (DARE) using iterative value iteration.

    Pure NumPy implementation:
        P_{k+1} = Ad^T P_k Ad - (Ad^T P_k Bd)(R + Bd^T P_k Bd)^(-1)(Bd^T P_k Ad) + Q

    Returns:
        Tuple (P, K) where K is the discrete feedback gain matrix (1, 4).
    """
    P = Q.copy().astype(np.float64)
    R_mat = R.astype(np.float64)

    for _ in range(max_iter):
        Bt_P = Bd.T @ P
        Bt_P_Bd = Bt_P @ Bd
        inv_term = np.linalg.inv(R_mat + Bt_P_Bd)

        P_next = Ad.T @ P @ Ad - (Ad.T @ P @ Bd) @ inv_term @ (Bt_P @ Ad) + Q
        diff = np.max(np.abs(P_next - P))
        P = P_next
        if diff < tol:
            break

    K = np.linalg.inv(R_mat + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    return P, K


def solve_lqr_gain(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    dt: float = 0.02,
    method: str = "continuous",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute LQR gain matrix K using continuous CARE or discrete DARE.

    Args:
        A: System dynamics matrix (4, 4)
        B: Input matrix (4, 1)
        Q: State cost matrix (4, 4)
        R: Control cost matrix (1, 1)
        dt: Discretization timestep (s)
        method: "continuous" or "discrete"

    Returns:
        Tuple (P, K) with optimal gain K of shape (1, 4).
    """
    if method == "continuous" and _HAS_SCIPY:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ (B.T @ P)
        return P, K

    # Discrete formulation (or fallback when scipy CARE is unavailable)
    Ad, Bd = discretize_linear_system(A, B, dt=dt)
    if _HAS_SCIPY and method == "discrete_scipy":
        P = solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
        return P, K

    # Iterative pure NumPy DARE solver
    P, K = solve_dare_iterative(Ad, Bd, Q, R)
    return P, K


# ── LQR Controller Class ─────────────────────────────────────────────────────


class LQRController:
    """Analytical Linear Quadratic Regulator (LQR) for inverted CartPole.

    Linearizes dynamics at the upright equilibrium [0, 0, 0, 0]^T and computes
    state-feedback law:
        u_force = - K * x

    Features:
        - Exact physical parameters matching CartPoleSwingUpEnv.
        - Robust coordinate mapping: handles 4D [x, x_dot, theta, theta_dot] or
          5D [x, x_dot, cos(theta), sin(theta), theta_dot].
        - Clamping: clamps control output to actuator limits [-10.0, 10.0] N or
          normalized [-1.0, 1.0] for direct use in gym environment step().
    """

    def __init__(
        self,
        Q: Optional[np.ndarray] = None,
        R: Optional[Union[np.ndarray, float]] = None,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        length: float = 0.5,
        gravity: float = 9.8,
        force_mag: float = 10.0,
        dt: float = 0.02,
        method: str = "continuous",
    ) -> None:
        self.mass_cart = mass_cart
        self.mass_pole = mass_pole
        self.length = length
        self.gravity = gravity
        self.force_mag = force_mag
        self.dt = dt
        self.method = method

        # Cost matrices
        if Q is None:
            self.Q = np.diag([1.0, 0.1, 10.0, 1.0]).astype(np.float64)
        else:
            self.Q = np.array(Q, dtype=np.float64)

        if R is None:
            self.R = np.array([[0.01]], dtype=np.float64)
        elif isinstance(R, (int, float)):
            self.R = np.array([[float(R)]], dtype=np.float64)
        else:
            self.R = np.array(R, dtype=np.float64)
            if self.R.ndim == 1:
                self.R = self.R.reshape(1, 1)

        # Compute linearized continuous state space
        self.A, self.B = get_cartpole_linearized_system(
            mass_cart=self.mass_cart,
            mass_pole=self.mass_pole,
            length=self.length,
            gravity=self.gravity,
        )

        # Solve for optimal feedback gain K (1, 4)
        self.P, self.K = solve_lqr_gain(
            self.A,
            self.B,
            self.Q,
            self.R,
            dt=self.dt,
            method=self.method,
        )

        # Flat 1D vector of gains [K_x, K_x_dot, K_theta, K_theta_dot]
        self.gain_vector = self.K.flatten()

    @property
    def active_parameters(self) -> int:
        """Total parameters (4 feedback gains)."""
        return 4

    @property
    def theoretical_macs(self) -> int:
        """Theoretical Multiply-Accumulate operations per control step."""
        return 4

    def extract_state(self, obs: Union[np.ndarray, list]) -> np.ndarray:
        """Extract [x, x_dot, theta, theta_dot] from 4D or 5D observation vector.

        Handles:
            - 4-dim: [x, x_dot, theta, theta_dot]
            - 5-dim: [x, x_dot, cos(theta), sin(theta), theta_dot]
        """
        arr = np.asarray(obs, dtype=np.float64).flatten()
        if len(arr) == 4:
            return arr
        elif len(arr) == 5:
            x = arr[0]
            x_dot = arr[1]
            cos_th = arr[2]
            sin_th = arr[3]
            th_dot = arr[4]
            theta = float(np.arctan2(sin_th, cos_th))
            return np.array([x, x_dot, theta, th_dot], dtype=np.float64)
        else:
            raise ValueError(
                f"Expected observation dimension 4 or 5, got {len(arr)}"
            )

    def compute_action(
        self,
        obs: Union[np.ndarray, list],
        normalized: bool = True,
    ) -> float:
        """Compute LQR feedback action for observation *obs*.

        Args:
            obs: 4D or 5D environment observation.
            normalized: If True (default), scales output to [-1.0, 1.0] for gym env.
                        If False, returns raw force in [-force_mag, force_mag] N.

        Returns:
            Scalar control action clamped to appropriate bounds.
        """
        state = self.extract_state(obs)
        # u = - K * x
        u_force = - float(np.dot(self.gain_vector, state))

        # Clamp raw force to [-force_mag, force_mag]
        u_force_clamped = float(np.clip(u_force, -self.force_mag, self.force_mag))

        if normalized:
            return float(np.clip(u_force_clamped / self.force_mag, -1.0, 1.0))
        return u_force_clamped

    def __call__(self, obs: Union[np.ndarray, list]) -> float:
        """Convenience alias for compute_action(obs, normalized=True)."""
        return self.compute_action(obs, normalized=True)
