"""CartPole Swing-Up environment.

A variant of the classic CartPole where the pole starts hanging downward
(θ = π rad) and the agent must swing it up to the inverted position (θ = 0)
while keeping the cart near the centre of the track.

Implements gymnasium.Env so it can be used with any standard RL or
neuroevolution loop.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CartPoleSwingUpEnv(gym.Env):
    """CartPole Swing-Up with continuous or discrete force control.

    Observation (5-dim continuous):
        Index  Meaning
        0      Cart position              x       ∈ [−4.8, 4.8]
        1      Cart velocity              ẋ
        2      cos(pole angle)            cos θ   ∈ [−1, 1]
        3      sin(pole angle)            sin θ   ∈ [−1, 1]
        4      Pole angular velocity      θ̇

    Action:
        continuous  →  scalar ∈ [−1, 1], scaled to [−force_mag, force_mag]
        discrete    →  {0, 1} mapped to {−force_mag, +force_mag}

    Reward per step:
        cos(θ) − 0.01·(x / x_threshold)²
        +1 when the pole is perfectly upright and the cart is centred;
        −1 when the pole hangs straight down.

    Termination:
        |x| > x_threshold  (cart fell off the track).

    Truncation:
        step count ≥ max_steps.
    """

    metadata: dict = {"render_modes": []}

    # ── Physics constants ────────────────────────────────────
    GRAVITY: float = 9.8
    MASS_CART: float = 1.0
    MASS_POLE: float = 0.1
    LENGTH: float = 0.5        # half-length of the pole
    FORCE_MAG: float = 10.0
    TAU: float = 0.02          # integration timestep (s)
    X_THRESHOLD: float = 2.4   # track half-width

    def __init__(
        self,
        max_steps: int = 200,
        continuous: bool = True,
    ) -> None:
        super().__init__()

        self.max_steps = max_steps
        self.continuous = continuous

        # Derived constants (cached for speed)
        self._total_mass = self.MASS_CART + self.MASS_POLE
        self._polemass_length = self.MASS_POLE * self.LENGTH

        # Action space
        if continuous:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(2)

        # Observation space: [x, x_dot, cos_theta, sin_theta, theta_dot]
        high = np.array(
            [
                self.X_THRESHOLD * 2.0,
                np.finfo(np.float32).max,
                1.0,
                1.0,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Internal state: [x, x_dot, theta, theta_dot]
        self._state: np.ndarray | None = None
        self._step_count: int = 0

    # ── Gymnasium API ────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Pole starts hanging down (θ = π) with a small random perturbation
        self._state = np.array(
            [0.0, 0.0, np.pi, 0.0], dtype=np.float64
        )
        self._state += self.np_random.uniform(
            low=-0.05, high=0.05, size=(4,)
        )
        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self, action
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()."

        x, x_dot, theta, theta_dot = self._state

        # ── Compute applied force ────────────────────────────
        if self.continuous:
            if isinstance(action, (list, np.ndarray)):
                action = action[0]
            force = float(np.clip(action, -1.0, 1.0)) * self.FORCE_MAG
        else:
            force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        # ── Cart-pole dynamics (semi-implicit Euler) ─────────
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (
            force + self._polemass_length * theta_dot ** 2 * sin_theta
        ) / self._total_mass

        theta_acc = (
            self.GRAVITY * sin_theta - cos_theta * temp
        ) / (
            self.LENGTH
            * (4.0 / 3.0 - self.MASS_POLE * cos_theta ** 2 / self._total_mass)
        )

        x_acc = (
            temp
            - self._polemass_length * theta_acc * cos_theta / self._total_mass
        )

        # Euler integration
        x_dot = x_dot + self.TAU * x_acc
        x = x + self.TAU * x_dot
        theta_dot = theta_dot + self.TAU * theta_acc
        theta = theta + self.TAU * theta_dot

        # Wrap θ to [−π, π]
        theta = ((theta + np.pi) % (2.0 * np.pi)) - np.pi

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self._step_count += 1

        # ── Reward ───────────────────────────────────────────
        reward = float(np.cos(theta)) - 0.01 * (x / self.X_THRESHOLD) ** 2

        # ── Done conditions ──────────────────────────────────
        terminated = bool(abs(x) > self.X_THRESHOLD)
        truncated = self._step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    # ── Helpers ──────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return the 5-dim observation from the internal 4-dim state."""
        x, x_dot, theta, theta_dot = self._state
        return np.array(
            [x, x_dot, np.cos(theta), np.sin(theta), theta_dot],
            dtype=np.float32,
        )
