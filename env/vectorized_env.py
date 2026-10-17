"""Vectorized CartPole Swing-Up Environment for High-Throughput Neuroevolution.

Simulates N environment instances concurrently in NumPy using float64 dynamics,
matching PerturbedCartPoleEnv step-by-step with zero divergence.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class VectorizedCartPoleEnv:
    """Vectorized Batch CartPole Simulator with Perturbation Support.

    Supports batch evaluation of B parallel environments across N timesteps.
    Maintains exact numerical float64 state parity with PerturbedCartPoleEnv.
    """

    GRAVITY: float = 9.8
    MASS_CART: float = 1.0
    MASS_POLE: float = 0.1
    LENGTH: float = 0.5
    FORCE_MAG: float = 10.0
    TAU: float = 0.02
    X_THRESHOLD: float = 2.4

    def __init__(
        self,
        num_envs: int,
        max_steps: int = 500,
        cart_mass: Union[float, np.ndarray] = 1.0,
        pole_mass: Union[float, np.ndarray] = 0.1,
        b_cart: Union[float, np.ndarray] = 0.0,
        b_pole: Union[float, np.ndarray] = 0.0,
        obs_noise_sigma: Union[float, np.ndarray] = 0.0,
        delay_steps: int = 0,
        noise_channels: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        reward_mode: str = "legacy",
    ):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.delay_steps = delay_steps
        self.reward_mode = reward_mode
        self.noise_channels = [0, 1, 2, 3, 4] if noise_channels is None else list(noise_channels)

        # Broadcast physical parameters to arrays of shape (num_envs,)
        self.cart_mass = np.full(num_envs, cart_mass, dtype=np.float64)
        self.pole_mass = np.full(num_envs, pole_mass, dtype=np.float64)
        self.b_cart = np.full(num_envs, b_cart, dtype=np.float64)
        self.b_pole = np.full(num_envs, b_pole, dtype=np.float64)
        self.obs_noise_sigma = np.full(num_envs, obs_noise_sigma, dtype=np.float64)

        self._total_mass = self.cart_mass + self.pole_mass
        self._polemass_length = self.pole_mass * self.LENGTH

        # State arrays of shape (num_envs, 4): [x, x_dot, theta, theta_dot]
        self._state = np.zeros((num_envs, 4), dtype=np.float64)
        self._step_count = np.zeros(num_envs, dtype=np.int32)
        self.boundary_violated = np.zeros(num_envs, dtype=bool)
        self.terminated = np.zeros(num_envs, dtype=bool)

    @property
    def state(self) -> np.ndarray:
        """State array of shape (num_envs, 4)."""
        return self._state

        # Delay queue: deque of arrays shape (num_envs,) matching PerturbedCartPoleEnv queue exactly
        self.action_queue = collections.deque(
            [np.zeros(num_envs, dtype=np.float64) for _ in range(delay_steps)]
        )

        # Per-environment independent RNG generators
        self.rngs = [np.random.default_rng(seeds[i] if seeds is not None else None) for i in range(num_envs)]

    def reset(self, seeds: Optional[List[int]] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset all environments in the batch with their respective seeds."""
        if seeds is not None:
            assert len(seeds) == self.num_envs
            self.rngs = [np.random.default_rng(s) for s in seeds]

        self._step_count.fill(0)
        self.boundary_violated.fill(False)
        self.terminated.fill(False)

        if self.delay_steps > 0:
            self.action_queue = collections.deque(
                [np.zeros(self.num_envs, dtype=np.float64) for _ in range(self.delay_steps)]
            )

        # Pole starts hanging down (theta = pi) with uniform noise [-0.05, 0.05]
        for i in range(self.num_envs):
            perturb = self.rngs[i].uniform(low=-0.05, high=0.05, size=(4,))
            self._state[i] = np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float64) + perturb

        # In PerturbedCartPoleEnv, super().reset() uses Gymnasium's internal RNG for the state perturbation,
        # and then PerturbedCartPoleEnv sets self._rng = np.random.default_rng(seed) for subsequent noise!
        if seeds is not None:
            self.rngs = [np.random.default_rng(s) for s in seeds]

        return self._get_noisy_obs(), [{} for _ in range(self.num_envs)]

    def _get_noisy_obs(self) -> np.ndarray:
        """Compute 5-dimensional observation vector with per-environment noise."""
        x = self._state[:, 0]
        x_dot = self._state[:, 1]
        theta = self._state[:, 2]
        theta_dot = self._state[:, 3]

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        obs = np.stack([x, x_dot, cos_th, sin_th, theta_dot], axis=1).astype(np.float32)

        # Apply noise if sigma > 0
        for i in range(self.num_envs):
            sig = self.obs_noise_sigma[i]
            if sig > 0.0:
                noise = np.zeros(5, dtype=np.float32)
                for ch in self.noise_channels:
                    noise[ch] = self.rngs[i].normal(0.0, sig)
                obs[i] += noise

        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Execute one simulation step across all environments concurrently.

        Args:
            actions: Array of shape (num_envs,) in [-1.0, 1.0]

        Returns:
            obs: (num_envs, 5) float32
            rewards: (num_envs,) float64
            terminated: (num_envs,) bool
            truncated: (num_envs,) bool
            infos: list of dicts with 'applied_action'
        """
        actions = np.asarray(actions, dtype=np.float64)

        if self.delay_steps > 0:
            self.action_queue.append(actions.copy())
            applied_actions = self.action_queue.popleft()
        else:
            applied_actions = actions

        force = np.clip(applied_actions, -1.0, 1.0) * self.FORCE_MAG

        x = self._state[:, 0]
        x_dot = self._state[:, 1]
        theta = self._state[:, 2]
        theta_dot = self._state[:, 3]

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        # Net cart horizontal force including rail viscous drag -b_cart * x_dot
        temp = (
            force - self.b_cart * x_dot + self._polemass_length * (theta_dot ** 2) * sin_th
        ) / self._total_mass

        # Rotational damping torque -b_pole * theta_dot
        pml = np.maximum(self._polemass_length, 1e-6)
        fric_pole = (self.b_pole / pml) * theta_dot

        theta_acc = (
            self.GRAVITY * sin_th - cos_th * temp - fric_pole
        ) / (
            self.LENGTH * (4.0 / 3.0 - self.pole_mass * (cos_th ** 2) / self._total_mass)
        )

        x_acc = temp - self._polemass_length * theta_acc * cos_th / self._total_mass

        # Semi-implicit Euler integration
        new_x_dot = x_dot + self.TAU * x_acc
        new_x = x + self.TAU * new_x_dot
        new_theta_dot = theta_dot + self.TAU * theta_acc
        new_theta = theta + self.TAU * new_theta_dot

        # Wrap theta to [-pi, pi]
        new_theta = ((new_theta + np.pi) % (2.0 * np.pi)) - np.pi

        # Update active states
        self._state[:, 0] = new_x
        self._state[:, 1] = new_x_dot
        self._state[:, 2] = new_theta
        self._state[:, 3] = new_theta_dot

        self._step_count += 1

        rail_hits = np.abs(new_x) >= self.X_THRESHOLD
        self.boundary_violated |= rail_hits

        term = np.abs(new_x) > self.X_THRESHOLD
        self.terminated |= term
        trunc = self._step_count >= self.max_steps

        if self.reward_mode == "stage3b":
            from env.rewards import compute_stage3b_batch_reward
            rewards = compute_stage3b_batch_reward(
                x=new_x,
                x_dot=new_x_dot,
                theta=new_theta,
                theta_dot=new_theta_dot,
                actions=force / self.FORCE_MAG,
                step_counts=self._step_count,
                max_steps=self.max_steps,
                x_threshold=self.X_THRESHOLD,
                boundary_violated=self.boundary_violated,
            )
        else:
            rewards = np.cos(new_theta) - 0.01 * (new_x / self.X_THRESHOLD) ** 2

        obs = self._get_noisy_obs()
        infos = [
            {
                "applied_action": float(applied_actions[i]),
                "boundary_violated": bool(self.boundary_violated[i]),
                "rail_hit": bool(rail_hits[i]),
            }
            for i in range(self.num_envs)
        ]

        return obs, rewards, term, trunc, infos
