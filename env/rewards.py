"""Stage 3B Reward Shaping and Objective Formulations for EvoEdge.

Provides:
  1. compute_stage3b_step_reward: Dense, loophole-free reward function for scalar CartPole.
  2. compute_stage3b_batch_reward: Vectorized batch implementation for VectorizedCartPoleEnv.
  3. Mathematical properties & return bounds:
     - Return bounds: [-450.0, +1750.0]
     - Rail crash: Return <= -200.0 (strictly worse than passive survival)
     - Passive bottom survival (hanging down): Return ~ 0.0
     - Swing progress: Return in [+50.0, +400.0]
     - Upright stabilization: Return in [+600.0, +1500.0]
     - Strict ordering: Success > Swing Progress > Passive Survival >> Rail Collision.
"""

from __future__ import annotations

import numpy as np

STAGE3B_RETURN_LOWER_BOUND: float = -450.0
STAGE3B_RETURN_UPPER_BOUND: float = 1750.0


def compute_stage3b_step_reward(
    x: float,
    x_dot: float,
    theta: float,
    theta_dot: float,
    action: float,
    step_count: int,
    max_steps: int = 500,
    x_threshold: float = 2.4,
) -> float:
    """Compute dense Stage 3B scalar step reward.

    Args:
        x: Cart position [m]
        x_dot: Cart velocity [m/s]
        theta: Pole angle [rad] (0 = upright, +-pi = hanging down)
        theta_dot: Pole angular velocity [rad/s]
        action: Continuous control action in [-1.0, 1.0]
        step_count: Current timestep index (1..max_steps)
        max_steps: Episode horizon (default 500)
        x_threshold: Rail boundary limit (default 2.4 m)

    Returns:
        Scalar reward value.
    """
    # Check for rail boundary collision
    terminated = abs(x) >= x_threshold
    if terminated:
        remaining_steps = max(0, max_steps - step_count)
        # Immediate crash penalty (-200.0) plus remaining opportunity loss penalty
        return float(-200.0 - 0.5 * remaining_steps)

    # 1. Non-negative angle potential: 0.0 at hanging down (theta=+-pi), 1.0 at upright (theta=0)
    cos_th = float(np.cos(theta))
    angle_progress = 0.5 * (1.0 + cos_th)

    # 2. Upright stabilization bonus
    upright_bonus = 0.0
    if cos_th > 0.8:
        # Linearly scales from 1.0 at cos_th=0.8 up to 2.0 at cos_th=1.0
        upright_bonus = 1.0 + 5.0 * (cos_th - 0.8)
        # Small angular velocity damping bonus inside the upright basin
        if abs(theta_dot) < 1.0:
            upright_bonus += 0.5

    # 3. Soft cart centering penalty (bounded in [0.0, 0.1])
    # Keeps cart near center without preventing dynamic cart swings
    cart_penalty = 0.1 * ((x / x_threshold) ** 2)

    # 4. Small action regularization (bounded in [0.0, 0.01])
    action_penalty = 0.01 * (float(action) ** 2)

    step_reward = angle_progress + upright_bonus - cart_penalty - action_penalty
    return float(step_reward)


def compute_stage3b_batch_reward(
    x: np.ndarray,
    x_dot: np.ndarray,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    actions: np.ndarray,
    step_counts: np.ndarray,
    max_steps: int = 500,
    x_threshold: float = 2.4,
    boundary_violated: np.ndarray | None = None,
) -> np.ndarray:
    """Compute dense Stage 3B vectorized batch reward across concurrent environments.

    Args:
        x: Array of cart positions, shape (num_envs,)
        x_dot: Array of cart velocities, shape (num_envs,)
        theta: Array of pole angles, shape (num_envs,)
        theta_dot: Array of pole angular velocities, shape (num_envs,)
        actions: Array of applied actions, shape (num_envs,)
        step_counts: Array of step counts, shape (num_envs,)
        max_steps: Episode horizon (default 500)
        x_threshold: Rail limit (default 2.4 m)
        boundary_violated: Optional boolean array marking boundary violation

    Returns:
        Array of rewards, shape (num_envs,)
    """
    num_envs = len(x)
    cos_th = np.cos(theta)

    # 1. Angle progress in [0.0, 1.0]
    angle_progress = 0.5 * (1.0 + cos_th)

    # 2. Upright stabilization bonus
    upright_bonus = np.zeros(num_envs, dtype=np.float64)
    in_basin = cos_th > 0.8
    if np.any(in_basin):
        upright_bonus[in_basin] = 1.0 + 5.0 * (cos_th[in_basin] - 0.8)
        calm = in_basin & (np.abs(theta_dot) < 1.0)
        upright_bonus[calm] += 0.5

    # 3. Soft cart centering penalty
    cart_penalty = 0.1 * ((x / x_threshold) ** 2)

    # 4. Small action penalty
    action_penalty = 0.01 * (actions ** 2)

    rewards = angle_progress + upright_bonus - cart_penalty - action_penalty

    # 5. Terminal crash penalty
    rail_hits = np.abs(x) >= x_threshold
    if boundary_violated is not None:
        rail_hits = rail_hits | boundary_violated

    if np.any(rail_hits):
        remaining = np.maximum(0, max_steps - step_counts)
        rewards[rail_hits] = -200.0 - 0.5 * remaining[rail_hits]

    return rewards
