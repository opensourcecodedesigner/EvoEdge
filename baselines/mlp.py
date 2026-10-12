"""Fixed-topology Dense Multilayer Perceptron (MLP) baseline controller.

Architecture:
    Input (dim 5) -> Dense(16, activation='tanh') -> Dense(1, activation='tanh')
    Total parameters: 5*16 + 16 + 16*1 + 1 = 113 (weights + biases).
    Theoretical MACs: 5*16 + 16*1 = 96 multiply-accumulates.

Optimized via lightweight Evolution Strategies (ES) under identical episode
reward metrics for fair, reproducible comparison against NEAT.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Observation Normalization (Matching NEAT & Experiments) ──────────────────


def normalize_obs(obs: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Normalize raw observation vector into neural-network-friendly range.

    Handles:
        - 5-dim: [x, x_dot, cos(theta), sin(theta), theta_dot]
        - 4-dim: [x, x_dot, theta, theta_dot] -> converted to 5-dim
    """
    arr = np.asarray(obs, dtype=np.float32).flatten()
    if len(arr) == 4:
        x, x_dot, theta, theta_dot = arr
        cos_th = float(np.cos(theta))
        sin_th = float(np.sin(theta))
    elif len(arr) == 5:
        x, x_dot, cos_th, sin_th, theta_dot = arr
    else:
        raise ValueError(f"Expected 4 or 5 observation dimensions, got {len(arr)}")

    return np.array([
        float(x / 2.4),
        float(np.clip(x_dot / 10.0, -1.0, 1.0)),
        float(cos_th),
        float(sin_th),
        float(np.clip(theta_dot / 10.0, -1.0, 1.0)),
    ], dtype=np.float32)


# ── Fixed MLP Controller ─────────────────────────────────────────────────────


class FixedMLPController:
    """Fixed-topology feedforward neural network controller.

    5 Inputs -> Dense(16, tanh) -> Dense(1, tanh).
    Parameter vector length: 113.
    """

    INPUT_DIM: int = 5
    HIDDEN_DIM: int = 16
    OUTPUT_DIM: int = 1
    TOTAL_PARAMS: int = 5 * 16 + 16 + 16 * 1 + 1  # 113
    TOTAL_MACS: int = 5 * 16 + 16 * 1             # 96

    def __init__(
        self,
        weights: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        weights_path: Optional[str] = None,
    ) -> None:
        self.W1 = np.zeros((self.INPUT_DIM, self.HIDDEN_DIM), dtype=np.float32)
        self.b1 = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
        self.W2 = np.zeros((self.HIDDEN_DIM, self.OUTPUT_DIM), dtype=np.float32)
        self.b2 = np.zeros(self.OUTPUT_DIM, dtype=np.float32)

        if weights_path is not None and os.path.exists(weights_path):
            self.load(weights_path)
        elif weights is not None:
            if isinstance(weights, dict):
                self.W1 = np.asarray(weights["W1"], dtype=np.float32)
                self.b1 = np.asarray(weights["b1"], dtype=np.float32)
                self.W2 = np.asarray(weights["W2"], dtype=np.float32)
                self.b2 = np.asarray(weights["b2"], dtype=np.float32)
            else:
                self.set_weights_vector(np.asarray(weights, dtype=np.float32))
        else:
            # Check default weights location in results/fixed_mlp/
            default_path = os.path.join("results", "fixed_mlp", "best_weights.npz")
            if os.path.exists(default_path):
                self.load(default_path)
            else:
                # Initialize with Xavier random weights
                rng = np.random.default_rng(42)
                limit1 = float(np.sqrt(6.0 / (self.INPUT_DIM + self.HIDDEN_DIM)))
                limit2 = float(np.sqrt(6.0 / (self.HIDDEN_DIM + self.OUTPUT_DIM)))
                self.W1 = rng.uniform(-limit1, limit1, (self.INPUT_DIM, self.HIDDEN_DIM)).astype(np.float32)
                self.b1 = np.zeros(self.HIDDEN_DIM, dtype=np.float32)
                self.W2 = rng.uniform(-limit2, limit2, (self.HIDDEN_DIM, self.OUTPUT_DIM)).astype(np.float32)
                self.b2 = np.zeros(self.OUTPUT_DIM, dtype=np.float32)

    @property
    def active_parameters(self) -> int:
        """Total trainable parameters (113)."""
        return self.TOTAL_PARAMS

    @property
    def theoretical_macs(self) -> int:
        """Theoretical Multiply-Accumulate operations per forward pass (96)."""
        return self.TOTAL_MACS

    def get_weights_vector(self) -> np.ndarray:
        """Flatten all parameters into a single 1D vector of shape (113,)."""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten(),
        ])

    def set_weights_vector(self, theta: np.ndarray) -> None:
        """Unpack a 1D vector of shape (113,) into layer weights and biases."""
        theta = np.asarray(theta, dtype=np.float32).flatten()
        if len(theta) != self.TOTAL_PARAMS:
            raise ValueError(
                f"Expected parameter vector of length {self.TOTAL_PARAMS}, got {len(theta)}"
            )

        idx = 0
        w1_size = self.INPUT_DIM * self.HIDDEN_DIM
        self.W1 = theta[idx:idx + w1_size].reshape(self.INPUT_DIM, self.HIDDEN_DIM)
        idx += w1_size

        self.b1 = theta[idx:idx + self.HIDDEN_DIM]
        idx += self.HIDDEN_DIM

        w2_size = self.HIDDEN_DIM * self.OUTPUT_DIM
        self.W2 = theta[idx:idx + w2_size].reshape(self.HIDDEN_DIM, self.OUTPUT_DIM)
        idx += w2_size

        self.b2 = theta[idx:idx + self.OUTPUT_DIM]

    def compute_action(
        self,
        obs: Union[np.ndarray, List[float]],
        normalized: bool = True,
    ) -> float:
        """Execute one forward inference pass and return control action in [-1.0, 1.0].

        Args:
            obs: Observation vector (4D or 5D).
            normalized: Ignored (MLP action is natively in [-1.0, 1.0] due to tanh output).

        Returns:
            Scalar control action in [-1.0, 1.0].
        """
        s = normalize_obs(obs)
        # Hidden layer: Dense(16, tanh)
        h = np.tanh(s @ self.W1 + self.b1)
        # Output layer: Dense(1, tanh)
        out = float(np.tanh(h @ self.W2 + self.b2)[0])
        return float(np.clip(out, -1.0, 1.0))

    def __call__(self, obs: Union[np.ndarray, List[float]]) -> float:
        """Convenience alias for compute_action(obs)."""
        return self.compute_action(obs)

    def save(self, filepath: str) -> None:
        """Serialize controller weights to .npz or .json."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        if filepath.endswith(".json"):
            data = {
                "W1": self.W1.tolist(),
                "b1": self.b1.tolist(),
                "W2": self.W2.tolist(),
                "b2": self.b2.tolist(),
                "total_params": self.TOTAL_PARAMS,
                "total_macs": self.TOTAL_MACS,
            }
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        else:
            np.savez_compressed(
                filepath,
                W1=self.W1,
                b1=self.b1,
                W2=self.W2,
                b2=self.b2,
            )

    def load(self, filepath: str) -> None:
        """Load controller weights from .npz or .json."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found: {filepath}")

        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.W1 = np.array(data["W1"], dtype=np.float32)
            self.b1 = np.array(data["b1"], dtype=np.float32)
            self.W2 = np.array(data["W2"], dtype=np.float32)
            self.b2 = np.array(data["b2"], dtype=np.float32)
        else:
            data = np.load(filepath)
            self.W1 = data["W1"].astype(np.float32)
            self.b1 = data["b1"].astype(np.float32)
            self.W2 = data["W2"].astype(np.float32)
            self.b2 = data["b2"].astype(np.float32)


# ── Lightweight Evolution Strategies (ES) Trainer ────────────────────────────


def _evaluate_candidate(
    theta: np.ndarray,
    env_class: Any,
    seed: int,
    eval_episodes: int = 3,
    max_steps: int = 200,
    operating_point: bool = True,
) -> float:
    """Evaluate one parameter vector across multiple deterministic episodes."""
    ctrl = FixedMLPController(weights=theta)
    env = env_class(max_steps=max_steps, continuous=True)
    rng = np.random.default_rng(seed)

    episode_rewards: List[float] = []
    for i in range(eval_episodes):
        ep_seed = (seed * 10007 + i * 31) % (2 ** 31)
        obs, _ = env.reset(seed=ep_seed)

        if operating_point:
            # Sample initial condition around operating point
            x0 = float(rng.uniform(-1.2, 1.2))
            th0 = float(rng.uniform(-0.15, 0.15))
            env._state = np.array([x0, 0.0, th0, 0.0], dtype=np.float64)
            obs = env._get_obs()

        tot_r = 0.0
        done = False
        step = 0
        while not done and step < max_steps:
            action = ctrl.compute_action(obs)
            obs, r, term, trunc, _ = env.step(action)
            tot_r += r
            done = term or trunc
            step += 1
        episode_rewards.append(tot_r)

    return float(np.mean(episode_rewards))


def train_mlp_es(
    env_class: Any = None,
    generations: int = 30,
    population_size: int = 30,
    sigma: float = 0.10,
    lr: float = 0.05,
    eval_episodes: int = 3,
    max_steps: int = 200,
    seed: int = 42,
    operating_point: bool = True,
    save_path: Optional[str] = "results/fixed_mlp/best_weights.npz",
    verbose: bool = True,
) -> Tuple[FixedMLPController, Dict[str, Any]]:
    """Train FixedMLPController using Natural Evolution Strategies with Adam.

    Optimizes the 113 parameters under identical episode reward metrics as NEAT.

    Args:
        env_class: Gymnasium environment class (defaults to CartPoleSwingUpEnv).
        generations: Number of optimization generations.
        population_size: Population size (must be even for antithetic sampling).
        sigma: Mutation perturbation scale.
        lr: Adam learning rate.
        eval_episodes: Number of episodes per candidate evaluation.
        max_steps: Maximum steps per episode.
        seed: Random seed for reproducibility.
        operating_point: If True, evaluates centered on task operating point.
        save_path: Path to save best weights (.npz). If None, does not save to disk.
        verbose: Print progress to stdout.

    Returns:
        Tuple of (trained_controller, history_dict).
    """
    if env_class is None:
        from env.cartpole_swingup import CartPoleSwingUpEnv
        env_class = CartPoleSwingUpEnv

    rng = np.random.default_rng(seed)
    n_params = FixedMLPController.TOTAL_PARAMS
    theta = rng.normal(0.0, 0.02, size=n_params).astype(np.float32)

    # Adam optimizer state
    m = np.zeros(n_params, dtype=np.float32)
    v = np.zeros(n_params, dtype=np.float32)
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8

    best_score = -float("inf")
    best_theta = theta.copy()
    history: Dict[str, List[float]] = {"generation": [], "best_reward": [], "mean_reward": []}

    half_pop = population_size // 2

    if verbose:
        print("\n" + "=" * 60, flush=True)
        print("  Training Fixed MLP Baseline (113 parameters) via ES", flush=True)
        print("=" * 60, flush=True)
        print(f"  Pop size: {population_size} | Gens: {generations} | Sigma: {sigma} | LR: {lr}", flush=True)

    for gen in range(1, generations + 1):
        # Antithetic perturbations
        eps = rng.standard_normal((half_pop, n_params)).astype(np.float32)
        perturbations = np.vstack([eps, -eps])

        # Evaluate population
        gen_seed = (seed + gen * 97) % (2 ** 31)
        rewards = np.array([
            _evaluate_candidate(
                theta + sigma * p,
                env_class,
                seed=gen_seed,
                eval_episodes=eval_episodes,
                max_steps=max_steps,
                operating_point=operating_point,
            )
            for p in perturbations
        ], dtype=np.float32)

        # Rank-based fitness shaping
        order = np.argsort(np.argsort(rewards))
        utilities = np.maximum(0.0, np.log(len(rewards) / 2.0 + 1.0) - np.log(len(rewards) - order))
        util_sum = np.sum(utilities)
        if util_sum > 0:
            utilities = utilities / util_sum - (1.0 / len(rewards))

        # Gradient estimate
        grad = (1.0 / (population_size * sigma)) * np.dot(utilities, perturbations)

        # Adam update
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad ** 2)
        m_hat = m / (1.0 - beta1 ** gen)
        v_hat = v / (1.0 - beta2 ** gen)
        theta += lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Validation check on fixed seed
        val_score = _evaluate_candidate(
            theta,
            env_class,
            seed=9999,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            operating_point=operating_point,
        )

        gen_max = float(np.max(rewards))
        gen_mean = float(np.mean(rewards))
        history["generation"].append(gen)
        history["best_reward"].append(gen_max)
        history["mean_reward"].append(gen_mean)

        if val_score > best_score:
            best_score = val_score
            best_theta = theta.copy()

        if verbose and (gen % 5 == 0 or gen == generations or gen == 1):
            print(
                f"  [Gen {gen:2d}/{generations}] pop_best: {gen_max:6.2f} | "
                f"pop_mean: {gen_mean:6.2f} | val: {val_score:6.2f} | best: {best_score:6.2f}",
                flush=True,
            )

    controller = FixedMLPController(weights=best_theta)

    if save_path:
        controller.save(save_path)
        if verbose:
            print(f"  [OK] Saved best MLP weights -> {save_path}", flush=True)

    return controller, history


# ── Module Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fixed-topology MLP baseline trainer")
    parser.add_argument("--train", action="store_true", help="Train MLP using Evolution Strategies")
    parser.add_argument("--generations", type=int, default=30, help="Number of ES generations (default: 30)")
    parser.add_argument("--pop-size", type=int, default=30, help="Population size (default: 30)")
    parser.add_argument("--output", type=str, default="results/fixed_mlp/best_weights.npz", help="Weights output path")
    args = parser.parse_args()

    if args.train:
        train_mlp_es(generations=args.generations, population_size=args.pop_size, save_path=args.output)
    else:
        ctrl = FixedMLPController(weights_path=args.output if os.path.exists(args.output) else None)
        print(f"FixedMLPController initialized: {ctrl.active_parameters} parameters, {ctrl.theoretical_macs} MACs.")
