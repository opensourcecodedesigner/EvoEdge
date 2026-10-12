"""Fitness evaluation with spatial centering and complexity penalties.

Exposes fitness calculation functions and GenomeEvaluator, logging raw reward
components (Base reward, Centering penalty, Boundary penalty) during evaluation.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import neat  # type: ignore

from env.cartpole_swingup import CartPoleSwingUpEnv
from metrics.complexity import (
    count_active_connections,
    count_hidden_neurons,
    estimate_operations,
)


# ── Observation normalisation ────────────────────────────────


def _normalize_obs(obs: np.ndarray) -> List[float]:
    """Scale the 5-dim observation into a NN-friendly range."""
    x, x_dot, cos_theta, sin_theta, theta_dot = obs
    return [
        x / 2.4,
        float(np.clip(x_dot / 10.0, -1.0, 1.0)),
        float(cos_theta),
        float(sin_theta),
        float(np.clip(theta_dot / 10.0, -1.0, 1.0)),
    ]


# ── Episode runner ───────────────────────────────────────────


def _run_episode(
    net: neat.nn.FeedForwardNetwork,
    env_seed: int,
    max_steps: int,
    verbose: bool = True,
) -> float:
    """Run a single episode, return cumulative reward with centering penalty.

    Logs raw reward components to terminal during evaluation.
    """
    env = CartPoleSwingUpEnv(max_steps=max_steps, continuous=True)
    obs, _ = env.reset(seed=env_seed)

    tot_base = 0.0
    tot_center = 0.0
    tot_bound = 0.0
    done = False
    step = 0
    while not done and step < max_steps:
        output = net.activate(_normalize_obs(obs))
        obs, _, terminated, truncated, _ = env.step(output[0])

        cart_x = obs[0] if len(obs) == 5 else env._state[0]
        cos_theta = obs[2] if len(obs) == 5 else np.cos(env._state[2])

        # 1. Posture & Centering: 
        # cos_theta ranges from -1 (down) to +1 (upright).
        # 0.1 * abs(cart_x) penalizes being off-center (max -0.24 at rails).
        step_reward = cos_theta - 0.1 * abs(cart_x)

        if abs(cart_x) >= 2.4:
            # 2. Suicide Prevention: 
            # If it crashes, we penalize it for all remaining steps it *could* have lived 
            # at a rate worse than just hanging safely at the bottom (-2.0 per step).
            steps_left = max_steps - step
            tot_base -= (steps_left * 2.0)
            tot_bound += (steps_left * 2.0)
            break

        tot_base += step_reward
        tot_center += 0.1 * abs(cart_x)

        done = terminated or truncated
        step += 1

    tot_base = round(tot_base, 2)
    tot_center = round(tot_center, 2)
    tot_bound = round(tot_bound, 2)

    if verbose:
        print(f"Base: {tot_base} | Center Pen: {tot_center} | Bound Pen: {tot_bound}", flush=True)

    return tot_base


# ── Genome evaluator (callable used by neat.Population.run) ──


class GenomeEvaluator:
    """Stateful callable that evaluates an entire generation.

    Instantiate once per evolution run; the ``__call__`` signature
    matches what ``neat.Population.run`` expects.
    """

    def __init__(self, experiment_config: Any, mode: str, base_seed: int) -> None:
        self.config = experiment_config
        self.mode = mode
        self.base_seed = base_seed
        self.generation: int = 0

    # ── neat.Population.run callback ─────────────────────────

    def __call__(
        self,
        genomes: List[Tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
    ) -> None:
        for genome_id, genome in genomes:
            self._evaluate_genome(genome, neat_config)
        self.generation += 1

    def eval_genomes(
        self,
        genomes: List[Tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
    ) -> None:
        """Alias for __call__ to match standard NEAT callback naming."""
        self(genomes, neat_config)

    # ── Internal ─────────────────────────────────────────────

    def _evaluate_genome(
        self,
        genome: neat.DefaultGenome,
        neat_config: neat.Config,
    ) -> None:
        net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

        # Deterministic env seeds: same across all genomes in a generation,
        # different across generations and experiment seeds.
        rewards = []
        for i in range(self.config.num_eval_episodes):
            env_seed = (
                self.base_seed + self.generation * 97 + i * 31
            ) % (2 ** 31)
            rewards.append(
                _run_episode(net, env_seed, self.config.max_episode_steps)
            )

        base_reward = float(np.mean(rewards))

        # ── Annealed complexity penalty ──────────────────────
        if self.mode == "performance":
            genome.fitness = base_reward

        elif self.mode == "complexity":
            if base_reward > self.config.survival_threshold:
                hidden = count_hidden_neurons(genome, neat_config)
                active = count_active_connections(genome)
                penalty = self.config.complexity_penalty_weight * (
                    active + hidden
                )
                genome.fitness = base_reward - penalty
            else:
                genome.fitness = base_reward

        elif self.mode == "hardware":
            if base_reward > self.config.survival_threshold:
                ops = estimate_operations(genome, neat_config)
                penalty = self.config.hardware_penalty_weight * ops
                genome.fitness = base_reward - penalty
            else:
                genome.fitness = base_reward

        else:
            raise ValueError(f"Unknown fitness mode: {self.mode!r}")


# ── Stand-alone raw-reward evaluation ────────────────────────


def evaluate_raw_reward(
    genome: neat.DefaultGenome,
    neat_config: neat.Config,
    experiment_config: Any,
    seed: int,
) -> float:
    """Evaluate *genome* without any penalty (for final reporting).

    Uses a *different* set of env seeds from the evolutionary loop to
    give an unbiased performance estimate.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    rewards = []
    for i in range(experiment_config.num_eval_episodes):
        env_seed = (seed * 10_007 + i * 31) % (2 ** 31)
        rewards.append(
            _run_episode(net, env_seed, experiment_config.max_episode_steps)
        )
    return float(np.mean(rewards))


# ── Stand-alone eval_genomes function ────────────────────────


def eval_genomes(
    genomes: List[Tuple[int, neat.DefaultGenome]],
    neat_config: neat.Config,
    experiment_config: Any = None,
    mode: str = "performance",
    base_seed: int = 42,
) -> None:
    """Evaluate a list of genomes under the configured fitness function."""
    if experiment_config is None:
        from config.experiment_config import ExperimentConfig
        experiment_config = ExperimentConfig()
    evaluator = GenomeEvaluator(experiment_config, mode=mode, base_seed=base_seed)
    evaluator(genomes, neat_config)


__all__ = [
    "GenomeEvaluator",
    "eval_genomes",
    "evaluate_raw_reward",
    "_run_episode",
    "_normalize_obs",
]
