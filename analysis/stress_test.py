r"""Robustness stress-testing for evolved NEAT controllers.

Evaluates champion genomes across four environmental shift scenarios:
  1. Baseline:        Nominal environment dynamics
  2. Heavy Cart:      Cart mass increased by 40% (masscart * 1.4)
  3. Long Pole:       Pole length increased by 30% (length * 1.3)
  4. Noisy Sensors:   Zero-mean Gaussian noise (\sigma = 0.05) added to observations

Generates:
  - Summary CSV: `results/stress_test_results.csv`
  - High-res bar plot: `results/robustness_comparison.png`
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import neat  # type: ignore

from env.cartpole_swingup import CartPoleSwingUpEnv
from utils.serialization import load_genome


# ── Perturbed Environment Wrapper ────────────────────────────────────────────


class PerturbedCartPoleEnv(CartPoleSwingUpEnv):
    """CartPole Swing-Up environment with parameter and sensor perturbations."""

    def __init__(
        self,
        max_steps: int = 200,
        continuous: bool = True,
        masscart_scale: float = 1.0,
        length_scale: float = 1.0,
        obs_noise_sigma: float = 0.0,
    ) -> None:
        super().__init__(max_steps=max_steps, continuous=continuous)

        # Apply physics parameter scaling
        self.MASS_CART = self.MASS_CART * masscart_scale
        self.LENGTH = self.LENGTH * length_scale

        # Recompute cached physical quantities used in step() dynamics
        self._total_mass = self.MASS_CART + self.MASS_POLE
        self._polemass_length = self.MASS_POLE * self.LENGTH

        self.obs_noise_sigma = obs_noise_sigma
        self._noise_rng = np.random.default_rng(42)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        obs, info = super().reset(seed=seed, options=options)
        if seed is not None:
            self._noise_rng = np.random.default_rng(seed + 9999)
        if self.obs_noise_sigma > 0.0:
            noise = self._noise_rng.normal(
                0.0, self.obs_noise_sigma, size=obs.shape
            ).astype(np.float32)
            obs = obs + noise
        return obs, info

    def step(
        self, action
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        if self.obs_noise_sigma > 0.0:
            noise = self._noise_rng.normal(
                0.0, self.obs_noise_sigma, size=obs.shape
            ).astype(np.float32)
            obs = obs + noise
        return obs, reward, terminated, truncated, info


# ── Observation Normalization ────────────────────────────────────────────────


def _normalize_obs(obs: np.ndarray) -> List[float]:
    """Scale the 5-dim observation into standard NN input range."""
    x, x_dot, cos_theta, sin_theta, theta_dot = obs
    return [
        float(x / 2.4),
        float(np.clip(x_dot / 10.0, -1.0, 1.0)),
        float(cos_theta),
        float(sin_theta),
        float(np.clip(theta_dot / 10.0, -1.0, 1.0)),
    ]


# ── Single Episode Evaluation ────────────────────────────────────────────────


def _run_single_episode(
    net: neat.nn.FeedForwardNetwork,
    env: PerturbedCartPoleEnv,
    env_seed: int,
) -> float:
    """Run one evaluation episode in *env* with deterministic *env_seed*."""
    obs, _ = env.reset(seed=env_seed)
    episode_reward = 0.0
    done = False

    while not done:
        action = net.activate(_normalize_obs(obs))
        obs, reward, terminated, truncated, _ = env.step(action[0])
        episode_reward += reward
        done = terminated or truncated

    return episode_reward


# ── Retention Metric ─────────────────────────────────────────────────────────


def compute_retention_pct(scenario_reward: float, baseline_reward: float) -> float:
    """Calculate percentage performance retention under shift.

    Defined as (scenario_reward / baseline_reward) * 100.
    If baseline_reward <= 0 (agent never learned to swing up even at baseline),
    retention is clamped to 0.0% to avoid misleading positive signs.
    """
    if baseline_reward <= 0.0:
        return 0.0
    return max(0.0, (scenario_reward / baseline_reward) * 100.0)


# ── Champion Discovery ───────────────────────────────────────────────────────


def find_champion_genomes(results_dir: str = "results") -> List[Dict[str, Any]]:
    """Scan *results_dir* for all evolved champion genomes across modes and seeds.

    Looks for paths matching `<results_dir>/<mode>/seed_<seed>/best_genome.pkl`.
    """
    champions: List[Dict[str, Any]] = []

    if not os.path.exists(results_dir):
        return champions

    modes = sorted(os.listdir(results_dir))
    for mode in modes:
        mode_dir = os.path.join(results_dir, mode)
        if not os.path.isdir(mode_dir):
            continue

        for seed_entry in sorted(os.listdir(mode_dir)):
            seed_dir = os.path.join(mode_dir, seed_entry)
            if not os.path.isdir(seed_dir) or not seed_entry.startswith("seed_"):
                continue

            pkl_path = os.path.join(seed_dir, "best_genome.pkl")
            if not os.path.exists(pkl_path):
                continue

            try:
                seed = int(seed_entry.split("_")[1])
            except ValueError:
                seed = 0

            loaded = load_genome(pkl_path)
            champions.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "genome": loaded["genome"],
                    "neat_config": loaded["config"],
                    "pkl_path": pkl_path,
                }
            )

    return champions


# ── Perturbation Scenarios Definition ────────────────────────────────────────

SCENARIOS = [
    {
        "name": "Baseline",
        "description": "Nominal dynamics",
        "masscart_scale": 1.0,
        "length_scale": 1.0,
        "obs_noise_sigma": 0.0,
    },
    {
        "name": "Heavy Cart",
        "description": "masscart * 1.4",
        "masscart_scale": 1.4,
        "length_scale": 1.0,
        "obs_noise_sigma": 0.0,
    },
    {
        "name": "Long Pole",
        "description": "length * 1.3",
        "masscart_scale": 1.0,
        "length_scale": 1.3,
        "obs_noise_sigma": 0.0,
    },
    {
        "name": "Noisy Sensors",
        "description": "Gaussian noise sigma = 0.05",
        "masscart_scale": 1.0,
        "length_scale": 1.0,
        "obs_noise_sigma": 0.05,
    },
]


# ── Stress Test Execution ────────────────────────────────────────────────────


def run_stress_test(
    results_dir: str = "results",
    num_episodes: int = 5,
    max_steps: int = 200,
) -> List[Dict[str, Any]]:
    """Execute robustness evaluation across all champion genomes and scenarios.

    Args:
        results_dir: Directory containing experimental results.
        num_episodes: Evaluation episodes per scenario.
        max_steps: Maximum environment steps per episode.

    Returns:
        List of result dictionaries.
    """
    champions = find_champion_genomes(results_dir)
    if not champions:
        print(f"Error: No champion genomes found in '{results_dir}'. Run experiments first.")
        return []

    print("\n" + "=" * 68)
    print("  NEAT Controller Robustness Stress Test")
    print("=" * 68)
    print(f"  Champions found : {len(champions)}")
    print(f"  Scenarios       : {len(SCENARIOS)} ({', '.join(s['name'] for s in SCENARIOS)})")
    print(f"  Eval episodes   : {num_episodes} per scenario")
    print("=" * 68 + "\n")

    results: List[Dict[str, Any]] = []

    for champ in champions:
        mode = champ["mode"]
        seed = champ["seed"]
        genome = champ["genome"]
        neat_config = champ["neat_config"]

        net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

        # Baseline evaluation first
        scenario_rewards: Dict[str, float] = {}

        for sc in SCENARIOS:
            sc_name = sc["name"]
            env = PerturbedCartPoleEnv(
                max_steps=max_steps,
                continuous=True,
                masscart_scale=sc["masscart_scale"],
                length_scale=sc["length_scale"],
                obs_noise_sigma=sc["obs_noise_sigma"],
            )

            # Use identical environment seeds across scenarios for paired comparison
            ep_rewards = []
            for ep_idx in range(num_episodes):
                env_seed = (seed * 10007 + ep_idx * 31) % (2 ** 31)
                reward = _run_single_episode(net, env, env_seed)
                ep_rewards.append(reward)

            scenario_rewards[sc_name] = float(np.mean(ep_rewards))

        baseline_reward = scenario_rewards["Baseline"]

        for sc in SCENARIOS:
            sc_name = sc["name"]
            mean_reward = scenario_rewards[sc_name]
            if sc_name == "Baseline":
                retention = 100.0 if baseline_reward > 0.0 else 0.0
            else:
                retention = compute_retention_pct(mean_reward, baseline_reward)

            record = {
                "mode": mode,
                "seed": seed,
                "scenario": sc_name,
                "mean_reward": round(mean_reward, 4),
                "baseline_reward": round(baseline_reward, 4),
                "retention_pct": round(retention, 2),
            }
            results.append(record)

            print(
                f"  [{mode:<12} | seed {seed:>4}] {sc_name:<14} -> "
                f"reward: {mean_reward:>7.2f} (base: {baseline_reward:>7.2f}) | "
                f"retention: {retention:>6.1f}%"
            )

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "stress_test_results.csv")
    fieldnames = ["mode", "seed", "scenario", "mean_reward", "baseline_reward", "retention_pct"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  [OK] Summary CSV saved -> {csv_path}")

    # ── Generate Plot ────────────────────────────────────────────────────────
    plot_path = os.path.join(results_dir, "robustness_comparison.png")
    generate_robustness_plot(results, plot_path)
    print(f"  [OK] Robustness plot saved -> {plot_path}\n")

    return results


# ── Visualisation ────────────────────────────────────────────────────────────


def generate_robustness_plot(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Generate and save a bar plot showing percentage retention under shift."""
    # Filter to the three perturbation scenarios (excluding Baseline)
    perturbation_names = ["Heavy Cart", "Long Pole", "Noisy Sensors"]

    # Desired mode display order
    mode_order = ["performance", "complexity", "hardware"]
    available_modes = [m for m in mode_order if any(r["mode"] == m for r in results)]
    if not available_modes:
        available_modes = sorted(list({r["mode"] for r in results}))

    # Aggregate retention per (mode, scenario)
    aggregated: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in results:
        if r["scenario"] in perturbation_names:
            aggregated[(r["mode"], r["scenario"])].append(r["retention_pct"])

    # Modern visual design & color palette
    scenario_colors = {
        "Heavy Cart": "#2563EB",     # Royal Blue
        "Long Pole": "#7C3AED",      # Deep Violet
        "Noisy Sensors": "#DB2777",  # Rose Pink
    }
    scenario_labels = {
        "Heavy Cart": "Heavy Cart (mass × 1.4)",
        "Long Pole": "Long Pole (length × 1.3)",
        "Noisy Sensors": "Noisy Sensors (σ = 0.05)",
    }

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=300)

    # Plot geometry
    n_modes = len(available_modes)
    n_scenarios = len(perturbation_names)
    bar_width = 0.24
    group_spacing = 1.0

    x_indices = np.arange(n_modes) * group_spacing

    max_height = 100.0

    for i, sc in enumerate(perturbation_names):
        offsets = x_indices + (i - (n_scenarios - 1) / 2.0) * bar_width
        means = []
        stds = []

        for mode in available_modes:
            vals = aggregated.get((mode, sc), [0.0])
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

        color = scenario_colors.get(sc, "#4F46E5")
        bars = ax.bar(
            offsets,
            means,
            width=bar_width * 0.9,
            yerr=stds,
            capsize=4,
            color=color,
            alpha=0.92,
            edgecolor="#1E293B",
            linewidth=0.8,
            label=scenario_labels.get(sc, sc),
            zorder=3,
        )

        # Annotate bars with retention percentage
        for bar, mean_val in zip(bars, means):
            h = bar.get_height()
            if h > max_height:
                max_height = h
            ax.annotate(
                f"{mean_val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2.0, h),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#1E293B",
            )

    # Reference line for 100% baseline retention
    ax.axhline(
        100.0,
        color="#64748B",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label="Baseline Performance (100%)",
        zorder=2,
    )

    # Formatting axes and labels
    mode_display_names = {
        "performance": "Performance\n(Unconstrained)",
        "complexity": "Complexity-Aware\n(Size Penalty)",
        "hardware": "Hardware-Aware\n(MAC Penalty)",
    }
    ax.set_xticks(x_indices)
    ax.set_xticklabels(
        [mode_display_names.get(m, m.capitalize()) for m in available_modes],
        fontsize=11,
        fontweight="bold",
        color="#0F172A",
    )

    ax.set_ylabel("Performance Retention (%)", fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_title(
        "NEAT Controller Robustness Under Environmental Shift",
        fontsize=13,
        fontweight="bold",
        pad=16,
        color="#0F172A",
    )

    y_upper = max(115.0, max_height * 1.15)
    ax.set_ylim(0.0, y_upper)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=1)

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#CBD5E1",
        fontsize=9.5,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# ── CLI Interface ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Robustness stress test for evolved NEAT CartPole controllers.",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experiment results (default: results).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per scenario (default: 5).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_stress_test(
        results_dir=args.results_dir,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
