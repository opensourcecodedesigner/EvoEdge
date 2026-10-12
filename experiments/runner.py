"""Multi-seed experiment orchestration.

Iterates over (mode × seed) combinations, runs NEAT evolution for each,
collects results, serialises best genomes, and writes a machine-readable
CSV incrementally (safe against mid-run crashes).
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional

from config.experiment_config import ExperimentConfig
from evolution.neat_evolve import run_evolution
from evolution.fitness import evaluate_raw_reward
from metrics.complexity import get_genome_metrics
from utils.serialization import save_genome, save_topology


# ── CSV schema ───────────────────────────────────────────────

CSV_FIELDS = [
    "seed",
    "mode",
    "best_reward",
    "mean_reward",
    "hidden_neurons",
    "active_connections",
    "estimated_operations",
    "success_flag",
]


# ── Single experiment ────────────────────────────────────────


def run_single_experiment(
    config: ExperimentConfig,
    mode: str,
    seed: int,
) -> Dict:
    """Run one full NEAT evolution and return a result dict."""
    print(f"\n{'=' * 64}")
    print(f"  Mode: {mode}  |  Seed: {seed}")
    print(f"{'=' * 64}\n")

    t0 = time.perf_counter()

    best_genome, neat_config, stats, _evaluator = run_evolution(
        config.neat_config_path, config, mode, seed
    )

    elapsed = time.perf_counter() - t0

    # ── Raw (unpenalised) reward of the champion ─────────────
    best_reward = evaluate_raw_reward(
        best_genome, neat_config, config, seed
    )

    # ── Population mean fitness in the final generation ──────
    mean_history = stats.get_fitness_mean()
    mean_reward = mean_history[-1] if mean_history else 0.0

    # ── Complexity metrics ───────────────────────────────────
    metrics = get_genome_metrics(best_genome, neat_config)

    # ── Persist best genome & topology ───────────────────────
    run_dir = os.path.join(config.results_dir, mode, f"seed_{seed}")
    save_genome(best_genome, neat_config, os.path.join(run_dir, "best_genome.pkl"))
    save_topology(best_genome, neat_config, os.path.join(run_dir, "topology.json"))

    result = {
        "seed": seed,
        "mode": mode,
        "best_reward": round(best_reward, 4),
        "mean_reward": round(mean_reward, 4),
        "hidden_neurons": metrics["hidden_neurons"],
        "active_connections": metrics["active_connections"],
        "estimated_operations": metrics["estimated_operations"],
        "success_flag": best_reward > config.success_threshold,
    }

    print(
        f"\n  [OK] reward={best_reward:+.2f}  hidden={metrics['hidden_neurons']}  "
        f"conns={metrics['active_connections']}  ops={metrics['estimated_operations']}  "
        f"success={result['success_flag']}  time={elapsed:.1f}s\n"
    )
    return result


# ── Full experiment sweep ────────────────────────────────────


def run_all_experiments(
    config: Optional[ExperimentConfig] = None,
) -> List[Dict]:
    """Run every (mode × seed) combination and write results to CSV."""
    if config is None:
        config = ExperimentConfig()

    csv_path = os.path.join(config.results_dir, "results.csv")

    # Write header
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    results: List[Dict] = []

    for mode in config.modes:
        for seed in config.seeds:
            result = run_single_experiment(config, mode, seed)
            results.append(result)

            # Append incrementally (crash-safe)
            row = {k: v for k, v in result.items() if k in CSV_FIELDS}
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)

    print(f"\n  Results CSV -> {csv_path}")
    return results
