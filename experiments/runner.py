"""Multi-seed experiment orchestration.

Iterates over (mode × seed) combinations, runs NEAT evolution for each,
collects results, serialises best genomes, and writes a machine-readable
CSV incrementally (safe against mid-run crashes).

Now integrated with unified PerturbationSuite for split integrity,
reproducible sampling, configuration hashing, and physical parameter provenance.
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
from utils.perturbation_loader import (
    load_perturbation_suite,
    PerturbationSuite,
    PerturbationConfigError,
)


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
    "config_hash",
    "model_artifact",
]


# ── Single experiment ────────────────────────────────────────


def run_single_experiment(
    config: ExperimentConfig,
    mode: str,
    seed: int,
    perturbation_config_path: str = "config/perturbations.yaml",
    split: str = "training",
    perturbation_suite: Optional[PerturbationSuite] = None,
) -> Dict:
    """Run one full NEAT evolution and return a result dict with provenance."""
    # ── Configuration & Split Validation ─────────────────────
    suite = perturbation_suite or load_perturbation_suite(perturbation_config_path)

    # Enforce split partition: Seeds must belong to declared split
    allowed_seeds = suite.get_split_seeds(split)
    if seed not in allowed_seeds:
        raise PerturbationConfigError(
            f"Seed {seed} violates partition constraint for split '{split}' "
            f"[{allowed_seeds[0]}..{allowed_seeds[-1]}]. Cannot train outside declared partition."
        )

    # Protect prospective final test seeds from accidental exposure
    final_test_range = suite.splits.get("supplementary_untouched_test", {}).get("seed_range", [2000, 2099])
    if final_test_range[0] <= seed <= final_test_range[1]:
        raise PerturbationConfigError(
            f"Seed {seed} belongs to supplementary untouched final test split "
            f"[{final_test_range[0]}..{final_test_range[1]}]. Access denied during evolution."
        )

    sampled_params = suite.sample_parameters(split=split, seed=seed)

    print(f"\n{'=' * 64}")
    print(f"  Mode: {mode}  |  Seed: {seed}  |  Split: {split}")
    print(f"  Config SHA-256: {suite.config_hash}")
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
    model_artifact = os.path.join(run_dir, "best_genome.pkl")
    save_genome(best_genome, neat_config, model_artifact)
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
        "config_hash": suite.config_hash,
        "model_artifact": model_artifact,
        "sampled_physical_params": sampled_params,
        "split": split,
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
    perturbation_config_path: str = "config/perturbations.yaml",
    split: str = "training",
) -> List[Dict]:
    """Run every (mode × seed) combination and write results to CSV with provenance."""
    if config is None:
        config = ExperimentConfig()

    suite = load_perturbation_suite(perturbation_config_path)

    csv_path = os.path.join(config.results_dir, "results.csv")

    # Write header
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    results: List[Dict] = []

    for mode in config.modes:
        for seed in config.seeds:
            result = run_single_experiment(
                config,
                mode,
                seed,
                perturbation_config_path=perturbation_config_path,
                split=split,
                perturbation_suite=suite,
            )
            results.append(result)

            # Append incrementally (crash-safe)
            row = {k: v for k, v in result.items() if k in CSV_FIELDS}
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)

    print(f"\n  Results CSV -> {csv_path}")
    return results

