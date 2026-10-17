"""Stage 3C Delay-Aware Domain-Randomized Neuroevolution Orchestrator.

Builds on Stage 3B by evolving compact, delay-aware neural controllers across
a domain-randomized distribution of conditions using NSGA2-WarmStart.

Enforces:
  1. Success criteria: Official CartPole swing-up predicate (500 steps, |x| < 2.4, cos(theta) > 0.9 for >=95 of last 100 steps).
  2. Partitions: Training seeds 0..199 only. Validation on exposed development seeds 200..299 & 300..319.
     Strict quarantine on seeds 600..749 and untouched seeds 2000..2099.
  3. Complexity: Active parameters <= 128, Weight MACs <= 128.
  4. Lifecycle Budget: Search <= 1,850,000 transitions, Validation <= 150,000 transitions, Total <= 2,000,000 transitions.
  5. Acceptance screening: Gate 1B nominal success >= 90%, Worst-condition success >= 40%.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from baselines.hybrid_controller import HybridController
from env.budget_tracker import TransitionBudgetTracker, BudgetedEnv, BudgetedVectorizedEnv, BudgetExhaustedError
from env.cartpole_swingup import CartPoleSwingUpEnv
from env.vectorized_env import VectorizedCartPoleEnv
from evolution.nsga2_engine import (
    MaskedGenome,
    STAGE3C_TRAIN_CONDITIONS,
    train_nsga2,
    validate_nsga2_champion,
    norm_batch_obs,
    norm_obs,
)
from experiments.run_phase_3a_evolution import DIAGNOSTIC_CONDITIONS
from experiments.run_robustness_experiment import evaluate_success_predicate
from utils.perturbation_loader import PerturbedCartPoleEnv


# ── Directory Layout ──────────────────────────────────────────────────────────

STAGE3C_DIR = os.path.join(_PROJECT_ROOT, "results", "stage3c")
MODELS_DIR = os.path.join(STAGE3C_DIR, "models")
LOGS_DIR = os.path.join(STAGE3C_DIR, "logs")

for d in [STAGE3C_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Strict Partition Safety Enforcer ──────────────────────────────────────────

FORBIDDEN_QUARANTINED = set(range(600, 750))
FORBIDDEN_UNTOUCHED = set(range(2000, 2100))


def assert_partition_safety(seeds: List[int], context_label: str) -> None:
    """Fail-closed assertion ensuring no prohibited seeds are ever accessed."""
    s_set = set(seeds)
    q_overlap = s_set.intersection(FORBIDDEN_QUARANTINED)
    u_overlap = s_set.intersection(FORBIDDEN_UNTOUCHED)
    if q_overlap:
        raise ValueError(f"CRITICAL SAFETY VIOLATION in {context_label}: Quarantined seeds {q_overlap} accessed!")
    if u_overlap:
        raise ValueError(f"CRITICAL SAFETY VIOLATION in {context_label}: Untouched seeds {u_overlap} accessed!")


# ── Classical Reference Baseline ──────────────────────────────────────────────


def evaluate_hybrid_baseline(max_steps: int = 500) -> Dict[str, Any]:
    """Evaluate classical reference HybridController on development partitions."""
    gate1b_seeds = list(range(200, 300))
    diag_seeds = list(range(300, 320))
    assert_partition_safety(gate1b_seeds, "Hybrid Baseline Gate 1B")
    assert_partition_safety(diag_seeds, "Hybrid Baseline Diagnostic")

    ctrl = HybridController()

    # 1. Gate 1B Nominal
    nominal_cond = DIAGNOSTIC_CONDITIONS["1_Nominal"]
    gate1b_succ = 0
    for s in gate1b_seeds:
        env = PerturbedCartPoleEnv(max_steps=max_steps, continuous=True, seed=s, **nominal_cond)
        obs, _ = env.reset(seed=s)
        cos_thetas = []
        rail_hit = False
        for _ in range(max_steps):
            action = ctrl.compute_action(obs)
            obs, _, term, trunc, _ = env.step(action)
            cos_thetas.append(float(obs[2]))
            if term:
                rail_hit = True
                break
        succ, _, _, _ = evaluate_success_predicate(cos_thetas, rail_hit, max_steps)
        if succ:
            gate1b_succ += 1

    gate1b_rate = gate1b_succ / len(gate1b_seeds)

    # 2. Diagnostic Conditions
    diag_rates = {}
    for cond_name, cond_params in DIAGNOSTIC_CONDITIONS.items():
        cond_succ = 0
        for s in diag_seeds:
            env = PerturbedCartPoleEnv(max_steps=max_steps, continuous=True, seed=s, **cond_params)
            obs, _ = env.reset(seed=s)
            cos_thetas = []
            rail_hit = False
            for _ in range(max_steps):
                action = ctrl.compute_action(obs)
                obs, _, term, trunc, _ = env.step(action)
                cos_thetas.append(float(obs[2]))
                if term:
                    rail_hit = True
                    break
            succ, _, _, _ = evaluate_success_predicate(cos_thetas, rail_hit, max_steps)
            if succ:
                cond_succ += 1
        diag_rates[cond_name] = cond_succ / len(diag_seeds)

    robust_succ = float(np.mean(list(diag_rates.values())))
    worst_succ = float(np.min(list(diag_rates.values())))

    return {
        "method": "Classical_HybridController",
        "gate1b_nominal_success": gate1b_rate,
        "robust_success": robust_succ,
        "worst_condition_success": worst_succ,
        "diagnostic_rates": diag_rates,
        "active_parameters": 0,  # Analytical classical
        "weight_macs": 0,
        "gate1b_pass": gate1b_rate >= 0.90,
        "worst_cond_pass": worst_succ >= 0.40,
        "complexity_pass": True,
    }


# ── Run Stage 3C Controlled Neuroevolution Experiment ─────────────────────────


def run_stage3c_experiment(
    optimizer_seeds: List[int] = [42, 43, 44, 45, 46],
    search_budget: int = 1_850_000,
    validation_budget: int = 150_000,
    total_budget: int = 2_000_000,
    in_dim: int = 7,
    warm_start_path: Optional[str] = "results/champion_neural_h8.pkl",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Execute Stage 3C training and validation across optimizer seeds."""
    train_seeds = list(range(0, 200))
    gate1b_seeds = list(range(200, 300))
    diag_seeds = list(range(300, 320))

    assert_partition_safety(train_seeds, "Stage 3C Training")
    assert_partition_safety(gate1b_seeds, "Stage 3C Gate 1B")
    assert_partition_safety(diag_seeds, "Stage 3C Diagnostic")

    results: List[Dict[str, Any]] = []

    print(f"\n{'='*75}")
    print(f"STAGE 3C EXPERIMENT: NSGA2-WarmStart-DelayAware (in_dim={in_dim})")
    print(f"Seeds: {optimizer_seeds} | Search Budget: {search_budget:,} | Validation Budget: {validation_budget:,}")
    print(f"{'='*75}")

    for seed in optimizer_seeds:
        print(f"\n>>> Running NSGA2-WarmStart-DelayAware with Optimizer Seed {seed}...")
        t_start = time.time()

        tracker = TransitionBudgetTracker(
            total_budget=total_budget,
            search_budget=search_budget,
            validation_budget=validation_budget,
        )

        train_out = train_nsga2(
            optimizer_seed=seed,
            pop_size=20,
            offspring_size=10,
            generations=1000,
            eval_episodes=10,
            max_steps=500,
            budget_tracker=tracker,
            search_budget=search_budget,
            validation_budget=validation_budget,
            total_budget=total_budget,
            train_seeds=train_seeds,
            conditions=STAGE3C_TRAIN_CONDITIONS,
            in_dim=in_dim,
            reward_mode="stage3b",
            warm_start_path=warm_start_path,
            vectorized=True,
            verbose=verbose,
        )

        pareto_front = train_out["pareto_front"]
        if pareto_front:
            # Filter for complexity feasibility
            feasible = [
                g for g in pareto_front
                if g.get_complexity_metrics()["active_parameters"] <= 128
                and g.get_complexity_metrics()["weight_macs"] <= 128
            ]
            pool = feasible if feasible else pareto_front
            best_genome = max(
                pool,
                key=lambda g: (
                    g.fitness.get("robust_success", 0.0),
                    g.fitness.get("worst_success", 0.0),
                    g.fitness.get("task_progress", 0.0),
                )
            )
        else:
            best_genome = train_out["population"][0]

        comp = best_genome.get_complexity_metrics()
        active_params = int(comp["active_parameters"])
        weight_macs = int(comp["weight_macs"])

        # Validation phase on development partitions
        val_out = validate_nsga2_champion(
            best_genome, tracker, max_steps=500, vectorized=True, verbose=verbose
        )

        # Model serialization (both pickle and JSON for full reproducibility)
        model_filename = f"stage3c_nsga2_delayaware_seed{seed}.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(best_genome.to_dict(), f)

        json_path = os.path.join(MODELS_DIR, f"stage3c_nsga2_delayaware_seed{seed}.json")
        with open(json_path, "w") as f:
            json.dump(best_genome.to_dict(), f, indent=2)

        gate1b_pass = bool(val_out["gate1b_rate"] >= 0.90)
        worst_cond_pass = bool(val_out["worst_success"] >= 0.40)
        complexity_pass = bool(active_params <= 128 and weight_macs <= 128)
        all_passed = bool(gate1b_pass and worst_cond_pass and complexity_pass)

        candidate_info = {
            "method": "NSGA2-WarmStart-DelayAware",
            "optimizer_seed": seed,
            "in_dim": in_dim,
            "model_path": model_path,
            "json_path": json_path,
            "active_parameters": active_params,
            "weight_macs": weight_macs,
            "training_search_transitions": tracker.search_consumed,
            "validation_transitions": tracker.validation_consumed,
            "total_transitions": tracker.consumed_transitions,
            "completed_generations": train_out["completed_generations"],
            "candidate_evaluations": train_out["candidate_evaluations"],
            "stopping_reason": train_out["stopping_reason"],
            "wall_clock_time_s": time.time() - t_start,
            "gate1b_nominal_success": val_out["gate1b_rate"],
            "robust_success": val_out["robust_success"],
            "worst_condition_success": val_out["worst_success"],
            "diagnostic_rates": val_out["diagnostic_rates"],
            "gate1b_pass": gate1b_pass,
            "worst_cond_pass": worst_cond_pass,
            "complexity_pass": complexity_pass,
            "all_gates_passed": all_passed,
        }

        print(f"[Seed {seed}] Gate 1B Nominal: {candidate_info['gate1b_nominal_success']*100:5.1f}% | "
              f"Robust: {candidate_info['robust_success']*100:5.1f}% | Worst: {candidate_info['worst_condition_success']*100:5.1f}% | "
              f"Params: {active_params} | MACs: {weight_macs} | Transitions: {tracker.consumed_transitions:,} | "
              f"All Gates Passed: {all_passed}")

        results.append(candidate_info)

    return results


# ── Main Entry Point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoEdge Stage 3C Delay-Aware Neuroevolution Orchestrator")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
                        help="Optimizer seeds (default: 42 43 44 45 46)")
    parser.add_argument("--search-budget", type=int, default=1_850_000,
                        help="Maximum search transitions per run (default: 1850000)")
    parser.add_argument("--validation-budget", type=int, default=150_000,
                        help="Maximum validation transitions per run (default: 150000)")
    parser.add_argument("--in-dim", type=int, default=7,
                        help="Controller input dimension (default: 7)")
    parser.add_argument("--quick", action="store_true",
                        help="Run bounded budget quick test")
    args = parser.parse_args()

    s_budget = 50_000 if args.quick else args.search_budget
    v_budget = args.validation_budget
    tot_budget = s_budget + v_budget

    print("Evaluating Classical Reference Baseline (HybridController)...")
    hybrid_res = evaluate_hybrid_baseline()
    print(f"Hybrid Baseline: Gate 1B: {hybrid_res['gate1b_nominal_success']*100:.1f}% | "
          f"Robust: {hybrid_res['robust_success']*100:.1f}% | Worst: {hybrid_res['worst_condition_success']*100:.1f}%")

    runs = run_stage3c_experiment(
        optimizer_seeds=args.seeds,
        search_budget=s_budget,
        validation_budget=v_budget,
        total_budget=tot_budget,
        in_dim=args.in_dim,
        verbose=True,
    )

    all_results: Dict[str, Any] = {
        "classical_reference": hybrid_res,
        "runs": runs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    summary_file = os.path.join(STAGE3C_DIR, "stage3c_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll Stage 3C experiments completed. Summary written to {summary_file}")


if __name__ == "__main__":
    main()
