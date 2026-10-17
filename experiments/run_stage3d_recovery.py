"""Stage 3D Robustness-First Neuroevolution Orchestrator.

Executes Phase 3D:
1. Enforces strict partition quarantine:
   - Search on training seeds 0..199 only.
   - Validation on exposed development seeds 200..299 (Gate 1B) & 300..319 (Diagnostics).
   - Strict assertion against accessing quarantined seeds 600..749 & untouched seeds 2000..2099.
2. Enforces Transition Budget Tracker:
   - Search <= 1,850,000 transitions.
   - Validation <= 150,000 transitions.
   - Total <= 2,000,000 transitions.
3. Architecture constraints:
   - Active parameters <= 128.
   - Weight MACs <= 128.
4. Acceptance criteria:
   - Gate 1B Nominal Success >= 90%.
   - Worst-Condition Success >= 40%.
5. Evaluates seeds 42, 43, 44, 45, 46.
6. Saves model checkpoints, training histories, summary JSON, and comparison visualization.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from baselines.hybrid_controller import HybridController
from env.budget_tracker import TransitionBudgetTracker, BudgetedEnv, BudgetedVectorizedEnv, BudgetExhaustedError
from evolution.nsga2_engine import (
    MaskedGenome,
    STAGE3D_TRAIN_CONDITIONS,
    train_nsga2_stage3d,
    validate_nsga2_champion,
    select_stage3d_champion,
    norm_batch_obs,
    norm_obs,
)
from experiments.run_phase_3a_evolution import DIAGNOSTIC_CONDITIONS
from experiments.run_robustness_experiment import evaluate_success_predicate
from utils.perturbation_loader import PerturbedCartPoleEnv

# ── Directory Layout ──────────────────────────────────────────────────────────

STAGE3D_DIR = os.path.join(_PROJECT_ROOT, "results", "stage3d")
MODELS_DIR = os.path.join(STAGE3D_DIR, "models")
LOGS_DIR = os.path.join(STAGE3D_DIR, "logs")

for d in [STAGE3D_DIR, MODELS_DIR, LOGS_DIR]:
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
        "active_parameters": 0,
        "weight_macs": 0,
        "gate1b_pass": gate1b_rate >= 0.90,
        "worst_cond_pass": worst_succ >= 0.40,
        "complexity_pass": True,
    }


# ── Run Stage 3D Controlled Neuroevolution Experiment ─────────────────────────


def run_stage3d_experiment(
    optimizer_seeds: List[int] = [42, 43, 44, 45, 46],
    search_budget: int = 1_850_000,
    validation_budget: int = 150_000,
    total_budget: int = 2_000_000,
    warm_start_path: Optional[str] = "results/stage3c/models/stage3c_nsga2_delayaware_seed44.pkl",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Execute Stage 3D training and validation across optimizer seeds."""
    train_seeds = list(range(0, 200))
    gate1b_seeds = list(range(200, 300))
    diag_seeds = list(range(300, 320))

    assert_partition_safety(train_seeds, "Stage 3D Training")
    assert_partition_safety(gate1b_seeds, "Stage 3D Gate 1B")
    assert_partition_safety(diag_seeds, "Stage 3D Diagnostic")

    results: List[Dict[str, Any]] = []

    print(f"\n{'='*78}")
    print(f"STAGE 3D EXPERIMENT: Robustness-First Neuroevolution")
    print(f"Seeds: {optimizer_seeds} | Search Budget: {search_budget:,} | Validation Budget: {validation_budget:,}")
    print(f"{'='*78}")

    for seed in optimizer_seeds:
        print(f"\n>>> Running Stage 3D Robustness-First with Optimizer Seed {seed}...")
        t_start = time.time()

        tracker = TransitionBudgetTracker(
            total_budget=total_budget,
            search_budget=search_budget,
            validation_budget=validation_budget,
        )

        train_out = train_nsga2_stage3d(
            optimizer_seed=seed,
            pop_size=20,
            offspring_size=10,
            generations=1000,
            search_budget=search_budget,
            validation_budget=validation_budget,
            total_budget=total_budget,
            budget_tracker=tracker,
            train_seeds=train_seeds,
            conditions=STAGE3D_TRAIN_CONDITIONS,
            vectorized=True,
            warm_start_path=warm_start_path,
            in_dim=7,
            verbose=verbose,
        )

        pareto_front = train_out["pareto_front"]
        if pareto_front:
            best_genome = select_stage3d_champion(pareto_front)
        else:
            best_genome = select_stage3d_champion(train_out["population"])

        comp = best_genome.get_complexity_metrics()
        active_params = int(comp["active_parameters"])
        weight_macs = int(comp["weight_macs"])

        # Validation phase on development partitions
        val_out = validate_nsga2_champion(
            best_genome, tracker, max_steps=500, vectorized=True, verbose=verbose
        )

        # Model serialization (both pickle and JSON for full reproducibility)
        model_filename = f"stage3d_nsga2_robust_seed{seed}.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(best_genome.to_dict(), f)

        json_path = os.path.join(MODELS_DIR, f"stage3d_nsga2_robust_seed{seed}.json")
        with open(json_path, "w") as f:
            json.dump(best_genome.to_dict(), f, indent=2)

        gate1b_pass = bool(val_out["gate1b_rate"] >= 0.90)
        worst_cond_pass = bool(val_out["worst_success"] >= 0.40)
        complexity_pass = bool(active_params <= 128 and weight_macs <= 128)
        all_passed = bool(gate1b_pass and worst_cond_pass and complexity_pass)

        candidate_info = {
            "method": "NSGA2-RobustnessFirst-Stage3D",
            "optimizer_seed": seed,
            "in_dim": 7,
            "parent_provenance": "results/stage3c/models/stage3c_nsga2_delayaware_seed44.pkl",
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


def plot_comparison_chart(all_results: Dict[str, Any], output_path: str) -> None:
    """Generate visual bar chart comparing Classical, Stage 3B, Stage 3C, and Phase 3D."""
    conditions = [
        "1_Nominal",
        "2_PoleMass_+30%",
        "3_CartMass_+30%",
        "4a_CartFriction_0.50",
        "4b_JointFriction_0.010",
        "4c_CombinedFriction",
        "5_SensorNoise_0.05",
        "6a_ActuatorDelay_1Step",
        "6b_ActuatorDelay_2Step",
        "7_CombinedHistoricalStress",
    ]
    cond_labels = [
        "Nominal", "+30% Pole", "+30% Cart", "Cart Fric",
        "Joint Fric", "Comb Fric", "Noise 0.05", "Delay 1-St",
        "Delay 2-St", "Comb Stress"
    ]

    # Classical Reference
    classical_rates = [all_results["classical_reference"]["diagnostic_rates"].get(c, 0.0) * 100 for c in conditions]

    # Stage 3C Champion (Seed 44)
    stage3c_seed44 = [100.0, 100.0, 5.0, 65.0, 70.0, 0.0, 60.0, 100.0, 100.0, 95.0]

    # Stage 3B Baseline (Seed 44)
    stage3b_seed44 = [100.0, 100.0, 10.0, 100.0, 0.0, 0.0, 75.0, 0.0, 0.0, 0.0]

    # Stage 3D Best Run
    runs = all_results["runs"]
    best_3d = max(runs, key=lambda r: (r["worst_condition_success"], r["robust_success"]))
    stage3d_rates = [best_3d["diagnostic_rates"].get(c, 0.0) * 100 for c in conditions]

    x = np.arange(len(conditions))
    width = 0.20

    plt.figure(figsize=(14, 7), dpi=300)
    plt.bar(x - 1.5 * width, classical_rates, width, label="Classical Hybrid Baseline (Analytical)", color="#4A90E2", alpha=0.9)
    plt.bar(x - 0.5 * width, stage3b_seed44, width, label="Stage 3B Champion (Memoryless)", color="#D0021B", alpha=0.8)
    plt.bar(x + 0.5 * width, stage3c_seed44, width, label="Stage 3C Champion (Delay-Aware)", color="#F5A623", alpha=0.85)
    plt.bar(x + 1.5 * width, stage3d_rates, width, label=f"Phase 3D Champion (Seed {best_3d['optimizer_seed']})", color="#2ECC71", alpha=0.95)

    plt.axhline(y=40.0, color="#E74C3C", linestyle="--", linewidth=1.5, label="Worst-Condition Screening Floor (40%)")
    plt.axhline(y=90.0, color="#27AE60", linestyle=":", linewidth=1.5, label="Gate 1B Nominal Floor (90%)")

    plt.ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    plt.title("EvoEdge Neuroevolution Evolution: Robustness Breakdown Across Diagnostic Conditions", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, cond_labels, rotation=25, ha="right", fontsize=10, fontweight="semibold")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Comparison chart saved to {output_path}")


# ── Main Entry Point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoEdge Phase 3D Robustness-First Neuroevolution Orchestrator")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
                        help="Optimizer seeds (default: 42 43 44 45 46)")
    parser.add_argument("--search-budget", type=int, default=1_850_000,
                        help="Maximum search transitions per run (default: 1850000)")
    parser.add_argument("--validation-budget", type=int, default=150_000,
                        help="Maximum validation transitions per run (default: 150000)")
    parser.add_argument("--warm-start-path", type=str,
                        default="results/stage3c/models/stage3c_nsga2_delayaware_seed44.pkl",
                        help="Path to Stage 3C warm-start model")
    args = parser.parse_args()

    s_budget = args.search_budget
    v_budget = args.validation_budget
    tot_budget = s_budget + v_budget

    print("Evaluating Classical Reference Baseline (HybridController)...")
    hybrid_res = evaluate_hybrid_baseline()
    print(f"Hybrid Baseline: Gate 1B: {hybrid_res['gate1b_nominal_success']*100:.1f}% | "
          f"Robust: {hybrid_res['robust_success']*100:.1f}% | Worst: {hybrid_res['worst_condition_success']*100:.1f}%")

    runs = run_stage3d_experiment(
        optimizer_seeds=args.seeds,
        search_budget=s_budget,
        validation_budget=v_budget,
        total_budget=tot_budget,
        warm_start_path=args.warm_start_path,
        verbose=True,
    )

    all_results: Dict[str, Any] = {
        "classical_reference": hybrid_res,
        "runs": runs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    summary_file = os.path.join(STAGE3D_DIR, "stage3d_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    chart_file = os.path.join(STAGE3D_DIR, "stage3d_comparison.png")
    plot_comparison_chart(all_results, chart_file)

    print(f"\nAll Phase 3D experiments completed. Summary written to {summary_file}")


if __name__ == "__main__":
    main()
