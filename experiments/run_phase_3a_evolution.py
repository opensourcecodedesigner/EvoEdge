"""Phase 3A: Multi-Objective NSGA-II Structural Neuroevolution Orchestrator.

Implements:
  1. Teacher audit on training seeds (0..49) across all 10 diagnostic conditions.
  2. Gate 0: Nominal-only evolution (Arm A: from-scratch, Arm B: warm-start).
     - Gate decision on validation seeds 200..249 (threshold >= 48/50 successes).
     - Validates that a non-identical evolved descendant satisfies threshold.
  3. Full Robust NSGA-II Evolution across Arm A and Arm B:
     - 5 independent optimizer seeds each: [42, 43, 44, 45, 46].
     - Multi-objective optimization:
         - Maximize robust task success (equal-weight mean of condition success rates).
         - Minimize active parameter count (weights + biases).
         - Minimize weight MACs per inference step.
     - Deb-style constrained dominance (success_floor = 0.4832).
     - Checkpoint/resume support.
  4. Baseline Evaluation on Phase 3A validation seeds 250..399:
     - Frozen H=8 controller.
     - Phase 2C model (Arm_C_seed46 / Arm_A_seed43).
     - 64x64 MLP dense & pruned at 50%, 75%, 90%.
     - Magnitude-pruned H=8 controller at 50%.
     - 6-input H=8 previous-action controller.
  5. Comprehensive artifact and ledger export.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.vectorized_env import VectorizedCartPoleEnv
from utils.perturbation_loader import PerturbedCartPoleEnv, load_perturbation_suite
from baselines.hybrid_controller import HybridController
from evolution.nsga2_engine import (
    MaskedGenome,
    constrained_dominates,
    fast_non_dominated_sort,
    compute_crowding_distance,
    nsga2_survivor_selection,
)
from experiments.run_robustness_experiment import TinyNet, evaluate_success_predicate
from experiments.run_baselines_pruning import MLP64, prune_model


# ── Canonical Constants & Partition Definitions ───────────────────────────────

DIAGNOSTIC_CONDITIONS = {
    "1_Nominal": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0, "obs_noise_sigma": 0.0, "delay_steps": 0
    },
    "2_PoleMass_+30%": {
        "cart_mass": 1.0, "pole_mass": 0.130, "b_cart": 0.0, "b_pole": 0.0, "obs_noise_sigma": 0.0, "delay_steps": 0
    },
    "3_CartMass_+30%": {
        "cart_mass": 1.30, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0, "obs_noise_sigma": 0.0, "delay_steps": 0
    },
    "4a_CartFriction_0.50": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.50, "b_pole": 0.0, "obs_noise_sigma": 0.0, "delay_steps": 0
    },
    "4b_JointFriction_0.010": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.010, "obs_noise_sigma": 0.0, "delay_steps": 0
    },
    "4c_CombinedFriction": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.50, "b_pole": 0.010, "obs_noise_sigma": 0.0, "delay_steps": 0
    },
    "5_SensorNoise_0.05": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0, "obs_noise_sigma": 0.050, "delay_steps": 0
    },
    "6a_ActuatorDelay_1Step": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0, "obs_noise_sigma": 0.0, "delay_steps": 1
    },
    "6b_ActuatorDelay_2Step": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0, "obs_noise_sigma": 0.0, "delay_steps": 2
    },
    "7_CombinedHistoricalStress": {
        "cart_mass": 1.30, "pole_mass": 0.130, "b_cart": 0.50, "b_pole": 0.010, "obs_noise_sigma": 0.050, "delay_steps": 1
    },
}

TRAIN_SEEDS = list(range(0, 200))
GATE0_VAL_SEEDS = list(range(200, 250))
PHASE3A_VAL_SEEDS = list(range(250, 400))
FINAL_HOLDOUT_SEEDS = list(range(2000, 2100))

SUCCESS_FLOOR = 0.4832  # 0.8 * 0.6040 (Frozen H=8 robust success)


def check_seed_safety(seeds: Sequence[int]) -> None:
    """Enforce strict seed boundary protection."""
    for s in seeds:
        if 2000 <= s <= 2099:
            raise ValueError(f"CRITICAL SAFETY VIOLATION: Access to final holdout seed {s} strictly forbidden!")


def norm_batch_obs(obs: np.ndarray) -> np.ndarray:
    """Standard observation normalization for neural evaluation."""
    return np.stack([
        obs[:, 0] / 2.4,
        np.clip(obs[:, 1] / 10.0, -1.0, 1.0),
        obs[:, 2],
        obs[:, 3],
        np.clip(obs[:, 4] / 10.0, -1.0, 1.0),
    ], axis=1)


# ── Evaluator Functions ───────────────────────────────────────────────────────


def evaluate_candidate_vectorized(
    genome: MaskedGenome,
    conditions: Dict[str, Dict[str, Any]],
    seeds: List[int],
    max_steps: int = 500,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate candidate genome over specified conditions and seeds using vectorized simulator.

    Returns:
        (robust_success, worst_condition_success, per_condition_rates)
    """
    check_seed_safety(seeds)
    num_envs = len(seeds)
    cond_rates: Dict[str, float] = {}

    for cond_name, cond_params in conditions.items():
        vec_env = VectorizedCartPoleEnv(num_envs=num_envs, max_steps=max_steps, **cond_params)
        obs, _ = vec_env.reset(seeds=seeds)
        cos_thetas: List[np.ndarray] = []

        for step in range(max_steps):
            norm_o = norm_batch_obs(obs)
            actions = genome.forward_batch(norm_o)
            obs, _, _, _, _ = vec_env.step(actions)
            cos_thetas.append(obs[:, 2])

        cos_arr = np.array(cos_thetas)  # shape (max_steps, num_envs)
        final_window = cos_arr[-100:]
        upright_count = np.sum(final_window > 0.9, axis=0)
        success = (upright_count >= 95) & (~vec_env.boundary_violated)
        rate = float(np.mean(success))
        cond_rates[cond_name] = rate

    robust_success = float(np.mean(list(cond_rates.values())))
    worst_success = float(np.min(list(cond_rates.values())))
    return robust_success, worst_success, cond_rates


def evaluate_genome_fitness(genome: MaskedGenome, fitness_conditions: Dict[str, Dict[str, Any]], eval_seeds: List[int]) -> None:
    """Compute and attach multi-objective fitness & Deb constraint violation to genome."""
    robust_succ, worst_succ, cond_rates = evaluate_candidate_vectorized(genome, fitness_conditions, eval_seeds)
    comp = genome.get_complexity_metrics()

    genome.fitness = {
        "robust_success": robust_succ,
        "worst_success": worst_succ,
        "active_params": comp["active_parameters"],
        "weight_macs": comp["weight_macs"],
        "active_neurons": comp["active_neurons"],
    }
    # Constraint violation: 0 if feasible (>= SUCCESS_FLOOR), positive if infeasible
    genome.constraint_violation = max(0.0, SUCCESS_FLOOR - robust_succ)


# ── Teacher Audit ─────────────────────────────────────────────────────────────


def audit_hand_designed_teacher(eval_seeds: List[int]) -> Dict[str, Any]:
    """Audit the hand-designed teacher used by DAgger on training seeds (0..49)."""
    check_seed_safety(eval_seeds)
    print("\n--- [Audit: Hand-Designed Teacher on Training Seeds] ---")
    teacher = HybridController()
    cond_rates: Dict[str, float] = {}
    episodes_per_cond = len(eval_seeds)

    for cond_name, cond_params in DIAGNOSTIC_CONDITIONS.items():
        succ_count = 0
        rail_count = 0
        other_fail_count = 0

        for s in eval_seeds:
            env = PerturbedCartPoleEnv(max_steps=500, continuous=True, seed=s, **cond_params)
            obs, _ = env.reset(seed=s)
            cos_th_list: List[float] = []
            rail_hit = False

            for step_i in range(500):
                action = teacher.compute_action(obs)
                obs, r, term, trunc, info = env.step(action)
                cos_th = float(obs[2])
                cos_th_list.append(cos_th)

                if term or abs(env._state[0]) >= env.X_THRESHOLD:
                    rail_hit = True
                    break

            is_succ, is_rail, is_other, _ = evaluate_success_predicate(cos_th_list, rail_hit, 500)
            if is_succ:
                succ_count += 1
            elif is_rail:
                rail_count += 1
            else:
                other_fail_count += 1

        rate = succ_count / episodes_per_cond
        cond_rates[cond_name] = rate
        print(f"  {cond_name:28s}: {succ_count}/{episodes_per_cond} ({rate*100:5.1f}%) | Rail: {rail_count} | Other: {other_fail_count}")

    mean_rate = float(np.mean(list(cond_rates.values())))
    worst_rate = float(np.min(list(cond_rates.values())))
    print(f"  >> Teacher Mean Success Rate: {mean_rate*100:.2f}% | Worst Condition: {worst_rate*100:.2f}%\n")

    return {
        "mean_success_rate": mean_rate,
        "worst_condition_success_rate": worst_rate,
        "condition_rates": cond_rates,
    }


# ── Gate 0: Nominal-Only Evolution ────────────────────────────────────────────


def run_gate0_nominal_evolution(
    warm_start_weights: Dict[str, np.ndarray],
    opt_seed: int = 42,
    pop_size: int = 20,
    generations: int = 5,
) -> Dict[str, Any]:
    """Run Gate 0 nominal-only evolution on warm-start and scratch arms."""
    print(f"\n--- [Gate 0: Nominal-Only Evolution (Optimizer Seed {opt_seed})] ---")
    rng = np.random.default_rng(opt_seed)
    train_eval_seeds = TRAIN_SEEDS[:50]  # Seeds 0..49 for training fitness
    nominal_cond = {"1_Nominal": DIAGNOSTIC_CONDITIONS["1_Nominal"]}

    # Warm-start population
    pop_warm: List[MaskedGenome] = []
    # Candidate 0 is the exact baseline
    g_base = MaskedGenome(rng=rng)
    g_base.load_from_dense_weights(
        warm_start_weights["w1"], warm_start_weights["b1"],
        warm_start_weights["w2"], warm_start_weights["b2"]
    )
    pop_warm.append(g_base)

    # Descendants: mutated / pruned copies
    for _ in range(pop_size - 1):
        g = g_base.copy()
        g.rng = rng
        g.mutate(weight_mutation_rate=0.7, connection_toggle_rate=0.3, neuron_toggle_rate=0.1)
        pop_warm.append(g)

    # Evolve warm-start population on nominal conditions
    best_descendant: Optional[MaskedGenome] = None
    for gen in range(generations):
        for ind in pop_warm:
            evaluate_genome_fitness(ind, nominal_cond, train_eval_seeds)

        survivors = nsga2_survivor_selection(pop_warm, pop_size // 2)

        # Offspring
        offspring: List[MaskedGenome] = []
        for p in survivors:
            child = p.copy()
            child.rng = rng
            child.mutate()
            offspring.append(child)

        pop_warm = survivors + offspring

    # Evaluate on Gate 0 validation seeds (200..249)
    # Filter for non-identical descendants (must differ in weights or masks)
    non_identical: List[MaskedGenome] = []
    for ind in pop_warm:
        w_diff = np.max(np.abs(ind.w1 - warm_start_weights["w1"]))
        mask_diff = np.any(ind.w1_mask[:8, :] != True) or np.any(ind.neuron_mask[:8] != True)
        if w_diff > 1e-6 or mask_diff:
            non_identical.append(ind)

    print(f"  Warm-start produced {len(non_identical)} non-identical candidate descendants.")

    gate_passed = False
    best_candidate_val_succ = 0
    best_candidate: Optional[MaskedGenome] = None

    for ind in non_identical:
        val_succ, _, _ = evaluate_candidate_vectorized(ind, nominal_cond, GATE0_VAL_SEEDS)
        succ_count = int(round(val_succ * len(GATE0_VAL_SEEDS)))
        if succ_count > best_candidate_val_succ:
            best_candidate_val_succ = succ_count
            best_candidate = ind

    print(f"  Best non-identical descendant validation score: {best_candidate_val_succ}/50")
    if best_candidate_val_succ >= 48:
        gate_passed = True
        print("  >> GATE 0 VERDICT: PASSED (>= 48/50 successes)")
    else:
        print("  >> GATE 0 VERDICT: FAILED (< 48/50 successes)")

    return {
        "gate_passed": gate_passed,
        "best_candidate_val_succ": best_candidate_val_succ,
        "threshold": 48,
        "total_val_episodes": 50,
        "best_candidate": best_candidate,
    }


# ── Full Robust NSGA-II Evolution ─────────────────────────────────────────────


def run_robust_nsga2_arm(
    arm_name: str,
    warm_start_weights: Optional[Dict[str, np.ndarray]],
    optimizer_seed: int,
    pop_size: int = 30,
    generations: int = 15,
    checkpoint_dir: str = "results/phase_3a/checkpoints",
) -> Tuple[List[MaskedGenome], Dict[str, Any]]:
    """Run full robust NSGA-II evolutionary optimization for a single arm and seed."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{arm_name}_seed{optimizer_seed}.pkl")

    rng = np.random.default_rng(optimizer_seed)
    train_seeds = TRAIN_SEEDS[:50]  # Evaluate fitness across 50 training seeds

    # Check for existing checkpoint
    start_gen = 0
    population: List[MaskedGenome] = []
    survivors: List[MaskedGenome] = []

    if os.path.exists(ckpt_path):
        print(f"  Resuming {arm_name} seed {optimizer_seed} from checkpoint {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            ckpt_data = pickle.load(f)
            start_gen = ckpt_data["generation"] + 1
            population = [MaskedGenome.from_dict(d, rng=rng) for d in ckpt_data["population"]]
            survivors = list(population)

    if not population:
        # Initialize population
        if arm_name == "Arm_B_WarmStart" and warm_start_weights is not None:
            # Seed 0: exact baseline
            g0 = MaskedGenome(rng=rng)
            g0.load_from_dense_weights(
                warm_start_weights["w1"], warm_start_weights["b1"],
                warm_start_weights["w2"], warm_start_weights["b2"]
            )
            population.append(g0)
            # Remaining: perturbed / pruned copies
            for _ in range(pop_size - 1):
                g = g0.copy()
                g.rng = rng
                g.mutate(weight_mutation_rate=0.7, connection_toggle_rate=0.3, neuron_toggle_rate=0.2)
                population.append(g)
        else:
            # From scratch
            for _ in range(pop_size):
                g = MaskedGenome(rng=rng)
                population.append(g)

    # Evolutionary loop
    t0 = time.time()
    for gen in range(start_gen, generations):
        gen_t0 = time.time()
        for ind in population:
            evaluate_genome_fitness(ind, DIAGNOSTIC_CONDITIONS, train_seeds)

        # Survivor selection
        survivors = nsga2_survivor_selection(population, pop_size // 2)

        # Offspring reproduction
        offspring: List[MaskedGenome] = []
        for p in survivors:
            child = p.copy()
            child.rng = rng
            child.mutate(
                weight_mutation_rate=0.6,
                weight_mutation_power=0.15,
                connection_toggle_rate=0.2,
                neuron_toggle_rate=0.1,
                activation_mutation_rate=0.05,
            )
            offspring.append(child)

        population = survivors + offspring
        gen_dt = time.time() - gen_t0

        # Best in generation by robust success
        best_gen = max(survivors, key=lambda x: x.fitness.get("robust_success", 0.0))
        print(f"    [{arm_name} | Seed {optimizer_seed}] Gen {gen+1:2d}/{generations} ({gen_dt:4.1f}s) | "
              f"Best Robust Succ: {best_gen.fitness['robust_success']*100:5.1f}% | "
              f"Active Params: {best_gen.fitness['active_params']} | MACs: {best_gen.fitness['weight_macs']}")

        # Save checkpoint
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "generation": gen,
                "population": [ind.to_dict() for ind in survivors],
            }, f)

    total_time = time.time() - t0

    # Final evaluation of survivors
    for ind in survivors:
        evaluate_genome_fitness(ind, DIAGNOSTIC_CONDITIONS, train_seeds)

    fronts = fast_non_dominated_sort(survivors)
    pareto_front = fronts[0] if fronts else []

    meta = {
        "arm_name": arm_name,
        "optimizer_seed": optimizer_seed,
        "wall_clock_seconds": round(total_time, 2),
        "population_size": pop_size,
        "generations": generations,
        "pareto_size": len(pareto_front),
    }

    return pareto_front, meta


# ── Baseline Evaluator on Phase 3A Validation Seeds (250..399) ────────────────


def evaluate_baselines_on_validation_seeds(
    warm_start_weights: Dict[str, np.ndarray],
    val_seeds: List[int],
) -> List[Dict[str, Any]]:
    """Evaluate all required baselines on Phase 3A validation partition (seeds 250..399)."""
    check_seed_safety(val_seeds)
    print("\n--- [Evaluating All Baselines on Phase 3A Validation Seeds (250..399)] ---")
    baseline_records: List[Dict[str, Any]] = []

    # 1. Frozen H=8 Controller
    g_h8 = MaskedGenome()
    g_h8.load_from_dense_weights(
        warm_start_weights["w1"], warm_start_weights["b1"],
        warm_start_weights["w2"], warm_start_weights["b2"]
    )
    r_succ, w_succ, cond_rates = evaluate_candidate_vectorized(g_h8, DIAGNOSTIC_CONDITIONS, val_seeds)
    comp_h8 = g_h8.get_complexity_metrics()
    baseline_records.append({
        "baseline_name": "Frozen_H8_Baseline",
        "robust_success": r_succ,
        "worst_success": w_succ,
        "active_parameters": comp_h8["active_parameters"],
        "weight_macs": comp_h8["weight_macs"],
        "num_activations": comp_h8["num_activations"],
        "dense_params": comp_h8["dense_params"],
        "dense_macs": comp_h8["dense_macs"],
        "format": "pickle / C float array",
        "condition_rates": cond_rates,
    })
    print(f"  Frozen H=8              : Robust: {r_succ*100:5.1f}% | Worst: {w_succ*100:5.1f}% | Params: {comp_h8['active_parameters']} | MACs: {comp_h8['weight_macs']}")

    # 2. Magnitude-Pruned H=8 Controller (50% weight sparsity)
    g_h8_pruned = g_h8.copy()
    all_w = np.concatenate([g_h8_pruned.w1.flatten(), g_h8_pruned.w2.flatten()])
    thresh = float(np.percentile(np.abs(all_w), 50))
    g_h8_pruned.w1_mask = (np.abs(g_h8_pruned.w1) > thresh)
    g_h8_pruned.w2_mask = (np.abs(g_h8_pruned.w2) > thresh)
    r_succ_p, w_succ_p, cond_rates_p = evaluate_candidate_vectorized(g_h8_pruned, DIAGNOSTIC_CONDITIONS, val_seeds)
    comp_p = g_h8_pruned.get_complexity_metrics()
    baseline_records.append({
        "baseline_name": "Magnitude_Pruned_H8_50%",
        "robust_success": r_succ_p,
        "worst_success": w_succ_p,
        "active_parameters": comp_p["active_parameters"],
        "weight_macs": comp_p["weight_macs"],
        "num_activations": comp_p["num_activations"],
        "dense_params": comp_p["dense_params"],
        "dense_macs": comp_p["dense_macs"],
        "format": "MaskedGenome JSON",
        "condition_rates": cond_rates_p,
    })
    print(f"  Pruned H=8 (50% sparse) : Robust: {r_succ_p*100:5.1f}% | Worst: {w_succ_p*100:5.1f}% | Params: {comp_p['active_parameters']} | MACs: {comp_p['weight_macs']}")

    # 3. Phase 2C Champion Model (Arm_C_seed46, 6-input H=8 previous action)
    p2c_path = os.path.join(_PROJECT_ROOT, "results", "replication_phase_2c", "models", "arm_c_seed46.pt")
    if os.path.exists(p2c_path):
        m_p2c = TinyNet(in_dim=6)
        m_p2c.load_state_dict(torch.load(p2c_path, weights_only=True))
        m_p2c.eval()

        # Evaluate serial wrapper
        succ_by_cond: Dict[str, float] = {}
        for c_name, c_params in DIAGNOSTIC_CONDITIONS.items():
            succ_cnt = 0
            for s in val_seeds:
                env = PerturbedCartPoleEnv(max_steps=500, continuous=True, seed=s, **c_params)
                obs, _ = env.reset(seed=s)
                prev_a = 0.0
                cos_list = []
                rail = False
                for _ in range(500):
                    feat = np.append(norm_batch_obs(obs[None, :])[0], prev_a)
                    t_feat = torch.from_numpy(feat).float().unsqueeze(0)
                    with torch.no_grad():
                        a = float(m_p2c(t_feat)[0, 0].item())
                    obs, r, term, trunc, info = env.step(a)
                    prev_a = info["applied_action"]
                    cos_list.append(float(obs[2]))
                    if term or abs(env._state[0]) >= env.X_THRESHOLD:
                        rail = True
                        break
                is_s, _, _, _ = evaluate_success_predicate(cos_list, rail, 500)
                if is_s:
                    succ_cnt += 1
            succ_by_cond[c_name] = succ_cnt / len(val_seeds)

        r_p2c = float(np.mean(list(succ_by_cond.values())))
        w_p2c = float(np.min(list(succ_by_cond.values())))
        baseline_records.append({
            "baseline_name": "Phase2C_Champion_ArmC_Seed46",
            "robust_success": r_p2c,
            "worst_success": w_p2c,
            "active_parameters": 65,
            "weight_macs": 56,
            "num_activations": 9,
            "dense_params": 65,
            "dense_macs": 56,
            "format": "PyTorch .pt",
            "condition_rates": succ_by_cond,
        })
        print(f"  Phase 2C Champion (Arm C): Robust: {r_p2c*100:5.1f}% | Worst: {w_p2c*100:5.1f}% | Params: 65 | MACs: 56")

    # 4. Dense 64x64 MLP and Pruned Variants (50%, 75%, 90%)
    mlp_path = os.path.join(_PROJECT_ROOT, "results", "mlp64_baseline.pt")
    if os.path.exists(mlp_path):
        m_mlp = MLP64()
        m_mlp.load_state_dict(torch.load(mlp_path, weights_only=True))
        m_mlp.eval()

        prune_ratios = [0.0, 0.50, 0.75, 0.90]
        for pr in prune_ratios:
            m_eval = prune_model(m_mlp, pr)[0] if pr > 0.0 else m_mlp
            m_eval.eval()

            active_weights = sum((p != 0).sum().item() for name, p in m_eval.named_parameters() if "weight" in name)
            active_biases = sum(p.numel() for name, p in m_eval.named_parameters() if "bias" in name)
            total_active_params = active_weights + active_biases

            succ_by_cond = {}
            for c_name, c_params in DIAGNOSTIC_CONDITIONS.items():
                succ_cnt = 0
                for s in val_seeds:
                    env = PerturbedCartPoleEnv(max_steps=500, continuous=True, seed=s, **c_params)
                    obs, _ = env.reset(seed=s)
                    cos_list = []
                    rail = False
                    for _ in range(500):
                        feat = norm_batch_obs(obs[None, :])[0]
                        t_feat = torch.from_numpy(feat).float().unsqueeze(0)
                        with torch.no_grad():
                            a = float(m_eval(t_feat)[0, 0].item())
                        obs, r, term, trunc, info = env.step(a)
                        cos_list.append(float(obs[2]))
                        if term or abs(env._state[0]) >= env.X_THRESHOLD:
                            rail = True
                            break
                    is_s, _, _, _ = evaluate_success_predicate(cos_list, rail, 500)
                    if is_s:
                        succ_cnt += 1
                succ_by_cond[c_name] = succ_cnt / len(val_seeds)

            r_mlp = float(np.mean(list(succ_by_cond.values())))
            w_mlp = float(np.min(list(succ_by_cond.values())))
            b_label = "Dense_64x64_MLP" if pr == 0.0 else f"Magnitude_Pruned_64x64_MLP_{int(pr*100)}%"
            baseline_records.append({
                "baseline_name": b_label,
                "robust_success": r_mlp,
                "worst_success": w_mlp,
                "active_parameters": total_active_params,
                "weight_macs": active_weights,
                "num_activations": 129,
                "dense_params": 4609,
                "dense_macs": 4480,
                "format": "PyTorch .pt",
                "condition_rates": succ_by_cond,
            })
            print(f"  {b_label:24s}: Robust: {r_mlp*100:5.1f}% | Worst: {w_mlp*100:5.1f}% | Params: {total_active_params} | MACs: {active_weights}")

    # 5. 6-input Previous Action H=8 (Phase 2C Nominal Arm C seed 42)
    p2c_nominal_path = os.path.join(_PROJECT_ROOT, "results", "replication_phase_2c", "models", "arm_c_seed42.pt")
    if os.path.exists(p2c_nominal_path):
        m_nom = TinyNet(in_dim=6)
        m_nom.load_state_dict(torch.load(p2c_nominal_path, weights_only=True))
        m_nom.eval()
        succ_by_cond = {}
        for c_name, c_params in DIAGNOSTIC_CONDITIONS.items():
            succ_cnt = 0
            for s in val_seeds:
                env = PerturbedCartPoleEnv(max_steps=500, continuous=True, seed=s, **c_params)
                obs, _ = env.reset(seed=s)
                prev_a = 0.0
                cos_list = []
                rail = False
                for _ in range(500):
                    feat = np.append(norm_batch_obs(obs[None, :])[0], prev_a)
                    t_feat = torch.from_numpy(feat).float().unsqueeze(0)
                    with torch.no_grad():
                        a = float(m_nom(t_feat)[0, 0].item())
                    obs, r, term, trunc, info = env.step(a)
                    prev_a = info["applied_action"]
                    cos_list.append(float(obs[2]))
                    if term or abs(env._state[0]) >= env.X_THRESHOLD:
                        rail = True
                        break
                is_s, _, _, _ = evaluate_success_predicate(cos_list, rail, 500)
                if is_s:
                    succ_cnt += 1
            succ_by_cond[c_name] = succ_cnt / len(val_seeds)
        r_nom = float(np.mean(list(succ_by_cond.values())))
        w_nom = float(np.min(list(succ_by_cond.values())))
        baseline_records.append({
            "baseline_name": "Six_Input_PrevAction_H8_ArmC_Seed42",
            "robust_success": r_nom,
            "worst_success": w_nom,
            "active_parameters": 65,
            "weight_macs": 56,
            "num_activations": 9,
            "dense_params": 65,
            "dense_macs": 56,
            "format": "PyTorch .pt",
            "condition_rates": succ_by_cond,
        })
        print(f"  Six-Input Prev-Action H=8: Robust: {r_nom*100:5.1f}% | Worst: {w_nom*100:5.1f}% | Params: 65 | MACs: 56")

    return baseline_records


# ── Main Phase 3A Orchestrator ────────────────────────────────────────────────


def main():
    print("=" * 80)
    print("=== EvoEdge: Phase 3A Multi-Objective NSGA-II Neuroevolution ===")
    print("=" * 80)

    # 1. Protection & Checkpoint Load
    h8_path = os.path.join(_PROJECT_ROOT, "results", "champion_neural_h8.pkl")
    with open(h8_path, "rb") as f:
        h8_data = pickle.load(f)
    print(f"Loaded Frozen H=8 Checkpoint: {h8_path}")

    # Output directories
    out_dir = os.path.join(_PROJECT_ROOT, "results", "phase_3a")
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 2. Section B: Measure the hand-designed teacher on training seeds (0..49)
    teacher_audit_res = audit_hand_designed_teacher(eval_seeds=list(range(0, 50)))

    # 3. Section E: Gate 0 Nominal Evolution
    gate0_res = run_gate0_nominal_evolution(
        warm_start_weights=h8_data,
        opt_seed=42,
        pop_size=20,
        generations=6,
    )
    if not gate0_res["gate_passed"]:
        print("\nCRITICAL: Gate 0 Nominal Validation failed. Halting before robust evolution.")
        sys.exit(1)

    # 4. Section G: Full Robust NSGA-II Evolution across Arm A and Arm B
    optimizer_seeds = [42, 43, 44, 45, 46]
    arm_a_pareto: List[Tuple[MaskedGenome, Dict[str, Any]]] = []
    arm_b_pareto: List[Tuple[MaskedGenome, Dict[str, Any]]] = []

    print("\n--- [Running Arm A: From-Scratch Structural Evolution (5 Seeds)] ---")
    for s in optimizer_seeds:
        front, meta = run_robust_nsga2_arm(
            arm_name="Arm_A_Scratch",
            warm_start_weights=None,
            optimizer_seed=s,
            pop_size=24,
            generations=10,
        )
        arm_a_pareto.append((front[0] if front else None, meta))

    print("\n--- [Running Arm B: Warm-Started Structural Evolution (5 Seeds)] ---")
    for s in optimizer_seeds:
        front, meta = run_robust_nsga2_arm(
            arm_name="Arm_B_WarmStart",
            warm_start_weights=h8_data,
            optimizer_seed=s,
            pop_size=24,
            generations=10,
        )
        arm_b_pareto.append((front[0] if front else None, meta))

    # 5. Section H: Evaluate Baselines and Pareto Candidates on Validation Seeds (250..399)
    val_seeds = PHASE3A_VAL_SEEDS  # Seeds 250..399 (150 seeds)
    baseline_records = evaluate_baselines_on_validation_seeds(h8_data, val_seeds)

    # Evaluate best Pareto candidates on validation seeds
    print("\n--- [Evaluating Top Evolved Pareto Candidates on Phase 3A Validation Seeds] ---")
    evolved_val_records: List[Dict[str, Any]] = []

    # Arm A candidates
    for idx, (best_g, meta) in enumerate(arm_a_pareto):
        if best_g is not None:
            r_succ, w_succ, c_rates = evaluate_candidate_vectorized(best_g, DIAGNOSTIC_CONDITIONS, val_seeds)
            comp = best_g.get_complexity_metrics()
            model_file = os.path.join(models_dir, f"arm_a_scratch_seed{meta['optimizer_seed']}.json")
            with open(model_file, "w") as f:
                json.dump(best_g.to_dict(), f, indent=2)

            rec = {
                "model_name": f"Arm_A_Scratch_Seed{meta['optimizer_seed']}",
                "arm": "Arm_A_Scratch",
                "seed": meta["optimizer_seed"],
                "robust_success": r_succ,
                "worst_success": w_succ,
                "active_parameters": comp["active_parameters"],
                "weight_macs": comp["weight_macs"],
                "num_activations": comp["num_activations"],
                "dense_params": comp["dense_params"],
                "dense_macs": comp["dense_macs"],
                "format": "MaskedGenome JSON",
                "condition_rates": c_rates,
            }
            evolved_val_records.append(rec)
            print(f"  {rec['model_name']:26s}: Robust: {r_succ*100:5.1f}% | Worst: {w_succ*100:5.1f}% | Params: {comp['active_parameters']} | MACs: {comp['weight_macs']}")

    # Arm B candidates
    for idx, (best_g, meta) in enumerate(arm_b_pareto):
        if best_g is not None:
            r_succ, w_succ, c_rates = evaluate_candidate_vectorized(best_g, DIAGNOSTIC_CONDITIONS, val_seeds)
            comp = best_g.get_complexity_metrics()
            model_file = os.path.join(models_dir, f"arm_b_warmstart_seed{meta['optimizer_seed']}.json")
            with open(model_file, "w") as f:
                json.dump(best_g.to_dict(), f, indent=2)

            rec = {
                "model_name": f"Arm_B_WarmStart_Seed{meta['optimizer_seed']}",
                "arm": "Arm_B_WarmStart",
                "seed": meta["optimizer_seed"],
                "robust_success": r_succ,
                "worst_success": w_succ,
                "active_parameters": comp["active_parameters"],
                "weight_macs": comp["weight_macs"],
                "num_activations": comp["num_activations"],
                "dense_params": comp["dense_params"],
                "dense_macs": comp["dense_macs"],
                "format": "MaskedGenome JSON",
                "condition_rates": c_rates,
            }
            evolved_val_records.append(rec)
            print(f"  {rec['model_name']:26s}: Robust: {r_succ*100:5.1f}% | Worst: {w_succ*100:5.1f}% | Params: {comp['active_parameters']} | MACs: {comp['weight_macs']}")

    # 6. Save Aggregated Summaries and JSON Ledger
    summary_path = os.path.join(out_dir, "phase_3a_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "teacher_audit": teacher_audit_res,
            "gate0_result": {
                "passed": gate0_res["gate_passed"],
                "val_success": gate0_res["best_candidate_val_succ"],
                "threshold": gate0_res["threshold"],
            },
            "success_floor": SUCCESS_FLOOR,
            "baselines": baseline_records,
            "evolved_candidates": evolved_val_records,
        }, f, indent=2)
    print(f"\nSaved Phase 3A Validation Summary: {summary_path}")

    # 7. Write Phase 3A Baseline Comparison CSV
    csv_path = os.path.join(out_dir, "phase_3a_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "robust_success_pct", "worst_condition_pct",
            "active_parameters", "weight_macs", "num_activations",
            "dense_params", "dense_macs", "format"
        ])
        for b in baseline_records:
            writer.writerow([
                b["baseline_name"], round(b["robust_success"] * 100, 2), round(b["worst_success"] * 100, 2),
                b["active_parameters"], b["weight_macs"], b["num_activations"],
                b["dense_params"], b["dense_macs"], b["format"]
            ])
        for e in evolved_val_records:
            writer.writerow([
                e["model_name"], round(e["robust_success"] * 100, 2), round(e["worst_success"] * 100, 2),
                e["active_parameters"], e["weight_macs"], e["num_activations"],
                e["dense_params"], e["dense_macs"], e["format"]
            ])
    print(f"Saved Phase 3A Comparison CSV: {csv_path}")

    print("\n=== Phase 3A Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
