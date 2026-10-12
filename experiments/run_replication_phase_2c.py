"""Phase 2C: Controlled Replication and Robustness Confirmation Suite.

Replicates the 2x2 experimental matrix under an explicitly frozen, configuration-driven
training protocol:
  - Arm A: 5-input H=8, nominal training
  - Arm B: 5-input H=8, robust training (domain randomization)
  - Arm C: 6-input H=8 with previous applied action, nominal training
  - Arm D: 6-input H=8 with previous applied action, robust training (domain randomization)
  - Frozen Baseline: Historical H=8 champion reference.

Protocol Controls:
  - 5 independent optimizer seeds per arm (seeds 42, 43, 44, 45, 46).
  - Training bounds dynamically queried from config/perturbations.yaml (SHA-256: 984053987fa92679).
  - Training seeds strictly within partition 0..199.
  - Validation seeds strictly within partition 200..249 (50 episodes/condition).
  - Final test seeds 2000..2099 strictly untouched and unexecuted.
  - Results saved under results/replication_phase_2c/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.perturbation_loader import (
    PerturbedCartPoleEnv,
    load_perturbation_suite,
    PerturbationSuite,
    PerturbationConfigError,
)
from baselines.hybrid_controller import EnergyShapingLinearFeedbackController
from metrics.ops_counter import compute_feedforward_ops
from experiments.run_robustness_experiment import (
    TinyNet,
    normalize_state,
    compute_wilson_ci,
    evaluate_success_predicate,
    sample_training_condition,
)


def train_arm_phase_2c(
    arm_name: str,
    in_dim: int,
    training_mode: str,
    optimizer_seed: int,
    suite: PerturbationSuite,
    epochs_per_iter: int = 15,
    num_dagger_iters: int = 6,
    batch_size: int = 256,
    lr: float = 0.01,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train one model arm via DAgger using configuration-driven bounds."""
    t0 = time.perf_counter()

    # Reproducibility seeds
    torch.manual_seed(optimizer_seed)
    np_rng = np.random.default_rng(optimizer_seed)

    teacher = EnergyShapingLinearFeedbackController(k_E=0.6, k_x=0.5, k_v=0.25)
    model = TinyNet(in_dim=in_dim, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_seeds = suite.get_split_seeds("training")
    all_X: List[np.ndarray] = []
    all_Y: List[float] = []
    sampled_training_params: List[Dict[str, Any]] = []

    # 1. Initial demonstration dataset (20 episodes from seeds 0..19)
    init_seeds = train_seeds[:20]
    for s in init_seeds:
        cond = sample_training_condition(training_mode, np_rng, suite=suite)
        sampled_training_params.append({"iter": -1, "seed": s, "params": cond})
        env = PerturbedCartPoleEnv(max_steps=500, continuous=True, seed=s, **cond)
        obs, _ = env.reset(seed=s)
        prev_action_applied = 0.0

        for _ in range(500):
            teacher_a = teacher.compute_action(obs)
            s_norm = normalize_state(obs)

            if in_dim == 6:
                feat = np.append(s_norm, prev_action_applied)
            else:
                feat = s_norm

            all_X.append(feat)
            all_Y.append(teacher_a)

            obs, r, term, trunc, info = env.step(teacher_a)
            prev_action_applied = info["applied_action"]
            if term:
                break

    # 2. DAgger Iterations (seeds 20..139)
    for dagger_iter in range(num_dagger_iters):
        X_t = torch.tensor(np.array(all_X), dtype=torch.float32)
        Y_t = torch.tensor(np.array(all_Y).reshape(-1, 1), dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs_per_iter):
            for bx, by in loader:
                pred = model(bx)
                loss = nn.functional.mse_loss(pred, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        start_idx = 20 + dagger_iter * 20
        batch_seeds = train_seeds[start_idx : start_idx + 20]

        for s in batch_seeds:
            cond = sample_training_condition(training_mode, np_rng, suite=suite)
            sampled_training_params.append({"iter": dagger_iter, "seed": s, "params": cond})
            env = PerturbedCartPoleEnv(max_steps=500, continuous=True, seed=s, **cond)
            obs, _ = env.reset(seed=s)
            prev_action_applied = 0.0

            for _ in range(500):
                teacher_a = teacher.compute_action(obs)
                s_norm = normalize_state(obs)

                if in_dim == 6:
                    feat = np.append(s_norm, prev_action_applied)
                else:
                    feat = s_norm

                all_X.append(feat)
                all_Y.append(teacher_a)

                with torch.no_grad():
                    tin = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                    student_a = float(model(tin).item())

                obs, r, term, trunc, info = env.step(student_a)
                prev_action_applied = info["applied_action"]
                if term:
                    break

    training_duration = time.perf_counter() - t0

    model_bytes = bytearray()
    for p in model.parameters():
        model_bytes.extend(p.detach().cpu().numpy().tobytes())
    model_hash = hashlib.sha256(model_bytes).hexdigest()[:16]

    ops_info = compute_feedforward_ops([in_dim, 8, 1])

    if verbose:
        print(f"  [{arm_name} | Seed {optimizer_seed}] Trained in {training_duration:.2f}s | "
              f"Data Points: {len(all_X)} | Model Hash: {model_hash} | Params: {ops_info['total_parameters']}")

    return {
        "arm_name": arm_name,
        "in_dim": in_dim,
        "training_mode": training_mode,
        "optimizer_seed": optimizer_seed,
        "model": model,
        "model_hash": model_hash,
        "config_hash": suite.config_hash,
        "training_duration_s": round(training_duration, 3),
        "total_training_samples": len(all_X),
        "parameters": ops_info["total_parameters"],
        "weight_macs": ops_info["weight_macs"],
        "ops_proxy": ops_info["operation_count_proxy"],
        "sampled_training_params": sampled_training_params,
    }


def evaluate_controller_on_condition(
    controller_name: str,
    act_fn: Callable[[np.ndarray, float], float],
    is_6dim: bool,
    condition_name: str,
    env_params: Dict[str, Any],
    val_seeds: List[int],
    max_steps: int = 500,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate a controller on a diagnostic validation condition across seeds."""
    episode_records: List[Dict[str, Any]] = []
    success_count = 0
    rail_count = 0
    other_fail_count = 0
    steps_to_upright_list: List[int] = []

    for s in val_seeds:
        env = PerturbedCartPoleEnv(max_steps=max_steps, continuous=True, seed=s, **env_params)
        obs, _ = env.reset(seed=s)
        prev_action_applied = 0.0

        cos_thetas: List[float] = []
        applied_actions: List[float] = []
        first_upright: Optional[int] = None
        rail_hit = False
        failure_step = max_steps

        for step_i in range(max_steps):
            s_norm = normalize_state(obs)
            action = act_fn(s_norm, prev_action_applied)

            obs, r, term, trunc, info = env.step(action)
            act_app = info["applied_action"]
            applied_actions.append(act_app)
            prev_action_applied = act_app

            cos_th = float(obs[2])
            cos_thetas.append(cos_th)

            if first_upright is None and cos_th > 0.9:
                first_upright = step_i + 1

            if term or abs(env._state[0]) >= env.X_THRESHOLD:
                rail_hit = True
                failure_step = step_i + 1
                break

        is_success, is_rail_hit, is_other_fail, upright_steps_in_window = evaluate_success_predicate(
            cos_thetas=cos_thetas,
            rail_hit=rail_hit,
            max_steps=max_steps,
            required_upright_in_window=95,
            upright_threshold=0.9,
            window_size=100,
        )

        if is_success:
            success_count += 1
            if first_upright is not None:
                steps_to_upright_list.append(first_upright)
        elif is_rail_hit:
            rail_count += 1
        else:
            other_fail_count += 1

        act_arr = np.array(applied_actions)
        saturation_frac = float(np.mean(np.abs(act_arr) >= 0.98)) if len(act_arr) > 0 else 0.0
        sign_flips = int(np.sum(np.diff(np.sign(act_arr)) != 0)) if len(act_arr) > 1 else 0

        episode_records.append({
            "controller": controller_name,
            "condition": condition_name,
            "seed": s,
            "success": int(is_success),
            "rail_hit": int(rail_hit),
            "other_failure": int(not is_success and not rail_hit),
            "steps_to_upright": first_upright if first_upright is not None else -1,
            "final_100_upright_steps": upright_steps_in_window,
            "failure_step": failure_step,
            "saturation_fraction": round(saturation_frac, 4),
            "switching_flips": sign_flips,
            "sampled_params": json.dumps(dict(env._last_sampled_params)),
        })

    n_tot = len(val_seeds)
    rate, ci_low, ci_high = compute_wilson_ci(success_count, n_tot)
    median_steps = float(np.median(steps_to_upright_list)) if steps_to_upright_list else float("nan")

    summary = {
        "controller": controller_name,
        "condition": condition_name,
        "episodes": n_tot,
        "success_count": success_count,
        "success_rate": round(rate * 100.0, 1),
        "ci_lower_pct": round(ci_low * 100.0, 1),
        "ci_upper_pct": round(ci_high * 100.0, 1),
        "rail_count": rail_count,
        "other_failure_count": other_fail_count,
        "median_steps_to_upright": round(median_steps, 1) if not math.isnan(median_steps) else -1,
    }
    return episode_records, summary


def run_phase_2c_experiment(
    smoke: bool = False,
    results_dir: str = "results/replication_phase_2c",
    perturbation_config_path: str = "config/perturbations.yaml",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute complete Phase 2C replication experiment."""
    suite = load_perturbation_suite(perturbation_config_path)
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Experiment Setup: 5 seeds per arm for replication
    opt_seeds = [42] if smoke else [42, 43, 44, 45, 46]
    val_seed_count = 5 if smoke else 50
    val_seeds = suite.get_split_seeds("validation")[:val_seed_count]
    epochs_per_iter = 5 if smoke else 15
    num_dagger_iters = 2 if smoke else 6

    print("\n" + "=" * 80)
    print(f"  EvoEdge Phase 2C: Controlled Replication Experiment {'(SMOKE TEST)' if smoke else ''}")
    print("=" * 80)
    print(f"  Configuration SHA-256 : {suite.config_hash}")
    print(f"  Protocol Version      : {suite.version}")
    print(f"  Optimizer Seeds       : {opt_seeds} ({len(opt_seeds)} seeds/arm)")
    print(f"  Validation Seeds      : {val_seeds[0]}..{val_seeds[-1]} ({len(val_seeds)} episodes/cond)")
    print(f"  Output Directory      : {results_dir}")
    print("=" * 80 + "\n")

    arms_spec = [
        {"name": "Arm_A", "label": "Arm A (5-In, Nominal)", "in_dim": 5, "mode": "nominal"},
        {"name": "Arm_B", "label": "Arm B (5-In, Robust)",  "in_dim": 5, "mode": "robust"},
        {"name": "Arm_C", "label": "Arm C (6-In, Nominal)", "in_dim": 6, "mode": "nominal"},
        {"name": "Arm_D", "label": "Arm D (6-In, Robust)",  "in_dim": 6, "mode": "robust"},
    ]

    trained_models: List[Dict[str, Any]] = []
    training_metadata: List[Dict[str, Any]] = []

    print("--- [1. Training 2x2 Experimental Arms] ---")
    for arm in arms_spec:
        for seed in opt_seeds:
            run_data = train_arm_phase_2c(
                arm_name=arm["name"],
                in_dim=arm["in_dim"],
                training_mode=arm["mode"],
                optimizer_seed=seed,
                suite=suite,
                epochs_per_iter=epochs_per_iter,
                num_dagger_iters=num_dagger_iters,
                verbose=verbose,
            )
            run_data["label"] = arm["label"]

            pt_path = os.path.join(models_dir, f"{arm['name'].lower()}_seed{seed}.pt")
            torch.save(run_data["model"].state_dict(), pt_path)
            run_data["pt_path"] = pt_path
            trained_models.append(run_data)

            training_metadata.append({
                "arm": arm["name"],
                "in_dim": arm["in_dim"],
                "mode": arm["mode"],
                "optimizer_seed": seed,
                "model_hash": run_data["model_hash"],
                "config_hash": run_data["config_hash"],
                "training_duration_s": run_data["training_duration_s"],
                "samples": run_data["total_training_samples"],
                "pt_path": pt_path,
            })

    # Save training metadata JSON
    meta_path = os.path.join(results_dir, "training_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2)

    # 4. Load Frozen Baseline Reference
    frozen_baseline_path = os.path.join("results", "champion_neural_h8.pkl")
    with open(frozen_baseline_path, "rb") as f:
        h8_data = pickle.load(f)
    w1_8, b1_8 = h8_data["w1"], h8_data["b1"]
    w2_8, b2_8 = h8_data["w2"], h8_data["b2"]

    def act_frozen_baseline(s_norm: np.ndarray, prev_a: float) -> float:
        h = np.tanh(2.5 * (w1_8 @ s_norm + b1_8))
        out = np.tanh(2.5 * (w2_8 @ h + b2_8))
        return float(out[0])

    # 5. Diagnostic Conditions
    validation_conditions = {
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

    print("\n--- [2. Evaluating Models Across 10 Validation Conditions] ---")
    all_episode_records: List[Dict[str, Any]] = []
    aggregated_summaries: List[Dict[str, Any]] = []

    # Evaluate Frozen Baseline
    for cond_name, cond_params in validation_conditions.items():
        ep_recs, summary = evaluate_controller_on_condition(
            controller_name="Frozen_Baseline_H8",
            act_fn=act_frozen_baseline,
            is_6dim=False,
            condition_name=cond_name,
            env_params=cond_params,
            val_seeds=val_seeds,
        )
        all_episode_records.extend(ep_recs)
        summary["in_dim"] = 5
        summary["training_mode"] = "nominal (frozen)"
        summary["optimizer_seed"] = 42
        aggregated_summaries.append(summary)

    # Evaluate all trained models
    for m_data in trained_models:
        model = m_data["model"]
        in_dim = m_data["in_dim"]
        c_name = f"{m_data['arm_name']}_seed{m_data['optimizer_seed']}"

        def make_act_fn(m=model, d=in_dim):
            def _fn(s_norm: np.ndarray, prev_a: float) -> float:
                if d == 6:
                    inp = np.append(s_norm, prev_a)
                else:
                    inp = s_norm
                with torch.no_grad():
                    tin = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
                    return float(m(tin).item())
            return _fn

        act_fn = make_act_fn()

        for cond_name, cond_params in validation_conditions.items():
            ep_recs, summary = evaluate_controller_on_condition(
                controller_name=c_name,
                act_fn=act_fn,
                is_6dim=(in_dim == 6),
                condition_name=cond_name,
                env_params=cond_params,
                val_seeds=val_seeds,
            )
            all_episode_records.extend(ep_recs)
            summary["in_dim"] = in_dim
            summary["training_mode"] = m_data["training_mode"]
            summary["optimizer_seed"] = m_data["optimizer_seed"]
            aggregated_summaries.append(summary)

    # Write CSVs
    ep_csv_path = os.path.join(results_dir, "phase_2c_episode_validation.csv")
    if all_episode_records:
        ep_fields = list(all_episode_records[0].keys())
        with open(ep_csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ep_fields)
            writer.writeheader()
            writer.writerows(all_episode_records)

    agg_csv_path = os.path.join(results_dir, "phase_2c_aggregated_summary.csv")
    if aggregated_summaries:
        agg_fields = list(aggregated_summaries[0].keys())
        with open(agg_csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=agg_fields)
            writer.writeheader()
            writer.writerows(aggregated_summaries)

    # Compute run-level summary table (5 runs per arm)
    run_level_rows = []
    # Group by arm and condition
    for arm in ["Arm_A", "Arm_B", "Arm_C", "Arm_D"]:
        for cond in validation_conditions.keys():
            arm_cond_runs = [
                s for s in aggregated_summaries
                if s["controller"].startswith(arm) and s["condition"] == cond
            ]
            rates = [r["success_rate"] for r in arm_cond_runs]
            succs = [r["success_count"] for r in arm_cond_runs]
            rails = [r["rail_count"] for r in arm_cond_runs]
            others = [r["other_failure_count"] for r in arm_cond_runs]
            n_runs = len(arm_cond_runs)
            tot_episodes = sum(r["episodes"] for r in arm_cond_runs)
            tot_succ = sum(succs)
            
            mean_r = float(np.mean(rates)) if rates else 0.0
            std_r = float(np.std(rates, ddof=1)) if n_runs > 1 else 0.0
            min_r = float(np.min(rates)) if rates else 0.0
            max_r = float(np.max(rates)) if rates else 0.0
            
            row = {
                "arm": arm,
                "condition": cond,
                "n_runs": n_runs,
                "mean_rate": round(mean_r, 2),
                "std_rate": round(std_r, 2),
                "min_rate": round(min_r, 2),
                "max_rate": round(max_r, 2),
                "total_episodes": tot_episodes,
                "total_success": tot_succ,
                "total_rail": sum(rails),
                "total_other": sum(others),
                "pooled_success_rate": round(tot_succ / tot_episodes * 100.0, 2) if tot_episodes > 0 else 0.0,
            }
            # Record individual seed rates
            for r in arm_cond_runs:
                row[f"seed{r['optimizer_seed']}_rate"] = r["success_rate"]
            run_level_rows.append(row)

    # Add baseline row
    for cond in validation_conditions.keys():
        b_run = next(s for s in aggregated_summaries if s["controller"] == "Frozen_Baseline_H8" and s["condition"] == cond)
        row = {
            "arm": "Frozen_Baseline",
            "condition": cond,
            "n_runs": 1,
            "mean_rate": b_run["success_rate"],
            "std_rate": 0.0,
            "min_rate": b_run["success_rate"],
            "max_rate": b_run["success_rate"],
            "total_episodes": b_run["episodes"],
            "total_success": b_run["success_count"],
            "total_rail": b_run["rail_count"],
            "total_other": b_run["other_failure_count"],
            "pooled_success_rate": b_run["success_rate"],
        }
        for s in opt_seeds:
            row[f"seed{s}_rate"] = b_run["success_rate"]
        run_level_rows.append(row)

    run_level_csv_path = os.path.join(results_dir, "phase_2c_run_level_summary.csv")
    if run_level_rows:
        rl_fields = list(run_level_rows[0].keys())
        with open(run_level_csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=rl_fields)
            writer.writeheader()
            writer.writerows(run_level_rows)

    print(f"\n  [OK] Saved Episode Validation CSV -> {ep_csv_path}")
    print(f"  [OK] Saved Aggregated Summary CSV -> {agg_csv_path}")
    print(f"  [OK] Saved Run-Level Summary CSV   -> {run_level_csv_path}\n")

    return {
        "status": "PASS",
        "smoke": smoke,
        "trained_models": trained_models,
        "total_episodes": len(all_episode_records),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2C Replication Experiment")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()
    run_phase_2c_experiment(smoke=args.smoke)
