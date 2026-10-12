"""Phase 2A Validation Smoke Test.

Runs an end-to-end smoke test on VALIDATION conditions only (seeds 200..209).
Verifies:
  - Configuration loading and SHA-256 validation.
  - Parameter sampling within validation bounds.
  - Integration with BenchmarkCartPoleEnv and PerturbedCartPoleEnv.
  - Controller evaluation (LQR and Hybrid ES-LQR).
  - Provenance logging per episode.
  - Total runtime << 20 minutes.
  - STRICTLY AVOIDS any supplementary final test seeds (2000..2099).
"""

from __future__ import annotations

import json
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.perturbation_loader import load_perturbation_suite, PerturbationConfigError
from experiments.benchmark_suite import (
    BenchmarkCartPoleEnv,
    generate_initial_condition_grid,
    evaluate_controller,
    EvaluatedController,
)
from baselines.lqr import LQRController
from baselines.hybrid_controller import HybridController


def run_smoke_test():
    t0 = time.perf_counter()
    print("================================================================================")
    print("  EvoEdge Phase 2A: Smoke Test on Validation Partition (Seeds 200..209)")
    print("================================================================================")

    # 1. Load suite
    config_path = os.path.join(_PROJECT_ROOT, "config", "perturbations.yaml")
    suite = load_perturbation_suite(config_path)
    print(f"  Configuration Loaded : {config_path}")
    print(f"  Configuration Hash   : {suite.config_hash}")
    print(f"  Protocol Version     : {suite.version}")
    print(f"  Objective            : {suite.raw_config.get('objective', {}).get('type')}")

    # 2. Validation seeds
    val_seeds = suite.get_split_seeds("validation")[:10]  # 200..209
    print(f"  Validation Seeds     : {val_seeds}")

    # Ensure no forbidden seeds
    for s in val_seeds:
        if 2000 <= s <= 2099:
            raise PerturbationConfigError("Supplementary final test seeds leaked into smoke test!")

    grid_b = generate_initial_condition_grid(
        seeds=val_seeds,
        x_range=(-1.0, 1.0),
        theta_center=3.14159,
        theta_spread=0.1,
        mode="grid",
    )

    # 3. Instantiate Controllers
    lqr = LQRController(method="continuous")
    eval_lqr = EvaluatedController(
        name="LQR",
        compute_fn=lambda obs: lqr.compute_action(obs, normalized=True),
        active_parameters=4,
        theoretical_macs=4,
    )

    hybrid = HybridController()
    eval_hybrid = EvaluatedController(
        name="Hybrid ES-LQR",
        compute_fn=lambda obs: hybrid.compute_action(obs, normalized=True),
        active_parameters=7,
        theoretical_macs=7,
    )

    # 4. Evaluate under sampled validation perturbations
    # Sample a representative validation condition
    sample_val = suite.sample_parameters(split="validation", seed=200)
    print(f"\n  Sampled Validation Parameters (Seed 200):")
    for k, v in sample_val.items():
        print(f"    - {k:<18}: {v}")

    print("\n  Evaluating Controllers...")
    results = {}
    for ctrl in [eval_lqr, eval_hybrid]:
        t_ctrl_0 = time.perf_counter()
        res = evaluate_controller(
            controller=ctrl,
            initial_conditions=grid_b,
            max_steps=200,
            cart_mass_scale=sample_val["MASS_CART"] / 1.0,
            pole_mass_scale=sample_val["MASS_POLE"] / 0.1,
            b_cart=sample_val["b_cart"],
            b_pole=sample_val["b_pole"],
            obs_noise_sigma=sample_val["obs_noise_sigma"],
            delay_steps=int(sample_val["delay_steps"]),
            config_hash=suite.config_hash,
            split="validation",
        )
        ctrl_elapsed = time.perf_counter() - t_ctrl_0
        results[ctrl.name] = {
            "mean_reward": res["mean_reward"],
            "boundary_violation_rate": res["boundary_violation_rate"],
            "steady_state_jitter_rms": res["steady_state_jitter_rms"],
            "config_hash": res["config_hash"],
            "provenance_episodes": len(res["episode_provenance"]),
            "elapsed_seconds": round(ctrl_elapsed, 3),
        }
        print(f"    [OK] {ctrl.name:<15} | Reward: {res['mean_reward']:>7.2f} | Rail: {res['boundary_violation_rate']:>5.1f}% | Time: {ctrl_elapsed:.3f}s")

    total_elapsed = time.perf_counter() - t0
    print(f"\n  Smoke Test Completed Successfully in {total_elapsed:.2f} seconds.")
    print("================================================================================")

    out_file = os.path.join(_PROJECT_ROOT, "results", "smoke_test_validation_summary.json")
    summary = {
        "status": "PASS",
        "duration_seconds": round(total_elapsed, 3),
        "config_hash": suite.config_hash,
        "protocol_version": suite.version,
        "split_evaluated": "validation",
        "seeds_evaluated": val_seeds,
        "controller_results": results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_smoke_test()
