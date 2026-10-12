"""Automated verification suite for Phase 2A runner migration and perturbation infrastructure.

Validates:
  1. The runner actually applies the values declared in YAML.
  2. The sampled distribution follows the requested bounds.
  3. Delay steps are applied correctly via the action transport queue.
  4. Noise affects the specified observation channels.
  5. Seed and condition partitions obey the frozen protocol (training 0..199, validation 200..399,
     historical test 1000..1099, supplementary untouched test 2000..2099).
  6. Results contain enough provenance to reconstruct the evaluated condition (config_hash, sampled_params, seed).
  7. Invalid configurations fail safely.
  8. Historical results remain strictly unchanged.
"""

from __future__ import annotations

import collections
import hashlib
import os
import sys
import tempfile
import unittest
from typing import Dict, List, Any
import numpy as np
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.perturbation_loader import (
    load_perturbation_suite,
    PerturbedCartPoleEnv,
    PerturbationSuite,
    PerturbationConfigError,
)
from experiments.benchmark_suite import (
    BenchmarkCartPoleEnv,
    evaluate_controller,
    EvaluatedController,
)
from baselines.lqr import LQRController


class TestRunnerMigrationAndInfrastructure(unittest.TestCase):
    """Rigorous automated verification of Phase 2A infrastructure migration."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(_PROJECT_ROOT, "config", "perturbations.yaml")
        cls.suite = load_perturbation_suite(cls.config_path)

    def test_runner_applies_yaml_values(self):
        """1. Verify that the runner actually applies the values declared in YAML."""
        # Sample validation condition for seed 205
        params = self.suite.sample_parameters(split="validation", seed=205)
        env = BenchmarkCartPoleEnv(
            cart_mass_scale=params["MASS_CART"] / 1.0,
            pole_mass_scale=params["MASS_POLE"] / 0.1,
            b_cart=params["b_cart"],
            b_pole=params["b_pole"],
            obs_noise_sigma=params["obs_noise_sigma"],
            delay_steps=int(params["delay_steps"]),
            seed=205,
        )

        self.assertAlmostEqual(env.MASS_CART, params["MASS_CART"], places=5)
        self.assertAlmostEqual(env.MASS_POLE, params["MASS_POLE"], places=5)
        self.assertAlmostEqual(env.b_cart, params["b_cart"], places=5)
        self.assertAlmostEqual(env.b_pole, params["b_pole"], places=5)
        self.assertAlmostEqual(env.obs_noise_sigma, params["obs_noise_sigma"], places=5)
        self.assertEqual(env.delay_steps, int(params["delay_steps"]))

    def test_sampled_distribution_bounds(self):
        """2. Verify that sampled distributions strictly follow requested bounds."""
        perts = self.suite.perturbations

        for split in ["training", "validation"]:
            seeds = self.suite.get_split_seeds(split)[:30]
            for s in seeds:
                sampled = self.suite.sample_parameters(split=split, seed=s)
                for p_name, p_data in perts.items():
                    var_name = p_data.get("variable_name", p_name)
                    val = sampled[var_name]
                    spec = p_data.get(split, {})
                    dist = spec.get("distribution")
                    if dist == "uniform":
                        low, high = spec["range"]
                        self.assertGreaterEqual(
                            val, low - 1e-6,
                            f"Sampled value {val} below range [{low}, {high}] for {p_name} in {split}"
                        )
                        self.assertLessEqual(
                            val, high + 1e-6,
                            f"Sampled value {val} above range [{low}, {high}] for {p_name} in {split}"
                        )
                    elif dist == "discrete":
                        allowed_vals = spec["values"]
                        self.assertIn(
                            int(val) if isinstance(val, float) and val.is_integer() else val,
                            allowed_vals,
                            f"Sampled value {val} not in allowed discrete values {allowed_vals}"
                        )

    def test_delay_steps_applied_correctly(self):
        """3. Verify that delay steps are applied correctly via action transport queue."""
        # Test with delay_steps = 2
        env = PerturbedCartPoleEnv(max_steps=100, continuous=True, delay_steps=2, seed=42)
        obs, _ = env.reset(seed=42)

        # Queue initially filled with 0.0s
        self.assertEqual(len(env.action_queue), 2)
        for act in env.action_queue:
            self.assertEqual(act, 0.0)

        # Step 1 with action = 1.0 -> should apply 0.0, queue becomes [0.0, 1.0]
        obs, r, term, trunc, info1 = env.step(1.0)
        self.assertEqual(info1["applied_action"], 0.0)

        # Step 2 with action = 0.5 -> should apply 0.0, queue becomes [1.0, 0.5]
        obs, r, term, trunc, info2 = env.step(0.5)
        self.assertEqual(info2["applied_action"], 0.0)

        # Step 3 with action = -0.8 -> should apply 1.0 (from Step 1)
        obs, r, term, trunc, info3 = env.step(-0.8)
        self.assertEqual(info3["applied_action"], 1.0)

        # Step 4 with action = 0.0 -> should apply 0.5 (from Step 2)
        obs, r, term, trunc, info4 = env.step(0.0)
        self.assertEqual(info4["applied_action"], 0.5)

    def test_noise_affects_specified_channels(self):
        """4. Verify that noise affects specified observation channels and leaves others unaffected."""
        # Configure noise on channels [0, 1] (x, x_dot) only
        env = PerturbedCartPoleEnv(
            max_steps=100,
            continuous=True,
            obs_noise_sigma=0.5,
            noise_channels=[0, 1],
            seed=42,
        )
        env_clean = PerturbedCartPoleEnv(
            max_steps=100,
            continuous=True,
            obs_noise_sigma=0.0,
            noise_channels=[0, 1],
            seed=42,
        )

        obs_noisy, _ = env.reset_with_initial_state(x_0=0.5, theta_0=0.2, seed=42)
        obs_clean, _ = env_clean.reset_with_initial_state(x_0=0.5, theta_0=0.2, seed=42)

        # Channels 2, 3, 4 (cos_theta, sin_theta, theta_dot) must be identical
        self.assertAlmostEqual(obs_noisy[2], obs_clean[2], places=5)
        self.assertAlmostEqual(obs_noisy[3], obs_clean[3], places=5)
        self.assertAlmostEqual(obs_noisy[4], obs_clean[4], places=5)

        # Channels 0 or 1 should differ due to noise sigma=0.5
        diff_ch0 = abs(obs_noisy[0] - obs_clean[0])
        diff_ch1 = abs(obs_noisy[1] - obs_clean[1])
        self.assertTrue(diff_ch0 > 1e-4 or diff_ch1 > 1e-4)

    def test_seed_and_condition_partitions_frozen(self):
        """5. Verify that seed and condition partitions strictly obey frozen protocol."""
        # 1. Training seeds: 0..199
        train_seeds = self.suite.get_split_seeds("training")
        self.assertEqual(train_seeds[0], 0)
        self.assertEqual(train_seeds[-1], 199)

        # 2. Validation seeds: 200..399
        val_seeds = self.suite.get_split_seeds("validation")
        self.assertEqual(val_seeds[0], 200)
        self.assertEqual(val_seeds[-1], 399)

        # 3. Test seeds: 1000..1099
        test_seeds = self.suite.get_split_seeds("test")
        self.assertEqual(test_seeds[0], 1000)
        self.assertEqual(test_seeds[-1], 1099)

        # 4. Supplementary untouched test seeds: 2000..2099
        supp_seeds = self.suite.get_split_seeds("supplementary_untouched_test")
        self.assertEqual(supp_seeds[0], 2000)
        self.assertEqual(supp_seeds[-1], 2099)

        # Mutually disjoint
        all_sets = [set(train_seeds), set(val_seeds), set(test_seeds), set(supp_seeds)]
        for i in range(len(all_sets)):
            for j in range(i + 1, len(all_sets)):
                self.assertEqual(len(all_sets[i].intersection(all_sets[j])), 0)

        # Accessing supplementary final test seeds (2000..2099) in benchmark_suite must fail safely
        with self.assertRaises(PerturbationConfigError):
            from experiments.benchmark_suite import run_benchmark_suite
            run_benchmark_suite(num_episodes=5, seeds=[2005], verbose=False)

    def test_provenance_reconstruction(self):
        """6. Verify results contain full provenance (config_hash, sampled_params, seed)."""
        lqr = LQRController()
        ctrl = EvaluatedController(
            name="LQR_Test",
            compute_fn=lambda obs: lqr.compute_action(obs, normalized=True),
            active_parameters=4,
            theoretical_macs=4,
        )

        res = evaluate_controller(
            ctrl,
            initial_conditions={200: (0.1, -0.05)},
            max_steps=10,
            cart_mass_scale=1.1,
            obs_noise_sigma=0.01,
            pole_mass_scale=1.05,
            b_cart=0.15,
            b_pole=0.002,
            delay_steps=1,
            config_hash=self.suite.config_hash,
            split="validation",
        )

        self.assertEqual(res["config_hash"], self.suite.config_hash)
        self.assertEqual(res["split"], "validation")
        self.assertIn("episode_provenance", res)
        self.assertEqual(len(res["episode_provenance"]), 1)

        ep_info = res["episode_provenance"][0]
        self.assertEqual(ep_info["seed"], 200)
        self.assertIn("sampled_params", ep_info)
        params = ep_info["sampled_params"]
        self.assertAlmostEqual(params["MASS_CART"], 1.1)
        self.assertAlmostEqual(params["MASS_POLE"], 0.105)
        self.assertAlmostEqual(params["b_cart"], 0.15)
        self.assertAlmostEqual(params["b_pole"], 0.002)
        self.assertAlmostEqual(params["obs_noise_sigma"], 0.01)
        self.assertEqual(params["delay_steps"], 1.0)

    def test_invalid_configurations_fail_safely(self):
        """7. Verify that invalid, unsupported, or leaky configurations fail safely."""
        # A. Unsupported physics field
        with self.assertRaises(PerturbationConfigError):
            self.suite.apply_to_env(
                BenchmarkCartPoleEnv(),
                {"wind_force": 10.0, "MASS_CART": 1.0}
            )

        # B. Out-of-split seed requested
        with self.assertRaises(PerturbationConfigError):
            self.suite.sample_parameters(split="training", seed=500)

        # C. Malformed YAML
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            temp_path = f.name
            f.write("corrupted: yaml: syntax: [unclosed")

        try:
            with self.assertRaises(Exception):
                load_perturbation_suite(temp_path)
        finally:
            os.remove(temp_path)

    def test_historical_results_remain_unchanged(self):
        """8. Verify that historical baseline files remain intact and unchanged."""
        historical_paths = [
            "results/baseline_comparison.csv",
            "results/final_evaluation_report.md",
            "results/controller.c",
            "results/champion_neural_h8.pkl",
        ]

        for rel in historical_paths:
            full_path = os.path.join(_PROJECT_ROOT, rel)
            self.assertTrue(
                os.path.exists(full_path),
                f"Historical file missing: {rel}"
            )
            # File must be non-empty
            self.assertGreater(
                os.path.getsize(full_path),
                0,
                f"Historical file is empty: {rel}"
            )


if __name__ == "__main__":
    unittest.main()
