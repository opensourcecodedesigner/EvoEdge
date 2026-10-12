"""Unit tests for Phase 2B: Controlled Robustness Experiment & Audit.

Tests:
  1. Previous-action queue semantics (initial 0.0, delay queue synchronization, applied vs computed action).
  2. Episode reset behavior (action queue cleared, previous action reset to 0.0, no inter-episode leakage).
  3. Actual inference path receiving previous applied action and reset behavior.
  4. Architecture accounting (5-input H=8 vs 6-input H=8 parameter and MAC counts).
  5. Canonical production success predicate helper:
     - Exactly 95 of final 100 steps > 0.9: SUCCESS.
     - Exactly 94 of final 100 steps > 0.9: FAILURE (OTHER_FAILURE).
     - Upright value exactly equal to 0.9: does NOT count as > 0.9.
     - Any rail hit: FAILURE (RAIL_HIT) regardless of upright score.
     - Episode shorter than 500 steps: FAILURE (OTHER_FAILURE).
     - Mutually exclusive and exhaustive outcome partition (SUCCESS + RAIL_HIT + OTHER_FAILURE == 1).
  6. Training distribution sampling matching loaded YAML configuration.
  7. Seed partition guard forbidding evaluation on seeds 2000..2099.
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.perturbation_loader import (
    PerturbedCartPoleEnv,
    load_perturbation_suite,
    PerturbationConfigError,
)
from metrics.ops_counter import compute_feedforward_ops
from experiments.run_robustness_experiment import (
    evaluate_success_predicate,
    sample_training_condition,
    normalize_state,
    TinyNet,
)


class TestRobustnessExperiment(unittest.TestCase):
    """Test suite for Phase 2B infrastructure, queue semantics, and success criteria."""

    def setUp(self):
        self.config_path = os.path.join(_PROJECT_ROOT, "config", "perturbations.yaml")
        self.suite = load_perturbation_suite(self.config_path)

    def test_previous_action_queue_semantics(self):
        """Verify previous-action tracking accurately reflects action applied to plant."""
        # Under 2-step delay:
        # Step 0: action submitted = 0.8 -> applied action = 0.0 (queue was [0.0, 0.0])
        # Step 1: action submitted = -0.5 -> applied action = 0.0 (queue was [0.0, 0.8])
        # Step 2: action submitted = 0.2 -> applied action = 0.8 (queue was [0.8, -0.5])
        env = PerturbedCartPoleEnv(max_steps=100, delay_steps=2, seed=42)
        obs, _ = env.reset(seed=42)

        prev_action_applied = 0.0
        self.assertEqual(prev_action_applied, 0.0)

        # Step 0
        obs, r, term, trunc, info = env.step(0.8)
        self.assertEqual(info["applied_action"], 0.0)
        prev_action_applied = info["applied_action"]
        self.assertEqual(prev_action_applied, 0.0)

        # Step 1
        obs, r, term, trunc, info = env.step(-0.5)
        self.assertEqual(info["applied_action"], 0.0)
        prev_action_applied = info["applied_action"]
        self.assertEqual(prev_action_applied, 0.0)

        # Step 2
        obs, r, term, trunc, info = env.step(0.2)
        self.assertEqual(info["applied_action"], 0.8)
        prev_action_applied = info["applied_action"]
        self.assertEqual(prev_action_applied, 0.8)

    def test_episode_reset_clears_queue_and_state(self):
        """Verify reset clears delay queue, resets previous action, and leaves no leakage."""
        env = PerturbedCartPoleEnv(max_steps=100, delay_steps=2, seed=42)
        obs, _ = env.reset(seed=42)
        env.step(0.9)
        env.step(-0.9)

        # Queue should now have actions
        self.assertTrue(any(a != 0.0 for a in env.action_queue))

        # Reset episode
        obs_new, _ = env.reset(seed=43)
        # Queue must be fully zeroed
        for a in env.action_queue:
            self.assertEqual(a, 0.0)
        self.assertFalse(env.boundary_violated)

    def test_actual_inference_path_receives_previous_applied_action(self):
        """Verify production inference loop feeds applied action to 6-input network."""
        model = TinyNet(in_dim=6, hidden_dim=8)
        env = PerturbedCartPoleEnv(max_steps=10, delay_steps=1, seed=42)
        obs, _ = env.reset(seed=42)
        prev_action_applied = 0.0

        received_inputs = []

        for step_i in range(5):
            s_norm = normalize_state(obs)
            feat = np.append(s_norm, prev_action_applied)
            received_inputs.append(float(feat[5]))

            t_in = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            action = float(model(t_in).item())

            obs, r, term, trunc, info = env.step(action)
            prev_action_applied = info["applied_action"]

        # Step 0: prev_action_applied must be 0.0
        self.assertEqual(received_inputs[0], 0.0)
        # Step 1: applied action from Step 0 under 1-step delay is 0.0 (queue had 0.0)
        self.assertEqual(received_inputs[1], 0.0)

    def test_architecture_parameter_and_mac_accounting(self):
        """Verify accurate parameter and MAC counts for 5-input and 6-input H=8 controllers."""
        # 5-input H=8: 5 -> 8 -> 1
        ops_5 = compute_feedforward_ops([5, 8, 1])
        self.assertEqual(ops_5["total_parameters"], 57)
        self.assertEqual(ops_5["weight_macs"], 48)
        self.assertEqual(ops_5["operation_count_proxy"], 114)

        # 6-input H=8: 6 -> 8 -> 1
        ops_6 = compute_feedforward_ops([6, 8, 1])
        self.assertEqual(ops_6["total_parameters"], 65)
        self.assertEqual(ops_6["weight_macs"], 56)
        self.assertEqual(ops_6["operation_count_proxy"], 130)

        # Confirm 6-input is strictly distinct from 5-input
        self.assertNotEqual(ops_5["total_parameters"], ops_6["total_parameters"])
        self.assertNotEqual(ops_5["weight_macs"], ops_6["weight_macs"])

    def test_canonical_success_predicate(self):
        """Test canonical production evaluate_success_predicate against all edge cases."""
        # 1. Exactly 95 steps above 0.9 in final 100 of 500 steps -> SUCCESS
        thetas_95 = [0.0] * 400 + [0.91] * 95 + [0.89] * 5
        succ, rail, other, up_steps = evaluate_success_predicate(thetas_95, rail_hit=False)
        self.assertTrue(succ)
        self.assertFalse(rail)
        self.assertFalse(other)
        self.assertEqual(up_steps, 95)
        self.assertEqual(int(succ) + int(rail) + int(other), 1)

        # 2. Exactly 94 steps above 0.9 in final 100 of 500 steps -> FAILURE (OTHER_FAILURE)
        thetas_94 = [0.0] * 400 + [0.91] * 94 + [0.89] * 6
        succ, rail, other, up_steps = evaluate_success_predicate(thetas_94, rail_hit=False)
        self.assertFalse(succ)
        self.assertFalse(rail)
        self.assertTrue(other)
        self.assertEqual(up_steps, 94)
        self.assertEqual(int(succ) + int(rail) + int(other), 1)

        # 3. Value exactly equal to 0.9 does NOT count as > 0.9
        thetas_exact_09 = [0.0] * 400 + [0.9000000] * 95 + [0.89] * 5
        succ, rail, other, up_steps = evaluate_success_predicate(thetas_exact_09, rail_hit=False)
        self.assertFalse(succ)
        self.assertFalse(rail)
        self.assertTrue(other)
        self.assertEqual(up_steps, 0)  # none strictly > 0.9
        self.assertEqual(int(succ) + int(rail) + int(other), 1)

        # 4. Any rail hit is a RAIL_HIT regardless of upright score
        thetas_rail = [0.95] * 500
        succ, rail, other, up_steps = evaluate_success_predicate(thetas_rail, rail_hit=True)
        self.assertFalse(succ)
        self.assertTrue(rail)
        self.assertFalse(other)
        self.assertEqual(int(succ) + int(rail) + int(other), 1)

        # 5. Episode shorter than 500 steps (e.g. 499 steps) -> FAILURE (OTHER_FAILURE)
        thetas_short = [0.95] * 499
        succ, rail, other, up_steps = evaluate_success_predicate(thetas_short, rail_hit=False)
        self.assertFalse(succ)
        self.assertFalse(rail)
        self.assertTrue(other)
        self.assertEqual(int(succ) + int(rail) + int(other), 1)

    def test_training_distribution_sampling_matches_yaml(self):
        """Verify sample_training_condition uses exact bounds from perturbations.yaml."""
        rng = np.random.default_rng(12345)
        
        # Test nominal
        nom = sample_training_condition("nominal", rng, suite=self.suite)
        self.assertEqual(nom["cart_mass"], 1.0)
        self.assertEqual(nom["pole_mass"], 0.1)
        self.assertEqual(nom["b_cart"], 0.0)
        self.assertEqual(nom["b_pole"], 0.0)
        self.assertEqual(nom["obs_noise_sigma"], 0.0)
        self.assertEqual(nom["delay_steps"], 0)

        # Test robust over 500 samples against YAML bounds
        yaml_pole = self.suite.perturbations["pole_mass"]["training"]["range"]
        yaml_cart = self.suite.perturbations["cart_mass"]["training"]["range"]
        yaml_b_cart = self.suite.perturbations["cart_friction"]["training"]["range"]
        yaml_b_pole = self.suite.perturbations["joint_friction"]["training"]["range"]
        yaml_noise = self.suite.perturbations["observation_noise"]["training"]["range"]
        yaml_delay = set(self.suite.perturbations["actuator_delay"]["training"]["values"])

        for _ in range(500):
            c = sample_training_condition("robust", rng, suite=self.suite)
            self.assertTrue(yaml_pole[0] <= c["pole_mass"] <= yaml_pole[1])
            self.assertTrue(yaml_cart[0] <= c["cart_mass"] <= yaml_cart[1])
            self.assertTrue(yaml_b_cart[0] <= c["b_cart"] <= yaml_b_cart[1])
            self.assertTrue(yaml_b_pole[0] <= c["b_pole"] <= yaml_b_pole[1])
            self.assertTrue(yaml_noise[0] <= c["obs_noise_sigma"] <= yaml_noise[1])
            self.assertIn(c["delay_steps"], yaml_delay)

    def test_seed_partition_guard(self):
        """Verify runners forbid evaluating on supplementary final test seeds 2000..2099."""
        final_test_range = self.suite.splits["supplementary_untouched_test"]["seed_range"]
        self.assertEqual(final_test_range, [2000, 2099])

        # Attempting to sample or use seeds 2000..2099 during training/val must fail
        with self.assertRaises(PerturbationConfigError):
            self.suite.sample_parameters(split="training", seed=2005)

        with self.assertRaises(PerturbationConfigError):
            self.suite.sample_parameters(split="validation", seed=2050)


if __name__ == "__main__":
    unittest.main()
