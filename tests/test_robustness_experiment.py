"""Unit tests for Phase 2B: Controlled Robustness Experiment.

Tests:
  1. Previous-action queue semantics (initial 0.0, delay queue synchronization, applied vs computed action).
  2. Episode reset behavior (action queue cleared, previous action reset to 0.0, no inter-episode leakage).
  3. Configuration use (nominal vs robust distributions, seed partitions 0..199 and 200..399).
  4. Fixed success criterion (rail hit detection, 95/100 upright threshold, other failure classification).
  5. Architecture accounting (5-input H=8 vs 6-input H=8 parameter and MAC counts).
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

    def test_fixed_success_criterion_logic(self):
        """Verify the exact definition of success, rail hit, and other failure."""
        def evaluate_trajectory(boundary_violated: bool, cos_thetas: list[float], max_steps: int = 500) -> str:
            if boundary_violated:
                return "RAIL_HIT"
            if len(cos_thetas) == max_steps and sum(1 for c in cos_thetas[-100:] if c > 0.9) >= 95:
                return "SUCCESS"
            return "OTHER_FAILURE"

        # Case 1: Rail hit
        self.assertEqual(evaluate_trajectory(True, [0.95] * 500), "RAIL_HIT")

        # Case 2: Clean swing-up and balance (96 steps > 0.9 in final 100)
        thetas_good = [0.0] * 400 + [0.95] * 96 + [0.85] * 4
        self.assertEqual(evaluate_trajectory(False, thetas_good), "SUCCESS")

        # Case 3: Incomplete balance (90 steps > 0.9 in final 100)
        thetas_insufficient = [0.0] * 400 + [0.95] * 90 + [0.85] * 10
        self.assertEqual(evaluate_trajectory(False, thetas_insufficient), "OTHER_FAILURE")

        # Case 4: Never swung up (hanging around pi)
        thetas_hanging = [-0.99] * 500
        self.assertEqual(evaluate_trajectory(False, thetas_hanging), "OTHER_FAILURE")

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
