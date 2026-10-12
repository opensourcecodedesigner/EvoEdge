"""Unit tests for unified perturbation configuration loading and validation."""

import os
import tempfile
import unittest
import numpy as np
import yaml

from utils.perturbation_loader import (
    load_perturbation_suite,
    PerturbationSuite,
    PerturbationConfigError,
    get_unmigrated_runners,
)
from env.cartpole_swingup import CartPoleSwingUpEnv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestPerturbationLoader(unittest.TestCase):
    """Verify loading, split partitioning, environment injection, and safe failure."""

    def setUp(self):
        self.config_path = os.path.join(_PROJECT_ROOT, "config", "perturbations.yaml")

    def test_load_valid_suite(self):
        """Standard perturbations.yaml must load successfully and return a valid suite."""
        suite = load_perturbation_suite(self.config_path)
        self.assertIn(suite.version, ["2.3", "2.4"])
        self.assertIn("training", suite.splits)
        self.assertIn("validation", suite.splits)
        self.assertIn("test", suite.splits)
        self.assertIn("supplementary_untouched_test", suite.splits)

    def test_seed_partition_enforcement(self):
        """Sampling with an out-of-split seed must raise PerturbationConfigError."""
        suite = load_perturbation_suite(self.config_path)
        # Training seed range is 0..199; passing 250 must fail
        with self.assertRaises(PerturbationConfigError):
            suite.sample_parameters(split="training", seed=250)

        # Test seed range is 1000..1099; passing 50 must fail
        with self.assertRaises(PerturbationConfigError):
            suite.sample_parameters(split="test", seed=50)

    def test_sample_and_apply_to_env(self):
        """Sampled parameters must correctly modify environment physics attributes."""
        suite = load_perturbation_suite(self.config_path)
        params = suite.sample_parameters(split="validation", seed=205)
        self.assertIn("MASS_CART", params)
        self.assertIn("MASS_POLE", params)
        self.assertIn("b_cart", params)
        self.assertIn("b_pole", params)

        env = CartPoleSwingUpEnv()
        applied = suite.apply_to_env(env, params)

        self.assertEqual(env.MASS_CART, params["MASS_CART"])
        self.assertEqual(env.MASS_POLE, params["MASS_POLE"])
        self.assertEqual(getattr(env, "b_cart", 0.0), params["b_cart"])
        self.assertEqual(getattr(env, "b_pole", 0.0), params["b_pole"])

    def test_invalid_config_fails_safely(self):
        """Malformed or leaky configuration must fail safely with an exception."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            temp_path = f.name
            yaml.dump({"version": "bad", "splits": {}}, f)

        try:
            with self.assertRaises(Exception):
                load_perturbation_suite(temp_path)
        finally:
            os.remove(temp_path)

    def test_runner_migration_registry(self):
        """Runner migration registry must list all runners with known valid status."""
        registry = get_unmigrated_runners()
        self.assertIn("experiments/benchmark_suite.py", registry)
        self.assertIn("experiments/runner.py", registry)
        self.assertIn("experiments/run_baselines_pruning.py", registry)
        self.assertIn("evolution/evolve_tinyml.py", registry)
        for runner, data in registry.items():
            self.assertIn(data["status"], ["MIGRATED", "UNMIGRATED"])


if __name__ == "__main__":
    unittest.main()
