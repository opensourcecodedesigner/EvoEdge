"""Unified Perturbation Configuration Loading and Validation Interface.

This module provides a single, validated entry point for loading perturbation configurations
from `config/perturbations.yaml`, verifying split and condition integrity, and sampling or
applying perturbation parameters to environment instances.

Features:
  1. Strict YAML parsing and schema validation.
  2. Split isolation verification (training, validation, test).
  3. Condition loading: nominal, previously exposed benchmarks, and unexposed final test suite.
  4. Parameter extraction and environment injection with explicit parameter logging per episode.
  5. Audit of unmigrated runners.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

from utils.split_integrity import validate_split_integrity, SplitIntegrityError

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Registry of runner scripts and their migration status
RUNNER_MIGRATION_REGISTRY = {
    "experiments/runner.py": {
        "status": "UNMIGRATED",
        "description": "Legacy multi-seed NEAT runner; does not consume config/perturbations.yaml.",
    },
    "experiments/benchmark_suite.py": {
        "status": "UNMIGRATED",
        "description": "Historical frozen benchmark suite; uses hardcoded parameter dictionaries (cart_mass=1.3, noise=0.05).",
    },
    "experiments/run_baselines_pruning.py": {
        "status": "UNMIGRATED",
        "description": "Pruning baseline evaluator; uses hardcoded parameter loops.",
    },
    "evolution/evolve_tinyml.py": {
        "status": "UNMIGRATED",
        "description": "Fixed-topology DAgger trainer; uses nominal physics defaults during training rollouts.",
    },
}


class PerturbationConfigError(ValueError):
    """Raised when perturbation configuration is invalid or inconsistent."""
    pass


class PerturbationSuite:
    """Loaded and validated perturbation suite."""

    def __init__(self, raw_config: Dict[str, Any], config_path: str = "config/perturbations.yaml") -> None:
        self.raw_config = raw_config
        self.config_path = config_path
        self.version = raw_config.get("version", "unknown")
        self.splits = raw_config.get("splits", {})
        self.perturbations = raw_config.get("perturbations", {})
        self.exposure_catalog = raw_config.get("exposure_catalog", {})

    def get_split_seeds(self, split: str) -> List[int]:
        """Return list of valid integer seeds for the given split partition."""
        if split not in self.splits:
            raise PerturbationConfigError(f"Unknown split '{split}'. Available: {list(self.splits.keys())}")
        start, end = self.splits[split]["seed_range"]
        return list(range(start, end + 1))

    def sample_parameters(
        self,
        split: str,
        seed: Optional[int] = None,
        use_test_benchmark: bool = False,
    ) -> Dict[str, float]:
        """Sample physical perturbation parameters for an episode.

        Returns a dictionary mapping physical variable names to their exact sampled values.
        """
        if split not in self.splits:
            raise PerturbationConfigError(f"Split '{split}' not defined in {self.config_path}")

        # Check seed partition constraint
        if seed is not None:
            allowed_seeds = self.get_split_seeds(split)
            if seed not in allowed_seeds:
                raise PerturbationConfigError(
                    f"Seed {seed} violates partition constraint for split '{split}' "
                    f"[{allowed_seeds[0]}..{allowed_seeds[-1]}]"
                )
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        sampled: Dict[str, float] = {}

        for p_name, p_data in self.perturbations.items():
            spec = p_data.get(split, {})
            dist = spec.get("distribution", "fixed")

            if split == "test" and use_test_benchmark:
                if "benchmark_fixed" in spec:
                    val = float(spec["benchmark_fixed"])
                elif "values" in spec:
                    val = float(spec["values"][0])
                else:
                    val = float(p_data.get("nominal_value", 0.0))
            elif dist == "uniform":
                low, high = spec["range"]
                val = float(rng.uniform(low, high))
            elif dist == "discrete":
                vals = spec["values"]
                val = float(rng.choice(vals))
            elif dist == "fixed":
                val = float(spec.get("benchmark_fixed", p_data.get("nominal_value", 0.0)))
            elif dist == "fixed_and_uniform":
                if use_test_benchmark and "benchmark_fixed" in spec:
                    val = float(spec["benchmark_fixed"])
                elif "range" in spec:
                    low, high = spec["range"]
                    val = float(rng.uniform(low, high))
                else:
                    val = float(p_data.get("nominal_value", 0.0))
            else:
                val = float(p_data.get("nominal_value", 0.0))

            var_name = p_data.get("variable_name", p_name)
            sampled[var_name] = val

        return sampled

    def apply_to_env(
        self,
        env: Any,
        params: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply sampled parameters to an environment instance and return the applied dict."""
        applied: Dict[str, float] = {}

        if "MASS_CART" in params:
            env.MASS_CART = float(params["MASS_CART"])
            applied["MASS_CART"] = env.MASS_CART
        if "MASS_POLE" in params:
            env.MASS_POLE = float(params["MASS_POLE"])
            applied["MASS_POLE"] = env.MASS_POLE
        if hasattr(env, "_total_mass"):
            env._total_mass = env.MASS_CART + env.MASS_POLE
        if hasattr(env, "_polemass_length"):
            env._polemass_length = env.MASS_POLE * getattr(env, "LENGTH", 0.5)

        if "b_cart" in params:
            env.b_cart = float(params["b_cart"])
            applied["b_cart"] = env.b_cart
        if "b_pole" in params:
            env.b_pole = float(params["b_pole"])
            applied["b_pole"] = env.b_pole
        if "obs_noise_sigma" in params:
            env.obs_noise_sigma = float(params["obs_noise_sigma"])
            applied["obs_noise_sigma"] = env.obs_noise_sigma
        if "delay_steps" in params:
            env.delay_steps = int(params["delay_steps"])
            applied["delay_steps"] = float(env.delay_steps)

        return applied


def load_perturbation_suite(config_path: str = "config/perturbations.yaml") -> PerturbationSuite:
    """Validate and load the perturbation configuration suite."""
    abs_path = os.path.abspath(config_path) if os.path.isabs(config_path) else os.path.join(_PROJECT_ROOT, config_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Perturbation config not found at: {abs_path}")

    # First validate with split_integrity
    integrity_res = validate_split_integrity(abs_path)
    if integrity_res.get("status") != "PASS":
        raise PerturbationConfigError(f"Split integrity validation failed for {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    return PerturbationSuite(raw_cfg, config_path=abs_path)


def get_unmigrated_runners() -> Dict[str, Dict[str, str]]:
    """Return the registry of runner scripts and their migration status."""
    return dict(RUNNER_MIGRATION_REGISTRY)
