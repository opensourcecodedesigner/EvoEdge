"""Unified Perturbation Configuration Loading, Validation, and Environment Interface.

This module provides a single, validated entry point for loading perturbation configurations
from `config/perturbations.yaml`, verifying split and condition integrity, and sampling or
applying perturbation parameters to environment instances.

Features:
  1. Strict YAML parsing, schema validation, and config hashing (SHA-256).
  2. Split isolation verification (training, validation, historical test, supplementary untouched test).
  3. Condition loading: nominal, previously exposed benchmarks, and unexposed final test suite.
  4. Parameter extraction, deterministic sampling, and environment injection with explicit
     provenance logging per episode.
  5. Built-in PerturbedCartPoleEnv standardizing friction dynamics, actuator transport delay,
     and channel-specific sensor noise across all runner scripts.
  6. Audit of runner migration statuses.
"""

from __future__ import annotations

import collections
import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import yaml

from env.cartpole_swingup import CartPoleSwingUpEnv
from utils.split_integrity import validate_split_integrity, SplitIntegrityError

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Registry of runner scripts and their migration status
RUNNER_MIGRATION_REGISTRY = {
    "experiments/runner.py": {
        "status": "MIGRATED",
        "description": "Multi-seed NEAT runner; consumes config/perturbations.yaml and logs sampled parameters and config hash per episode.",
    },
    "experiments/benchmark_suite.py": {
        "status": "MIGRATED",
        "description": "Deterministic benchmark suite; consumes config/perturbations.yaml via PerturbedCartPoleEnv and logs provenance.",
    },
    "experiments/run_baselines_pruning.py": {
        "status": "MIGRATED",
        "description": "Pruning baseline evaluator; loads conditions and splits directly from PerturbationSuite.",
    },
    "evolution/evolve_tinyml.py": {
        "status": "MIGRATED",
        "description": "Fixed-topology DAgger trainer; validates splits and uses PerturbedCartPoleEnv with provenance logging.",
    },
}


class PerturbationConfigError(ValueError):
    """Raised when perturbation configuration is invalid or inconsistent."""
    pass


class PerturbedCartPoleEnv(CartPoleSwingUpEnv):
    """Standardized CartPole Swing-Up Environment supporting physical perturbations.

    Supports:
      - Variable cart and pole mass.
      - Cart rail viscous friction (b_cart) and pole joint rotational friction (b_pole).
      - Actuator transport delay queue (delay_steps).
      - Channel-specific Gaussian observation noise (obs_noise_sigma applied to target_channels).
      - Step-by-step and episode-level provenance logging of sampled parameters.
    """

    def __init__(
        self,
        max_steps: int = 500,
        continuous: bool = True,
        cart_mass: float = 1.0,
        pole_mass: float = 0.1,
        b_cart: float = 0.0,
        b_pole: float = 0.0,
        obs_noise_sigma: float = 0.0,
        delay_steps: int = 0,
        noise_channels: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(max_steps=max_steps, continuous=continuous)
        self.boundary_violated: bool = False
        self.MASS_CART = float(cart_mass)
        self.MASS_POLE = float(pole_mass)
        self._total_mass = self.MASS_CART + self.MASS_POLE
        self._polemass_length = self.MASS_POLE * self.LENGTH

        self.b_cart = float(b_cart)
        self.b_pole = float(b_pole)
        self.obs_noise_sigma = float(obs_noise_sigma)
        self.delay_steps = int(delay_steps)
        self.noise_channels = list(noise_channels) if noise_channels is not None else list(range(5))

        self.action_queue: collections.deque = collections.deque([0.0] * self.delay_steps, maxlen=max(1, self.delay_steps + 1))
        self._rng = np.random.default_rng(seed if seed is not None else 42)
        self._last_sampled_params: Dict[str, float] = {
            "MASS_CART": self.MASS_CART,
            "MASS_POLE": self.MASS_POLE,
            "b_cart": self.b_cart,
            "b_pole": self.b_pole,
            "obs_noise_sigma": self.obs_noise_sigma,
            "delay_steps": float(self.delay_steps),
        }

    def reset_with_initial_state(
        self,
        x_0: float,
        theta_0: float,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to exact deterministic initial state [x_0, 0.0, theta_0, 0.0]."""
        super().reset(seed=seed)
        wrapped_th = ((float(theta_0) + np.pi) % (2.0 * np.pi)) - np.pi
        self._state = np.array([float(x_0), 0.0, wrapped_th, 0.0], dtype=np.float64)
        self._step_count = 0
        self.boundary_violated = False

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.action_queue.clear()
        for _ in range(self.delay_steps):
            self.action_queue.append(0.0)

        obs = self._get_noisy_obs()
        return obs, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.boundary_violated = False
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.action_queue.clear()
        for _ in range(self.delay_steps):
            self.action_queue.append(0.0)

        obs = self._get_noisy_obs()
        return obs, info

    def _get_noisy_obs(self) -> np.ndarray:
        obs = self._get_obs()
        if self.obs_noise_sigma > 0.0:
            noise = np.zeros_like(obs)
            for ch in self.noise_channels:
                if 0 <= ch < len(obs):
                    noise[ch] = self._rng.normal(0.0, self.obs_noise_sigma)
            obs = obs + noise.astype(np.float32)
        return obs

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()."

        # ── Transport delay queue ────────────────────────────
        if self.delay_steps > 0:
            self.action_queue.append(action)
            act_applied = self.action_queue.popleft()
        else:
            act_applied = action

        # ── Compute applied force ────────────────────────────
        if self.continuous:
            if isinstance(act_applied, (list, np.ndarray)):
                act_applied = act_applied[0]
            force = float(np.clip(act_applied, -1.0, 1.0)) * self.FORCE_MAG
        else:
            force = self.FORCE_MAG if act_applied == 1 else -self.FORCE_MAG

        # ── Cart-pole dynamics with viscous rail & joint friction ──
        x, x_dot, theta, theta_dot = self._state
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Net cart horizontal force including rail viscous drag -b_cart * x_dot
        temp = (
            force - self.b_cart * x_dot + self._polemass_length * (theta_dot ** 2) * sin_theta
        ) / self._total_mass

        # Rotational damping torque -b_pole * theta_dot
        fric_pole = (self.b_pole / max(self._polemass_length, 1e-6)) * theta_dot

        theta_acc = (
            self.GRAVITY * sin_theta - cos_theta * temp - fric_pole
        ) / (
            self.LENGTH
            * (4.0 / 3.0 - self.MASS_POLE * (cos_theta ** 2) / self._total_mass)
        )

        x_acc = (
            temp
            - self._polemass_length * theta_acc * cos_theta / self._total_mass
        )

        # Semi-implicit Euler integration
        x_dot = x_dot + self.TAU * x_acc
        x = x + self.TAU * x_dot
        theta_dot = theta_dot + self.TAU * theta_acc
        theta = theta + self.TAU * theta_dot

        # Wrap theta to [-pi, pi]
        theta = ((theta + np.pi) % (2.0 * np.pi)) - np.pi
        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self._step_count += 1

        if abs(x) >= self.X_THRESHOLD:
            self.boundary_violated = True

        reward = float(np.cos(theta)) - 0.01 * (x / self.X_THRESHOLD) ** 2
        terminated = bool(abs(x) > self.X_THRESHOLD)
        truncated = self._step_count >= self.max_steps

        obs = self._get_noisy_obs()
        return obs, reward, terminated, truncated, {
            "applied_action": act_applied,
            "boundary_violated": self.boundary_violated,
            "sampled_params": dict(self._last_sampled_params),
        }


class PerturbationSuite:
    """Loaded and validated perturbation suite."""

    def __init__(self, raw_config: Dict[str, Any], config_path: str = "config/perturbations.yaml") -> None:
        self.raw_config = raw_config
        self.config_path = config_path
        self.version = raw_config.get("version", "unknown")
        self.splits = raw_config.get("splits", {})
        self.perturbations = raw_config.get("perturbations", {})
        self.exposure_catalog = raw_config.get("exposure_catalog", {})
        self.config_hash = hashlib.sha256(yaml.dump(raw_config, sort_keys=True).encode("utf-8")).hexdigest()[:16]

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

            if split in ["test", "supplementary_untouched_test"] and use_test_benchmark:
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

    def create_env(
        self,
        split: str,
        seed: int,
        use_test_benchmark: bool = False,
        max_steps: int = 500,
        continuous: bool = True,
    ) -> Tuple[PerturbedCartPoleEnv, Dict[str, float]]:
        """Create a fully configured PerturbedCartPoleEnv instance with sampled parameters."""
        params = self.sample_parameters(split=split, seed=seed, use_test_benchmark=use_test_benchmark)
        noise_channels = self.perturbations.get("observation_noise", {}).get("target_channels", [])
        ch_indices = [c["index"] for c in noise_channels] if noise_channels else [0, 1, 2, 3, 4]

        env = PerturbedCartPoleEnv(
            max_steps=max_steps,
            continuous=continuous,
            cart_mass=params.get("MASS_CART", 1.0),
            pole_mass=params.get("MASS_POLE", 0.1),
            b_cart=params.get("b_cart", 0.0),
            b_pole=params.get("b_pole", 0.0),
            obs_noise_sigma=params.get("obs_noise_sigma", 0.0),
            delay_steps=int(params.get("delay_steps", 0)),
            noise_channels=ch_indices,
            seed=seed,
        )
        env._last_sampled_params = dict(params)
        return env, params

    def apply_to_env(
        self,
        env: Any,
        params: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply sampled parameters to an environment instance and return the applied dict."""
        SUPPORTED_PARAMS = {"MASS_CART", "MASS_POLE", "b_cart", "b_pole", "obs_noise_sigma", "delay_steps"}
        for k in params:
            if k not in SUPPORTED_PARAMS:
                raise PerturbationConfigError(
                    f"Unsupported perturbation parameter '{k}'. "
                    f"Supported fields: {sorted(list(SUPPORTED_PARAMS))}. "
                    f"Runner cannot support this field without an explicit physics extension."
                )

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

        if hasattr(env, "_last_sampled_params"):
            env._last_sampled_params.update(applied)

        return applied


def load_perturbation_suite(config_path: str = "config/perturbations.yaml") -> PerturbationSuite:
    """Validate and load the perturbation configuration suite."""
    abs_path = os.path.abspath(config_path) if os.path.isabs(config_path) else os.path.join(_PROJECT_ROOT, config_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Perturbation config not found at: {abs_path}")

    integrity_res = validate_split_integrity(abs_path)
    if integrity_res.get("status") != "PASS":
        raise PerturbationConfigError(f"Split integrity validation failed for {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    return PerturbationSuite(raw_cfg, config_path=abs_path)


def get_unmigrated_runners() -> Dict[str, Dict[str, str]]:
    """Return the registry of runner scripts and their migration status."""
    return dict(RUNNER_MIGRATION_REGISTRY)
