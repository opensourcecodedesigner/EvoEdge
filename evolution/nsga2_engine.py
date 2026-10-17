"""Masked Neural Genome and Multi-Objective NSGA-II Evolution Engine.

Implements:
  1. MaskedGenome: Evolvable feedforward neural controller with evolvable weights,
     biases, connection masks, neuron activation flags, and activation functions.
  2. Multi-Objective Objectives:
     - Robust task success rate (equal-weight mean of condition success rates).
     - Active parameter count (weights + biases with active connections/neurons).
     - Active weight MACs per inference step.
  3. Deb-style Constrained Dominance:
     - Feasible candidates dominate infeasible candidates (robust_success >= success_floor).
     - Infeasible candidates sorted by constraint violation magnitude.
     - Feasible candidates sorted by Pareto dominance and crowding distance.
  4. Fast Non-Dominated Sorting and Crowding Distance computation.
  5. Deterministic seed control, mutation, and reproducible serialization.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.budget_tracker import (
    BudgetExhaustedError,
    TransitionBudgetTracker,
    BudgetedEnv,
    BudgetedVectorizedEnv,
)

# Documented deterministic integer namespace for NSGA-II optimizer
NAMESPACE_NSGA2: int = 2


def get_nsga2_rng(optimizer_seed: int) -> Tuple[np.random.Generator, np.random.SeedSequence]:
    """Construct an isolated, deterministic NumPy Generator for NSGA-II optimizer.

    Args:
        optimizer_seed: Base integer seed (e.g. 42..46).

    Returns:
        Tuple of (numpy Generator, SeedSequence).
    """
    if not isinstance(optimizer_seed, (int, np.integer)):
        raise TypeError(f"optimizer_seed must be an integer, got {type(optimizer_seed)}")
    spawn_key = (NAMESPACE_NSGA2, int(optimizer_seed))
    seed_seq = np.random.SeedSequence(entropy=int(optimizer_seed), spawn_key=spawn_key)
    return np.random.default_rng(seed_seq), seed_seq


def norm_batch_obs(
    obs: np.ndarray,
    u_prev1: Optional[np.ndarray] = None,
    u_prev2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize batch of observation vectors for neural forward pass.

    If u_prev1 and u_prev2 are provided, returns (B, 7) array with:
      [x/2.4, clip(x_dot/10), cos(th), sin(th), clip(th_dot/10), u_{t-1}, u_{t-2}]
    Otherwise returns (B, 5) physical observation array.
    """
    phys = np.stack([
        obs[:, 0] / 2.4,
        np.clip(obs[:, 1] / 10.0, -1.0, 1.0),
        obs[:, 2],
        obs[:, 3],
        np.clip(obs[:, 4] / 10.0, -1.0, 1.0),
    ], axis=1)
    if u_prev1 is not None and u_prev2 is not None:
        u1 = np.asarray(u_prev1, dtype=np.float64).reshape(-1, 1)
        u2 = np.asarray(u_prev2, dtype=np.float64).reshape(-1, 1)
        return np.concatenate([phys, u1, u2], axis=1)
    return phys


def norm_obs(
    obs: np.ndarray,
    u_prev1: float = 0.0,
    u_prev2: float = 0.0,
    include_delay: bool = False,
) -> np.ndarray:
    """Normalize single observation vector for neural forward pass.

    If include_delay is True, returns 7D vector with action history.
    Otherwise returns 5D physical observation vector.
    """
    phys = [
        obs[0] / 2.4,
        np.clip(obs[1] / 10.0, -1.0, 1.0),
        obs[2],
        obs[3],
        np.clip(obs[4] / 10.0, -1.0, 1.0),
    ]
    if include_delay:
        return np.array(phys + [float(u_prev1), float(u_prev2)], dtype=np.float64)
    return np.array(phys, dtype=np.float64)


class MaskedGenome:
    """Fixed-capacity masked neural network genome supporting topology evolution."""

    ACTIVATION_FUNCTIONS = ["tanh", "relu", "linear"]

    def __init__(
        self,
        in_dim: int = 5,
        max_hidden: int = 8,
        out_dim: int = 1,
        rng: Optional[np.random.Generator] = None,
    ):
        self.in_dim = in_dim
        self.max_hidden = max_hidden
        self.out_dim = out_dim
        self.rng = rng if rng is not None else np.random.default_rng()

        # Action history state for delay-aware sequential inference
        self._u_prev1: float = 0.0
        self._u_prev2: float = 0.0

        # Weight matrices: Layer 1 (max_hidden, in_dim), Layer 2 (out_dim, max_hidden)
        self.w1 = self.rng.normal(0.0, 0.5, size=(max_hidden, in_dim)).astype(np.float64)
        self.b1 = np.zeros(max_hidden, dtype=np.float64)
        self.w2 = self.rng.normal(0.0, 0.5, size=(out_dim, max_hidden)).astype(np.float64)
        self.b2 = np.zeros(out_dim, dtype=np.float64)

        # Binary Masks: 1 = active connection/neuron, 0 = inactive
        self.w1_mask = np.ones((max_hidden, in_dim), dtype=bool)
        self.b1_mask = np.ones(max_hidden, dtype=bool)
        self.w2_mask = np.ones((out_dim, max_hidden), dtype=bool)
        self.b2_mask = np.ones(out_dim, dtype=bool)
        self.neuron_mask = np.ones(max_hidden, dtype=bool)

        # Per-neuron activation functions
        self.hidden_activations = ["tanh"] * max_hidden

        # Multi-objective metrics & fitness
        comp = self.get_complexity_metrics()
        self.fitness: Dict[str, float] = {
            "robust_success": 0.0,
            "worst_success": 0.0,
            "active_params": float(comp["active_parameters"]),
            "weight_macs": float(comp["weight_macs"]),
            "active_neurons": float(comp["active_neurons"]),
        }
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.constraint_violation: float = 0.0  # Feasible within capacity by default

    def reset_history(self) -> None:
        """Reset action history queue to initial state (zero)."""
        self._u_prev1 = 0.0
        self._u_prev2 = 0.0

    def compute_action(self, obs: np.ndarray) -> float:
        """Compute action from raw unnormalized observation with automatic history tracking."""
        if self.in_dim == 7:
            x = norm_obs(obs, u_prev1=self._u_prev1, u_prev2=self._u_prev2, include_delay=True)
            action = self.forward(x)
            self._u_prev2 = self._u_prev1
            self._u_prev1 = action
            return action
        else:
            x = norm_obs(obs)
            return self.forward(x)

    def copy(self) -> MaskedGenome:
        """Create a deep clone of the genome."""
        clone = MaskedGenome(
            in_dim=self.in_dim,
            max_hidden=self.max_hidden,
            out_dim=self.out_dim,
            rng=copy.copy(self.rng),
        )
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        clone.w1_mask = self.w1_mask.copy()
        clone.b1_mask = self.b1_mask.copy()
        clone.w2_mask = self.w2_mask.copy()
        clone.b2_mask = self.b2_mask.copy()
        clone.neuron_mask = self.neuron_mask.copy()
        clone.hidden_activations = list(self.hidden_activations)
        clone.fitness = dict(self.fitness)
        clone.rank = self.rank
        clone.crowding_distance = self.crowding_distance
        clone.constraint_violation = self.constraint_violation
        clone._u_prev1 = self._u_prev1
        clone._u_prev2 = self._u_prev2
        return clone

    def load_from_dense_weights(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> None:
        """Warm-start the genome from pre-existing dense weights (e.g. Frozen H=8).

        Zeroes out weights for delay inputs beyond the pre-existing input dimension,
        ensuring initial inference produces identical outputs to the teacher.
        """
        H = w1.shape[0]
        assert H <= self.max_hidden
        self.w1.fill(0.0)
        in_cols = min(self.in_dim, w1.shape[1])
        self.w1[:H, :in_cols] = w1[:, :in_cols].astype(np.float64)
        self.b1[:H] = b1.astype(np.float64)
        self.w2[:, :H] = w2.astype(np.float64)
        self.b2[:] = b2.astype(np.float64)

        self.neuron_mask[:H] = True
        self.neuron_mask[H:] = False
        self.w1_mask[:H, :] = True
        self.w1_mask[H:, :] = False
        self.b1_mask[:H] = True
        self.b1_mask[H:] = False
        self.w2_mask[:, :H] = True
        self.w2_mask[:, H:] = False

    def forward(self, x: np.ndarray) -> float:
        """Compute feedforward inference step on 1D observation vector (float64)."""
        # Active neurons
        active_h = self.neuron_mask

        # Layer 1
        effective_w1 = self.w1 * self.w1_mask
        effective_b1 = self.b1 * self.b1_mask

        # Pre-activations
        z1 = effective_w1 @ x + effective_b1
        h = np.zeros(self.max_hidden, dtype=np.float64)

        for i in range(self.max_hidden):
            if active_h[i]:
                act_fn = self.hidden_activations[i]
                val = 2.5 * z1[i]
                if act_fn == "tanh":
                    h[i] = np.tanh(val)
                elif act_fn == "relu":
                    h[i] = max(0.0, val)
                elif act_fn == "linear":
                    h[i] = val

        # Layer 2
        effective_w2 = self.w2 * self.w2_mask
        effective_b2 = self.b2 * self.b2_mask

        # Zero out inactive hidden neurons in layer 2 projection
        h = h * active_h
        out = np.tanh(2.5 * (effective_w2 @ h + effective_b2))
        return float(out[0])

    def forward_batch(self, x_batch: np.ndarray) -> np.ndarray:
        """Compute batch inference across B parallel observations of shape (B, in_dim)."""
        active_h = self.neuron_mask
        effective_w1 = self.w1 * self.w1_mask
        effective_b1 = self.b1 * self.b1_mask

        z1 = x_batch @ effective_w1.T + effective_b1  # shape (B, max_hidden)
        h = np.zeros_like(z1)

        for i in range(self.max_hidden):
            if active_h[i]:
                act_fn = self.hidden_activations[i]
                val = 2.5 * z1[:, i]
                if act_fn == "tanh":
                    h[:, i] = np.tanh(val)
                elif act_fn == "relu":
                    h[:, i] = np.maximum(0.0, val)
                elif act_fn == "linear":
                    h[:, i] = val

        effective_w2 = self.w2 * self.w2_mask
        effective_b2 = self.b2 * self.b2_mask

        h = h * active_h
        z2 = h @ effective_w2.T + effective_b2  # shape (B, 1)
        out = np.tanh(2.5 * z2[:, 0])
        return out

    def get_complexity_metrics(self) -> Dict[str, int]:
        """Compute active nonzero parameters, active weight MACs, and activation count."""
        active_h = self.neuron_mask
        # Layer 1 active weights: active neuron AND active connection AND nonzero weight
        l1_w_active = (self.w1_mask & active_h[:, None] & (np.abs(self.w1) > 1e-8))
        l1_b_active = (self.b1_mask & active_h & (np.abs(self.b1) > 1e-8))

        # Layer 2 active weights: active input neuron AND active connection AND nonzero weight
        l2_w_active = (self.w2_mask & active_h[None, :] & (np.abs(self.w2) > 1e-8))
        l2_b_active = (self.b2_mask & (np.abs(self.b2) > 1e-8))

        total_active_weights = int(np.sum(l1_w_active) + np.sum(l2_w_active))
        total_active_biases = int(np.sum(l1_b_active) + np.sum(l2_b_active))
        total_active_params = total_active_weights + total_active_biases

        # Weight MACs: each active weight multiplication
        weight_macs = total_active_weights

        # Total nonlinear activations evaluated
        num_activations = int(np.sum(active_h)) + self.out_dim

        dense_equiv_params = (self.in_dim * self.max_hidden + self.max_hidden +
                              self.max_hidden * self.out_dim + self.out_dim)
        dense_equiv_macs = self.in_dim * self.max_hidden + self.max_hidden * self.out_dim

        return {
            "active_parameters": total_active_params,
            "active_weights": total_active_weights,
            "active_biases": total_active_biases,
            "weight_macs": weight_macs,
            "num_activations": num_activations,
            "dense_params": dense_equiv_params,
            "dense_macs": dense_equiv_macs,
            "active_neurons": int(np.sum(active_h)),
        }

    def mutate(
        self,
        weight_mutation_rate: float = 0.8,
        weight_mutation_power: float = 0.2,
        connection_toggle_rate: float = 0.1,
        neuron_toggle_rate: float = 0.05,
        activation_mutation_rate: float = 0.05,
    ) -> None:
        """Mutate weights, topology masks, and activation types."""
        # 1. Weight perturbations
        if self.rng.random() < weight_mutation_rate:
            mask1 = self.rng.random(self.w1.shape) < 0.3
            self.w1 += mask1 * self.rng.normal(0.0, weight_mutation_power, size=self.w1.shape)

            mask_b1 = self.rng.random(self.b1.shape) < 0.3
            self.b1 += mask_b1 * self.rng.normal(0.0, weight_mutation_power, size=self.b1.shape)

            mask2 = self.rng.random(self.w2.shape) < 0.3
            self.w2 += mask2 * self.rng.normal(0.0, weight_mutation_power, size=self.w2.shape)

            mask_b2 = self.rng.random(self.b2.shape) < 0.3
            self.b2 += mask_b2 * self.rng.normal(0.0, weight_mutation_power, size=self.b2.shape)

        # 2. Connection toggles (addition / removal)
        if self.rng.random() < connection_toggle_rate:
            # Layer 1 toggle
            i = self.rng.integers(0, self.max_hidden)
            j = self.rng.integers(0, self.in_dim)
            self.w1_mask[i, j] = not self.w1_mask[i, j]

            # Layer 2 toggle
            k = self.rng.integers(0, self.max_hidden)
            self.w2_mask[0, k] = not self.w2_mask[0, k]

        # 3. Neuron activation flag toggle
        if self.rng.random() < neuron_toggle_rate:
            idx = self.rng.integers(0, self.max_hidden)
            # Ensure at least 1 neuron remains active
            if self.neuron_mask[idx] and np.sum(self.neuron_mask) > 1:
                self.neuron_mask[idx] = False
            else:
                self.neuron_mask[idx] = True

        # 4. Activation type mutation
        if self.rng.random() < activation_mutation_rate:
            idx = self.rng.integers(0, self.max_hidden)
            self.hidden_activations[idx] = str(self.rng.choice(self.ACTIVATION_FUNCTIONS))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome state to JSON-serializable dictionary."""
        return {
            "in_dim": self.in_dim,
            "max_hidden": self.max_hidden,
            "out_dim": self.out_dim,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w1_mask": self.w1_mask.tolist(),
            "b1_mask": self.b1_mask.tolist(),
            "w2_mask": self.w2_mask.tolist(),
            "b2_mask": self.b2_mask.tolist(),
            "neuron_mask": self.neuron_mask.tolist(),
            "hidden_activations": self.hidden_activations,
            "fitness": self.fitness,
            "complexity": self.get_complexity_metrics(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> MaskedGenome:
        """Restore genome from dictionary."""
        g = cls(
            in_dim=data["in_dim"],
            max_hidden=data["max_hidden"],
            out_dim=data["out_dim"],
            rng=rng,
        )
        g.w1 = np.array(data["w1"], dtype=np.float64)
        g.b1 = np.array(data["b1"], dtype=np.float64)
        g.w2 = np.array(data["w2"], dtype=np.float64)
        g.b2 = np.array(data["b2"], dtype=np.float64)
        g.w1_mask = np.array(data["w1_mask"], dtype=bool)
        g.b1_mask = np.array(data["b1_mask"], dtype=bool)
        g.w2_mask = np.array(data["w2_mask"], dtype=bool)
        g.b2_mask = np.array(data["b2_mask"], dtype=bool)
        g.neuron_mask = np.array(data["neuron_mask"], dtype=bool)
        g.hidden_activations = list(data["hidden_activations"])
        g.fitness = dict(data.get("fitness", {}))
        return g


# ── Deb Constrained Domination & NSGA-II Utilities ────────────────────────────


def get_complexity_violation(ind: MaskedGenome, max_params: int = 128, max_macs: int = 128) -> float:
    """Compute complexity constraint violation magnitude."""
    params = float(ind.fitness.get("active_params", 0.0))
    macs = float(ind.fitness.get("weight_macs", 0.0))
    return max(0.0, params - max_params) + max(0.0, macs - max_macs)


def constrained_dominates(
    ind1: MaskedGenome,
    ind2: MaskedGenome,
    max_params: int = 128,
    max_macs: int = 128,
    progress_tolerance: float = 1.0,
) -> bool:
    """Deb-style constrained domination comparator with Stage 3B non-degeneracy:

    1. Complexity Constraint Violation:
       - Candidates exceeding 128 active parameters or 128 weight MACs are infeasible.
       - A feasible candidate strictly dominates an infeasible candidate.
       - Between two infeasible candidates, the one with smaller violation dominates.

    2. Functional vs Nonfunctional:
       - A functional candidate (robust_success > 0) strictly dominates a nonfunctional
         candidate (robust_success == 0), preventing trivial zero-success networks from
         dominating simply because they are small.

    3. Both Nonfunctional (robust_success == 0):
       - If ind1 has strictly better task progress (> ind2.task_progress + progress_tolerance):
         ind1 dominates ind2 (progress guides early acquisition).
       - If ind2 has strictly better task progress (> ind1.task_progress + progress_tolerance):
         ind1 does not dominate ind2 (ind2 dominates ind1).
       - If task progress is comparable (|ind1.task_progress - ind2.task_progress| <= progress_tolerance):
         Pareto comparison on complexity: smaller active_params and weight_macs dominate.

    4. Both Functional (robust_success > 0):
       - Pareto dominance across:
         Objective 1: Maximize robust_success (or task_progress)
         Objective 2: Minimize active_parameters
         Objective 3: Minimize weight_macs
    """
    cv1 = float(getattr(ind1, "constraint_violation", 0.0))
    cv2 = float(getattr(ind2, "constraint_violation", 0.0))
    v1 = max(cv1, get_complexity_violation(ind1, max_params, max_macs))
    v2 = max(cv2, get_complexity_violation(ind2, max_params, max_macs))

    # 1. Constraint checks
    if v1 <= 0.0 and v2 > 0.0:
        return True
    if v1 > 0.0 and v2 <= 0.0:
        return False
    if v1 > 0.0 and v2 > 0.0:
        if abs(v1 - v2) > 1e-6:
            return v1 < v2
        # If constraint violations are equal, proceed to task performance comparison below!

    succ1 = float(ind1.fitness.get("robust_success", 0.0))
    succ2 = float(ind2.fitness.get("robust_success", 0.0))
    prog1 = float(ind1.fitness.get("task_progress", 0.0))
    prog2 = float(ind2.fitness.get("task_progress", 0.0))
    p1 = float(ind1.fitness.get("active_params", 0.0))
    p2 = float(ind2.fitness.get("active_params", 0.0))
    m1 = float(ind1.fitness.get("weight_macs", 0.0))
    m2 = float(ind2.fitness.get("weight_macs", 0.0))

    # 2. Functional vs Nonfunctional
    if succ1 > 0.0 and succ2 <= 0.0:
        return True
    if succ1 <= 0.0 and succ2 > 0.0:
        return False

    # 3. Both Nonfunctional: Dense task progress guides search
    if succ1 <= 0.0 and succ2 <= 0.0:
        if prog1 > prog2 + progress_tolerance:
            return True
        if prog2 > prog1 + progress_tolerance:
            return False
        # Comparable progress: smaller complexity dominates
        at_least_as_good = (p1 <= p2) and (m1 <= m2)
        strictly_better = (p1 < p2) or (m1 < m2)
        return at_least_as_good and strictly_better

    # 4. Both Functional: Standard Pareto dominance
    perf1 = succ1
    perf2 = succ2
    if abs(perf1 - perf2) < 1e-6:
        # Same success rate: use task progress as fine-grained tie-breaker
        perf1 = prog1
        perf2 = prog2

    at_least_as_good = (perf1 >= perf2) and (p1 <= p2) and (m1 <= m2)
    strictly_better = (perf1 > perf2) or (p1 < p2) or (m1 < m2)
    return at_least_as_good and strictly_better


def fast_non_dominated_sort(population: List[MaskedGenome]) -> List[List[MaskedGenome]]:
    """Partition population into Pareto fronts using Deb's fast non-dominated sort."""
    fronts: List[List[MaskedGenome]] = [[]]
    sp: Dict[int, List[int]] = {i: [] for i in range(len(population))}
    np_dom_count = [0] * len(population)

    for p_idx, p in enumerate(population):
        for q_idx, q in enumerate(population):
            if p_idx == q_idx:
                continue
            if constrained_dominates(p, q):
                sp[p_idx].append(q_idx)
            elif constrained_dominates(q, p):
                np_dom_count[p_idx] += 1

        if np_dom_count[p_idx] == 0:
            p.rank = 0
            fronts[0].append(p)

    cur_front = 0
    # Map object id to index for robust O(1) index lookup
    id_to_idx = {id(ind): idx for idx, ind in enumerate(population)}

    while len(fronts[cur_front]) > 0:
        next_front: List[MaskedGenome] = []
        for p in fronts[cur_front]:
            p_idx = id_to_idx[id(p)]
            for q_idx in sp[p_idx]:
                np_dom_count[q_idx] -= 1
                if np_dom_count[q_idx] == 0:
                    q = population[q_idx]
                    q.rank = cur_front + 1
                    next_front.append(q)
        cur_front += 1
        fronts.append(next_front)

    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts


def compute_crowding_distance(front: List[MaskedGenome]) -> None:
    """Calculate crowding distance for individuals within a non-dominated front."""
    n = len(front)
    if n == 0:
        return
    for ind in front:
        ind.crowding_distance = 0.0

    if n <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    # Objectives: task_progress (max, captures dense progress & success), active_params (min), weight_macs (min)
    objectives = ["task_progress", "active_params", "weight_macs"]

    for obj in objectives:
        front.sort(key=lambda ind: ind.fitness.get(obj, 0.0))
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        obj_min = front[0].fitness.get(obj, 0.0)
        obj_max = front[-1].fitness.get(obj, 0.0)
        denom = obj_max - obj_min

        if denom > 1e-9:
            for i in range(1, n - 1):
                front[i].crowding_distance += (front[i + 1].fitness.get(obj, 0.0) - front[i - 1].fitness.get(obj, 0.0)) / denom


def nsga2_survivor_selection(population: List[MaskedGenome], pop_size: int) -> List[MaskedGenome]:
    """Select the top pop_size individuals using elitist NSGA-II survivor selection."""
    fronts = fast_non_dominated_sort(population)
    survivors: List[MaskedGenome] = []

    for front in fronts:
        compute_crowding_distance(front)
        if len(survivors) + len(front) <= pop_size:
            survivors.extend(front)
        else:
            # Sort by crowding distance descending (breaking ties)
            front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
            needed = pop_size - len(survivors)
            survivors.extend(front[:needed])
            break

    return survivors


# ── Stage 3C Domain-Randomized Training Distribution ──────────────────────────
STAGE3C_TRAIN_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "1_Nominal": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0,
        "obs_noise_sigma": 0.0, "delay_steps": 0,
    },
    "2_ActuatorDelay_1Step": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0,
        "obs_noise_sigma": 0.0, "delay_steps": 1,
    },
    "3_ActuatorDelay_2Step": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0,
        "obs_noise_sigma": 0.0, "delay_steps": 2,
    },
    "4_Mass_Friction": {
        "cart_mass": 1.25, "pole_mass": 0.125, "b_cart": 0.35, "b_pole": 0.006,
        "obs_noise_sigma": 0.0, "delay_steps": 0,
    },
    "5_CombinedStress": {
        "cart_mass": 1.20, "pole_mass": 0.120, "b_cart": 0.40, "b_pole": 0.008,
        "obs_noise_sigma": 0.03, "delay_steps": 1,
    },
}


# ── Budgeted Evaluation and Training Entry Point ──────────────────────────────


def evaluate_genome_budgeted(
    genome: MaskedGenome,
    eval_seeds: List[int],
    conditions: Optional[Dict[str, Dict[str, Any]]] = None,
    max_steps: int = 500,
    budget_tracker: Optional[TransitionBudgetTracker] = None,
    vectorized: bool = True,
    success_floor: float = 0.40,
    reward_mode: str = "stage3b",
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate candidate genome over specified conditions and seeds using a budgeted environment.

    Bills physical simulator transitions monotonically to budget_tracker.
    Raises BudgetExhaustedError if the budget ceiling is reached.
    """
    if conditions is None:
        conditions = STAGE3C_TRAIN_CONDITIONS

    num_envs = len(eval_seeds)
    cond_rates: Dict[str, float] = {}
    cond_returns: Dict[str, float] = {}

    if vectorized:
        from env.vectorized_env import VectorizedCartPoleEnv

        for cond_name, cond_params in conditions.items():
            if budget_tracker is not None and budget_tracker.is_exhausted:
                raise BudgetExhaustedError("Budget exhausted before starting condition evaluation.")

            vec_env = VectorizedCartPoleEnv(num_envs=num_envs, max_steps=max_steps, reward_mode=reward_mode, **cond_params)
            bvec_env = BudgetedVectorizedEnv(vec_env, budget_tracker) if budget_tracker is not None else vec_env

            obs, _ = bvec_env.reset(seeds=eval_seeds)
            cos_thetas: List[np.ndarray] = []
            batch_returns = np.zeros(num_envs, dtype=np.float64)
            u_prev1 = np.zeros(num_envs, dtype=np.float64)
            u_prev2 = np.zeros(num_envs, dtype=np.float64)

            for step in range(max_steps):
                if genome.in_dim == 7:
                    norm_o = norm_batch_obs(obs, u_prev1=u_prev1, u_prev2=u_prev2)
                    actions = genome.forward_batch(norm_o)
                    u_prev2 = u_prev1.copy()
                    u_prev1 = actions.copy()
                else:
                    norm_o = norm_batch_obs(obs)
                    actions = genome.forward_batch(norm_o)
                obs, rewards, term, trunc, _ = bvec_env.step(actions)
                cos_thetas.append(obs[:, 2])
                batch_returns += np.asarray(rewards, dtype=np.float64)

            cos_arr = np.array(cos_thetas)
            final_window = cos_arr[-100:]
            upright_count = np.sum(final_window > 0.9, axis=0)
            boundary_violated = vec_env.boundary_violated if hasattr(vec_env, "boundary_violated") else bvec_env.env.boundary_violated
            success = (upright_count >= 95) & (~boundary_violated)
            rate = float(np.mean(success))
            cond_rates[cond_name] = rate
            cond_returns[cond_name] = float(np.mean(batch_returns))
    else:
        from env.cartpole_swingup import CartPoleSwingUpEnv

        for cond_name, cond_params in conditions.items():
            succ_count = 0
            ep_returns: List[float] = []
            for s in eval_seeds:
                if budget_tracker is not None and budget_tracker.is_exhausted:
                    raise BudgetExhaustedError("Budget exhausted before starting episode.")

                base_env = CartPoleSwingUpEnv(max_steps=max_steps, continuous=True, reward_mode=reward_mode)
                benv = BudgetedEnv(base_env, budget_tracker) if budget_tracker is not None else base_env
                obs, _ = benv.reset(seed=s)
                cos_th_list: List[float] = []
                rail_hit = False
                ep_r = 0.0
                genome.reset_history()

                for step_i in range(max_steps):
                    action = genome.compute_action(obs)
                    obs, r, term, trunc, _ = benv.step(action)
                    ep_r += float(r)
                    cos_th_list.append(float(obs[2]))
                    if term:
                        rail_hit = True
                        break

                ep_returns.append(ep_r)
                if not rail_hit and len(cos_th_list) == max_steps:
                    if sum(1 for c in cos_th_list[-100:] if c > 0.9) >= 95:
                        succ_count += 1

            cond_rates[cond_name] = succ_count / max(1, len(eval_seeds))
            cond_returns[cond_name] = float(np.mean(ep_returns)) if ep_returns else 0.0

    robust_success = float(np.mean(list(cond_rates.values())))
    worst_success = float(np.min(list(cond_rates.values())))
    mean_return = float(np.mean(list(cond_returns.values())))
    # Task progress prioritizes robust success and worst condition improvement
    task_progress = float(1000.0 * robust_success + 500.0 * worst_success + mean_return)
    comp = genome.get_complexity_metrics()

    genome.fitness = {
        "robust_success": robust_success,
        "worst_success": worst_success,
        "mean_return": mean_return,
        "task_progress": task_progress,
        "active_params": comp["active_parameters"],
        "weight_macs": comp["weight_macs"],
        "active_neurons": comp["active_neurons"],
    }
    genome.constraint_violation = get_complexity_violation(genome)

    return robust_success, worst_success, cond_rates


def train_nsga2(
    optimizer_seed: int,
    pop_size: int = 20,
    offspring_size: int = 10,
    generations: int = 1000,
    eval_episodes: int = 10,
    max_steps: int = 500,
    budget_limit: Optional[int] = None,
    total_budget: int = 2_000_000,
    search_budget: int = 1_850_000,
    validation_budget: int = 150_000,
    budget_tracker: Optional[TransitionBudgetTracker] = None,
    train_seeds: Optional[List[int]] = None,
    conditions: Optional[Dict[str, Dict[str, Any]]] = None,
    vectorized: bool = True,
    success_floor: float = 0.40,
    reward_mode: str = "stage3b",
    warm_start_weights: Optional[Dict[str, np.ndarray]] = None,
    warm_start_path: Optional[str] = None,
    in_dim: int = 7,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 3B NSGA-II compact neuroevolution with strict transition budget enforcement.

    Args:
        optimizer_seed: Integer seed for optimizer RNG.
        pop_size: Population size (default 20).
        offspring_size: Offspring generated per gen (default 10).
        generations: Maximum generations (default 1000, loops until search budget exhausted).
        eval_episodes: Episodes per evaluation (default 10).
        max_steps: Maximum steps per episode (default 500).
        budget_limit: Optional legacy single budget limit.
        total_budget: Maximum transitions across search + validation lifecycle (default 2,000,000).
        search_budget: Maximum transitions allowed for search phase (default 1,850,000).
        validation_budget: Maximum transitions reserved for validation phase (default 150,000).
        budget_tracker: Optional existing TransitionBudgetTracker.
        train_seeds: Optional list of training seeds (defaults to seeds 0..199).
        conditions: Optional condition dict (defaults to nominal condition).
        vectorized: Use vectorized simulator for batch rollouts.
        success_floor: Deb constraint floor.
        reward_mode: Reward formulation ('stage3b' or 'legacy').
        warm_start_weights: Optional pre-trained dense weights dict.
        warm_start_path: Optional pickle path to warm-start controller.
        verbose: Print progress.

    Returns:
        Dictionary containing Pareto front, population, tracker, and stop metadata.
    """
    import pickle
    import time
    t0 = time.time()

    if budget_tracker is None:
        if budget_limit is not None:
            budget_tracker = TransitionBudgetTracker(
                budget_limit=budget_limit,
                search_budget=budget_limit,
                validation_budget=0,
            )
        else:
            budget_tracker = TransitionBudgetTracker(
                budget_limit=total_budget,
                search_budget=search_budget,
                validation_budget=validation_budget,
            )

    rng, seed_seq = get_nsga2_rng(optimizer_seed)

    if train_seeds is None:
        train_seeds = list(range(0, 200))

    if conditions is None:
        conditions = STAGE3C_TRAIN_CONDITIONS

    if warm_start_path is None:
        default_teacher_path = os.path.join(_PROJECT_ROOT, "results", "champion_neural_h8.pkl")
        if os.path.exists(default_teacher_path):
            warm_start_path = default_teacher_path

    # Load warm-start weights if requested
    if warm_start_weights is None and warm_start_path is not None and os.path.exists(warm_start_path):
        with open(warm_start_path, "rb") as f:
            warm_data = pickle.load(f)
            if isinstance(warm_data, dict):
                warm_start_weights = warm_data
            elif hasattr(warm_data, "w1"):
                warm_start_weights = {
                    "w1": warm_data.w1, "b1": warm_data.b1,
                    "w2": warm_data.w2, "b2": warm_data.b2,
                }

    # Initialize Population
    if warm_start_weights is not None:
        population: List[MaskedGenome] = []
        g_base = MaskedGenome(in_dim=in_dim, rng=rng)
        g_base.load_from_dense_weights(
            warm_start_weights["w1"], warm_start_weights["b1"],
            warm_start_weights["w2"], warm_start_weights["b2"],
        )
        population.append(g_base)
        for _ in range(pop_size - 1):
            child = g_base.copy()
            child.rng = rng
            child.mutate(
                weight_mutation_rate=0.7,
                weight_mutation_power=0.15,
                connection_toggle_rate=0.2,
                neuron_toggle_rate=0.1,
                activation_mutation_rate=0.05,
            )
            population.append(child)
    else:
        population = [MaskedGenome(in_dim=in_dim, rng=rng) for _ in range(pop_size)]

    stopped_by_budget = False
    stopping_reason = "max_generations_reached"
    completed_generations = 0
    candidate_evaluations = 0

    num_conds = len(conditions) if conditions is not None else 1
    cond_seeds_count = max(1, eval_episodes // num_conds)

    try:
        # Initial Population Evaluation
        init_seeds = [train_seeds[(optimizer_seed + i) % len(train_seeds)] for i in range(cond_seeds_count)]
        for ind in population:
            if budget_tracker.is_exhausted or budget_tracker.remaining_transitions < 1:
                stopped_by_budget = True
                stopping_reason = "search_budget_exhausted"
                break
            evaluate_genome_budgeted(
                ind,
                eval_seeds=init_seeds,
                conditions=conditions,
                max_steps=max_steps,
                budget_tracker=budget_tracker,
                vectorized=vectorized,
                success_floor=success_floor,
                reward_mode=reward_mode,
            )
            candidate_evaluations += 1

        # Generational Search Loop (continues until search budget is exhausted)
        for gen in range(1, generations + 1):
            if budget_tracker.is_exhausted or budget_tracker.remaining_transitions < 1:
                stopped_by_budget = True
                stopping_reason = "search_budget_exhausted"
                break

            # Survivor selection
            survivors = nsga2_survivor_selection(population, pop_size // 2)

            # Produce offspring
            offspring: List[MaskedGenome] = []
            for p in survivors[:offspring_size]:
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

            gen_seeds = [
                train_seeds[(optimizer_seed + gen * 37 + i) % len(train_seeds)]
                for i in range(cond_seeds_count)
            ]

            # Evaluate offspring with strict pre-checks
            aborted_generation = False
            for child in offspring:
                if budget_tracker.is_exhausted or budget_tracker.remaining_transitions < 1:
                    stopped_by_budget = True
                    stopping_reason = "search_budget_exhausted"
                    aborted_generation = True
                    break
                evaluate_genome_budgeted(
                    child,
                    eval_seeds=gen_seeds,
                    conditions=conditions,
                    max_steps=max_steps,
                    budget_tracker=budget_tracker,
                    vectorized=vectorized,
                    success_floor=success_floor,
                    reward_mode=reward_mode,
                )
                candidate_evaluations += 1

            if aborted_generation or len(offspring) == 0:
                break

            population = survivors + offspring
            completed_generations = gen

            if verbose and (gen % 5 == 0 or gen == 1):
                best_ind = max(population, key=lambda x: (x.fitness.get("robust_success", 0.0), x.fitness.get("task_progress", 0.0)))
                succ_val = best_ind.fitness.get('robust_success', 0.0) * 100
                prog_val = best_ind.fitness.get('task_progress', 0.0)
                print(f"[NSGA-II Gen {gen:3d}] Best Succ: {succ_val:5.1f}% | Progress: {prog_val:7.1f} | Consumed: {budget_tracker.consumed_transitions:,} / {budget_tracker.search_budget:,}")

    except BudgetExhaustedError:
        stopped_by_budget = True
        stopping_reason = "search_budget_exhausted"

    fronts = fast_non_dominated_sort(population)
    pareto_front = fronts[0] if fronts else []

    wall_clock_time = time.time() - t0
    spawn_key = (NAMESPACE_NSGA2, optimizer_seed)
    return {
        "pareto_front": pareto_front,
        "population": population,
        "tracker": budget_tracker,
        "completed_generations": completed_generations,
        "stopped_by_budget": stopped_by_budget,
        "stopping_reason": stopping_reason,
        "actual_transitions": budget_tracker.consumed_transitions,
        "search_transitions": budget_tracker.search_consumed,
        "candidate_evaluations": candidate_evaluations,
        "nominal_rollout_capacity": candidate_evaluations * eval_episodes * max_steps,
        "wall_clock_time": wall_clock_time,
        "optimizer_seed": optimizer_seed,
        "spawn_key": spawn_key,
        "namespace": NAMESPACE_NSGA2,
        "reward_mode": reward_mode,
    }


def validate_nsga2_champion(
    genome: MaskedGenome,
    tracker: TransitionBudgetTracker,
    max_steps: int = 500,
    vectorized: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 2 validation screening under the validation phase budget.

    Charges simulator transitions to tracker under 'validation' phase.
    1. Gate 1B Nominal: 100 episodes on seeds 200..299 (max 50,000 transitions).
    2. Multi-condition diagnostic screening: 20 seeds (300..319) x 10 conditions (max 100,000 transitions).
    """
    from experiments.run_phase_3a_evolution import DIAGNOSTIC_CONDITIONS
    from experiments.run_robustness_experiment import evaluate_success_predicate
    from utils.perturbation_loader import PerturbedCartPoleEnv

    tracker.start_validation_phase()

    gate1b_seeds = list(range(200, 300))
    diag_seeds = list(range(300, 320))
    nominal_cond = {"1_Nominal": DIAGNOSTIC_CONDITIONS["1_Nominal"]}

    # 1. Gate 1B Nominal
    if vectorized:
        from env.vectorized_env import VectorizedCartPoleEnv
        vec_env = VectorizedCartPoleEnv(num_envs=len(gate1b_seeds), max_steps=max_steps, **nominal_cond["1_Nominal"])
        bvec_env = BudgetedVectorizedEnv(vec_env, tracker)
        obs, _ = bvec_env.reset(seeds=gate1b_seeds)
        cos_thetas = []
        u_prev1 = np.zeros(len(gate1b_seeds), dtype=np.float64)
        u_prev2 = np.zeros(len(gate1b_seeds), dtype=np.float64)
        for _ in range(max_steps):
            if genome.in_dim == 7:
                norm_o = norm_batch_obs(obs, u_prev1=u_prev1, u_prev2=u_prev2)
                actions = genome.forward_batch(norm_o)
                u_prev2 = u_prev1.copy()
                u_prev1 = actions.copy()
            else:
                norm_o = norm_batch_obs(obs)
                actions = genome.forward_batch(norm_o)
            try:
                obs, _, _, _, _ = bvec_env.step(actions)
                cos_thetas.append(obs[:, 2])
            except BudgetExhaustedError:
                break
        cos_arr = np.array(cos_thetas)
        final_window = cos_arr[-100:] if len(cos_arr) >= 100 else cos_arr
        upright_count = np.sum(final_window > 0.9, axis=0) if len(cos_arr) > 0 else np.zeros(len(gate1b_seeds))
        success = (upright_count >= 95) & (~vec_env.boundary_violated)
        gate1b_rate = float(np.mean(success))
        gate1b_succ = int(np.sum(success))
    else:
        gate1b_succ = 0
        for s in gate1b_seeds:
            if tracker.is_exhausted or tracker.is_total_exhausted:
                break
            base_env = PerturbedCartPoleEnv(max_steps=max_steps, continuous=True, seed=s, **nominal_cond["1_Nominal"])
            benv = BudgetedEnv(base_env, tracker)
            obs, _ = benv.reset(seed=s)
            cos_list = []
            rail = False
            genome.reset_history()
            for _ in range(max_steps):
                action = genome.compute_action(obs)
                try:
                    obs, _, term, trunc, _ = benv.step(action)
                    cos_list.append(float(obs[2]))
                    if term:
                        rail = True
                        break
                except BudgetExhaustedError:
                    break
            is_s, _, _, _ = evaluate_success_predicate(cos_list, rail, max_steps)
            if is_s:
                gate1b_succ += 1
        gate1b_rate = gate1b_succ / len(gate1b_seeds)

    # 2. Multi-condition diagnostic screening
    diag_rates: Dict[str, float] = {}
    if vectorized:
        from env.vectorized_env import VectorizedCartPoleEnv
        for cond_name, cond_params in DIAGNOSTIC_CONDITIONS.items():
            if tracker.is_exhausted or tracker.is_total_exhausted:
                break
            vec_env = VectorizedCartPoleEnv(num_envs=len(diag_seeds), max_steps=max_steps, **cond_params)
            bvec_env = BudgetedVectorizedEnv(vec_env, tracker)
            obs, _ = bvec_env.reset(seeds=diag_seeds)
            cos_thetas = []
            u_prev1 = np.zeros(len(diag_seeds), dtype=np.float64)
            u_prev2 = np.zeros(len(diag_seeds), dtype=np.float64)
            for _ in range(max_steps):
                if genome.in_dim == 7:
                    norm_o = norm_batch_obs(obs, u_prev1=u_prev1, u_prev2=u_prev2)
                    actions = genome.forward_batch(norm_o)
                    u_prev2 = u_prev1.copy()
                    u_prev1 = actions.copy()
                else:
                    norm_o = norm_batch_obs(obs)
                    actions = genome.forward_batch(norm_o)
                try:
                    obs, _, _, _, _ = bvec_env.step(actions)
                    cos_thetas.append(obs[:, 2])
                except BudgetExhaustedError:
                    break
            cos_arr = np.array(cos_thetas)
            final_window = cos_arr[-100:] if len(cos_arr) >= 100 else cos_arr
            upright_count = np.sum(final_window > 0.9, axis=0) if len(cos_arr) > 0 else np.zeros(len(diag_seeds))
            success = (upright_count >= 95) & (~vec_env.boundary_violated)
            diag_rates[cond_name] = float(np.mean(success))
    else:
        for cond_name, cond_params in DIAGNOSTIC_CONDITIONS.items():
            if tracker.is_exhausted or tracker.is_total_exhausted:
                break
            cond_succ = 0
            for s in diag_seeds:
                if tracker.is_exhausted or tracker.is_total_exhausted:
                    break
                base_env = PerturbedCartPoleEnv(max_steps=max_steps, continuous=True, seed=s, **cond_params)
                benv = BudgetedEnv(base_env, tracker)
                obs, _ = benv.reset(seed=s)
                cos_list = []
                rail = False
                genome.reset_history()
                for _ in range(max_steps):
                    action = genome.compute_action(obs)
                    try:
                        obs, _, term, trunc, _ = benv.step(action)
                        cos_list.append(float(obs[2]))
                        if term:
                            rail = True
                            break
                    except BudgetExhaustedError:
                        break
                is_s, _, _, _ = evaluate_success_predicate(cos_list, rail, max_steps)
                if is_s:
                    cond_succ += 1
            diag_rates[cond_name] = cond_succ / len(diag_seeds)

    robust_success = float(np.mean(list(diag_rates.values()))) if diag_rates else 0.0
    worst_success = float(np.min(list(diag_rates.values()))) if diag_rates else 0.0

    if verbose:
        print(f"NSGA-II Validation complete: Gate 1B: {gate1b_rate*100:.1f}% | Robust: {robust_success*100:.1f}% | Worst: {worst_success*100:.1f}%")
        print(f"Validation transitions: {tracker.validation_consumed:,} | Total consumed: {tracker.consumed_transitions:,}")

    return {
        "gate1b_rate": gate1b_rate,
        "gate1b_success_count": gate1b_succ,
        "gate1b_total": len(gate1b_seeds),
        "robust_success": robust_success,
        "worst_success": worst_success,
        "diagnostic_rates": diag_rates,
        "validation_transitions_used": tracker.validation_consumed,
        "total_transitions_used": tracker.consumed_transitions,
    }


# ── Stage 3D Robustness-First Training Distribution ───────────────────────────
STAGE3D_TRAIN_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "1_Nominal": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0,
        "obs_noise_sigma": 0.0, "delay_steps": 0,
    },
    "2_CartMass_+30%": {
        "cart_mass": 1.30, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0,
        "obs_noise_sigma": 0.0, "delay_steps": 0,
    },
    "3_CartFriction_0.50": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.50, "b_pole": 0.0,
        "obs_noise_sigma": 0.0, "delay_steps": 0,
    },
    "4_JointFriction_0.010": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.010,
        "obs_noise_sigma": 0.0, "delay_steps": 0,
    },
    "5_CombinedFriction": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.50, "b_pole": 0.010,
        "obs_noise_sigma": 0.0, "delay_steps": 0,
    },
    "6_SensorNoise_0.05": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0,
        "obs_noise_sigma": 0.050, "delay_steps": 0,
    },
    "7_ActuatorDelay_2Step": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0,
        "obs_noise_sigma": 0.0, "delay_steps": 2,
    },
    "8_CombinedStress": {
        "cart_mass": 1.30, "pole_mass": 0.130, "b_cart": 0.50, "b_pole": 0.010,
        "obs_noise_sigma": 0.050, "delay_steps": 1,
    },
}


def constrained_dominates_stage3d(
    ind1: MaskedGenome,
    ind2: MaskedGenome,
    max_params: int = 128,
    max_macs: int = 128,
    progress_tolerance: float = 5.0,
) -> bool:
    """Stage 3D Robustness-First Constrained Domination Comparator.

    Priority Hierarchy:
    1. Feasibility: Active params <= 128 and weight MACs <= 128.
    2. Functional vs Non-functional: robust_success > 0 strictly dominates robust_success == 0.
    3. Both Non-functional: Dense task progress guides search towards initial acquisition.
    4. Both Functional:
       4.1: Worst-Condition Binary Success: Higher worst_success strictly dominates.
       4.2: Worst-Condition Success Tied (w1 == w2 == 0.0):
            - Compare robust_success (candidate solving more conditions dominates).
            - If robust_success tied: compare worst_return (dense return on failing condition).
            - If worst_return tied: compare mean_return and complexity.
       4.3: Worst-Condition Success Tied (w1 == w2 > 0.0):
            - Pareto dominance across: Maximize robust_success, Minimize active_params, Minimize weight_macs.
    """
    cv1 = float(getattr(ind1, "constraint_violation", 0.0))
    cv2 = float(getattr(ind2, "constraint_violation", 0.0))
    v1 = max(cv1, get_complexity_violation(ind1, max_params, max_macs))
    v2 = max(cv2, get_complexity_violation(ind2, max_params, max_macs))

    # 1. Complexity Constraint Violation
    if v1 <= 0.0 and v2 > 0.0:
        return True
    if v1 > 0.0 and v2 <= 0.0:
        return False
    if v1 > 0.0 and v2 > 0.0:
        if abs(v1 - v2) > 1e-6:
            return v1 < v2

    r1 = float(ind1.fitness.get("robust_success", 0.0))
    r2 = float(ind2.fitness.get("robust_success", 0.0))
    w1 = float(ind1.fitness.get("worst_success", 0.0))
    w2 = float(ind2.fitness.get("worst_success", 0.0))
    wp1 = float(ind1.fitness.get("worst_return", 0.0))
    wp2 = float(ind2.fitness.get("worst_return", 0.0))
    mp1 = float(ind1.fitness.get("mean_return", 0.0))
    mp2 = float(ind2.fitness.get("mean_return", 0.0))
    p1 = float(ind1.fitness.get("active_params", 0.0))
    p2 = float(ind2.fitness.get("active_params", 0.0))
    m1 = float(ind1.fitness.get("weight_macs", 0.0))
    m2 = float(ind2.fitness.get("weight_macs", 0.0))

    # 2. Functional vs Non-functional
    if r1 > 0.0 and r2 <= 0.0:
        return True
    if r1 <= 0.0 and r2 > 0.0:
        return False

    # 3. Both Non-functional: Dense progress guides early acquisition
    if r1 <= 0.0 and r2 <= 0.0:
        prog1 = float(ind1.fitness.get("task_progress", 0.0))
        prog2 = float(ind2.fitness.get("task_progress", 0.0))
        if prog1 > prog2 + progress_tolerance:
            return True
        if prog2 > prog1 + progress_tolerance:
            return False
        at_least_as_good = (p1 <= p2) and (m1 <= m2)
        strictly_better = (p1 < p2) or (m1 < m2)
        return at_least_as_good and strictly_better

    # 4. Both Functional (r1 > 0 and r2 > 0):
    # 4.1: Worst-Condition Binary Success Priority
    if w1 > w2 + 1e-6:
        return True
    if w2 > w1 + 1e-6:
        return False

    # 4.2: Worst-Condition Success Tied at 0.0 (both have failing conditions)
    if w1 <= 0.0 and w2 <= 0.0:
        if r1 > r2 + 1e-6:
            return True
        if r2 > r1 + 1e-6:
            return False

        # Same number of conditions solved: compare dense progress on failing condition
        if wp1 > wp2 + progress_tolerance:
            return True
        if wp2 > wp1 + progress_tolerance:
            return False

        # Failing condition progress tied: compare mean_return and complexity
        perf1 = mp1
        perf2 = mp2
        at_least_as_good = (perf1 >= perf2) and (p1 <= p2) and (m1 <= m2)
        strictly_better = (perf1 > perf2) or (p1 < p2) or (m1 < m2)
        return at_least_as_good and strictly_better

    # 4.3: Worst-Condition Success Tied > 0.0 (both passed all conditions to level w1)
    perf1 = r1 if abs(r1 - r2) > 1e-6 else mp1
    perf2 = r2 if abs(r1 - r2) > 1e-6 else mp2
    at_least_as_good = (perf1 >= perf2) and (p1 <= p2) and (m1 <= m2)
    strictly_better = (perf1 > perf2) or (p1 < p2) or (m1 < m2)
    return at_least_as_good and strictly_better


def fast_non_dominated_sort_stage3d(population: List[MaskedGenome]) -> List[List[MaskedGenome]]:
    """Partition population into Pareto fronts using Stage 3D Robustness-First dominance."""
    fronts: List[List[MaskedGenome]] = [[]]
    sp: Dict[int, List[int]] = {i: [] for i in range(len(population))}
    np_dom_count = [0] * len(population)

    for p_idx, p in enumerate(population):
        for q_idx, q in enumerate(population):
            if p_idx == q_idx:
                continue
            if constrained_dominates_stage3d(p, q):
                sp[p_idx].append(q_idx)
            elif constrained_dominates_stage3d(q, p):
                np_dom_count[p_idx] += 1

        if np_dom_count[p_idx] == 0:
            p.rank = 0
            fronts[0].append(p)

    cur_front = 0
    id_to_idx = {id(ind): idx for idx, ind in enumerate(population)}

    while len(fronts[cur_front]) > 0:
        next_front: List[MaskedGenome] = []
        for p in fronts[cur_front]:
            p_idx = id_to_idx[id(p)]
            for q_idx in sp[p_idx]:
                np_dom_count[q_idx] -= 1
                if np_dom_count[q_idx] == 0:
                    q = population[q_idx]
                    q.rank = cur_front + 1
                    next_front.append(q)
        cur_front += 1
        fronts.append(next_front)

    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts


def compute_crowding_distance_stage3d(front: List[MaskedGenome]) -> None:
    """Calculate crowding distance along worst_success, robust_success, active_params, weight_macs."""
    n = len(front)
    if n == 0:
        return
    for ind in front:
        ind.crowding_distance = 0.0

    if n <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    objectives = ["worst_success", "robust_success", "worst_return", "active_params", "weight_macs"]

    for obj in objectives:
        front.sort(key=lambda ind: ind.fitness.get(obj, 0.0))
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        obj_min = front[0].fitness.get(obj, 0.0)
        obj_max = front[-1].fitness.get(obj, 0.0)
        denom = obj_max - obj_min

        if denom > 1e-9:
            for i in range(1, n - 1):
                front[i].crowding_distance += (front[i + 1].fitness.get(obj, 0.0) - front[i - 1].fitness.get(obj, 0.0)) / denom


def nsga2_survivor_selection_stage3d(population: List[MaskedGenome], pop_size: int) -> List[MaskedGenome]:
    """Select the top pop_size individuals using Stage 3D non-dominated sorting and crowding distance."""
    fronts = fast_non_dominated_sort_stage3d(population)
    survivors: List[MaskedGenome] = []

    for front in fronts:
        compute_crowding_distance_stage3d(front)
        if len(survivors) + len(front) <= pop_size:
            survivors.extend(front)
        else:
            front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
            needed = pop_size - len(survivors)
            survivors.extend(front[:needed])
            break

    return survivors


def evaluate_genome_stage3d(
    genome: MaskedGenome,
    eval_seeds: List[int],
    conditions: Optional[Dict[str, Dict[str, Any]]] = None,
    max_steps: int = 500,
    budget_tracker: Optional[TransitionBudgetTracker] = None,
    vectorized: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate candidate genome across Stage 3D domain-randomized conditions with transition tracking."""
    if conditions is None:
        conditions = STAGE3D_TRAIN_CONDITIONS

    num_envs = len(eval_seeds)
    cond_rates: Dict[str, float] = {}
    cond_returns: Dict[str, float] = {}

    if vectorized:
        from env.vectorized_env import VectorizedCartPoleEnv

        for cond_name, cond_params in conditions.items():
            if budget_tracker is not None and budget_tracker.is_exhausted:
                raise BudgetExhaustedError("Budget exhausted before starting condition evaluation.")

            vec_env = VectorizedCartPoleEnv(num_envs=num_envs, max_steps=max_steps, reward_mode="stage3b", **cond_params)
            bvec_env = BudgetedVectorizedEnv(vec_env, budget_tracker) if budget_tracker is not None else vec_env

            obs, _ = bvec_env.reset(seeds=eval_seeds)
            cos_thetas: List[np.ndarray] = []
            batch_returns = np.zeros(num_envs, dtype=np.float64)
            u_prev1 = np.zeros(num_envs, dtype=np.float64)
            u_prev2 = np.zeros(num_envs, dtype=np.float64)

            for step in range(max_steps):
                norm_o = norm_batch_obs(obs, u_prev1=u_prev1, u_prev2=u_prev2)
                actions = genome.forward_batch(norm_o)
                u_prev2 = u_prev1.copy()
                u_prev1 = actions.copy()
                obs, rewards, term, trunc, _ = bvec_env.step(actions)
                cos_thetas.append(obs[:, 2])
                batch_returns += np.asarray(rewards, dtype=np.float64)

            cos_arr = np.array(cos_thetas)
            final_window = cos_arr[-100:]
            upright_count = np.sum(final_window > 0.9, axis=0)
            boundary_violated = vec_env.boundary_violated if hasattr(vec_env, "boundary_violated") else bvec_env.env.boundary_violated
            success = (upright_count >= 95) & (~boundary_violated)
            rate = float(np.mean(success))
            cond_rates[cond_name] = rate
            cond_returns[cond_name] = float(np.mean(batch_returns))
    else:
        from utils.perturbation_loader import PerturbedCartPoleEnv
        from experiments.run_robustness_experiment import evaluate_success_predicate

        for cond_name, cond_params in conditions.items():
            cond_succ = 0
            ep_returns: List[float] = []
            for s in eval_seeds:
                if budget_tracker is not None and budget_tracker.is_exhausted:
                    raise BudgetExhaustedError("Budget exhausted before starting episode.")

                base_env = PerturbedCartPoleEnv(max_steps=max_steps, continuous=True, seed=s, reward_mode="stage3b", **cond_params)
                benv = BudgetedEnv(base_env, budget_tracker) if budget_tracker is not None else base_env
                obs, _ = benv.reset(seed=s)
                cos_th_list: List[float] = []
                rail_hit = False
                ep_r = 0.0
                genome.reset_history()

                for _ in range(max_steps):
                    action = genome.compute_action(obs)
                    obs, r, term, trunc, _ = benv.step(action)
                    ep_r += float(r)
                    cos_th_list.append(float(obs[2]))
                    if term:
                        rail_hit = True
                        break

                ep_returns.append(ep_r)
                is_s, _, _, _ = evaluate_success_predicate(cos_th_list, rail_hit, max_steps)
                if is_s:
                    cond_succ += 1

            cond_rates[cond_name] = cond_succ / max(1, len(eval_seeds))
            cond_returns[cond_name] = float(np.mean(ep_returns)) if ep_returns else 0.0

    robust_success = float(np.mean(list(cond_rates.values())))
    worst_success = float(np.min(list(cond_rates.values())))
    mean_return = float(np.mean(list(cond_returns.values())))
    worst_return = float(np.min(list(cond_returns.values())))
    task_progress = float(1000.0 * robust_success + 2000.0 * worst_success + 0.1 * worst_return + 0.01 * mean_return)
    comp = genome.get_complexity_metrics()

    genome.fitness = {
        "robust_success": robust_success,
        "worst_success": worst_success,
        "worst_return": worst_return,
        "mean_return": mean_return,
        "task_progress": task_progress,
        "active_params": comp["active_parameters"],
        "weight_macs": comp["weight_macs"],
        "active_neurons": comp["active_neurons"],
    }
    genome.constraint_violation = get_complexity_violation(genome)
    return robust_success, worst_success, cond_rates


def select_stage3d_champion(candidates: List[MaskedGenome]) -> MaskedGenome:
    """Select the development champion from Pareto candidates under Stage 3D acceptance rules.

    Constraints: active_parameters <= 128, weight_macs <= 128.
    Lexicographic Priority:
    1. worst_condition success rate (maximize)
    2. robust success rate (maximize)
    3. worst_return (maximize)
    4. mean_return (maximize)
    5. active_parameters (minimize)
    6. weight_macs (minimize)
    """
    feasible = [
        g for g in candidates
        if g.get_complexity_metrics()["active_parameters"] <= 128
        and g.get_complexity_metrics()["weight_macs"] <= 128
    ]
    pool = feasible if feasible else candidates
    return max(
        pool,
        key=lambda g: (
            g.fitness.get("worst_success", 0.0),
            g.fitness.get("robust_success", 0.0),
            g.fitness.get("worst_return", 0.0),
            g.fitness.get("mean_return", 0.0),
            -g.fitness.get("active_params", 0.0),
            -g.fitness.get("weight_macs", 0.0),
        )
    )


def train_nsga2_stage3d(
    optimizer_seed: int,
    pop_size: int = 20,
    offspring_size: int = 10,
    generations: int = 1000,
    total_budget: int = 2_000_000,
    search_budget: int = 1_850_000,
    validation_budget: int = 150_000,
    budget_tracker: Optional[TransitionBudgetTracker] = None,
    train_seeds: Optional[List[int]] = None,
    conditions: Optional[Dict[str, Dict[str, Any]]] = None,
    vectorized: bool = True,
    warm_start_path: Optional[str] = "results/stage3c/models/stage3c_nsga2_delayaware_seed44.pkl",
    in_dim: int = 7,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute Stage 3D Robustness-First Neuroevolution search."""
    import time
    t0 = time.time()

    if budget_tracker is None:
        budget_tracker = TransitionBudgetTracker(
            total_budget=total_budget,
            search_budget=search_budget,
            validation_budget=validation_budget,
        )

    rng, seed_seq = get_nsga2_rng(optimizer_seed)

    if train_seeds is None:
        train_seeds = list(range(0, 200))

    if conditions is None:
        conditions = STAGE3D_TRAIN_CONDITIONS

    # Load warm-start champion from Stage 3C
    if warm_start_path is None or not os.path.exists(warm_start_path):
        warm_start_path = os.path.join(_PROJECT_ROOT, "results", "stage3c", "models", "stage3c_nsga2_delayaware_seed44.pkl")

    with open(warm_start_path, "rb") as f:
        warm_data = pickle.load(f)
        if isinstance(warm_data, dict):
            g_champ = MaskedGenome.from_dict(warm_data, rng=rng)
        else:
            g_champ = warm_data
            g_champ.rng = rng

    # Construct balanced initial population
    population: List[MaskedGenome] = []
    # 1. Exact champion
    c0 = g_champ.copy()
    c0.rng = rng
    population.append(c0)

    # 2. 8 fine mutants for subtle weight and centering adjustments
    for _ in range(8):
        c = g_champ.copy()
        c.rng = rng
        c.mutate(weight_mutation_rate=0.4, weight_mutation_power=0.03, connection_toggle_rate=0.02, neuron_toggle_rate=0.0)
        population.append(c)

    # 3. 7 moderate mutants for authority on high mass and friction
    for _ in range(7):
        c = g_champ.copy()
        c.rng = rng
        c.mutate(weight_mutation_rate=0.6, weight_mutation_power=0.08, connection_toggle_rate=0.08, neuron_toggle_rate=0.0)
        population.append(c)

    # 4. 4 exploratory mutants to avoid premature convergence
    for _ in range(4):
        c = g_champ.copy()
        c.rng = rng
        c.mutate(weight_mutation_rate=0.8, weight_mutation_power=0.20, connection_toggle_rate=0.15, neuron_toggle_rate=0.05)
        population.append(c)

    stopped_by_budget = False
    stopping_reason = "max_generations_reached"
    completed_generations = 0
    candidate_evaluations = 0

    num_conds = len(conditions)
    # Use 2 rotating training seeds per condition
    cond_seeds_count = 2

    try:
        # Initial Population Evaluation
        init_seeds = [
            train_seeds[(optimizer_seed + i) % len(train_seeds)]
            for i in range(cond_seeds_count)
        ]
        for ind in population:
            if budget_tracker.is_exhausted or budget_tracker.remaining_transitions < 1:
                stopped_by_budget = True
                stopping_reason = "search_budget_exhausted"
                break
            evaluate_genome_stage3d(
                ind,
                eval_seeds=init_seeds,
                conditions=conditions,
                max_steps=500,
                budget_tracker=budget_tracker,
                vectorized=vectorized,
            )
            candidate_evaluations += 1

        # Generational Search Loop
        for gen in range(1, generations + 1):
            if budget_tracker.is_exhausted or budget_tracker.remaining_transitions < 1:
                stopped_by_budget = True
                stopping_reason = "search_budget_exhausted"
                break

            # Survivor selection
            survivors = nsga2_survivor_selection_stage3d(population, pop_size // 2)

            # Produce offspring
            offspring: List[MaskedGenome] = []
            for p in survivors[:offspring_size]:
                child = p.copy()
                child.rng = rng
                child.mutate(
                    weight_mutation_rate=0.4,
                    weight_mutation_power=0.05,
                    connection_toggle_rate=0.05,
                    neuron_toggle_rate=0.0,
                )
                offspring.append(child)

            gen_seeds = [
                train_seeds[(optimizer_seed + gen * 37 + i) % len(train_seeds)]
                for i in range(cond_seeds_count)
            ]

            aborted_generation = False
            for child in offspring:
                if budget_tracker.is_exhausted or budget_tracker.remaining_transitions < 1:
                    stopped_by_budget = True
                    stopping_reason = "search_budget_exhausted"
                    aborted_generation = True
                    break
                evaluate_genome_stage3d(
                    child,
                    eval_seeds=gen_seeds,
                    conditions=conditions,
                    max_steps=500,
                    budget_tracker=budget_tracker,
                    vectorized=vectorized,
                )
                candidate_evaluations += 1

            if aborted_generation or len(offspring) == 0:
                break

            population = survivors + offspring
            completed_generations = gen

            if verbose and (gen % 5 == 0 or gen == 1):
                best_cand = select_stage3d_champion(population)
                w_succ = best_cand.fitness.get("worst_success", 0.0) * 100
                r_succ = best_cand.fitness.get("robust_success", 0.0) * 100
                w_ret = best_cand.fitness.get("worst_return", 0.0)
                print(f"[Stage 3D Gen {gen:3d}] Worst Succ: {w_succ:5.1f}% | Robust: {r_succ:5.1f}% | Worst Ret: {w_ret:8.1f} | Consumed: {budget_tracker.consumed_transitions:,} / {budget_tracker.search_budget:,}")

    except BudgetExhaustedError:
        stopped_by_budget = True
        stopping_reason = "search_budget_exhausted"

    fronts = fast_non_dominated_sort_stage3d(population)
    pareto_front = fronts[0] if fronts else []

    wall_clock_time = time.time() - t0
    spawn_key = (NAMESPACE_NSGA2, optimizer_seed)

    return {
        "pareto_front": pareto_front,
        "population": population,
        "tracker": budget_tracker,
        "completed_generations": completed_generations,
        "stopped_by_budget": stopped_by_budget,
        "stopping_reason": stopping_reason,
        "actual_transitions": budget_tracker.consumed_transitions,
        "search_transitions": budget_tracker.search_consumed,
        "candidate_evaluations": candidate_evaluations,
        "wall_clock_time": wall_clock_time,
        "optimizer_seed": optimizer_seed,
        "spawn_key": spawn_key,
        "namespace": NAMESPACE_NSGA2,
    }

    import argparse

    parser = argparse.ArgumentParser(description="Stage 2 NSGA-II Neuroevolution Trainer")
    parser.add_argument("--seed", type=int, default=42, help="Optimizer seed (default: 42)")
    parser.add_argument("--generations", type=int, default=35, help="Number of generations (default: 35)")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size (default: 20)")
    parser.add_argument("--offspring-size", type=int, default=10, help="Offspring size (default: 10)")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes per eval (default: 10)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode (default: 500)")
    parser.add_argument("--total-budget", type=int, default=2_000_000, help="Total lifecycle budget limit (default: 2000000)")
    parser.add_argument("--search-budget", type=int, default=1_850_000, help="Search phase budget limit (default: 1850000)")
    parser.add_argument("--validation-budget", type=int, default=150_000, help="Validation phase budget limit (default: 150000)")
    parser.add_argument("--budget-limit", type=int, default=None, help="Legacy alias: sets search budget ceiling")
    parser.add_argument("--run-validation", action="store_true", help="Execute post-search validation screening")
    parser.add_argument("--output-dir", type=str, default="results/stage2/models", help="Directory to save model checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Print verbose progress")
    args = parser.parse_args()

    s_budget = args.budget_limit if args.budget_limit is not None else args.search_budget
    t_budget = max(s_budget + args.validation_budget, args.total_budget) if args.budget_limit is not None else args.total_budget

    res = train_nsga2(
        optimizer_seed=args.seed,
        pop_size=args.pop_size,
        offspring_size=args.offspring_size,
        generations=args.generations,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        total_budget=t_budget,
        search_budget=s_budget,
        validation_budget=args.validation_budget,
        vectorized=True,
        verbose=args.verbose,
    )

    val_res = None
    if args.run_validation and res["pareto_front"]:
        # Select champion candidate from Pareto front using proposed selection rule
        # Filter for feasibility: params <= 128, macs <= 128
        feasible = [
            g for g in res["pareto_front"]
            if g.get_complexity_metrics()["active_parameters"] <= 128
            and g.get_complexity_metrics()["weight_macs"] <= 128
        ]
        pool = feasible if feasible else res["pareto_front"]
        val_res = validate_nsga2_champion(champ, res["tracker"], max_steps=args.max_steps, vectorized=True, verbose=args.verbose)
        val_res["search_transitions_used"] = res["tracker"].search_consumed
        val_res["champion_complexity"] = champ.get_complexity_metrics()
        val_res["champion_fitness"] = champ.fitness
        val_res["champion_genome"] = champ.to_dict()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        save_file = os.path.join(args.output_dir, f"nsga2_pareto_seed{args.seed}.json")
        with open(save_file, "w") as f:
            json.dump([ind.to_dict() for ind in res["pareto_front"]], f, indent=2)
        print(f"Saved NSGA-II Pareto front to {save_file} (Consumed transitions: {res['tracker'].consumed_transitions:,})")
        if val_res:
            meta_file = os.path.join(args.output_dir, f"nsga2_champion_seed{args.seed}_validation.json")
            with open(meta_file, "w") as f:
                json.dump(val_res, f, indent=2)
            print(f"Saved validation metadata to {meta_file}")


