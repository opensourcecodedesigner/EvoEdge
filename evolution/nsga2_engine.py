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
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


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
        self.fitness: Dict[str, float] = {}
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.constraint_violation: float = 0.0  # 0 if feasible, >0 if infeasible

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
        return clone

    def load_from_dense_weights(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> None:
        """Warm-start the genome from pre-existing dense weights (e.g. Frozen H=8)."""
        H = w1.shape[0]
        assert H <= self.max_hidden
        self.w1[:H, :] = w1.astype(np.float64)
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


def constrained_dominates(ind1: MaskedGenome, ind2: MaskedGenome) -> bool:
    """Deb-style constrained domination comparator:

    1. A feasible candidate always dominates an infeasible candidate.
    2. Between infeasible candidates, lower constraint violation wins.
    3. Between feasible candidates, Pareto dominance over objectives:
       - Objective 1: Maximize robust_success (higher is better)
       - Objective 2: Minimize active_parameters (lower is better)
       - Objective 3: Minimize weight_macs (lower is better)
    """
    feas1 = (ind1.constraint_violation <= 0.0)
    feas2 = (ind2.constraint_violation <= 0.0)

    if feas1 and not feas2:
        return True
    if not feas1 and feas2:
        return False
    if not feas1 and not feas2:
        return ind1.constraint_violation < ind2.constraint_violation

    # Both are feasible: evaluate Pareto dominance
    # Higher robust success, lower active parameters, lower weight MACs
    succ1, succ2 = ind1.fitness["robust_success"], ind2.fitness["robust_success"]
    p1, p2 = ind1.fitness["active_params"], ind2.fitness["active_params"]
    m1, m2 = ind1.fitness["weight_macs"], ind2.fitness["weight_macs"]

    at_least_as_good = (succ1 >= succ2) and (p1 <= p2) and (m1 <= m2)
    strictly_better = (succ1 > succ2) or (p1 < p2) or (m1 < m2)

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

    # Objectives: robust_success (max), active_params (min), weight_macs (min)
    objectives = ["robust_success", "active_params", "weight_macs"]

    for obj in objectives:
        front.sort(key=lambda ind: ind.fitness[obj])
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        obj_min = front[0].fitness[obj]
        obj_max = front[-1].fitness[obj]
        denom = obj_max - obj_min

        if denom > 1e-9:
            for i in range(1, n - 1):
                front[i].crowding_distance += (front[i + 1].fitness[obj] - front[i - 1].fitness[obj]) / denom


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
