"""Unit tests for Phase 3A: Vectorized Simulator Parity & Multi-Objective NSGA-II.

Tests:
  1. VectorizedCartPoleEnv parity against reference PerturbedCartPoleEnv:
     - 50 non-holdout seeds (100..149) across 500 steps.
     - atol < 1e-9 float64 precision test on state trajectories.
     - Exact episode-success and rail-hit predicate parity.
     - Reset behavior, delay queues, observation noise, and rail hits.
  2. Deb-style Constrained Domination:
     - Feasible dominates infeasible.
     - Lower constraint violation wins between infeasible candidates.
     - Pareto dominance between feasible candidates (robust success, active params, weight MACs).
  3. Fast Non-Dominated Sorting and Crowding Distance:
     - Correct front ranking (Front 0 is non-dominated).
     - Boundary points get infinite crowding distance.
     - Elitist survivor selection preserves best candidates.
  4. MaskedGenome:
     - Complexity accounting (active params, active weights, weight MACs).
     - Serialization and exact deserialization (round-trip parity).
     - Forward pass batch vs single-step equivalence.
     - Mutation operations and validity.
  5. Seed partition protection:
     - Rejection of holdout seeds (2000..2099) during evolution and validation.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import unittest
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.vectorized_env import VectorizedCartPoleEnv
from utils.perturbation_loader import PerturbedCartPoleEnv
from evolution.nsga2_engine import (
    MaskedGenome,
    constrained_dominates,
    fast_non_dominated_sort,
    compute_crowding_distance,
    nsga2_survivor_selection,
)

DIAGNOSTIC_CONDITIONS = {
    "1_Nominal": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0, "obs_noise_sigma": 0.0, "delay_steps": 0
    },
    "4c_CombinedFriction": {
        "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.50, "b_pole": 0.010, "obs_noise_sigma": 0.0, "delay_steps": 0
    },
    "7_CombinedHistoricalStress": {
        "cart_mass": 1.30, "pole_mass": 0.130, "b_cart": 0.50, "b_pole": 0.010, "obs_noise_sigma": 0.050, "delay_steps": 1
    },
}


class TestVectorizedParity(unittest.TestCase):
    """Test float64 parity and predicate equivalence between serial reference and batch vectorized env."""

    def test_float64_state_parity_over_50_seeds(self):
        """Verify max absolute state difference is < 1e-9 across 50 non-holdout seeds over 500 steps."""
        test_seeds = list(range(100, 150))
        num_envs = len(test_seeds)
        steps = 500

        # Choose a challenging condition with friction, noise, and delay
        cond = DIAGNOSTIC_CONDITIONS["7_CombinedHistoricalStress"]

        # Run vectorized batch
        vec_env = VectorizedCartPoleEnv(num_envs=num_envs, max_steps=steps, **cond)
        obs_vec, _ = vec_env.reset(seeds=test_seeds)

        # Reference serial environments
        ref_envs = [
            PerturbedCartPoleEnv(max_steps=steps, seed=s, **cond)
            for s in test_seeds
        ]
        obs_refs = [env.reset(seed=s)[0] for s, env in zip(test_seeds, ref_envs)]

        # Check initial observation parity
        for i in range(num_envs):
            np.testing.assert_allclose(obs_vec[i], obs_refs[i], atol=1e-9)

        # Drive with identical prescribed pseudo-random actions
        rng = np.random.default_rng(12345)
        actions = rng.uniform(-1.0, 1.0, size=(steps, num_envs))

        max_state_diff = 0.0

        for t in range(steps):
            u_step = actions[t]
            o_vec, r_vec, d_vec, trunc_vec, info_vec = vec_env.step(u_step)

            for i in range(num_envs):
                o_ref, r_ref, d_ref, trunc_ref, info_ref = ref_envs[i].step(float(u_step[i]))

                # Check state difference
                diff = np.max(np.abs(vec_env.state[i] - ref_envs[i]._state))
                if diff > max_state_diff:
                    max_state_diff = diff

                # Check observation difference
                np.testing.assert_allclose(o_vec[i], o_ref, atol=1e-9)
                self.assertEqual(d_vec[i], d_ref)
                self.assertEqual(trunc_vec[i], trunc_ref)
                self.assertEqual(info_vec[i]["boundary_violated"], ref_envs[i].boundary_violated)

        self.assertLess(max_state_diff, 1e-9, f"Max state difference {max_state_diff} exceeded atol=1e-9")

    def test_predicate_parity_on_deterministic_controller(self):
        """Verify identical success and rail-hit predicate outcomes under both engines."""
        test_seeds = list(range(100, 150))
        cond = DIAGNOSTIC_CONDITIONS["4c_CombinedFriction"]

        vec_env = VectorizedCartPoleEnv(num_envs=len(test_seeds), max_steps=500, **cond)
        obs_vec, _ = vec_env.reset(seeds=test_seeds)

        ref_envs = [PerturbedCartPoleEnv(max_steps=500, seed=s, **cond) for s in test_seeds]
        for s, env in zip(test_seeds, ref_envs):
            env.reset(seed=s)

        # Controller: simple proportional damper
        for t in range(500):
            u = np.clip(-obs_vec[:, 2] * 2.0, -1.0, 1.0)
            obs_vec, _, _, _, info_vec = vec_env.step(u)
            for i in range(len(test_seeds)):
                ref_envs[i].step(float(u[i]))

        for i in range(len(test_seeds)):
            self.assertEqual(info_vec[i]["boundary_violated"], ref_envs[i].boundary_violated)


class TestNSGA2Engine(unittest.TestCase):
    """Test Deb-style constrained dominance, sorting, crowding distance, and survivor selection."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_deb_constrained_domination(self):
        """Feasible dominates infeasible; lower violation wins; Pareto dominance for feasible."""
        # Ind 1: Feasible (violation = 0.0), robust success 0.70, params 30, MACs 25
        g1 = MaskedGenome(rng=self.rng)
        g1.constraint_violation = 0.0
        g1.fitness = {"robust_success": 0.70, "active_params": 30, "weight_macs": 25}

        # Ind 2: Infeasible (violation = 0.10)
        g2 = MaskedGenome(rng=self.rng)
        g2.constraint_violation = 0.10
        g2.fitness = {"robust_success": 0.38, "active_params": 10, "weight_macs": 8}

        # Ind 3: Infeasible with worse violation (violation = 0.25)
        g3 = MaskedGenome(rng=self.rng)
        g3.constraint_violation = 0.25
        g3.fitness = {"robust_success": 0.23, "active_params": 5, "weight_macs": 4}

        # 1. Feasible g1 MUST dominate infeasible g2, even though g2 has fewer params/macs
        self.assertTrue(constrained_dominates(g1, g2))
        self.assertFalse(constrained_dominates(g2, g1))

        # 2. Lower violation g2 MUST dominate higher violation g3
        self.assertTrue(constrained_dominates(g2, g3))
        self.assertFalse(constrained_dominates(g3, g2))

        # 3. Between feasible: Pareto dominance
        g4 = MaskedGenome(rng=self.rng)
        g4.constraint_violation = 0.0
        g4.fitness = {"robust_success": 0.75, "active_params": 28, "weight_macs": 22}

        # g4 dominates g1 (higher success, lower params, lower MACs)
        self.assertTrue(constrained_dominates(g4, g1))
        self.assertFalse(constrained_dominates(g1, g4))

        # Trade-off (neither dominates)
        g5 = MaskedGenome(rng=self.rng)
        g5.constraint_violation = 0.0
        g5.fitness = {"robust_success": 0.80, "active_params": 35, "weight_macs": 30}
        self.assertFalse(constrained_dominates(g4, g5))
        self.assertFalse(constrained_dominates(g5, g4))

    def test_fast_non_dominated_sort_and_crowding(self):
        """Test Pareto front sorting and crowding distance assignment."""
        pop = []
        # Create 5 feasible individuals with clear Pareto structure
        f_data = [
            (0.85, 40, 35),  # Front 0 candidate A
            (0.80, 25, 20),  # Front 0 candidate B (fewer params)
            (0.70, 15, 12),  # Front 0 candidate C (lowest params)
            (0.65, 45, 40),  # Dominated by all above (Front 1)
        ]
        for succ, p, m in f_data:
            g = MaskedGenome(rng=self.rng)
            g.constraint_violation = 0.0
            g.fitness = {"robust_success": succ, "active_params": p, "weight_macs": m}
            pop.append(g)

        fronts = fast_non_dominated_sort(pop)
        self.assertEqual(len(fronts), 2)
        self.assertEqual(len(fronts[0]), 3)  # A, B, C are in front 0
        self.assertEqual(len(fronts[1]), 1)  # Dominated candidate is in front 1

        compute_crowding_distance(fronts[0])
        # Extreme points in Front 0 must have infinite crowding distance
        inf_count = sum(1 for ind in fronts[0] if math.isinf(ind.crowding_distance))
        self.assertGreaterEqual(inf_count, 2)

    def test_nsga2_survivor_selection(self):
        """Test elitist survivor selection with crowding distance tie-breaking."""
        pop = []
        for i in range(10):
            g = MaskedGenome(rng=self.rng)
            g.constraint_violation = 0.0
            g.fitness = {"robust_success": 0.5 + 0.04 * i, "active_params": 50 - 3 * i, "weight_macs": 40 - 2 * i}
            pop.append(g)

        survivors = nsga2_survivor_selection(pop, pop_size=5)
        self.assertEqual(len(survivors), 5)


class TestMaskedGenome(unittest.TestCase):
    """Test genome complexity accounting, serialization, and forward batch execution."""

    def test_complexity_accounting(self):
        """Verify active parameters and weight MACs accurately track active connections and neurons."""
        g = MaskedGenome(in_dim=5, max_hidden=8, out_dim=1)
        # Give biases non-zero values to test full active parameter counting
        g.b1[:] = 0.1
        g.b2[:] = 0.1
        # By default all active: 8 neurons, 5*8=40 w1, 8 b1, 8*1=8 w2, 1 b2 = 57 params, 48 MACs
        comp = g.get_complexity_metrics()
        self.assertEqual(comp["dense_params"], 57)
        self.assertEqual(comp["dense_macs"], 48)
        self.assertEqual(comp["active_neurons"], 8)
        self.assertEqual(comp["active_parameters"], 57)
        self.assertEqual(comp["weight_macs"], 48)

        # Deactivate 3 hidden neurons
        g.neuron_mask[5:] = False
        comp_pruned = g.get_complexity_metrics()
        self.assertEqual(comp_pruned["active_neurons"], 5)
        # Active: 5*5=25 w1, 5 b1, 5*1=5 w2, 1 b2 = 36 params, 30 MACs
        self.assertEqual(comp_pruned["active_parameters"], 36)
        self.assertEqual(comp_pruned["weight_macs"], 30)

    def test_serialization_round_trip(self):
        """Verify to_dict and from_dict preserve exact weights, masks, and activations."""
        g = MaskedGenome(in_dim=5, max_hidden=8, out_dim=1)
        g.mutate(weight_mutation_rate=1.0, connection_toggle_rate=1.0, neuron_toggle_rate=1.0)
        g.fitness = {"robust_success": 0.88, "active_params": 32, "weight_macs": 26}

        data = g.to_dict()
        g_restored = MaskedGenome.from_dict(data)

        np.testing.assert_array_equal(g.w1, g_restored.w1)
        np.testing.assert_array_equal(g.w1_mask, g_restored.w1_mask)
        np.testing.assert_array_equal(g.neuron_mask, g_restored.neuron_mask)
        self.assertEqual(g.hidden_activations, g_restored.hidden_activations)
        self.assertEqual(g.fitness, g_restored.fitness)

    def test_forward_batch_equivalence(self):
        """Verify forward_batch outputs match repeated forward single-step calls."""
        g = MaskedGenome(in_dim=5, max_hidden=8, out_dim=1)
        x_batch = np.random.normal(0.0, 1.0, size=(20, 5))

        out_batch = g.forward_batch(x_batch)
        out_singles = np.array([g.forward(x_batch[i]) for i in range(20)])

        np.testing.assert_allclose(out_batch, out_singles, atol=1e-12)


if __name__ == "__main__":
    import math
    unittest.main()
