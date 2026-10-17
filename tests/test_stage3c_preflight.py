"""Stage 3C Delay-Aware Neuroevolution Preflight & Regression Test Suite.

Verifies:
  1. Delay-aware 7-dimensional observation interface and state ordering.
  2. Simulator transport delay queue alignment (0, 1, 2 steps).
  3. Teacher warm-start initialization (zeroed delay weights matching 5D teacher).
  4. Genome serialization (to_dict/from_dict) and deterministic inference.
  5. Deb-style constrained dominance with delay-sensitive candidate fixtures.
  6. Transition budget tracker enforcement and zero overshoot.
  7. Strict partition isolation safety (rejection of seeds 600..749 and 2000..2099).
  8. Bounded preflight optimization on randomized training distribution.
"""

from __future__ import annotations

import collections
import copy
import os
import pickle
import sys
from typing import Dict, List
import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.budget_tracker import TransitionBudgetTracker, BudgetExhaustedError
from env.cartpole_swingup import CartPoleSwingUpEnv
from env.vectorized_env import VectorizedCartPoleEnv
from evolution.nsga2_engine import (
    MaskedGenome,
    STAGE3C_TRAIN_CONDITIONS,
    constrained_dominates,
    evaluate_genome_budgeted,
    get_nsga2_rng,
    norm_batch_obs,
    norm_obs,
    train_nsga2,
    validate_nsga2_champion,
)
from utils.perturbation_loader import PerturbedCartPoleEnv


def test_delay_aware_observation_and_genome_interface():
    """Verify 7D observation interface: 5 physical + 2 action history states."""
    g = MaskedGenome(in_dim=7, max_hidden=8, out_dim=1)
    assert g.in_dim == 7
    assert g.max_hidden == 8
    assert g.out_dim == 1
    assert g.w1.shape == (8, 7)
    assert g.w1_mask.shape == (8, 7)

    comp = g.get_complexity_metrics()
    assert comp["active_parameters"] <= 128
    assert comp["weight_macs"] <= 128

    # Test normalization: single observation
    raw_5d = np.array([1.2, 5.0, 0.8, 0.6, -2.5], dtype=np.float64)
    norm_5d = norm_obs(raw_5d, include_delay=False)
    assert norm_5d.shape == (5,)
    assert np.isclose(norm_5d[0], 1.2 / 2.4)
    assert np.isclose(norm_5d[1], 5.0 / 10.0)
    assert np.isclose(norm_5d[2], 0.8)
    assert np.isclose(norm_5d[3], 0.6)
    assert np.isclose(norm_5d[4], -2.5 / 10.0)

    # 7D normalization with action history
    norm_7d = norm_obs(raw_5d, u_prev1=0.75, u_prev2=-0.50, include_delay=True)
    assert norm_7d.shape == (7,)
    assert np.allclose(norm_7d[:5], norm_5d)
    assert np.isclose(norm_7d[5], 0.75)
    assert np.isclose(norm_7d[6], -0.50)

    # Batch normalization
    raw_batch = np.array([
        [1.2, 5.0, 0.8, 0.6, -2.5],
        [-0.6, -3.0, 0.0, 1.0, 1.5],
    ], dtype=np.float64)
    u1_batch = np.array([0.75, -0.25])
    u2_batch = np.array([-0.50, 0.10])
    norm_b = norm_batch_obs(raw_batch, u_prev1=u1_batch, u_prev2=u2_batch)
    assert norm_b.shape == (2, 7)
    assert np.isclose(norm_b[0, 5], 0.75)
    assert np.isclose(norm_b[0, 6], -0.50)
    assert np.isclose(norm_b[1, 5], -0.25)
    assert np.isclose(norm_b[1, 6], 0.10)


def test_action_history_and_delay_queue_semantics():
    """Verify that controller history tracking exactly mirrors simulator delay queues."""
    # Test 1-step delay queue
    env_1step = PerturbedCartPoleEnv(delay_steps=1, seed=42)
    obs, _ = env_1step.reset(seed=42)

    # Simulator queue initially contains [0.0]
    assert len(env_1step.action_queue) == 1
    assert env_1step.action_queue[0] == 0.0

    # At step 0: controller emits action a0 = 0.6
    obs, _, _, _, info = env_1step.step(0.6)
    # Applied action was 0.0 (popped), pending action in queue is 0.6
    assert env_1step.action_queue[0] == 0.6

    # At step 1: controller emits action a1 = -0.4
    obs, _, _, _, info = env_1step.step(-0.4)
    # Applied action was 0.6, pending action in queue is -0.4
    assert env_1step.action_queue[0] == -0.4

    # Verify controller compute_action tracks this exact sequence:
    g = MaskedGenome(in_dim=7)
    g.reset_history()
    assert g._u_prev1 == 0.0
    assert g._u_prev2 == 0.0

    a0 = g.compute_action(obs)
    assert g._u_prev1 == a0
    assert g._u_prev2 == 0.0

    a1 = g.compute_action(obs)
    assert g._u_prev1 == a1
    assert g._u_prev2 == a0


def test_two_step_delay_queue_semantics():
    """Verify 2-step delay queue requires both u_{t-1} and u_{t-2}."""
    env_2step = PerturbedCartPoleEnv(delay_steps=2, seed=42)
    obs, _ = env_2step.reset(seed=42)

    # Initially queue has [0.0, 0.0]
    assert len(env_2step.action_queue) == 2
    assert env_2step.action_queue[0] == 0.0
    assert env_2step.action_queue[1] == 0.0

    # Step 0: emit a0 = 0.5. Applied = 0.0. Queue = [0.0, 0.5]
    env_2step.step(0.5)
    assert list(env_2step.action_queue) == [0.0, 0.5]

    # Step 1: emit a1 = 0.8. Applied = 0.0. Queue = [0.5, 0.8]
    env_2step.step(0.8)
    assert list(env_2step.action_queue) == [0.5, 0.8]

    # Step 2: emit a2 = -0.3. Applied = 0.5 (which was a0). Queue = [0.8, -0.3]
    env_2step.step(-0.3)
    assert list(env_2step.action_queue) == [0.8, -0.3]


def test_teacher_warm_start_zero_delay_initialization():
    """Verify loading 5D teacher into 7D genome zeroes delay weights and matches output."""
    teacher_path = os.path.join(_PROJECT_ROOT, "results", "champion_neural_h8.pkl")
    if not os.path.exists(teacher_path):
        pytest.skip("champion_neural_h8.pkl not found")

    with open(teacher_path, "rb") as f:
        t_data = pickle.load(f)

    w1 = t_data["w1"] if isinstance(t_data, dict) else t_data.w1
    b1 = t_data["b1"] if isinstance(t_data, dict) else t_data.b1
    w2 = t_data["w2"] if isinstance(t_data, dict) else t_data.w2
    b2 = t_data["b2"] if isinstance(t_data, dict) else t_data.b2

    g5 = MaskedGenome(in_dim=5, max_hidden=8, out_dim=1)
    g5.load_from_dense_weights(w1, b1, w2, b2)

    g7 = MaskedGenome(in_dim=7, max_hidden=8, out_dim=1)
    g7.load_from_dense_weights(w1, b1, w2, b2)

    # Weights for input 5 and 6 must be 0.0
    assert np.all(g7.w1[:, 5:] == 0.0)

    # For any 5D observation (with u_prev1=0, u_prev2=0), outputs must match exactly
    rng = np.random.default_rng(123)
    for _ in range(20):
        x5 = rng.uniform(-1.0, 1.0, size=5)
        out5 = g5.forward(x5)
        x7 = np.concatenate([x5, [0.0, 0.0]])
        out7 = g7.forward(x7)
        assert np.isclose(out5, out7, atol=1e-12)


def test_serialization_and_deterministic_inference():
    """Verify genome serialization to/from dictionary and deterministic cloning."""
    rng, _ = get_nsga2_rng(42)
    g = MaskedGenome(in_dim=7, max_hidden=8, out_dim=1, rng=rng)
    g.mutate(weight_mutation_rate=1.0, connection_toggle_rate=0.5)

    g_dict = g.to_dict()
    g_restored = MaskedGenome.from_dict(g_dict)

    assert g_restored.in_dim == 7
    assert np.allclose(g.w1, g_restored.w1)
    assert np.all(g.w1_mask == g_restored.w1_mask)
    assert np.all(g.neuron_mask == g_restored.neuron_mask)

    x = np.random.default_rng(99).normal(size=7)
    assert np.isclose(g.forward(x), g_restored.forward(x))


def test_deb_constrained_dominance_fixtures():
    """Verify Deb constrained dominance logic for delay-sensitive candidates."""
    g_feasible = MaskedGenome(in_dim=7)
    g_feasible.fitness = {
        "robust_success": 0.85,
        "worst_success": 0.50,
        "task_progress": 1100.0,
        "active_params": 55,
        "weight_macs": 48,
    }
    g_feasible.constraint_violation = 0.0

    g_infeasible = MaskedGenome(in_dim=7)
    g_infeasible.fitness = {
        "robust_success": 0.20,
        "worst_success": 0.00,
        "task_progress": 250.0,
        "active_params": 20,
        "weight_macs": 15,
    }
    g_infeasible.constraint_violation = 0.20  # Violates success floor (0.40 - 0.20 = 0.20)

    # Feasible candidate MUST dominate infeasible candidate regardless of lower complexity
    assert constrained_dominates(g_feasible, g_infeasible)
    assert not constrained_dominates(g_infeasible, g_feasible)


def test_partition_isolation_safety():
    """Verify fail-closed rejection of quarantined seeds (600..749) and untouched seeds (2000..2099)."""
    with pytest.raises(ValueError, match="SAFETY VIOLATION"):
        from experiments.run_stage3b_recovery import assert_partition_safety
        assert_partition_safety([650], "Test Context")

    with pytest.raises(ValueError, match="SAFETY VIOLATION"):
        from experiments.run_stage3b_recovery import assert_partition_safety
        assert_partition_safety([2050], "Test Context")


def test_transition_tracker_budget_enforcement():
    """Verify transition budget tracker strictly raises BudgetExhaustedError and limits usage."""
    tracker = TransitionBudgetTracker(budget_limit=1000, search_budget=1000, validation_budget=0)
    tracker.record_transitions(500)
    assert tracker.consumed_transitions == 500
    assert tracker.remaining_transitions == 500
    assert not tracker.is_exhausted

    # Reaching budget ceiling strictly triggers BudgetExhaustedError
    with pytest.raises(BudgetExhaustedError):
        tracker.record_transitions(500)
    assert tracker.consumed_transitions == 1000
    assert tracker.is_exhausted


def test_bounded_training_preflight():
    """Run a small bounded preflight (2 generations, 50,000 transitions) on seeds 0..199 only."""
    tracker = TransitionBudgetTracker(
        budget_limit=50_000,
        search_budget=40_000,
        validation_budget=10_000,
    )
    res = train_nsga2(
        optimizer_seed=42,
        pop_size=6,
        offspring_size=4,
        generations=2,
        eval_episodes=5,
        max_steps=500,
        budget_tracker=tracker,
        search_budget=40_000,
        validation_budget=10_000,
        conditions=STAGE3C_TRAIN_CONDITIONS,
        in_dim=7,
        vectorized=True,
        verbose=False,
    )

    assert tracker.search_consumed <= 40_000
    assert tracker.consumed_transitions <= 40_000
    assert len(res["pareto_front"]) > 0
    champ = res["pareto_front"][0]
    assert champ.in_dim == 7
    assert champ.get_complexity_metrics()["active_parameters"] <= 128
    assert champ.get_complexity_metrics()["weight_macs"] <= 128
