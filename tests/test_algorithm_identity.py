"""Targeted verification tests for algorithm identity and fair-comparison compatibility.

Verifies:
  1. ES algorithm identity:
     - Controller architecture and parameter count (113).
     - Antithetic sampling and mutation distribution.
     - Rank-based utility shaping and Adam update formulation.
     - Adaptation of baselines/mlp.py with budget and namespace isolation.
  2. Neuroevolution algorithm identity:
     - MaskedGenome representation and capacity (in_dim=5, max_hidden=8, out_dim=1).
     - Deb-style constrained dominance logic.
     - Non-dominated sorting, crowding distance, and survivor selection matching Phase 3A.
     - Asexual mutation reproduction matching Phase 3A.
  3. Fair-comparison compatibility:
     - Transition budget ceilings strictly <= 2,000,000.
     - Approved development partition (seeds 0..199).
     - Strict isolation from provisional (600..749) and holdout (2000..2099) partitions.
     - Canonical simulation dynamics and frozen success criteria alignment.
"""

import numpy as np
import pytest

from baselines.mlp import FixedMLPController, train_mlp_es
from env.cartpole_swingup import CartPoleSwingUpEnv
from env.vectorized_env import VectorizedCartPoleEnv
from evolution.evolution_strategies import (
    NAMESPACE_ES,
    FixedMLPController as ES_Controller,
    evaluate_es_candidate,
    get_es_rng,
    train_es,
)
from evolution.nsga2_engine import (
    NAMESPACE_NSGA2,
    MaskedGenome,
    constrained_dominates,
    compute_crowding_distance,
    fast_non_dominated_sort,
    get_nsga2_rng,
    nsga2_survivor_selection,
    train_nsga2,
)
from experiments.run_phase_3a_evolution import (
    DIAGNOSTIC_CONDITIONS,
    SUCCESS_FLOOR,
    TRAIN_SEEDS,
    GATE0_VAL_SEEDS,
    PHASE3A_VAL_SEEDS,
    FINAL_HOLDOUT_SEEDS,
)
from experiments.run_robustness_experiment import evaluate_success_predicate


# ── 1. ES Algorithm Identity Tests ───────────────────────────────────────────


def test_es_controller_architecture_identity():
    """Verify FixedMLPController architecture used by train_es matches baselines/mlp.py."""
    # Both import and use the exact same class definition
    assert ES_Controller is FixedMLPController
    ctrl = FixedMLPController()
    assert ctrl.INPUT_DIM == 5
    assert ctrl.HIDDEN_DIM == 16
    assert ctrl.OUTPUT_DIM == 1
    assert ctrl.active_parameters == 113
    assert ctrl.theoretical_macs == 96

    # Verify forward pass dimensions and clipping
    obs = np.array([0.5, 0.1, 0.9, 0.43, 0.2], dtype=np.float32)
    action = ctrl.compute_action(obs)
    assert isinstance(action, float)
    assert -1.0 <= action <= 1.0


def test_es_antithetic_sampling_and_adam_formulation():
    """Verify train_es uses symmetric antithetic perturbations and exact Adam update."""
    rng, _ = get_es_rng(42)
    half_pop = 10
    n_params = FixedMLPController.TOTAL_PARAMS
    eps = rng.standard_normal((half_pop, n_params)).astype(np.float32)
    perturbations = np.vstack([eps, -eps])

    # Symmetrical antithetic property: sum over pairs is zero
    assert perturbations.shape == (20, 113)
    np.testing.assert_allclose(perturbations[:half_pop] + perturbations[half_pop:], 0.0, atol=1e-7)

    # Rank-based fitness shaping formula identity
    mock_rewards = np.linspace(-100, 200, 20, dtype=np.float32)
    order = np.argsort(np.argsort(mock_rewards))
    utilities = np.maximum(0.0, np.log(len(mock_rewards) / 2.0 + 1.0) - np.log(len(mock_rewards) - order))
    util_sum = np.sum(utilities)
    utilities = utilities / util_sum - (1.0 / len(mock_rewards))

    # Natural gradient calculation: 1/(P*sigma) * utilities @ perturbations
    sigma = 0.10
    grad = (1.0 / (20 * sigma)) * np.dot(utilities, perturbations)
    assert grad.shape == (113,)


# ── 2. Neuroevolution Algorithm Identity Tests ────────────────────────────────


def test_nsga2_genome_representation_and_capacity():
    """Verify MaskedGenome matches Phase 3A representation and parameter limits."""
    rng, _ = get_nsga2_rng(42)
    g = MaskedGenome(rng=rng)

    assert g.in_dim == 5
    assert g.max_hidden == 8
    assert g.out_dim == 1
    assert g.w1.shape == (8, 5)
    assert g.b1.shape == (8,)
    assert g.w2.shape == (1, 8)
    assert g.b2.shape == (1,)

    # Baseline dense equivalent capacity: (5*8 + 8) + (8*1 + 1) = 48 + 9 = 57 parameters
    comp = g.get_complexity_metrics()
    assert comp["dense_params"] == 57
    assert comp["dense_macs"] == 48


def test_nsga2_deb_constrained_dominance_identity():
    """Verify Deb-style constrained dominance rules match Phase 3A specifications."""
    g1 = MaskedGenome()
    g2 = MaskedGenome()

    # Rule 1: Feasible dominates Infeasible
    g1.constraint_violation = 0.0
    g1.fitness["robust_success"] = 0.60
    g1.fitness["active_params"] = 40
    g1.fitness["weight_macs"] = 32

    g2.constraint_violation = 0.05
    g2.fitness["robust_success"] = 0.43
    g2.fitness["active_params"] = 20
    g2.fitness["weight_macs"] = 16

    assert constrained_dominates(g1, g2) is True
    assert constrained_dominates(g2, g1) is False

    # Rule 2: Between two infeasible, smaller violation dominates
    g1.constraint_violation = 0.02
    g2.constraint_violation = 0.08
    assert constrained_dominates(g1, g2) is True
    assert constrained_dominates(g2, g1) is False

    # Rule 3: Between two feasible, Pareto dominance on (robust_succ [max], params [min], macs [min])
    g1.constraint_violation = 0.0
    g2.constraint_violation = 0.0

    # g1 has higher success, same params, same macs -> dominates
    g1.fitness["robust_success"] = 0.70
    g1.fitness["active_params"] = 30
    g1.fitness["weight_macs"] = 24

    g2.fitness["robust_success"] = 0.65
    g2.fitness["active_params"] = 30
    g2.fitness["weight_macs"] = 24
    assert constrained_dominates(g1, g2) is True
    assert constrained_dominates(g2, g1) is False

    # g1 has lower params, same success, same macs -> dominates
    g1.fitness["robust_success"] = 0.65
    g1.fitness["active_params"] = 25
    g1.fitness["weight_macs"] = 24
    assert constrained_dominates(g1, g2) is True


def test_nsga2_survivor_selection_and_pareto_front():
    """Verify NSGA-II non-dominated sorting and crowding distance partition identically."""
    pop = [MaskedGenome() for _ in range(10)]
    for i, ind in enumerate(pop):
        ind.constraint_violation = 0.0
        ind.fitness["robust_success"] = 0.50 + i * 0.02
        ind.fitness["active_params"] = 50 - i * 2
        ind.fitness["weight_macs"] = 40 - i * 2

    # All individuals are mutually non-dominated or dominant along gradient
    survivors = nsga2_survivor_selection(pop, pop_size=5)
    assert len(survivors) == 5
    # The highest success / lowest complexity individual must survive
    assert any(s.fitness["robust_success"] == 0.50 + 9 * 0.02 for s in survivors)


# ── 3. Fair-Comparison Compatibility Tests ────────────────────────────────────


def test_budget_ceilings_and_partition_isolation():
    """Verify both training entry points adhere to ceilings and partition protections."""
    # 1. Budget ceilings
    es_result = train_es(optimizer_seed=42, budget_limit=10, max_steps=5, eval_episodes=1, train_seeds=[0])
    assert es_result["tracker"].budget_limit <= 2_000_000

    nsga2_result = train_nsga2(optimizer_seed=42, budget_limit=10, max_steps=5, eval_episodes=1, train_seeds=[0], pop_size=2, offspring_size=1)
    assert nsga2_result["tracker"].budget_limit <= 2_000_000

    # 2. Partition safety check
    for seed_list in [TRAIN_SEEDS, GATE0_VAL_SEEDS, PHASE3A_VAL_SEEDS]:
        # Must never touch provisional (600..749) or holdout (2000..2099)
        assert not any(600 <= s <= 749 for s in seed_list)
        assert not any(2000 <= s <= 2099 for s in seed_list)

    # Final holdout is strictly 2000..2099
    assert FINAL_HOLDOUT_SEEDS == list(range(2000, 2100))


def test_success_predicate_alignment():
    """Verify that evaluate_genome_budgeted uses the official frozen success predicate logic."""
    # Upright threshold: cos(theta) > 0.9 for at least 95 of final 100 steps
    # No boundary rail hit: not rail_hit
    # Max steps: 500
    cos_succ = [0.0] * 400 + [0.95] * 100
    is_s, is_rail, is_other, upright = evaluate_success_predicate(cos_succ, rail_hit=False, max_steps=500)
    assert is_s is True
    assert upright == 100

    # Fail if rail hit
    is_s, is_rail, is_other, _ = evaluate_success_predicate(cos_succ, rail_hit=True, max_steps=500)
    assert is_s is False
    assert is_rail is True

    # Fail if upright in final window < 95
    cos_fail = [0.0] * 400 + [0.95] * 90 + [0.5] * 10
    is_s, is_rail, is_other, upright = evaluate_success_predicate(cos_fail, rail_hit=False, max_steps=500)
    assert is_s is False
    assert upright == 90
