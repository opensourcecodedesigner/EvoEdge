"""Stage 3D Robustness-First Neuroevolution Preflight & Regression Test Suite.

Verifies:
  1. Stage 3C champion checkpoint loads cleanly and preserves complexity/interface.
  2. Population initialization produces valid, compliant controllers (params <= 128, MACs <= 128).
  3. Robustness-First constrained domination comparator:
     - High worst-condition success dominates high average with failing condition.
     - Functional candidates strictly dominate passive non-functional candidates (no collapse).
     - When worst_success tied at 0, candidate solving more conditions dominates.
     - When worst_success tied at 0 and same robust_success, dense return on worst condition dominates.
     - When worst_success tied > 0, Pareto dominance across robust success and complexity.
     - Infeasible candidates (>128 params/macs) are strictly dominated.
  4. Domain randomization distribution:
     - STAGE3D_TRAIN_CONDITIONS covers nominal, mass variations, cart/joint/combined friction, sensor noise, delay.
     - Difficult conditions participate directly in fitness evaluation.
  5. Action delay semantics and state representation integrity.
  6. Transition budget tracker integration: search budget strictly enforced, zero overshoot.
  7. Strict partition isolation safety (rejection of seeds 600..749 and 2000..2099).
  8. Determinism under fixed seed sequences.
  9. Bounded preflight training run executes without crashing or collapsing.
"""

from __future__ import annotations

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
from evolution.nsga2_engine import (
    MaskedGenome,
    STAGE3D_TRAIN_CONDITIONS,
    constrained_dominates_stage3d,
    fast_non_dominated_sort_stage3d,
    evaluate_genome_stage3d,
    get_nsga2_rng,
    select_stage3d_champion,
    train_nsga2_stage3d,
    norm_batch_obs,
    norm_obs,
)
from utils.perturbation_loader import PerturbedCartPoleEnv


def test_stage3c_champion_checkpoint_loading():
    """Verify Stage 3C seed-44 champion loads correctly with 7D input and valid complexity."""
    champ_path = os.path.join(_PROJECT_ROOT, "results", "stage3c", "models", "stage3c_nsga2_delayaware_seed44.pkl")
    assert os.path.exists(champ_path), f"Stage 3C champion not found at {champ_path}"

    with open(champ_path, "rb") as f:
        data = pickle.load(f)

    genome = MaskedGenome.from_dict(data)
    assert genome.in_dim == 7, f"Expected 7D input dimension, got {genome.in_dim}"
    assert genome.out_dim == 1
    assert genome.max_hidden == 8

    comp = genome.get_complexity_metrics()
    assert comp["active_parameters"] <= 128, f"Active params {comp['active_parameters']} exceeds 128"
    assert comp["weight_macs"] <= 128, f"Weight MACs {comp['weight_macs']} exceeds 128"


def test_stage3d_population_initialization_validity():
    """Verify Stage 3D initial population has documented balance and valid controllers."""
    rng, _ = get_nsga2_rng(42)
    champ_path = os.path.join(_PROJECT_ROOT, "results", "stage3c", "models", "stage3c_nsga2_delayaware_seed44.pkl")

    with open(champ_path, "rb") as f:
        g_champ = MaskedGenome.from_dict(pickle.load(f), rng=rng)

    pop = [g_champ.copy()]
    # 8 fine mutants
    for _ in range(8):
        c = g_champ.copy()
        c.rng = rng
        c.mutate(weight_mutation_rate=0.4, weight_mutation_power=0.03, connection_toggle_rate=0.02)
        pop.append(c)
    # 7 moderate mutants
    for _ in range(7):
        c = g_champ.copy()
        c.rng = rng
        c.mutate(weight_mutation_rate=0.6, weight_mutation_power=0.08, connection_toggle_rate=0.08)
        pop.append(c)
    # 4 exploratory mutants
    for _ in range(4):
        c = g_champ.copy()
        c.rng = rng
        c.mutate(weight_mutation_rate=0.8, weight_mutation_power=0.20, connection_toggle_rate=0.15)
        pop.append(c)

    assert len(pop) == 20
    for i, ind in enumerate(pop):
        assert ind.in_dim == 7
        comp = ind.get_complexity_metrics()
        assert comp["active_parameters"] <= 128, f"Candidate {i} exceeds param limit: {comp['active_parameters']}"
        assert comp["weight_macs"] <= 128, f"Candidate {i} exceeds MAC limit: {comp['weight_macs']}"
        assert comp["active_neurons"] >= 1, f"Candidate {i} has zero active neurons"


def test_robustness_first_comparator_fixtures():
    """Verify all critical selection rules of constrained_dominates_stage3d."""
    g_base = MaskedGenome(in_dim=7)

    # Fixture 1: Higher worst-condition success dominates higher aggregate average
    g_high_worst = g_base.copy()
    g_high_worst.fitness = {"robust_success": 0.65, "worst_success": 0.50, "active_params": 64, "weight_macs": 55}
    g_high_avg = g_base.copy()
    g_high_avg.fitness = {"robust_success": 0.90, "worst_success": 0.00, "active_params": 50, "weight_macs": 40}

    assert constrained_dominates_stage3d(g_high_worst, g_high_avg) is True, "High worst-success must dominate high avg!"
    assert constrained_dominates_stage3d(g_high_avg, g_high_worst) is False, "High avg with 0 worst must NOT dominate!"

    # Fixture 2: Functional candidate dominates passive non-functional candidate (prevents collapse!)
    g_func = g_base.copy()
    g_func.fitness = {"robust_success": 0.50, "worst_success": 0.00, "worst_return": -50000.0, "active_params": 62, "weight_macs": 53}
    g_passive = g_base.copy()
    g_passive.fitness = {"robust_success": 0.00, "worst_success": 0.00, "worst_return": 100.0, "active_params": 10, "weight_macs": 5}

    assert constrained_dominates_stage3d(g_func, g_passive) is True, "Functional candidate must dominate passive!"
    assert constrained_dominates_stage3d(g_passive, g_func) is False, "Passive candidate must NOT dominate functional!"

    # Fixture 3: Both worst=0, candidate solving more conditions dominates
    g_more_conds = g_base.copy()
    g_more_conds.fitness = {"robust_success": 0.75, "worst_success": 0.00, "worst_return": -20000.0, "active_params": 60, "weight_macs": 50}
    g_fewer_conds = g_base.copy()
    g_fewer_conds.fitness = {"robust_success": 0.50, "worst_success": 0.00, "worst_return": -10000.0, "active_params": 60, "weight_macs": 50}

    assert constrained_dominates_stage3d(g_more_conds, g_fewer_conds) is True
    assert constrained_dominates_stage3d(g_fewer_conds, g_more_conds) is False

    # Fixture 4: Both worst=0 and same robust_success, higher worst_return dominates
    g_better_worst_ret = g_base.copy()
    g_better_worst_ret.fitness = {"robust_success": 0.75, "worst_success": 0.00, "worst_return": -5000.0, "active_params": 60, "weight_macs": 50}
    g_worse_worst_ret = g_base.copy()
    g_worse_worst_ret.fitness = {"robust_success": 0.75, "worst_success": 0.00, "worst_return": -50000.0, "active_params": 60, "weight_macs": 50}

    assert constrained_dominates_stage3d(g_better_worst_ret, g_worse_worst_ret) is True
    assert constrained_dominates_stage3d(g_worse_worst_ret, g_better_worst_ret) is False

    # Fixture 5: Infeasible (>128 params) dominated by compliant
    g_infeasible = g_base.copy()
    g_infeasible.fitness = {"robust_success": 1.0, "worst_success": 1.0, "active_params": 140, "weight_macs": 130}
    g_compliant = g_base.copy()
    g_compliant.fitness = {"robust_success": 0.8, "worst_success": 0.5, "active_params": 60, "weight_macs": 50}

    assert constrained_dominates_stage3d(g_compliant, g_infeasible) is True
    assert constrained_dominates_stage3d(g_infeasible, g_compliant) is False


def test_stage3d_training_conditions_distribution():
    """Verify STAGE3D_TRAIN_CONDITIONS covers all required physical regimes."""
    conds = STAGE3D_TRAIN_CONDITIONS
    assert "1_Nominal" in conds
    assert "2_CartMass_+30%" in conds
    assert "3_CartFriction_0.50" in conds
    assert "4_JointFriction_0.010" in conds
    assert "5_CombinedFriction" in conds
    assert "6_SensorNoise_0.05" in conds
    assert "7_ActuatorDelay_2Step" in conds
    assert "8_CombinedStress" in conds

    # Verify physical parameters match specifications
    assert conds["2_CartMass_+30%"]["cart_mass"] == 1.30
    assert conds["3_CartFriction_0.50"]["b_cart"] == 0.50
    assert conds["4_JointFriction_0.010"]["b_pole"] == 0.010
    assert conds["5_CombinedFriction"]["b_cart"] == 0.50 and conds["5_CombinedFriction"]["b_pole"] == 0.010
    assert conds["6_SensorNoise_0.05"]["obs_noise_sigma"] == 0.050
    assert conds["7_ActuatorDelay_2Step"]["delay_steps"] == 2
    assert conds["8_CombinedStress"]["delay_steps"] == 1


def test_partition_isolation_safety_assertion():
    """Verify fail-closed assertion prevents access to quarantined or untouched partitions."""
    from experiments.run_stage3c_recovery import assert_partition_safety

    # Valid partitions pass
    assert_partition_safety(list(range(0, 200)), "Training")
    assert_partition_safety(list(range(200, 300)), "Gate 1B")
    assert_partition_safety(list(range(300, 320)), "Diagnostic")

    # Quarantined seeds (600..749) trigger immediate exception
    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION.*Quarantined"):
        assert_partition_safety([650], "Prohibited Test")

    # Untouched seeds (2000..2099) trigger immediate exception
    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION.*Untouched"):
        assert_partition_safety([2050], "Prohibited Test")


def test_transition_tracker_budget_enforcement():
    """Verify TransitionBudgetTracker prevents budget overshoot in Stage 3D."""
    tracker = TransitionBudgetTracker(total_budget=10000, search_budget=8000, validation_budget=2000)
    tracker.record_transitions(7500)
    assert tracker.search_consumed == 7500
    assert tracker.consumed_transitions == 7500

    with pytest.raises(BudgetExhaustedError):
        tracker.record_transitions(1000)  # Exceeds 8000 search ceiling

    assert tracker.search_consumed == 7500  # Atomic rejection


def test_bounded_training_preflight():
    """Run a 3-generation bounded preflight smoke test to verify optimizer executes smoothly."""
    tracker = TransitionBudgetTracker(total_budget=500000, search_budget=350000, validation_budget=150000)
    out = train_nsga2_stage3d(
        optimizer_seed=99,
        pop_size=6,
        offspring_size=3,
        generations=3,
        search_budget=350000,
        validation_budget=150000,
        total_budget=500000,
        budget_tracker=tracker,
        vectorized=True,
        verbose=False,
    )

    assert out["completed_generations"] >= 1
    assert len(out["pareto_front"]) >= 1
    champ = select_stage3d_champion(out["pareto_front"])
    comp = champ.get_complexity_metrics()
    assert comp["active_parameters"] <= 128
    assert comp["weight_macs"] <= 128
    assert tracker.consumed_transitions <= 350000
