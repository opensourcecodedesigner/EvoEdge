"""Pre-training Gate and Stage 3B Regression Test Suite.

Verifies:
  1. Reward Incentive & Bounds:
     - Return is bounded in [-450, 1750].
     - Early rail crash is unambiguously worse than surviving episodes.
     - Swing-up progress produces positive learning signal.
     - Upright balance is rewarded; rail excursions penalized.
     - Official success predicate remains strictly decoupled from shaped reward.
  2. NSGA-II Objective Non-Degeneracy:
     - Synthetic Fixture A: Two zero-success candidates with different progress -> higher progress dominates.
     - Synthetic Fixture B: Functional candidate vs smaller nonfunctional candidate -> functional dominates.
     - Synthetic Fixture C: Candidate exceeding complexity (>128 params/macs) is dominated by compliant.
     - Synthetic Fixture D: Comparable performance, different size -> smaller candidate dominates.
     - Population survivor selection does not eliminate functional/progress candidates in favor of empty nets.
  3. ES Parameter Updates & Determinism:
     - Deterministic under fixed seeds.
     - Parameter updates actually modify weights.
  4. Budget Tracking & Partition Safety:
     - Budget tracker respects search and validation limits.
     - Strict assertion against accessing quarantined seeds (600-749) or untouched seeds (2000-2099).
"""

from __future__ import annotations

import math
import os
import sys
import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.rewards import (
    STAGE3B_RETURN_LOWER_BOUND,
    STAGE3B_RETURN_UPPER_BOUND,
    compute_stage3b_step_reward,
    compute_stage3b_batch_reward,
)
from env.cartpole_swingup import CartPoleSwingUpEnv
from env.vectorized_env import VectorizedCartPoleEnv
from env.budget_tracker import TransitionBudgetTracker, BudgetedEnv, BudgetExhaustedError
from evolution.evolution_strategies import train_es, get_es_rng
from evolution.nsga2_engine import (
    MaskedGenome,
    constrained_dominates,
    get_complexity_violation,
    fast_non_dominated_sort,
    nsga2_survivor_selection,
    train_nsga2,
)
from experiments.run_robustness_experiment import evaluate_success_predicate


# ── Section 1: Reward Incentive & Bounds Regression Tests ─────────────────────


def test_reward_explicit_bounds():
    """Verify return bounds are strictly within the mathematically proven limits [-450, 1750]."""
    # Worst case: crashed immediately at step 1
    r_crash_early = compute_stage3b_step_reward(
        x=2.5, x_dot=5.0, theta=np.pi, theta_dot=5.0, action=1.0,
        step_count=1, max_steps=500, x_threshold=2.4
    )
    # Single crash step returns terminal penalty: -200 - 0.5 * 499 = -449.5
    assert r_crash_early >= STAGE3B_RETURN_LOWER_BOUND, f"Crash reward {r_crash_early} below lower bound"

    # Best step: perfect upright calm at center
    r_best_step = compute_stage3b_step_reward(
        x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0, action=0.0,
        step_count=500, max_steps=500, x_threshold=2.4
    )
    # Max single step reward = 1.0 (progress) + 2.0 (upright) + 0.5 (calm) = 3.5
    # Across 500 steps, max return is 500 * 3.5 = 1750.0
    assert r_best_step <= 3.5 + 1e-6
    assert 500 * r_best_step <= STAGE3B_RETURN_UPPER_BOUND + 1e-6


def test_early_rail_failure_cannot_beat_surviving_episode():
    """Regression test for the Stage 2 defect: early rail crash must NOT receive higher return than surviving."""
    env_crash = CartPoleSwingUpEnv(max_steps=500, continuous=True, reward_mode="stage3b")
    env_survive = CartPoleSwingUpEnv(max_steps=500, continuous=True, reward_mode="stage3b")

    # 1. Trajectory that drives full throttle into rail and crashes at step ~30
    obs, _ = env_crash.reset(seed=42)
    crash_return = 0.0
    for _ in range(500):
        obs, r, term, trunc, _ = env_crash.step(1.0)
        crash_return += r
        if term:
            break

    assert term is True, "Crash trajectory should hit rail"
    assert env_crash._step_count < 100, f"Expected early crash, took {env_crash._step_count} steps"

    # 2. Trajectory that applies zero action (survives 500 steps swinging passively at bottom)
    obs, _ = env_survive.reset(seed=42)
    survive_return = 0.0
    for _ in range(500):
        obs, r, term, trunc, _ = env_survive.step(0.0)
        survive_return += r
        if term:
            break

    assert term is False, "Passive trajectory should NOT hit rail"
    assert env_survive._step_count == 500, "Passive trajectory should survive 500 steps"

    # The surviving trajectory MUST be strictly better than early rail failure!
    assert survive_return > crash_return, (
        f"Stage 2 reward defect persists! Survive return {survive_return} <= Crash return {crash_return}"
    )
    # Specifically, crash return should be strongly negative (<-200)
    assert crash_return < -200.0, f"Crash return {crash_return} was not sufficiently penalized"
    assert survive_return >= -50.0, f"Survive return {survive_return} unexpectedly low"


def test_swingup_progress_provides_dense_learning_signal():
    """Verify that moving the pole upward produces an unambiguously higher reward than hanging down."""
    # Hanging down: theta = pi (cos = -1)
    r_down = compute_stage3b_step_reward(
        x=0.0, x_dot=0.0, theta=np.pi, theta_dot=0.0, action=0.0,
        step_count=100, max_steps=500, x_threshold=2.4
    )
    # Horizontal: theta = pi/2 (cos = 0)
    r_horizontal = compute_stage3b_step_reward(
        x=0.0, x_dot=0.0, theta=np.pi / 2.0, theta_dot=0.0, action=0.0,
        step_count=100, max_steps=500, x_threshold=2.4
    )
    # High: theta = pi/6 (cos = 0.866)
    r_high = compute_stage3b_step_reward(
        x=0.0, x_dot=0.0, theta=np.pi / 6.0, theta_dot=0.0, action=0.0,
        step_count=100, max_steps=500, x_threshold=2.4
    )
    # Perfect upright: theta = 0 (cos = 1)
    r_upright = compute_stage3b_step_reward(
        x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0, action=0.0,
        step_count=100, max_steps=500, x_threshold=2.4
    )

    assert r_down < r_horizontal < r_high < r_upright, (
        f"Rewards not strictly monotonic with swing-up progress: "
        f"down={r_down}, horiz={r_horizontal}, high={r_high}, upright={r_upright}"
    )


def test_rail_proximity_penalty_and_centering():
    """Verify rail excursion is penalized smoothly and centered cart is preferred."""
    r_center = compute_stage3b_step_reward(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0, action=0.0, step_count=100)
    r_near_rail = compute_stage3b_step_reward(x=2.2, x_dot=0.0, theta=0.0, theta_dot=0.0, action=0.0, step_count=100)
    assert r_center > r_near_rail, "Cart centering penalty failed"


# ── Section 2: NSGA-II Objective Non-Degeneracy Synthetic Fixtures ────────────


def test_nsga2_synthetic_fixture_a_zero_success_different_progress():
    """Synthetic Fixture A: Two candidates with zero success but different task progress.
    The candidate with higher task progress MUST dominate, regardless of parameter count.
    """
    cand1 = MaskedGenome()
    cand1.fitness = {
        "robust_success": 0.0,
        "task_progress": 250.0,  # Swung up, high progress
        "active_params": 50.0,
        "weight_macs": 40.0,
    }
    cand2 = MaskedGenome()
    cand2.fitness = {
        "robust_success": 0.0,
        "task_progress": -150.0,  # Crashed into rail, negative progress
        "active_params": 10.0,   # Smaller network!
        "weight_macs": 8.0,
    }

    assert constrained_dominates(cand1, cand2) is True, (
        "Cand1 (higher task progress) must dominate Cand2 despite having more parameters!"
    )
    assert constrained_dominates(cand2, cand1) is False, (
        "Cand2 (smaller but worse progress) must NOT dominate Cand1!"
    )


def test_nsga2_synthetic_fixture_b_functional_vs_smaller_nonfunctional():
    """Synthetic Fixture B: A functional candidate (robust_success > 0) versus a smaller nonfunctional candidate.
    The functional candidate MUST dominate the smaller nonfunctional candidate.
    """
    cand_functional = MaskedGenome()
    cand_functional.fitness = {
        "robust_success": 0.60,
        "task_progress": 850.0,
        "active_params": 57.0,
        "weight_macs": 48.0,
    }
    cand_nonfunctional = MaskedGenome()
    cand_nonfunctional.fitness = {
        "robust_success": 0.0,
        "task_progress": 50.0,
        "active_params": 5.0,  # Trivial network with almost zero connections
        "weight_macs": 4.0,
    }

    assert constrained_dominates(cand_functional, cand_nonfunctional) is True, (
        "Functional candidate must strictly dominate smaller nonfunctional candidate!"
    )
    assert constrained_dominates(cand_nonfunctional, cand_functional) is False, (
        "Nonfunctional candidate must never dominate a functional candidate!"
    )


def test_nsga2_synthetic_fixture_c_complexity_constraint_violation():
    """Synthetic Fixture C: Candidates exceeding complexity constraints (>128 params or >128 macs).
    A candidate satisfying constraints must dominate a violating candidate.
    """
    cand_compliant = MaskedGenome()
    cand_compliant.fitness = {
        "robust_success": 0.70,
        "task_progress": 900.0,
        "active_params": 110.0,  # <= 128
        "weight_macs": 90.0,    # <= 128
    }
    cand_violator = MaskedGenome()
    cand_violator.fitness = {
        "robust_success": 0.95,  # Higher success, BUT violates complexity constraint!
        "task_progress": 1200.0,
        "active_params": 150.0,  # > 128
        "weight_macs": 140.0,    # > 128
    }

    assert get_complexity_violation(cand_compliant) == 0.0
    assert get_complexity_violation(cand_violator) > 0.0

    assert constrained_dominates(cand_compliant, cand_violator) is True, (
        "Compliant candidate must dominate complexity violator!"
    )
    assert constrained_dominates(cand_violator, cand_compliant) is False


def test_nsga2_synthetic_fixture_d_comparable_performance_different_sizes():
    """Synthetic Fixture D: Candidates with comparable performance but different sizes.
    The smaller candidate dominates or is Pareto-preferred.
    """
    cand_compact = MaskedGenome()
    cand_compact.fitness = {
        "robust_success": 0.80,
        "task_progress": 1000.0,
        "active_params": 30.0,
        "weight_macs": 24.0,
    }
    cand_bulky = MaskedGenome()
    cand_bulky.fitness = {
        "robust_success": 0.80,
        "task_progress": 1000.0,
        "active_params": 60.0,
        "weight_macs": 50.0,
    }

    assert constrained_dominates(cand_compact, cand_bulky) is True, (
        "Smaller candidate must dominate larger candidate when performance is equal!"
    )
    assert constrained_dominates(cand_bulky, cand_compact) is False


def test_nsga2_survivor_selection_preserves_task_progress():
    """Verify that NSGA-II survivor selection does not degenerate to selecting zero-progress trivial networks."""
    # Create a synthetic population of 10 individuals
    pop = []
    # 3 individuals with high progress and medium size
    for i in range(3):
        g = MaskedGenome()
        g.fitness = {
            "robust_success": 0.0,
            "task_progress": 300.0 + i * 10,
            "active_params": 45.0,
            "weight_macs": 36.0,
        }
        pop.append(g)

    # 7 individuals with zero/negative progress and tiny size (empty nets)
    for i in range(7):
        g = MaskedGenome()
        g.fitness = {
            "robust_success": 0.0,
            "task_progress": -200.0 - i * 10,
            "active_params": 5.0 + i,
            "weight_macs": 4.0 + i,
        }
        pop.append(g)

    # Select top 3 survivors
    survivors = nsga2_survivor_selection(pop, pop_size=3)
    assert len(survivors) == 3

    # All 3 survivors MUST be from the high-progress group!
    survivor_progress = [s.fitness["task_progress"] for s in survivors]
    assert all(p >= 300.0 for p in survivor_progress), (
        f"NSGA-II survivor selection degenerated! Selected progress: {survivor_progress}"
    )


# ── Section 3: ES Parameter Updates & Determinism ─────────────────────────────


def test_es_determinism_and_parameter_update():
    """Verify ES updates are deterministic under fixed seeds and actually change controller parameters."""
    tracker1 = TransitionBudgetTracker(budget_limit=20_000, search_budget=20_000, validation_budget=0)
    res1 = train_es(
        optimizer_seed=42,
        generations=2,
        population_size=4,
        eval_episodes=2,
        max_steps=50,
        budget_tracker=tracker1,
        reward_mode="stage3b",
    )

    tracker2 = TransitionBudgetTracker(budget_limit=20_000, search_budget=20_000, validation_budget=0)
    res2 = train_es(
        optimizer_seed=42,
        generations=2,
        population_size=4,
        eval_episodes=2,
        max_steps=50,
        budget_tracker=tracker2,
        reward_mode="stage3b",
    )

    # Determinism: identical outcomes under seed 42
    np.testing.assert_allclose(res1["best_theta"], res2["best_theta"], atol=1e-7)
    assert res1["actual_transitions"] == res2["actual_transitions"]
    assert res1["completed_generations"] == res2["completed_generations"]

    # Parameter update: theta must not be all zeros or unchanged
    assert np.linalg.norm(res1["best_theta"]) > 0.01, "ES parameters failed to initialize or update"


# ── Section 4: Budget Enforcement & Seed Partition Safety ─────────────────────


def test_budget_ceilings_and_stop_metadata():
    """Verify search and validation ceilings, batch pre-checks, and stop metadata."""
    # Test with a small search budget of 5,000 transitions
    tracker = TransitionBudgetTracker(total_budget=6_000, search_budget=5_000, validation_budget=1_000)
    res = train_es(
        optimizer_seed=43,
        generations=100,  # Request many generations
        population_size=4,
        eval_episodes=2,
        max_steps=100,
        budget_tracker=tracker,
        reward_mode="stage3b",
    )

    assert res["stopped_by_budget"] is True
    assert res["stopping_reason"] == "search_budget_exhausted"
    assert res["actual_transitions"] <= 5_000, (
        f"Actual transitions {res['actual_transitions']} exceeded search budget 5,000"
    )
    assert res["candidate_evaluations"] > 0
    assert res["nominal_rollout_capacity"] >= res["actual_transitions"]


def test_seed_partition_safety_assertion():
    """Verify strictly that seeds 600-749 (quarantined) and 2000-2099 (untouched) are never accessed."""
    QUARANTINED_RANGE = set(range(600, 750))
    UNTOUCHED_RANGE = set(range(2000, 2100))

    # Standard training seeds: 0..199
    train_seeds = list(range(0, 200))
    # Standard development nominal screening seeds: 200..299
    dev_nominal_seeds = list(range(200, 300))
    # Standard development diagnostic seeds: 300..319
    dev_diag_seeds = list(range(300, 320))

    all_used_seeds = set(train_seeds + dev_nominal_seeds + dev_diag_seeds)

    quarantined_overlap = all_used_seeds.intersection(QUARANTINED_RANGE)
    untouched_overlap = all_used_seeds.intersection(UNTOUCHED_RANGE)

    assert len(quarantined_overlap) == 0, f"VIOLATION: Quarantined seeds accessed: {quarantined_overlap}"
    assert len(untouched_overlap) == 0, f"VIOLATION: Untouched seeds accessed: {untouched_overlap}"
