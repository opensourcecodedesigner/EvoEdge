"""RNG Construction, Seed Independence, and Training Entry Point Budget Integration Tests.

Validates:
  1. NumPy SeedSequence construction uses strictly valid integer types (no strings).
  2. Documented integer namespaces: NAMESPACE_ES = 1, NAMESPACE_NSGA2 = 2.
  3. Determinism: same registered run seed repeats identically.
  4. Stream independence: ES and NSGA-II with the same seed produce distinct streams.
  5. Real training entry points (train_es, train_nsga2) stop at tiny budget ceilings.
  6. Vectorized batch execution billing and fail-closed stopping.
  7. Exception and retry persistence (transitions remain billed fail-closed).
  8. Early termination billing (halted episodes bill only executed steps).
"""

import copy
import numpy as np
import pytest

from env.budget_tracker import (
    BudgetExhaustedError,
    TransitionBudgetTracker,
    BudgetedEnv,
    BudgetedVectorizedEnv,
)
from env.cartpole_swingup import CartPoleSwingUpEnv
from env.vectorized_env import VectorizedCartPoleEnv
from evolution.evolution_strategies import (
    NAMESPACE_ES,
    get_es_rng,
    train_es,
)
from evolution.nsga2_engine import (
    NAMESPACE_NSGA2,
    get_nsga2_rng,
    train_nsga2,
    MaskedGenome,
)


# ── 1. RNG Construction and Namespace Tests ───────────────────────────────────


def test_rng_initialization_supported_types():
    """Verify that both ES and NSGA-II SeedSequence calls use valid integer argument types."""
    for s_opt in [42, 43, 44, 45, 46]:
        rng_es, seq_es = get_es_rng(s_opt)
        rng_nsga2, seq_nsga2 = get_nsga2_rng(s_opt)

        assert isinstance(seq_es.entropy, int)
        assert isinstance(seq_es.spawn_key, tuple)
        assert all(isinstance(k, int) for k in seq_es.spawn_key)
        assert seq_es.spawn_key == (NAMESPACE_ES, s_opt)

        assert isinstance(seq_nsga2.entropy, int)
        assert isinstance(seq_nsga2.spawn_key, tuple)
        assert all(isinstance(k, int) for k in seq_nsga2.spawn_key)
        assert seq_nsga2.spawn_key == (NAMESPACE_NSGA2, s_opt)

        # Generate numbers to prove bit generator operates without error
        vals_es = rng_es.standard_normal(5)
        vals_nsga2 = rng_nsga2.standard_normal(5)
        assert len(vals_es) == 5
        assert len(vals_nsga2) == 5


def test_rng_determinism_and_reproducibility():
    """Verify that the same registered seed repeats identically on independent calls."""
    s_opt = 42
    rng1_es, _ = get_es_rng(s_opt)
    rng2_es, _ = get_es_rng(s_opt)
    np.testing.assert_array_equal(rng1_es.random(20), rng2_es.random(20))

    rng1_nsga2, _ = get_nsga2_rng(s_opt)
    rng2_nsga2, _ = get_nsga2_rng(s_opt)
    np.testing.assert_array_equal(rng1_nsga2.random(20), rng2_nsga2.random(20))


def test_rng_stream_independence_between_algorithms():
    """Verify that ES and NSGA-II use completely distinct random streams for identical seeds."""
    for s_opt in [42, 43, 44, 45, 46]:
        rng_es, _ = get_es_rng(s_opt)
        rng_nsga2, _ = get_nsga2_rng(s_opt)

        v_es = rng_es.random(50)
        v_nsga2 = rng_nsga2.random(50)

        # Streams must never be identical
        assert not np.allclose(v_es, v_nsga2)
        # Even the very first float draw must differ
        assert v_es[0] != v_nsga2[0]


def test_rng_invalid_type_raises():
    """Verify that passing non-integer seeds raises TypeError rather than silently converting."""
    with pytest.raises(TypeError):
        get_es_rng("42")  # type: ignore

    with pytest.raises(TypeError):
        get_nsga2_rng("42")  # type: ignore


# ── 2. Training Entry Point Budget Integration Tests ──────────────────────────


def test_es_entry_point_stops_at_tiny_budget():
    """Verify that train_es instantiates and respects TransitionBudgetTracker on a tiny ceiling."""
    tiny_budget = 45
    result = train_es(
        optimizer_seed=42,
        generations=10,
        population_size=4,
        eval_episodes=2,
        max_steps=50,
        budget_limit=tiny_budget,
        train_seeds=[10, 11, 12, 13],
    )

    tracker = result["tracker"]
    assert result["stopped_by_budget"] is True
    assert tracker.is_exhausted is True
    assert tracker.consumed_transitions == tiny_budget
    assert tracker.remaining_transitions == 0

    # Verify fail-closed blocking after exhaustion
    with pytest.raises(BudgetExhaustedError):
        tracker.record_transitions(1)


def test_nsga2_entry_point_vectorized_stops_at_tiny_budget():
    """Verify that train_nsga2 with vectorized simulator stops exactly at a tiny budget ceiling."""
    # With batch size = 2 (2 eval episodes), each step bills 2 transitions
    tiny_budget = 30
    result = train_nsga2(
        optimizer_seed=42,
        pop_size=4,
        offspring_size=2,
        generations=5,
        eval_episodes=2,
        max_steps=50,
        budget_limit=tiny_budget,
        train_seeds=[10, 11],
        vectorized=True,
    )

    tracker = result["tracker"]
    assert result["stopped_by_budget"] is True
    assert tracker.is_exhausted is True
    assert tracker.consumed_transitions == tiny_budget
    assert tracker.remaining_transitions == 0

    with pytest.raises(BudgetExhaustedError):
        tracker.record_transitions(1)


def test_nsga2_entry_point_scalar_stops_at_tiny_budget():
    """Verify that train_nsga2 with scalar simulator stops at tiny budget ceiling."""
    tiny_budget = 25
    result = train_nsga2(
        optimizer_seed=43,
        pop_size=2,
        offspring_size=2,
        generations=5,
        eval_episodes=2,
        max_steps=50,
        budget_limit=tiny_budget,
        train_seeds=[14, 15],
        vectorized=False,
    )

    tracker = result["tracker"]
    assert result["stopped_by_budget"] is True
    assert tracker.is_exhausted is True
    assert tracker.consumed_transitions == tiny_budget


# ── 3. Exception, Retry, and Early Termination Budget Tests ───────────────────


def test_exception_and_retry_budget_persistence():
    """Verify that transitions consumed before an exception remain billed on retry."""
    tracker = TransitionBudgetTracker(budget_limit=50)

    class FaultyEnv:
        def __init__(self):
            self.step_count = 0

        def reset(self, **kwargs):
            return np.zeros(5), {}

        def step(self, action):
            self.step_count += 1
            if self.step_count == 20:
                raise RuntimeError("Simulated transient actuator communication error")
            return np.zeros(5), 1.0, False, False, {}

    faulty_env = FaultyEnv()
    budgeted = BudgetedEnv(faulty_env, tracker)

    # 1. First run raises at step 20
    with pytest.raises(RuntimeError, match="Simulated transient"):
        for _ in range(30):
            budgeted.step(0.0)

    # 20 transitions MUST remain billed despite exception
    assert tracker.consumed_transitions == 20
    assert tracker.remaining_transitions == 30

    # 2. Retry execution without resetting tracker
    fresh_env = CartPoleSwingUpEnv(max_steps=500, continuous=True)
    retry_budgeted = BudgetedEnv(fresh_env, tracker)
    retry_budgeted.reset(seed=100)

    # Execute remaining 30 transitions
    with pytest.raises(BudgetExhaustedError):
        for _ in range(40):
            retry_budgeted.step(0.0)

    assert tracker.consumed_transitions == 50
    assert tracker.is_exhausted is True


def test_early_termination_bills_only_executed_steps():
    """Verify that episodes ending early due to rail collision bill only executed steps."""
    tracker = TransitionBudgetTracker(budget_limit=1000)
    base_env = CartPoleSwingUpEnv(max_steps=500, continuous=True)
    benv = BudgetedEnv(base_env, tracker)

    # Force continuous maximum positive action -> quickly hits rail |x| >= 2.4m
    benv.reset(seed=42)
    steps_executed = 0
    done = False
    while not done:
        _, _, term, trunc, _ = benv.step(1.0)
        steps_executed += 1
        done = term or trunc

    # Early boundary termination must happen well before 500 steps
    assert steps_executed < 150
    # The tracker must reflect EXACTLY the executed steps, not 500
    assert tracker.consumed_transitions == steps_executed
    assert tracker.remaining_transitions == 1000 - steps_executed


# ── 4. Lifecycle Budgeting (Search, Validation, and Total Ceilings) ───────────


def test_lifecycle_budget_enforcement_search_and_validation():
    """Verify that search halts at search_budget, validation at validation_budget, and total never exceeds total_budget."""
    tracker = TransitionBudgetTracker(total_budget=50, search_budget=30, validation_budget=20)
    assert tracker.budget_limit == 30
    assert tracker.phase == "search"

    # Step 29 times during search
    for _ in range(29):
        tracker.check_can_step(1)
        tracker.record_transitions(1)

    # 30th step reaches ceiling and raises BudgetExhaustedError
    tracker.check_can_step(1)
    with pytest.raises(BudgetExhaustedError):
        tracker.record_transitions(1)

    assert tracker.search_consumed == 30
    assert tracker.consumed_transitions == 30
    assert tracker.is_exhausted is True

    # 31st step in search must be rejected before stepping
    with pytest.raises(BudgetExhaustedError):
        tracker.check_can_step(1)

    # Transition to validation phase
    tracker.start_validation_phase()
    assert tracker.phase == "validation"
    assert tracker.budget_limit == 50  # 30 search + 20 validation = 50 total
    assert tracker.remaining_transitions == 20

    # Execute 19 validation steps
    for _ in range(19):
        tracker.check_can_step(1)
        tracker.record_transitions(1)

    # 20th validation step reaches ceiling and raises BudgetExhaustedError
    tracker.check_can_step(1)
    with pytest.raises(BudgetExhaustedError):
        tracker.record_transitions(1)

    assert tracker.validation_consumed == 20
    assert tracker.consumed_transitions == 50
    assert tracker.is_exhausted is True
    assert tracker.is_total_exhausted is True

    # Next step must be rejected fail-closed
    with pytest.raises(BudgetExhaustedError):
        tracker.check_can_step(1)


def test_search_early_stop_does_not_expand_validation_beyond_cap():
    """Verify that an early-stopping search does NOT allow validation to exceed its own allocation."""
    tracker = TransitionBudgetTracker(total_budget=100, search_budget=60, validation_budget=20)

    # Search stops early after only 15 transitions
    for _ in range(15):
        tracker.record_transitions(1)

    assert tracker.search_consumed == 15
    tracker.start_validation_phase()

    # Validation active ceiling should be min(100, 15 + 20) = 35 transitions total
    assert tracker.budget_limit == 35
    assert tracker.remaining_transitions == 20

    # Validation can execute 19 steps without hitting ceiling
    for _ in range(19):
        tracker.record_transitions(1)

    # 20th step reaches 20 validation transitions (35 total) and raises BudgetExhaustedError
    with pytest.raises(BudgetExhaustedError):
        tracker.record_transitions(1)

    assert tracker.validation_consumed == 20
    assert tracker.consumed_transitions == 35  # <= 100 total, exactly 20 validation

    # 21st validation step must be refused even though total remaining is 65!
    with pytest.raises(BudgetExhaustedError):
        tracker.check_can_step(1)


def test_entry_points_respect_lifecycle_ceilings():
    """Verify that real training entry points respect search_budget and total_budget."""
    # ES
    es_res = train_es(
        optimizer_seed=42,
        generations=5,
        population_size=2,
        eval_episodes=1,
        max_steps=10,
        total_budget=50,
        search_budget=25,
        validation_budget=25,
        train_seeds=[0],
    )
    assert es_res["tracker"].search_consumed <= 25
    assert es_res["tracker"].consumed_transitions <= 25

    # NSGA-II
    nsga2_res = train_nsga2(
        optimizer_seed=42,
        pop_size=2,
        offspring_size=1,
        generations=5,
        eval_episodes=1,
        max_steps=10,
        total_budget=50,
        search_budget=20,
        validation_budget=30,
        train_seeds=[0],
        vectorized=True,
    )
    assert nsga2_res["tracker"].search_consumed <= 20
    assert nsga2_res["tracker"].consumed_transitions <= 20

