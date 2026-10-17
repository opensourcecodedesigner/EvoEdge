"""Comprehensive tests for simulation transition budget tracking and enforcement.

Verifies:
  1. Scalar execution accounting (1-to-1 step accounting).
  2. Vectorized execution accounting (B * steps accounting).
  3. Action repeat / substeps accounting.
  4. Early termination billing (halted episodes bill only executed steps).
  5. Exceptions persistence (transitions executed prior to exception remain billed).
  6. Retries persistence (counter does not reset across retries).
  7. Fail-closed blocking on attempted execution after exhaustion.
  8. Environment resets do not consume transitions.
"""

from __future__ import annotations

import unittest
import numpy as np

from env.cartpole_swingup import CartPoleSwingUpEnv
from env.vectorized_env import VectorizedCartPoleEnv
from env.budget_tracker import (
    BudgetExhaustedError,
    TransitionBudgetTracker,
    BudgetedEnv,
    BudgetedVectorizedEnv,
)


class TestBudgetTracker(unittest.TestCase):
    """Test suite for TransitionBudgetTracker, BudgetedEnv, and BudgetedVectorizedEnv."""

    def test_scalar_execution_accounting(self):
        """Verify scalar environment steps bill exactly 1 transition per step."""
        tracker = TransitionBudgetTracker(budget_limit=100)
        raw_env = CartPoleSwingUpEnv(max_steps=50, continuous=True)
        env = BudgetedEnv(raw_env, tracker)

        env.reset(seed=42)
        self.assertEqual(tracker.consumed_transitions, 0, "Resets must not consume transitions.")

        for _ in range(15):
            env.step(np.array([0.0], dtype=np.float32))

        self.assertEqual(tracker.consumed_transitions, 15)
        self.assertEqual(tracker.remaining_transitions, 85)
        self.assertFalse(tracker.is_exhausted)

    def test_vectorized_execution_accounting(self):
        """Verify vectorized environment steps bill num_envs transitions per step."""
        tracker = TransitionBudgetTracker(budget_limit=500)
        num_envs = 8
        raw_vec_env = VectorizedCartPoleEnv(num_envs=num_envs, max_steps=50)
        vec_env = BudgetedVectorizedEnv(raw_vec_env, tracker)

        vec_env.reset(seeds=[100 + i for i in range(num_envs)])
        self.assertEqual(tracker.consumed_transitions, 0)

        num_steps = 10
        actions = np.zeros(num_envs, dtype=np.float64)
        for _ in range(num_steps):
            vec_env.step(actions)

        expected_transitions = num_envs * num_steps  # 8 * 10 = 80
        self.assertEqual(tracker.consumed_transitions, expected_transitions)
        self.assertEqual(tracker.remaining_transitions, 500 - expected_transitions)

    def test_substeps_accounting(self):
        """Verify action repeat / substeps multiplier is billed accurately."""
        tracker = TransitionBudgetTracker(budget_limit=100)
        raw_env = CartPoleSwingUpEnv(max_steps=50, continuous=True)
        env = BudgetedEnv(raw_env, tracker, substeps=4)

        env.reset(seed=42)
        for _ in range(5):
            env.step(np.array([0.0], dtype=np.float32))

        # 5 steps * 4 substeps = 20 transitions
        self.assertEqual(tracker.consumed_transitions, 20)

    def test_early_termination_accounting(self):
        """Verify early terminated episodes bill only actual executed steps, not the full horizon."""
        tracker = TransitionBudgetTracker(budget_limit=1000)
        raw_env = CartPoleSwingUpEnv(max_steps=500, continuous=True)
        env = BudgetedEnv(raw_env, tracker)

        env.reset(seed=42)
        # Apply full force to quickly cause rail collision and terminate early
        steps_executed = 0
        for _ in range(500):
            obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
            steps_executed += 1
            if terminated or truncated or abs(raw_env._state[0]) >= 2.4:
                break

        self.assertLess(steps_executed, 100, "Full force should cause early collision in under 100 steps.")
        self.assertEqual(
            tracker.consumed_transitions,
            steps_executed,
            f"Billed transitions ({tracker.consumed_transitions}) must match actual executed steps ({steps_executed}), not 500.",
        )

    def test_exceptions_accounting(self):
        """Verify that transitions executed prior to an exception remain billed to the tracker."""
        tracker = TransitionBudgetTracker(budget_limit=100)

        class FaultyEnv:
            def __init__(self):
                self.steps = 0

            def reset(self, **kwargs):
                return None, {}

            def step(self, action):
                self.steps += 1
                if self.steps == 7:
                    raise RuntimeError("Simulated physical or sensor hardware fault.")
                return None, 0.0, False, False, {}

        env = BudgetedEnv(FaultyEnv(), tracker)
        env.reset()

        with self.assertRaises(RuntimeError):
            for _ in range(10):
                env.step(0.0)

        # 7 steps were executed before the exception raised; all 7 must be billed
        self.assertEqual(tracker.consumed_transitions, 7)

    def test_retries_persistence(self):
        """Verify that trial retries do not reset the budget counter."""
        tracker = TransitionBudgetTracker(budget_limit=100)
        raw_env = CartPoleSwingUpEnv(max_steps=50, continuous=True)
        env = BudgetedEnv(raw_env, tracker)

        # Trial 1: 5 steps
        env.reset(seed=1)
        for _ in range(5):
            env.step(np.array([0.0], dtype=np.float32))
        self.assertEqual(tracker.consumed_transitions, 5)

        # Retry / Trial 2: 8 steps
        env.reset(seed=2)
        for _ in range(8):
            env.step(np.array([0.0], dtype=np.float32))

        # Total billed across both trials must be 5 + 8 = 13 (retries do not reset counter)
        self.assertEqual(tracker.consumed_transitions, 13)

    def test_attempted_execution_after_exhaustion(self):
        """Verify fail-closed behavior: calling step after exhaustion immediately raises BudgetExhaustedError."""
        tracker = TransitionBudgetTracker(budget_limit=10)
        raw_env = CartPoleSwingUpEnv(max_steps=50, continuous=True)
        env = BudgetedEnv(raw_env, tracker)

        env.reset(seed=42)
        # Step 9 times: consumes 9 transitions
        for _ in range(9):
            env.step(np.array([0.0], dtype=np.float32))
        self.assertEqual(tracker.consumed_transitions, 9)
        self.assertFalse(tracker.is_exhausted)

        # Step 10: reaches budget limit (10), raises BudgetExhaustedError
        with self.assertRaises(BudgetExhaustedError):
            env.step(np.array([0.0], dtype=np.float32))

        self.assertEqual(tracker.consumed_transitions, 10)
        self.assertTrue(tracker.is_exhausted)

        # Attempt step 11 after exhaustion: fails closed immediately
        with self.assertRaises(BudgetExhaustedError):
            env.step(np.array([0.0], dtype=np.float32))

        # Total transitions remain 10 (no extra physics steps executed)
        self.assertEqual(tracker.consumed_transitions, 10)
        self.assertEqual(tracker.transitions_used, 10)

    def test_vectorized_batch_refuses_when_active_size_exceeds_remaining_budget(self):
        """Verify wrapper refuses to execute a batch that exceeds remaining budget without advancing simulator."""
        budget_limit = 11  # Indivisible by batch size 4 (2 full steps = 8, leaving 3 remaining)
        tracker = TransitionBudgetTracker(budget_limit=budget_limit)
        raw_vec_env = VectorizedCartPoleEnv(num_envs=4, max_steps=50)
        vec_env = BudgetedVectorizedEnv(raw_vec_env, tracker)

        vec_env.reset(seeds=[10, 11, 12, 13])
        actions = np.zeros(4, dtype=np.float64)

        # Step 1: 4 transitions consumed, 7 remaining
        vec_env.step(actions)
        self.assertEqual(tracker.transitions_used, 4)
        self.assertEqual(tracker.remaining_transitions, 7)
        self.assertTrue(np.all(raw_vec_env._step_count == 1))

        # Step 2: 4 transitions consumed, 3 remaining
        vec_env.step(actions)
        self.assertEqual(tracker.transitions_used, 8)
        self.assertEqual(tracker.remaining_transitions, 3)
        self.assertTrue(np.all(raw_vec_env._step_count == 2))

        # Step 3: 4 active environments, but only 3 transitions remain in budget!
        # The wrapper MUST refuse execution before calling raw_vec_env.step()
        with self.assertRaises(BudgetExhaustedError):
            vec_env.step(actions)

        # Crucial assertions:
        # 1. Underlying simulator did NOT advance beyond step 2
        self.assertTrue(np.all(raw_vec_env._step_count == 2), "Underlying simulator must NOT advance when batch exceeds budget.")
        # 2. transitions_used remains 8 and never exceeds budget_limit (11)
        self.assertEqual(tracker.transitions_used, 8)
        self.assertLessEqual(tracker.transitions_used, tracker.budget_limit)
        self.assertEqual(tracker.remaining_transitions, 3)

    def test_changing_active_batch_size_and_indivisible_budget_boundary(self):
        """Test changing active batch size after early termination and an indivisible budget limit."""
        budget_limit = 9  # 9 is not divisible by initial batch size 4
        tracker = TransitionBudgetTracker(budget_limit=budget_limit)
        raw_vec_env = VectorizedCartPoleEnv(num_envs=4, max_steps=50)
        vec_env = BudgetedVectorizedEnv(raw_vec_env, tracker, mask_terminated=True)

        vec_env.reset(seeds=[20, 21, 22, 23])
        actions = np.zeros(4, dtype=np.float64)

        # Step 1: all 4 active -> costs 4 transitions. 5 remaining.
        vec_env.step(actions)
        self.assertEqual(tracker.transitions_used, 4)
        self.assertEqual(tracker.remaining_transitions, 5)
        self.assertTrue(np.all(raw_vec_env._step_count == 1))

        # Simulate early termination of 2 environments (e.g. boundary violations)
        raw_vec_env.terminated[0] = True
        raw_vec_env.terminated[1] = True
        # Active size changes dynamically from 4 to 2

        # Step 2: 2 active environments -> costs 2 transitions. 3 remaining.
        vec_env.step(actions)
        self.assertEqual(tracker.transitions_used, 6)
        self.assertEqual(tracker.remaining_transitions, 3)
        self.assertTrue(np.all(raw_vec_env._step_count == 2))

        # Step 3: 2 active environments -> costs 2 transitions. 1 remaining.
        vec_env.step(actions)
        self.assertEqual(tracker.transitions_used, 8)
        self.assertEqual(tracker.remaining_transitions, 1)
        self.assertTrue(np.all(raw_vec_env._step_count == 3))

        # Step 4: 2 active environments, but only 1 transition remains!
        # Wrapper must refuse batch of 2 when only 1 remains
        with self.assertRaises(BudgetExhaustedError):
            vec_env.step(actions)

        self.assertTrue(np.all(raw_vec_env._step_count == 3), "Simulator must not advance on rejected batch.")
        self.assertEqual(tracker.transitions_used, 8)
        self.assertLessEqual(tracker.transitions_used, tracker.budget_limit)

        # Now 1 more environment terminates, reducing active size to 1!
        raw_vec_env.terminated[2] = True
        # Now exactly 1 active environment remains, matching the 1 remaining transition!
        with self.assertRaises(BudgetExhaustedError):
            # Step 5: Active size 1 fits remaining 1 transition, reaching ceiling (9)
            vec_env.step(actions)

        self.assertTrue(np.all(raw_vec_env._step_count == 4), "Simulator advanced 1 step for single active environment.")
        self.assertEqual(tracker.transitions_used, 9)
        self.assertEqual(tracker.remaining_transitions, 0)
        self.assertTrue(tracker.is_exhausted)
        self.assertLessEqual(tracker.transitions_used, tracker.budget_limit)

    def test_transitions_used_never_exceeds_budget_under_exceptions_and_retries(self):
        """Assert transitions_used never exceeds budget_limit under exceptions and retries."""
        budget_limit = 7
        tracker = TransitionBudgetTracker(budget_limit=budget_limit)

        class MockFailingBatchEnv:
            def __init__(self, num_envs=3):
                self.num_envs = num_envs
                self.step_count = 0

            def reset(self, **kwargs):
                return np.zeros((self.num_envs, 5)), {}

            def step(self, actions):
                self.step_count += 1
                if self.step_count == 2:
                    raise RuntimeError("Transient actuator communication failure")
                return np.zeros((self.num_envs, 5)), np.ones(self.num_envs), np.zeros(self.num_envs, bool), False, []

        mock_env = MockFailingBatchEnv(num_envs=3)
        vec_env = BudgetedVectorizedEnv(mock_env, tracker)

        # Step 1: succeeds, bills 3 transitions
        vec_env.step(np.zeros(3))
        self.assertEqual(tracker.transitions_used, 3)
        self.assertEqual(tracker.remaining_transitions, 4)

        # Step 2: raises RuntimeError inside step, but transitions executed remain billed (3 + 3 = 6)
        with self.assertRaises(RuntimeError):
            vec_env.step(np.zeros(3))

        self.assertEqual(tracker.transitions_used, 6)
        self.assertEqual(tracker.remaining_transitions, 1)

        # Step 3 (Retry): Attempt to retry with full batch of 3
        # Requested 3 > remaining 1 -> must raise BudgetExhaustedError before stepping
        with self.assertRaises(BudgetExhaustedError):
            vec_env.step(np.zeros(3))

        # Simulator step_count must not have incremented on rejected retry
        self.assertEqual(mock_env.step_count, 2)
        # transitions_used must remain 6, strictly <= budget_limit (7)
        self.assertEqual(tracker.transitions_used, 6)
        self.assertLessEqual(tracker.transitions_used, tracker.budget_limit)


if __name__ == "__main__":
    unittest.main()
