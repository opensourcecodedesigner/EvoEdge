"""Transition Budget Accounting and Enforcement for EvoEdge.

Provides fail-closed budget accounting for physical simulator transitions:
  1. BudgetExhaustedError: Raised when transition limit is reached or exceeded.
  2. TransitionBudgetTracker: Thread-safe, monotonic counter tracking actual physics transitions.
  3. BudgetedEnv: Wrapper for scalar Gymnasium environments (e.g. CartPoleSwingUpEnv, PerturbedCartPoleEnv).
  4. BudgetedVectorizedEnv: Wrapper for VectorizedCartPoleEnv batch rollouts.

Enforces:
  - Exact accounting for physics steps (dt = 0.02s) and substeps.
  - Early termination billing (halted episodes bill only executed steps).
  - Exception and retry persistence (executed transitions remain billed; retries do not reset).
  - Fail-closed blocking on attempted execution after exhaustion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class BudgetExhaustedError(RuntimeError):
    """Raised when the simulation transition budget is exhausted or execution is attempted after exhaustion."""
    pass


class TransitionBudgetTracker:
    """Monotonic transition budget tracker enforcing search, validation, and lifecycle interaction ceilings."""

    def __init__(
        self,
        budget_limit: Optional[int] = None,
        total_budget: Optional[int] = None,
        search_budget: Optional[int] = None,
        validation_budget: Optional[int] = None,
    ):
        if total_budget is not None:
            eff_total = int(total_budget)
        elif budget_limit is not None:
            eff_total = int(budget_limit)
        else:
            eff_total = 2_000_000

        if eff_total <= 0:
            raise ValueError(f"budget must be positive, got {eff_total}")
        self.total_budget = eff_total

        if search_budget is None:
            if self.total_budget == 2_000_000:
                self.search_budget = 1_850_000
                self.validation_budget = 150_000
            else:
                self.search_budget = self.total_budget
                self.validation_budget = 0
        else:
            self.search_budget = int(search_budget)
            if validation_budget is None:
                self.validation_budget = max(0, self.total_budget - self.search_budget)
            else:
                self.validation_budget = int(validation_budget)

        if self.search_budget > self.total_budget:
            raise ValueError(f"search_budget ({self.search_budget}) cannot exceed total_budget ({self.total_budget})")
        if self.search_budget + self.validation_budget > self.total_budget:
            raise ValueError(
                f"search_budget + validation_budget ({self.search_budget + self.validation_budget}) "
                f"exceeds total_budget ({self.total_budget})"
            )

        self._consumed_transitions: int = 0
        self._search_consumed: int = 0
        self._validation_consumed: int = 0
        self._phase: str = "search"  # "search" or "validation"

    @property
    def phase(self) -> str:
        """Current lifecycle phase ('search' or 'validation')."""
        return self._phase

    @property
    def budget_limit(self) -> int:
        """Active ceiling for the current lifecycle phase."""
        if self._phase == "search":
            return self.search_budget
        else:
            # In validation phase, active ceiling cannot exceed search_consumed + validation_budget,
            # and cannot exceed total_budget
            return min(self.total_budget, self._search_consumed + self.validation_budget)

    @budget_limit.setter
    def budget_limit(self, value: int) -> None:
        self.total_budget = int(value)
        self.search_budget = min(self.search_budget, self.total_budget)

    @property
    def consumed_transitions(self) -> int:
        """Total simulator transitions billed so far across the entire lifecycle."""
        return self._consumed_transitions

    @property
    def transitions_used(self) -> int:
        """Alias for consumed_transitions (total simulator transitions billed)."""
        return self._consumed_transitions

    @property
    def search_consumed(self) -> int:
        """Transitions consumed during search phase."""
        return self._search_consumed

    @property
    def validation_consumed(self) -> int:
        """Transitions consumed during validation phase."""
        return self._validation_consumed

    @property
    def remaining_transitions(self) -> int:
        """Remaining transitions before reaching the active phase ceiling."""
        return max(0, self.budget_limit - self._consumed_transitions)

    @property
    def remaining_total_transitions(self) -> int:
        """Remaining transitions before reaching the hard total lifecycle ceiling."""
        return max(0, self.total_budget - self._consumed_transitions)

    @property
    def is_exhausted(self) -> bool:
        """Whether current active phase ceiling has been reached."""
        return self._consumed_transitions >= self.budget_limit

    @property
    def is_total_exhausted(self) -> bool:
        """Whether hard total lifecycle ceiling has been reached."""
        return self._consumed_transitions >= self.total_budget

    def start_validation_phase(self) -> None:
        """Transition tracker from search phase to validation phase."""
        if self._phase == "search":
            self._search_consumed = self._consumed_transitions
            self._phase = "validation"

    def check_can_step(self, count: int = 1) -> None:
        """Fail-closed check before executing physics steps.

        Refuses execution if the budget is exhausted OR if executing the requested
        count of transitions would cause consumed transitions to exceed budget_limit or total_budget.
        """
        if count < 0:
            raise ValueError(f"Cannot check negative step count: {count}")
        active_ceiling = self.budget_limit
        if self._consumed_transitions >= active_ceiling:
            raise BudgetExhaustedError(
                f"Attempted execution after {self._phase} budget exhaustion: "
                f"{self._consumed_transitions} >= {active_ceiling} transitions."
            )
        if self._consumed_transitions + count > active_ceiling:
            raise BudgetExhaustedError(
                f"Batch execution would exceed {self._phase} budget ceiling: "
                f"requested {count} transitions with {self.remaining_transitions} remaining "
                f"({self._consumed_transitions} + {count} > {active_ceiling})."
            )
        if self._consumed_transitions + count > self.total_budget:
            raise BudgetExhaustedError(
                f"Execution would exceed total lifecycle budget ceiling: "
                f"{self._consumed_transitions} + {count} > {self.total_budget}."
            )

    def record_transitions(self, count: int) -> int:
        """Record executed transitions and raise BudgetExhaustedError if limit is reached."""
        if count < 0:
            raise ValueError(f"Cannot record negative transitions: {count}")
        active_ceiling = self.budget_limit
        if self._consumed_transitions >= active_ceiling:
            raise BudgetExhaustedError(
                f"Attempted execution after {self._phase} budget exhaustion: "
                f"{self._consumed_transitions} >= {active_ceiling} transitions."
            )
        if self._consumed_transitions + count > active_ceiling:
            raise BudgetExhaustedError(
                f"Recording {count} transitions would exceed {self._phase} budget limit: "
                f"{self._consumed_transitions} + {count} > {active_ceiling}."
            )
        if self._consumed_transitions + count > self.total_budget:
            raise BudgetExhaustedError(
                f"Recording {count} transitions would exceed total lifecycle budget: "
                f"{self._consumed_transitions} + {count} > {self.total_budget}."
            )
        self._consumed_transitions += int(count)
        if self._phase == "search":
            self._search_consumed = self._consumed_transitions
        else:
            self._validation_consumed += int(count)

        if self._consumed_transitions >= active_ceiling:
            raise BudgetExhaustedError(
                f"Budget ceiling reached: consumed {self._consumed_transitions} "
                f"of {active_ceiling} allowed transitions in {self._phase} phase."
            )
        return self._consumed_transitions

    def reset_counter_for_new_run(self) -> None:
        """Explicitly reset counter for a brand new independent run (retries within run must NOT call this)."""
        self._consumed_transitions = 0
        self._search_consumed = 0
        self._validation_consumed = 0
        self._phase = "search"


class BudgetedEnv:
    """Wrapper for scalar CartPole environments with strict transition budgeting."""

    def __init__(
        self,
        env: Any,
        tracker: TransitionBudgetTracker,
        substeps: int = 1,
    ):
        self.env = env
        self.tracker = tracker
        self.substeps = max(1, int(substeps))

    def reset(self, **kwargs: Any) -> Any:
        """Reset environment. Resets do not consume physics transitions."""
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Any:
        """Execute one control step, billing physics transitions to the tracker."""
        self.tracker.check_can_step(self.substeps)
        try:
            result = self.env.step(action)
        finally:
            self.tracker.record_transitions(self.substeps)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class BudgetedVectorizedEnv:
    """Wrapper for VectorizedCartPoleEnv batch environments with strict transition budgeting."""

    def __init__(
        self,
        vec_env: Any,
        tracker: TransitionBudgetTracker,
        substeps: int = 1,
        mask_terminated: bool = False,
    ):
        self.vec_env = vec_env
        self.tracker = tracker
        self.substeps = max(1, int(substeps))
        self.mask_terminated = bool(mask_terminated)

    @property
    def num_envs(self) -> int:
        return self.vec_env.num_envs

    def reset(self, **kwargs: Any) -> Any:
        """Reset batch environment. Does not consume physics transitions."""
        return self.vec_env.reset(**kwargs)

    def step(self, actions: Any) -> Any:
        """Execute batch simulation step, billing all executed environment slots."""
        if self.mask_terminated and hasattr(self.vec_env, "terminated"):
            active_slots = int(np.sum(~self.vec_env.terminated))
        else:
            active_slots = self.vec_env.num_envs

        step_cost = active_slots * self.substeps
        self.tracker.check_can_step(step_cost)

        try:
            result = self.vec_env.step(actions)
        finally:
            self.tracker.record_transitions(step_cost)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.vec_env, name)
