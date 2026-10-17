"""Evolution Strategies (ES) Optimizer and Training Entry Point for EvoEdge Stage 2.

Implements Natural Evolution Strategies with antithetic sampling and Adam optimization:
  1. Documented deterministic integer namespace for NumPy SeedSequence:
     NAMESPACE_ES = 1
     spawn_key = (NAMESPACE_ES, optimizer_seed)
     Guarantees strict separation from NSGA-II streams while repeating deterministically.
  2. Fail-closed transition budgeting:
     Integrates TransitionBudgetTracker, BudgetedEnv, and BudgetedVectorizedEnv.
     Guarantees search halts immediately upon reaching budget ceiling.
  3. Exact transition arithmetic:
     1 initial center eval (10 eps x 500 = 5,000 transitions)
     18 generations x 20 mutants (18 x 20 x 10 eps x 500 = 1,800,000 transitions)
     9 biennial center tracking evals (9 x 10 eps x 500 = 45,000 transitions)
     Total Search Allocation = 370 candidate evals = exactly 1,850,000 transitions.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from baselines.mlp import FixedMLPController, normalize_obs
from env.cartpole_swingup import CartPoleSwingUpEnv
from env.budget_tracker import (
    BudgetExhaustedError,
    TransitionBudgetTracker,
    BudgetedEnv,
    BudgetedVectorizedEnv,
)

# Documented deterministic integer namespace for ES optimizer
NAMESPACE_ES: int = 1


def get_es_rng(optimizer_seed: int) -> Tuple[np.random.Generator, np.random.SeedSequence]:
    """Construct an isolated, deterministic NumPy Generator for ES optimizer.

    Args:
        optimizer_seed: Base integer seed (e.g., 42..46).

    Returns:
        Tuple of (numpy Generator, SeedSequence).
    """
    if not isinstance(optimizer_seed, (int, np.integer)):
        raise TypeError(f"optimizer_seed must be an integer, got {type(optimizer_seed)}")
    spawn_key = (NAMESPACE_ES, int(optimizer_seed))
    seed_seq = np.random.SeedSequence(entropy=int(optimizer_seed), spawn_key=spawn_key)
    return np.random.default_rng(seed_seq), seed_seq


def evaluate_es_candidate(
    theta: np.ndarray,
    env: Any,
    ep_seeds: List[int],
    max_steps: int = 500,
    tracker: Optional[TransitionBudgetTracker] = None,
) -> float:
    """Evaluate one parameter vector across deterministic episodes using a budgeted environment.

    Bills simulator transitions monotonically to the budget tracker.
    """
    ctrl = FixedMLPController(weights=theta)
    rewards: List[float] = []

    for s in ep_seeds:
        if tracker is not None and tracker.is_exhausted:
            raise BudgetExhaustedError("Budget exhausted before starting episode.")

        obs, _ = env.reset(seed=s)
        ep_r = 0.0
        done = False
        step_i = 0

        while not done and step_i < max_steps:
            action = ctrl.compute_action(obs)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += float(r)
            done = term or trunc
            step_i += 1

        rewards.append(ep_r)

    return float(np.mean(rewards)) if rewards else -1000.0


def train_es(
    optimizer_seed: int,
    generations: int = 1000,
    population_size: int = 20,
    eval_episodes: int = 10,
    max_steps: int = 500,
    sigma: float = 0.10,
    lr: float = 0.05,
    budget_limit: Optional[int] = None,
    total_budget: int = 2_000_000,
    search_budget: int = 1_850_000,
    validation_budget: int = 150_000,
    budget_tracker: Optional[TransitionBudgetTracker] = None,
    train_seeds: Optional[List[int]] = None,
    env_factory: Optional[Callable[[], Any]] = None,
    reward_mode: str = "stage3b",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 3B Evolution Strategies with strict transition budget enforcement.

    Args:
        optimizer_seed: Integer seed for optimizer RNG.
        generations: Maximum generations (default 1000, loops until search budget exhausted).
        population_size: Number of mutant candidates per gen (must be even for antithetic sampling).
        eval_episodes: Number of episodes per candidate eval (default 10).
        max_steps: Maximum steps per episode (default 500).
        sigma: Mutation noise scale.
        lr: Adam learning rate.
        budget_limit: Optional legacy single budget limit.
        total_budget: Maximum transitions across search + validation lifecycle (default 2,000,000).
        search_budget: Maximum transitions allowed for search phase (default 1,850,000).
        validation_budget: Maximum transitions reserved for validation phase (default 150,000).
        budget_tracker: Optional existing TransitionBudgetTracker. If None, creates one.
        train_seeds: Optional list of training seeds (defaults to seeds 0..199).
        env_factory: Optional factory returning an environment instance.
        reward_mode: Reward function formulation ('stage3b' or 'legacy').
        verbose: Print progress.

    Returns:
        Dictionary containing best parameters, history, tracker, and stop metadata.
    """
    import time
    t0 = time.time()

    if population_size % 2 != 0:
        raise ValueError(f"population_size must be even for antithetic sampling, got {population_size}")

    if budget_tracker is None:
        if budget_limit is not None:
            # Legacy or test override
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

    rng, seed_seq = get_es_rng(optimizer_seed)

    if train_seeds is None:
        train_seeds = list(range(0, 200))

    if env_factory is None:
        base_env = CartPoleSwingUpEnv(max_steps=max_steps, continuous=True, reward_mode=reward_mode)
    else:
        base_env = env_factory()

    # Wrap base environment in BudgetedEnv
    budgeted_env = BudgetedEnv(base_env, budget_tracker)

    n_params = FixedMLPController.TOTAL_PARAMS
    theta = rng.normal(0.0, 0.02, size=n_params).astype(np.float32)

    # Adam state
    m = np.zeros(n_params, dtype=np.float32)
    v = np.zeros(n_params, dtype=np.float32)
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8

    best_score = -float("inf")
    best_theta = theta.copy()
    history: Dict[str, List[float]] = {"generation": [], "best_reward": [], "mean_reward": []}
    half_pop = population_size // 2

    stopped_by_budget = False
    stopping_reason = "max_generations_reached"
    completed_generations = 0
    candidate_evaluations = 0

    try:
        # 1. Initial Center Evaluation (1 evaluation: eval_episodes x max_steps = 5,000 transitions max)
        if budget_tracker.remaining_transitions >= 1:
            ep_seeds_init = [train_seeds[(optimizer_seed + i) % len(train_seeds)] for i in range(eval_episodes)]
            init_score = evaluate_es_candidate(theta, budgeted_env, ep_seeds_init, max_steps=max_steps, tracker=budget_tracker)
            candidate_evaluations += 1
            best_score = init_score
            best_theta = theta.copy()
        else:
            stopped_by_budget = True
            stopping_reason = "search_budget_exhausted"

        # 2. Generational Search Loop (continues until search budget is exhausted)
        for gen in range(1, generations + 1):
            if budget_tracker.is_exhausted or budget_tracker.remaining_transitions < 1:
                stopped_by_budget = True
                stopping_reason = "search_budget_exhausted"
                break

            # Antithetic perturbations from namespaced RNG
            eps = rng.standard_normal((half_pop, n_params)).astype(np.float32)
            perturbations = np.vstack([eps, -eps])

            gen_rewards: List[float] = []
            gen_ep_seeds = [
                train_seeds[(optimizer_seed + gen * 37 + i) % len(train_seeds)]
                for i in range(eval_episodes)
            ]

            # Evaluate each mutant candidate with strict pre-check
            aborted_generation = False
            for p in perturbations:
                if budget_tracker.is_exhausted or budget_tracker.remaining_transitions < 1:
                    stopped_by_budget = True
                    stopping_reason = "search_budget_exhausted"
                    aborted_generation = True
                    break
                cand_theta = theta + sigma * p
                r = evaluate_es_candidate(cand_theta, budgeted_env, gen_ep_seeds, max_steps=max_steps, tracker=budget_tracker)
                candidate_evaluations += 1
                gen_rewards.append(r)

            if aborted_generation or len(gen_rewards) < population_size:
                # If generation could not evaluate full population before budget exhaustion, break
                break

            rewards_arr = np.array(gen_rewards, dtype=np.float32)

            # Rank-based fitness shaping
            order = np.argsort(np.argsort(rewards_arr))
            utilities = np.maximum(0.0, np.log(len(rewards_arr) / 2.0 + 1.0) - np.log(len(rewards_arr) - order))
            util_sum = np.sum(utilities)
            if util_sum > 0:
                utilities = utilities / util_sum - (1.0 / len(rewards_arr))

            # Gradient estimate and Adam update
            grad = (1.0 / (population_size * sigma)) * np.dot(utilities, perturbations)
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - beta1 ** gen)
            v_hat = v / (1.0 - beta2 ** gen)
            theta += lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            # Biennial center tracking if budget remains
            if gen % 2 == 0 and budget_tracker.remaining_transitions >= 1:
                center_score = evaluate_es_candidate(theta, budgeted_env, gen_ep_seeds, max_steps=max_steps, tracker=budget_tracker)
                candidate_evaluations += 1
                if center_score > best_score:
                    best_score = center_score
                    best_theta = theta.copy()

            gen_max = float(np.max(rewards_arr))
            gen_mean = float(np.mean(rewards_arr))
            history["generation"].append(gen)
            history["best_reward"].append(gen_max)
            history["mean_reward"].append(gen_mean)
            completed_generations = gen

            if gen_max > best_score:
                best_idx = int(np.argmax(rewards_arr))
                best_score = gen_max
                best_theta = (theta + sigma * perturbations[best_idx]).copy()

            if verbose and (gen % 5 == 0 or gen == 1):
                print(f"[ES Gen {gen:3d}] Best: {gen_max:7.2f} | Mean: {gen_mean:7.2f} | Consumed: {budget_tracker.consumed_transitions:,} / {budget_tracker.search_budget:,}")

    except BudgetExhaustedError:
        stopped_by_budget = True
        stopping_reason = "search_budget_exhausted"

    wall_clock_time = time.time() - t0
    spawn_key = (NAMESPACE_ES, optimizer_seed)
    return {
        "best_theta": best_theta,
        "best_score": best_score,
        "history": history,
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
        "namespace": NAMESPACE_ES,
        "reward_mode": reward_mode,
    }


def validate_es_champion(
    controller: FixedMLPController,
    tracker: TransitionBudgetTracker,
    max_steps: int = 500,
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

    # 1. Gate 1B: Nominal screening
    gate1b_succ = 0
    nominal_cond = DIAGNOSTIC_CONDITIONS["1_Nominal"]
    for s in gate1b_seeds:
        if tracker.is_exhausted or tracker.is_total_exhausted:
            raise BudgetExhaustedError("Budget exhausted during Gate 1B validation screening.")
        base_env = PerturbedCartPoleEnv(max_steps=max_steps, continuous=True, seed=s, **nominal_cond)
        benv = BudgetedEnv(base_env, tracker)
        obs, _ = benv.reset(seed=s)
        cos_thetas: List[float] = []
        rail_hit = False
        for _ in range(max_steps):
            action = controller.compute_action(obs)
            obs, r, term, trunc, _ = benv.step(action)
            cos_thetas.append(float(obs[2]))
            if term:
                rail_hit = True
                break
        is_succ, _, _, _ = evaluate_success_predicate(cos_thetas, rail_hit, max_steps)
        if is_succ:
            gate1b_succ += 1

    gate1b_rate = gate1b_succ / len(gate1b_seeds)

    # 2. Multi-condition diagnostic screening
    diag_rates: Dict[str, float] = {}
    for cond_name, cond_params in DIAGNOSTIC_CONDITIONS.items():
        cond_succ = 0
        for s in diag_seeds:
            if tracker.is_exhausted or tracker.is_total_exhausted:
                raise BudgetExhaustedError("Budget exhausted during diagnostic validation screening.")
            base_env = PerturbedCartPoleEnv(max_steps=max_steps, continuous=True, seed=s, **cond_params)
            benv = BudgetedEnv(base_env, tracker)
            obs, _ = benv.reset(seed=s)
            cos_thetas = []
            rail_hit = False
            for _ in range(max_steps):
                action = controller.compute_action(obs)
                obs, r, term, trunc, _ = benv.step(action)
                cos_thetas.append(float(obs[2]))
                if term:
                    rail_hit = True
                    break
            is_succ, _, _, _ = evaluate_success_predicate(cos_thetas, rail_hit, max_steps)
            if is_succ:
                cond_succ += 1
        diag_rates[cond_name] = cond_succ / len(diag_seeds)

    robust_success = float(np.mean(list(diag_rates.values()))) if diag_rates else 0.0
    worst_success = float(np.min(list(diag_rates.values()))) if diag_rates else 0.0

    if verbose:
        print(f"Validation complete: Gate 1B: {gate1b_rate*100:.1f}% | Robust: {robust_success*100:.1f}% | Worst: {worst_success*100:.1f}%")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2 Evolution Strategies (ES) Trainer")
    parser.add_argument("--seed", type=int, default=42, help="Optimizer seed (default: 42)")
    parser.add_argument("--generations", type=int, default=18, help="Number of generations (default: 18)")
    parser.add_argument("--pop-size", type=int, default=20, help="Mutant population size (default: 20)")
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

    res = train_es(
        optimizer_seed=args.seed,
        generations=args.generations,
        population_size=args.pop_size,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        total_budget=t_budget,
        search_budget=s_budget,
        validation_budget=args.validation_budget,
        verbose=args.verbose,
    )

    ctrl = FixedMLPController(weights=res["best_theta"])

    val_res = None
    if args.run_validation:
        val_res = validate_es_champion(ctrl, res["tracker"], max_steps=args.max_steps, verbose=args.verbose)
        val_res["search_transitions_used"] = res["tracker"].search_consumed
        val_res["champion_complexity"] = {"active_parameters": 113, "weight_macs": 96}
        val_res["champion_fitness"] = {"best_score": float(res["best_score"])}
        val_res["history"] = {
            "generation": [int(x) for x in res["history"]["generation"]],
            "best_reward": [float(x) for x in res["history"]["best_reward"]],
            "mean_reward": [float(x) for x in res["history"]["mean_reward"]],
        }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        save_file = os.path.join(args.output_dir, f"es_champion_seed{args.seed}.npz")
        ctrl.save(save_file)
        print(f"Saved ES champion to {save_file} (Consumed transitions: {res['tracker'].consumed_transitions:,})")
        if val_res:
            meta_file = os.path.join(args.output_dir, f"es_champion_seed{args.seed}_validation.json")
            with open(meta_file, "w") as f:
                json.dump(val_res, f, indent=2)
            print(f"Saved validation metadata to {meta_file}")


