"""Centralized experiment configuration.

All tuneable parameters for the NEAT CartPole Swing-Up experiments live here
so that every module pulls from a single source of truth.
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class ExperimentConfig:
    """Immutable-ish bag of experiment hyperparameters."""

    # ── Evolution ──────────────────────────────────────────────
    population_size: int = 150
    generations: int = 100

    # ── Environment ────────────────────────────────────────────
    max_episode_steps: int = 200
    num_eval_episodes: int = 3  # episodes per genome evaluation

    # ── Experiment design ──────────────────────────────────────
    modes: List[str] = field(
        default_factory=lambda: ["performance", "complexity", "hardware"]
    )
    seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 1024]
    )

    # ── Fitness / penalty ──────────────────────────────────────
    #  survival_threshold: a genome must exceed this *raw* reward
    #  before any complexity or hardware penalty is applied.
    #  0.0 ≈ "pole above horizontal on average".
    survival_threshold: float = 0.0
    complexity_penalty_weight: float = 0.5   # λ for complexity mode
    hardware_penalty_weight: float = 0.3     # λ for hardware mode
    success_threshold: float = 100.0         # raw reward → success flag

    # ── Paths ──────────────────────────────────────────────────
    results_dir: str = "results"
    neat_config_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "neat_config.ini"
    )

    def __post_init__(self) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
