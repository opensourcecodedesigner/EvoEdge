"""NEAT evolution engine.

Thin orchestration layer: loads config, seeds RNGs, wires reporters,
and delegates fitness evaluation to :mod:`evolution.fitness`.
"""

from __future__ import annotations

import random as py_random
from typing import Tuple

import numpy as np
import neat  # type: ignore

from evolution.fitness import GenomeEvaluator


def run_evolution(
    neat_config_path: str,
    experiment_config,
    mode: str,
    seed: int,
) -> Tuple[neat.DefaultGenome, neat.Config, neat.StatisticsReporter, GenomeEvaluator]:
    """Run a single NEAT evolution and return the best genome + metadata.

    Parameters
    ----------
    neat_config_path : str
        Path to the ``neat_config.ini`` file.
    experiment_config : ExperimentConfig
        Centralised experiment hyperparameters.
    mode : str
        One of ``"performance"``, ``"complexity"``, ``"hardware"``.
    seed : int
        Random seed – controls genome initialisation, mutation, *and*
        environment episode seeds so the run is fully reproducible.

    Returns
    -------
    best_genome : neat.DefaultGenome
    neat_config : neat.Config
    stats       : neat.StatisticsReporter
    evaluator   : GenomeEvaluator
    """
    # ── Reproducibility ──────────────────────────────────────
    py_random.seed(seed)
    np.random.seed(seed)

    # ── Load & patch NEAT config ─────────────────────────────
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )
    config.pop_size = experiment_config.population_size

    # ── Build population ─────────────────────────────────────
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # ── Evolve ───────────────────────────────────────────────
    evaluator = GenomeEvaluator(experiment_config, mode, seed)
    best_genome = population.run(evaluator, experiment_config.generations)

    return best_genome, config, stats, evaluator
