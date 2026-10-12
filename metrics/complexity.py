"""Network complexity and inference-cost metrics for NEAT genomes.

All functions operate directly on `neat.DefaultGenome` objects and the
corresponding `neat.Config`, keeping this module independent of any
particular fitness function or environment.
"""

from __future__ import annotations

from typing import Any, Dict

import neat  # type: ignore


# ── Topology metrics ────────────────────────────────────────


def count_hidden_neurons(genome: neat.DefaultGenome, config: neat.Config) -> int:
    """Return the number of hidden (non-input, non-output) nodes.

    In neat-python input nodes are *not* stored in ``genome.nodes``, so
    hidden count = len(genome.nodes) − num_outputs.
    """
    return max(0, len(genome.nodes) - config.genome_config.num_outputs)


def count_active_connections(genome: neat.DefaultGenome) -> int:
    """Return the number of enabled (active) synaptic connections."""
    return sum(1 for c in genome.connections.values() if c.enabled)


# ── Inference cost estimation ───────────────────────────────


def estimate_operations(genome: neat.DefaultGenome, config: neat.Config) -> int:
    """Estimate single-sample inference cost in proxy MACs.

    Heuristic:
        ops = active_connections          (one multiply-accumulate each)
            + hidden_neurons              (one activation evaluation each)
            + num_outputs                 (one activation evaluation each)

    This intentionally *does not* count input nodes (they are pass-through).
    """
    active = count_active_connections(genome)
    hidden = count_hidden_neurons(genome, config)
    num_out = config.genome_config.num_outputs
    return active + hidden + num_out


# ── Aggregate convenience ───────────────────────────────────


def get_genome_metrics(
    genome: neat.DefaultGenome,
    config: neat.Config,
) -> Dict[str, Any]:
    """Return a dict of all complexity / cost metrics for *genome*."""
    return {
        "hidden_neurons": count_hidden_neurons(genome, config),
        "active_connections": count_active_connections(genome),
        "estimated_operations": estimate_operations(genome, config),
    }
