"""Network complexity, inference cost, and fitness metrics."""

from metrics.complexity import (
    count_hidden_neurons,
    count_active_connections,
    estimate_operations,
    get_genome_metrics,
)

__all__ = [
    "count_hidden_neurons",
    "count_active_connections",
    "estimate_operations",
    "get_genome_metrics",
]
