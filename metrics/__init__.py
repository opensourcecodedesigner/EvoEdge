"""Network complexity, inference cost, and fitness metrics."""

from metrics.complexity import (
    count_hidden_neurons,
    count_active_connections,
    estimate_operations,
    get_genome_metrics,
)
from metrics.ops_counter import (
    compute_feedforward_ops,
    count_dense_layer_ops,
    audit_model_accounting,
)

__all__ = [
    "count_hidden_neurons",
    "count_active_connections",
    "estimate_operations",
    "get_genome_metrics",
    "compute_feedforward_ops",
    "count_dense_layer_ops",
    "audit_model_accounting",
]
