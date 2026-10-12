"""Standalone C Header and Source Exporter for NEAT Neural Networks.

Compiles a NEAT network topology (from best_genome.pkl or topology.json) into a self-contained,
zero-dependency ANSI C header and source file (`controller.h`, `controller.c`).

Features:
- Static memory only: local stack arrays, zero heap allocations.
- Resolves node computation order via topological sorting (DAG dependency graph).
- Fast approximation of tanh / standard math.h tanhf.
- Inlines theoretical complexity metrics: total parameters, active synapses,
  hidden neurons, and theoretical MAC operations.
- Generates `float compute_action(const float obs[5]);` and `float run_controller(const float state[5]);`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from c_exporter import (
    load_model_topology,
    calculate_metrics,
    topological_sort_nodes,
    generate_c_header,
    generate_c_source,
    export_controller_c,
)

def load_topology(path: str) -> Dict[str, Any]:
    top, _ = load_model_topology(path)
    return top


def export_c(topology_path: str, output_path: str = "controller.h") -> str:
    """Compile a topology or pickle file into C files."""
    output_dir = os.path.dirname(output_path) or "."
    header_code, source_code, _ = export_controller_c(topology_path, output_dir=output_dir)
    # If a specific filename was requested other than controller.h, write to it
    if os.path.basename(output_path) != "controller.h":
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(header_code)
    return header_code


def main() -> None:
    from c_exporter import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
