"""Genome serialisation and topology extraction.

Provides pickle-based save/load for NEAT genomes and a JSON-serialisable
topology representation for downstream analysis or visualisation.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict

import neat  # type: ignore

from metrics.complexity import count_active_connections, count_hidden_neurons


# ── Pickle-based genome persistence ─────────────────────────


def save_genome(
    genome: neat.DefaultGenome,
    config: neat.Config,
    filepath: str,
) -> None:
    """Serialise *genome* (and its config) to *filepath* via pickle."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "wb") as fh:
        pickle.dump({"genome": genome, "config": config}, fh)


def load_genome(filepath: str) -> dict:
    """Load a ``{genome, config}`` dict previously written by :func:`save_genome`."""
    with open(filepath, "rb") as fh:
        return pickle.load(fh)


# ── Topology extraction ─────────────────────────────────────


def extract_topology(
    genome: neat.DefaultGenome,
    config: neat.Config,
) -> Dict[str, Any]:
    """Return the network graph as a JSON-friendly dictionary.

    Structure::

        {
          "num_inputs":  int,
          "num_outputs": int,
          "num_hidden":  int,
          "num_active_connections": int,
          "nodes": [
            {"id": int, "type": "input"|"output"|"hidden",
             "activation": str|null, "bias": float|null, "response": float|null},
            ...
          ],
          "connections": [
            {"from": int, "to": int, "weight": float, "enabled": bool},
            ...
          ]
        }
    """
    gc = config.genome_config
    num_inputs = gc.num_inputs
    num_outputs = gc.num_outputs

    nodes = []

    # Input nodes (not stored in genome.nodes)
    for i in range(num_inputs):
        nodes.append(
            {
                "id": -(i + 1),
                "type": "input",
                "activation": None,
                "bias": None,
                "response": None,
            }
        )

    # Output and hidden nodes
    for key, node in genome.nodes.items():
        node_type = "output" if key < num_outputs else "hidden"
        nodes.append(
            {
                "id": key,
                "type": node_type,
                "activation": node.activation,
                "bias": round(node.bias, 6),
                "response": round(node.response, 6),
            }
        )

    connections = []
    for (in_node, out_node), conn in genome.connections.items():
        connections.append(
            {
                "from": in_node,
                "to": out_node,
                "weight": round(conn.weight, 6),
                "enabled": conn.enabled,
            }
        )

    return {
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "num_hidden": count_hidden_neurons(genome, config),
        "num_active_connections": count_active_connections(genome),
        "nodes": nodes,
        "connections": connections,
    }


def save_topology(
    genome: neat.DefaultGenome,
    config: neat.Config,
    filepath: str,
) -> Dict[str, Any]:
    """Extract topology and write it to *filepath* as JSON."""
    topology = extract_topology(genome, config)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(topology, fh, indent=2)
    return topology
