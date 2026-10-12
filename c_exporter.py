"""Standalone C Exporter for NEAT Neural Controllers (Microcontroller / ESP32).

Compiles a NEAT network (from best_genome.pkl or topology.json) into standalone,
zero-dynamic-allocation C files:
  - controller.h
  - controller.c

Features:
- Static memory only: local stack variables, 0 bytes dynamic allocation (malloc/free).
- Fast approximation of tanh / standard math.h tanhf.
- Inlines theoretical complexity metrics: total parameters, active synapses,
  hidden neurons, and theoretical MAC operations for embedded firmware engineers.
- Exposes `float compute_action(const float obs[5]);`
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.serialization import extract_topology, load_genome


def load_model_topology(path: str) -> Tuple[Dict[str, Any], str]:
    """Load topology from either a .pkl genome or .json topology file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input model file not found: {path}")

    if path.endswith(".pkl"):
        data = load_genome(path)
        genome = data["genome"]
        config = data["config"]
        topology = extract_topology(genome, config)
        return topology, os.path.basename(path)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            topology = json.load(fh)
        return topology, os.path.basename(path)
    else:
        # Try as pickle first, then json
        try:
            data = load_genome(path)
            topology = extract_topology(data["genome"], data["config"])
            return topology, os.path.basename(path)
        except Exception:
            with open(path, "r", encoding="utf-8") as fh:
                topology = json.load(fh)
            return topology, os.path.basename(path)


def calculate_metrics(topology: Dict[str, Any]) -> Dict[str, int]:
    """Calculate theoretical complexity metrics from network topology."""
    num_inputs = topology.get("num_inputs", 5)
    num_outputs = topology.get("num_outputs", 1)

    nodes = topology.get("nodes", [])
    conns = topology.get("connections", [])

    active_conns = [c for c in conns if c.get("enabled", True)]
    active_synapses = len(active_conns)

    hidden_nodes = [n for n in nodes if n.get("type") == "hidden"]
    output_nodes = [n for n in nodes if n.get("type") == "output"]
    hidden_neurons = len(hidden_nodes)

    total_parameters = active_synapses + len(hidden_nodes) + len(output_nodes)
    theoretical_macs = active_synapses + hidden_neurons + num_outputs

    return {
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "hidden_neurons": hidden_neurons,
        "active_synapses": active_synapses,
        "total_parameters": total_parameters,
        "theoretical_macs": theoretical_macs,
    }


def topological_sort_nodes(topology: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sort non-input nodes so dependencies are computed in feedforward order."""
    nodes = topology.get("nodes", [])
    conns = topology.get("connections", [])
    active_conns = [c for c in conns if c.get("enabled", True)]

    node_map = {n["id"]: n for n in nodes}
    input_ids = {n["id"] for n in nodes if n.get("type") == "input"}
    non_input_nodes = {n["id"]: n for n in nodes if n.get("type") != "input"}

    in_deps: Dict[int, Set[int]] = {nid: set() for nid in non_input_nodes}
    for c in active_conns:
        to_id = c["to"]
        from_id = c["from"]
        if to_id in in_deps:
            in_deps[to_id].add(from_id)

    resolved: Set[int] = set(input_ids)
    sorted_nodes: List[Dict[str, Any]] = []
    remaining = set(non_input_nodes.keys())

    while remaining:
        ready = [nid for nid in remaining if in_deps[nid].issubset(resolved)]
        if not ready:
            ready = sorted(remaining, key=lambda x: len(in_deps[x] - resolved))[:1]

        ready.sort(key=lambda nid: (0 if node_map[nid].get("type") == "hidden" else 1, nid))
        chosen_id = ready[0]
        sorted_nodes.append(node_map[chosen_id])
        resolved.add(chosen_id)
        remaining.remove(chosen_id)

    return sorted_nodes


def generate_c_header(topology: Dict[str, Any], source_name: str) -> str:
    """Generate header controller.h content."""
    metrics = calculate_metrics(topology)

    lines = [
        "/* ========================================================================",
        " * NEAT CartPole Swing-Up — Embedded C Controller Header",
        " * ========================================================================",
        f" * Generated from: {source_name}",
        " * Target Platform: Standalone Microcontroller / ESP32 / Bare-Metal C",
        " *",
        " * Theoretical Complexity & Hardware Footprint:",
        f" *   - Total Parameters:           {metrics['total_parameters']}",
        f" *   - Active Synapses:            {metrics['active_synapses']}",
        f" *   - Hidden Neurons:             {metrics['hidden_neurons']}",
        f" *   - Theoretical MAC Operations: {metrics['theoretical_macs']}",
        f" *   - Memory Footprint:           0 bytes heap (Static stack allocation only)",
        f" *   - Observation Dimension:      {metrics['num_inputs']} floats",
        f" *   - Action Dimension:           {metrics['num_outputs']} float in [-1.0, 1.0]",
        " *",
        " * Observation Vector Format (const float obs[5]):",
        " *   obs[0] : Cart Position (normalized: x / 2.4)",
        " *   obs[1] : Cart Velocity (normalized: clip(x_dot / 10.0, -1, 1))",
        " *   obs[2] : Pole Cosine Angle (cos(theta))",
        " *   obs[3] : Pole Sine Angle (sin(theta))",
        " *   obs[4] : Pole Angular Velocity (normalized: clip(theta_dot / 10.0, -1, 1))",
        " * ======================================================================== */",
        "",
        "#ifndef CONTROLLER_H",
        "#define CONTROLLER_H",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "/* Architectural Metrics (Compile-Time Constants) */",
        f"#define CONTROLLER_TOTAL_PARAMS     {metrics['total_parameters']}",
        f"#define CONTROLLER_ACTIVE_SYNAPSES  {metrics['active_synapses']}",
        f"#define CONTROLLER_HIDDEN_NEURONS   {metrics['hidden_neurons']}",
        f"#define CONTROLLER_THEORETICAL_MACS {metrics['theoretical_macs']}",
        "",
        "/**",
        " * Compute continuous control action from normalized observations.",
        " * Zero heap allocation; completely thread-safe and re-entrant.",
        " *",
        " * @param obs   Input observation array of length 5 (normalized).",
        " * @return      Control action in [-1.0, 1.0] (force command).",
        " */",
        "float compute_action(const float obs[5]);",
        "",
        "/**",
        " * Compatibility alias matching legacy run_controller signature.",
        " */",
        "static inline float run_controller(const float state[5]) {",
        "    return compute_action(state);",
        "}",
        "",
        "/**",
        " * Helper to normalize raw Gym/Sensor states to obs[5] format.",
        " *",
        " * @param raw_state   [x (m), x_dot (m/s), cos_theta, sin_theta, theta_dot (rad/s)]",
        " * @param normalized  Output array of length 5.",
        " */",
        "void normalize_state(const float raw_state[5], float normalized[5]);",
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        "#endif /* CONTROLLER_H */",
        "",
    ]
    return "\n".join(lines)


def generate_c_source(topology: Dict[str, Any], source_name: str) -> str:
    """Generate source controller.c content."""
    metrics = calculate_metrics(topology)
    sorted_nodes = topological_sort_nodes(topology)
    num_inputs = metrics["num_inputs"]

    conns = topology.get("connections", [])
    active_conns = [c for c in conns if c.get("enabled", True)]

    incoming_by_target: Dict[int, List[Dict[str, Any]]] = {}
    for c in active_conns:
        to_id = c["to"]
        incoming_by_target.setdefault(to_id, []).append(c)

    for to_id in incoming_by_target:
        incoming_by_target[to_id].sort(key=lambda x: x["from"])

    lines = [
        "/* ========================================================================",
        " * NEAT CartPole Swing-Up — Standalone C Controller Implementation",
        " * ========================================================================",
        f" * Generated from: {source_name}",
        f" * Parameters: {metrics['total_parameters']} | MACs: {metrics['theoretical_macs']}",
        " * ======================================================================== */",
        "",
        '#include "controller.h"',
        "#include <math.h>",
        "",
        "/* ── Fast / Embedded Activation Approximations ────────────────────────── */",
        "",
        "/* Fast rational Pade approximation for tanh on low-power microcontrollers */",
        "#if defined(USE_FAST_MATH)",
        "static inline float fast_tanhf(float x) {",
        "    float x2 = x * x;",
        "    return x * (27.0f + x2) / (27.0f + 9.0f * x2);",
        "}",
        "#else",
        "static inline float fast_tanhf(float x) {",
        "    return tanhf(x);",
        "}",
        "#endif",
        "",
        "static inline float _neat_act_tanh(float z) {",
        "    float scaled = 2.5f * z;",
        "    if (scaled > 60.0f) scaled = 60.0f;",
        "    else if (scaled < -60.0f) scaled = -60.0f;",
        "    return fast_tanhf(scaled);",
        "}",
        "",
        "static inline float _neat_act_identity(float z) {",
        "    return z;",
        "}",
        "",
        "static inline float _neat_act_relu(float z) {",
        "    return (z > 0.0f) ? z : 0.0f;",
        "}",
        "",
        "/* ── State Normalization Helper ────────────────────────────────────────── */",
        "",
        "void normalize_state(const float raw_state[5], float normalized[5]) {",
        "    normalized[0] = raw_state[0] / 2.4f;",
        "    float x_dot_norm = raw_state[1] / 10.0f;",
        "    normalized[1] = (x_dot_norm > 1.0f) ? 1.0f : ((x_dot_norm < -1.0f) ? -1.0f : x_dot_norm);",
        "    normalized[2] = raw_state[2];",
        "    normalized[3] = raw_state[3];",
        "    float th_dot_norm = raw_state[4] / 10.0f;",
        "    normalized[4] = (th_dot_norm > 1.0f) ? 1.0f : ((th_dot_norm < -1.0f) ? -1.0f : th_dot_norm);",
        "}",
        "",
        "/* ── Forward Action Computation (Static Memory Only) ───────────────────── */",
        "",
        "float compute_action(const float obs[5]) {",
    ]

    def _format_source_ref(from_id: int) -> str:
        if from_id < 0:
            idx = -(from_id) - 1
            if 0 <= idx < num_inputs:
                return f"obs[{idx}]"
            return f"0.0f /* invalid input node {from_id} */"
        return f"node_{from_id}"

    act_map = {
        "tanh": "_neat_act_tanh",
        "identity": "_neat_act_identity",
        "linear": "_neat_act_identity",
        "relu": "_neat_act_relu",
    }

    for node in sorted_nodes:
        nid = node["id"]
        ntype = node.get("type", "node")
        raw_act = node.get("activation") or "tanh"
        act_func = act_map.get(raw_act, "_neat_act_tanh")
        bias = float(node.get("bias") if node.get("bias") is not None else 0.0)
        response = float(node.get("response") if node.get("response") is not None else 1.0)

        in_conns = incoming_by_target.get(nid, [])

        lines.append(f"    /* Node {nid} ({ntype}, bias: {bias:+.6f}f) */")
        lines.append(f"    float node_{nid}_sum = 0.0f;")
        for conn in in_conns:
            from_ref = _format_source_ref(conn["from"])
            w = float(conn["weight"])
            lines.append(f"    node_{nid}_sum += {from_ref} * ({w:+.7f}f);")

        if abs(response - 1.0) < 1e-7:
            eval_expr = f"{bias:+.7f}f + node_{nid}_sum"
        else:
            eval_expr = f"{bias:+.7f}f + ({response:.7f}f) * node_{nid}_sum"

        lines.append(f"    float node_{nid} = {act_func}({eval_expr});")
        lines.append("")

    output_nodes = [n["id"] for n in sorted_nodes if n.get("type") == "output"]
    if output_nodes:
        ret_id = output_nodes[0]
        lines.append(f"    return node_{ret_id};")
    elif sorted_nodes:
        ret_id = sorted_nodes[-1]["id"]
        lines.append(f"    return node_{ret_id};")
    else:
        lines.append("    return 0.0f;")

    lines.extend([
        "}",
        "",
    ])

    return "\n".join(lines)


def export_controller_c(
    model_path: str,
    output_dir: str = "results",
) -> Tuple[str, str, str]:
    """Export controller.h and controller.c from model_path to output_dir."""
    topology, source_name = load_model_topology(model_path)
    metrics = calculate_metrics(topology)

    header_code = generate_c_header(topology, source_name)
    source_code = generate_c_source(topology, source_name)

    os.makedirs(output_dir, exist_ok=True)
    header_path = os.path.join(output_dir, "controller.h")
    source_path = os.path.join(output_dir, "controller.c")

    with open(header_path, "w", encoding="utf-8") as fh:
        fh.write(header_code)

    with open(source_path, "w", encoding="utf-8") as fh:
        fh.write(source_code)

    print(f"\n[OK] Exported C controller files:")
    print(f"     Header:     {header_path}")
    print(f"     Source:     {source_path}")
    print(f"     Parameters: {metrics['total_parameters']}")
    print(f"     Synapses:   {metrics['active_synapses']}")
    print(f"     MACs:       {metrics['theoretical_macs']}")

    return header_code, source_code, source_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile NEAT champion genome into standalone C code for microcontrollers."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Path to best_genome.pkl or topology.json file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="results",
        help="Output directory (default: results/).",
    )
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        # Search for best genome in results/
        candidates = [
            "results/performance/seed_42/best_genome.pkl",
            "results/hardware/seed_42/best_genome.pkl",
            "results/best_genome.pkl",
            "results/swingup/seed_42/best_genome.pkl",
            "results/performance/seed_42/topology.json",
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

        if not model_path:
            for root, _, files in os.walk(args.output_dir):
                if "best_genome.pkl" in files:
                    model_path = os.path.join(root, "best_genome.pkl")
                    break
                elif "topology.json" in files:
                    model_path = os.path.join(root, "topology.json")
                    break

    if not model_path or not os.path.exists(model_path):
        print(f"Error: No model found at '{model_path}'. Run evolution first.", file=sys.stderr)
        sys.exit(1)

    header_code, source_code, source_path = export_controller_c(model_path, args.output_dir)

    print("\n" + "=" * 70)
    print("  Generated Standalone C Function (compute_action):")
    print("=" * 70)
    print(source_code)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
