"""Statistical comparison module.

Reads the machine-readable ``results.csv``, computes per-mode aggregates
(mean ± std), and emits a Markdown comparison table.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


# ── CSV loader ───────────────────────────────────────────────


def load_results(csv_path: str) -> List[Dict[str, Any]]:
    """Parse ``results.csv`` into a list of typed dicts."""
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                {
                    "seed": int(raw["seed"]),
                    "mode": raw["mode"],
                    "best_reward": float(raw["best_reward"]),
                    "mean_reward": float(raw["mean_reward"]),
                    "hidden_neurons": int(raw["hidden_neurons"]),
                    "active_connections": int(raw["active_connections"]),
                    "estimated_operations": int(raw["estimated_operations"]),
                    "success_flag": raw["success_flag"] == "True",
                }
            )
    return rows


# ── Statistics ───────────────────────────────────────────────


def compute_statistics(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return ``{mode: {metric_mean, metric_std, ...}}`` aggregates."""
    by_mode: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    stats: Dict[str, Dict[str, Any]] = {}
    numeric_keys = [
        "best_reward",
        "mean_reward",
        "hidden_neurons",
        "active_connections",
        "estimated_operations",
    ]

    for mode, runs in by_mode.items():
        n = len(runs)
        entry: Dict[str, Any] = {
            "n_runs": n,
            "success_rate": sum(1 for r in runs if r["success_flag"]) / max(n, 1),
        }
        for key in numeric_keys:
            vals = np.array([r[key] for r in runs], dtype=float)
            entry[f"{key}_mean"] = float(np.mean(vals))
            entry[f"{key}_std"] = float(np.std(vals))
        stats[mode] = entry

    return stats


# ── Markdown report ──────────────────────────────────────────

_MODE_ORDER = ["performance", "complexity", "hardware"]


def generate_markdown_table(
    stats: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
) -> str:
    """Build a Markdown comparison table and optionally write it to disk."""
    lines = [
        "# Experimental Results – NEAT CartPole Swing-Up",
        "",
        "## Statistical Comparison Across Modes",
        "",
        (
            "| Mode | Best Reward | Mean Reward | Hidden Neurons "
            "| Active Conns | Est. Ops | Success Rate |"
        ),
        (
            "|------|-------------|-------------|----------------"
            "|--------------|----------|--------------|"
        ),
    ]

    for mode in _MODE_ORDER:
        if mode not in stats:
            continue
        s = stats[mode]
        n = s["n_runs"]
        succ = int(s["success_rate"] * n)
        lines.append(
            f"| {mode} "
            f"| {s['best_reward_mean']:.2f} ± {s['best_reward_std']:.2f} "
            f"| {s['mean_reward_mean']:.2f} ± {s['mean_reward_std']:.2f} "
            f"| {s['hidden_neurons_mean']:.1f} ± {s['hidden_neurons_std']:.1f} "
            f"| {s['active_connections_mean']:.1f} ± {s['active_connections_std']:.1f} "
            f"| {s['estimated_operations_mean']:.1f} ± {s['estimated_operations_std']:.1f} "
            f"| {s['success_rate']:.0%} ({succ}/{n}) |"
        )

    lines += [
        "",
        "### Column Definitions",
        "",
        "| Column | Meaning |",
        "|--------|---------|",
        "| **Best Reward** | Raw (unpenalised) reward of the best evolved genome |",
        "| **Mean Reward** | Population mean fitness in the final generation (includes any penalty) |",
        "| **Hidden Neurons** | Hidden nodes in the best genome's topology |",
        "| **Active Conns** | Enabled synaptic connections in the best genome |",
        "| **Est. Ops** | Estimated single-sample inference cost (connections + nodes) |",
        "| **Success Rate** | Fraction of seeds where best reward exceeded the success threshold |",
        "",
    ]

    text = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"  Report -> {output_path}")

    return text


# ── High-level entry point ───────────────────────────────────


def compare_results(
    csv_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load CSV → aggregate → print & write Markdown table."""
    results = load_results(csv_path)
    stats = compute_statistics(results)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(csv_path), "comparison_report.md"
        )

    table = generate_markdown_table(stats, output_path)
    print(table)
    return stats
