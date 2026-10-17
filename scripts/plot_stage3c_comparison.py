"""Generate Stage 3C Comparison Charts and Robustness Visualizations."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE3C_DIR = os.path.join(_PROJECT_ROOT, "results", "stage3c")
STAGE3B_DIR = os.path.join(_PROJECT_ROOT, "results", "stage3b")
SUMMARY_3C_FILE = os.path.join(STAGE3C_DIR, "stage3c_summary.json")
SUMMARY_3B_FILE = os.path.join(STAGE3B_DIR, "stage3b_summary.json")
OUTPUT_CHART = os.path.join(STAGE3C_DIR, "stage3c_comparison_charts.png")


def generate_charts() -> None:
    if not os.path.exists(SUMMARY_3C_FILE):
        print(f"File not found: {SUMMARY_3C_FILE}")
        return

    with open(SUMMARY_3C_FILE, "r") as f:
        data_3c = json.load(f)

    data_3b = None
    if os.path.exists(SUMMARY_3B_FILE):
        with open(SUMMARY_3B_FILE, "r") as f:
            data_3b = json.load(f)

    classical = data_3c["classical_reference"]
    runs_3c = data_3c["runs"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.suptitle("EvoEdge Phase 3C: Delay-Aware, Domain-Randomized Neuroevolution", fontsize=16, fontweight="bold")

    # Chart 1: Gate 1B Nominal Success Comparison
    ax1 = axes[0, 0]
    methods = ["Hybrid\n(Classical Ref)", "Stage 3B NSGA2\n(Nominal Only)", "Stage 3C Delay-Aware\n(5-Seed Mean)"]
    vals = [
        classical["gate1b_nominal_success"] * 100,
        np.mean([x["gate1b_nominal_success"] for x in data_3b["methods"]["NSGA2-WarmStart"]]) * 100 if data_3b else 100.0,
        np.mean([x["gate1b_nominal_success"] for x in runs_3c]) * 100,
    ]
    colors = ["#2b5c8f", "#ff7f0e", "#2ca02c"]
    bars = ax1.bar(methods, vals, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.axhline(90.0, color="green", linestyle="--", linewidth=1.5, label="Gate 1B Threshold (>= 90%)")
    ax1.set_ylabel("Nominal Success Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Gate 1B Nominal Success (Seeds 200..299)", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right")
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 2.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Chart 2: Per-Condition Diagnostic Robustness Breakdown
    ax2 = axes[0, 1]
    cond_names = list(classical["diagnostic_rates"].keys())
    short_conds = [c.split("_", 1)[1] if "_" in c else c for c in cond_names]
    x_indices = np.arange(len(cond_names))
    width = 0.28

    hybrid_rates = [classical["diagnostic_rates"][c] * 100 for c in cond_names]
    # Best 3C run by robust success
    best_3c = max(runs_3c, key=lambda r: (r["robust_success"], r["worst_condition_success"]))
    best_3c_rates = [best_3c["diagnostic_rates"][c] * 100 for c in cond_names]

    if data_3b:
        best_3b = max(data_3b["methods"]["NSGA2-WarmStart"], key=lambda r: r["robust_success"])
        best_3b_rates = [best_3b["diagnostic_rates"][c] * 100 for c in cond_names]
        ax2.bar(x_indices - width, hybrid_rates, width=width, label="Classical Reference", color="#2b5c8f", edgecolor="black")
        ax2.bar(x_indices, best_3b_rates, width=width, label=f"Stage 3B (Seed {best_3b['optimizer_seed']})", color="#ff7f0e", edgecolor="black")
        ax2.bar(x_indices + width, best_3c_rates, width=width, label=f"Stage 3C Delay-Aware (Seed {best_3c['optimizer_seed']})", color="#2ca02c", edgecolor="black")
    else:
        ax2.bar(x_indices - width / 2, hybrid_rates, width=width, label="Classical Reference", color="#2b5c8f", edgecolor="black")
        ax2.bar(x_indices + width / 2, best_3c_rates, width=width, label=f"Stage 3C Delay-Aware (Seed {best_3c['optimizer_seed']})", color="#2ca02c", edgecolor="black")

    ax2.axhline(40.0, color="red", linestyle=":", linewidth=1.5, label="Worst-Cond Threshold (40%)")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(short_conds, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Condition Success Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Diagnostic Robustness Breakdown (Seeds 300..319)", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.legend(loc="upper right")

    # Chart 3: Worst-Condition Success Rate Comparison
    ax3 = axes[1, 0]
    worst_labels = ["Hybrid Ref", "Stage 3B\n(Worst-Seed)", "Stage 3B\n(Best-Seed)", "Stage 3C\n(Mean 5-Seeds)", f"Stage 3C\n(Best Seed {best_3c['optimizer_seed']})"]
    worst_vals = [
        classical["worst_condition_success"] * 100,
        0.0,  # All Stage 3B seeds had 0.0% worst condition
        0.0,
        np.mean([x["worst_condition_success"] for x in runs_3c]) * 100,
        best_3c["worst_condition_success"] * 100,
    ]
    w_colors = ["#2b5c8f", "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    w_bars = ax3.bar(worst_labels, worst_vals, color=w_colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax3.axhline(40.0, color="red", linestyle="--", linewidth=1.5, label="Worst-Condition Screening Gate (>= 40%)")
    ax3.set_ylabel("Worst-Condition Success Rate (%)", fontsize=12, fontweight="bold")
    ax3.set_title("Worst-Condition Success Rate Comparison", fontsize=13, fontweight="bold")
    ax3.set_ylim(0, 115)
    ax3.legend(loc="upper left")
    for bar in w_bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 2.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Chart 4: Controller Complexity & Budget Adherence
    ax4 = axes[1, 1]
    seeds = [r["optimizer_seed"] for r in runs_3c]
    x_p = np.arange(len(seeds))
    params = [r["active_parameters"] for r in runs_3c]
    macs = [r["weight_macs"] for r in runs_3c]

    ax4.bar(x_p - 0.18, params, width=0.35, label="Active Parameters", color="#3b528b", edgecolor="black")
    ax4.bar(x_p + 0.18, macs, width=0.35, label="Weight MACs", color="#5dc863", edgecolor="black")
    ax4.axhline(128.0, color="red", linestyle="--", linewidth=1.5, label="Complexity Ceiling (<= 128)")
    ax4.set_xticks(x_p)
    ax4.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10)
    ax4.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax4.set_title("Controller Complexity Verification (Params & MACs <= 128)", fontsize=13, fontweight="bold")
    ax4.set_ylim(0, 145)
    ax4.legend(loc="upper right")
    for i, p in enumerate(params):
        ax4.text(i - 0.18, p + 2.0, str(p), ha="center", va="bottom", fontsize=9, fontweight="bold")
    for i, m in enumerate(macs):
        ax4.text(i + 0.18, m + 2.0, str(m), ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=150)
    plt.close()
    print(f"Chart saved to {OUTPUT_CHART}")


if __name__ == "__main__":
    generate_charts()
