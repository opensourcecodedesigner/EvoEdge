"""Generate Stage 3B Per-Condition Comparison Charts and Visualizations."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE3B_DIR = os.path.join(_PROJECT_ROOT, "results", "stage3b")
SUMMARY_FILE = os.path.join(STAGE3B_DIR, "stage3b_summary.json")
OUTPUT_CHART = os.path.join(STAGE3B_DIR, "stage3b_comparison_charts.png")

with open(SUMMARY_FILE, "r") as f:
    data = json.load(f)

classical = data["classical_reference"]
warmstart = data["methods"]["NSGA2-WarmStart"]
es_scratch = data["methods"]["ES-FromScratch"]
nsga2_scratch = data["methods"]["NSGA2-FromScratch"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
fig.suptitle("EvoEdge Phase 3B: Neuroevolution Recovery & Robust Controller Discovery", fontsize=16, fontweight="bold")

# Chart 1: Gate 1B Nominal Success Comparison
ax1 = axes[0, 0]
methods = ["Hybrid (Baseline)", "NSGA2-WarmStart\n(Mean of 5 Seeds)", "ES-FromScratch\n(Mean of 5 Seeds)", "NSGA2-FromScratch\n(Mean of 5 Seeds)"]
gate1b_vals = [
    classical["gate1b_nominal_success"] * 100,
    np.mean([x["gate1b_nominal_success"] for x in warmstart]) * 100,
    np.mean([x["gate1b_nominal_success"] for x in es_scratch]) * 100,
    np.mean([x["gate1b_nominal_success"] for x in nsga2_scratch]) * 100,
]
colors = ["#2b5c8f", "#2ca02c", "#d62728", "#ff7f0e"]
bars = ax1.bar(methods, gate1b_vals, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
ax1.axhline(90.0, color="green", linestyle="--", linewidth=1.5, label="Gate 1B Target (>= 90%)")
ax1.set_ylabel("Nominal Success Rate (%)", fontsize=12, fontweight="bold")
ax1.set_title("Gate 1B Nominal Success Rate (Seeds 200..299)", fontsize=13, fontweight="bold")
ax1.set_ylim(0, 115)
ax1.legend(loc="upper right")
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 2.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

# Chart 2: Per-Condition Diagnostic Breakdown (Hybrid vs NSGA2-WarmStart)
ax2 = axes[0, 1]
cond_names = list(classical["diagnostic_rates"].keys())
short_conds = [c.split("_", 1)[1] if "_" in c else c for c in cond_names]
x_indices = np.arange(len(cond_names))
width = 0.35

hybrid_rates = [classical["diagnostic_rates"][c] * 100 for c in cond_names]
ws_seed46 = warmstart[4]  # Seed 46 (best performer, robust 44.5%)
ws_rates = [ws_seed46["diagnostic_rates"][c] * 100 for c in cond_names]

ax2.bar(x_indices - width / 2, hybrid_rates, width=width, label="Classical Hybrid", color="#2b5c8f", edgecolor="black")
ax2.bar(x_indices + width / 2, ws_rates, width=width, label="NSGA2-WarmStart (Seed 46)", color="#2ca02c", edgecolor="black")
ax2.axhline(40.0, color="red", linestyle=":", linewidth=1.5, label="Worst-Cond Floor (40%)")
ax2.set_xticks(x_indices)
ax2.set_xticklabels(short_conds, rotation=35, ha="right", fontsize=9)
ax2.set_ylabel("Condition Success Rate (%)", fontsize=12, fontweight="bold")
ax2.set_title("Per-Condition Diagnostic Robustness (Seeds 300..319)", fontsize=13, fontweight="bold")
ax2.set_ylim(0, 115)
ax2.legend(loc="upper right")

# Chart 3: Transition Budget Utilization Accounting
ax3 = axes[1, 0]
runs = [
    "NSGA2-WS\nSeed 42", "NSGA2-WS\nSeed 46",
    "ES-Scratch\nSeed 42", "ES-Scratch\nSeed 46",
    "NSGA2-Scratch\nSeed 42", "NSGA2-Scratch\nSeed 46"
]
search_cons = [
    warmstart[0]["training_search_transitions"], warmstart[4]["training_search_transitions"],
    es_scratch[0]["training_search_transitions"], es_scratch[4]["training_search_transitions"],
    nsga2_scratch[0]["training_search_transitions"], nsga2_scratch[4]["training_search_transitions"],
]
val_cons = [
    warmstart[0]["validation_transitions"], warmstart[4]["validation_transitions"],
    es_scratch[0]["validation_transitions"], es_scratch[4]["validation_transitions"],
    nsga2_scratch[0]["validation_transitions"], nsga2_scratch[4]["validation_transitions"],
]
x_pos = np.arange(len(runs))
p1 = ax3.bar(x_pos, search_cons, width=0.5, label="Search Phase Transitions", color="#1f77b4", edgecolor="black")
p2 = ax3.bar(x_pos, val_cons, width=0.5, bottom=search_cons, label="Validation Transitions", color="#aec7e8", edgecolor="black")
ax3.axhline(1_850_000, color="orange", linestyle="--", linewidth=1.5, label="Search Ceiling (1.85M)")
ax3.axhline(2_000_000, color="purple", linestyle="-.", linewidth=1.5, label="Total Ceiling (2.0M)")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(runs, fontsize=9)
ax3.set_ylabel("Simulator Transitions", fontsize=12, fontweight="bold")
ax3.set_title("Simulator Transition Accounting & Fail-Closed Ceilings", fontsize=13, fontweight="bold")
ax3.set_ylim(0, 2_350_000)
ax3.legend(loc="lower right")

# Chart 4: Controller Complexity vs Performance (Pareto Front Projection)
ax4 = axes[1, 1]
macs_hybrid = 0
gate1b_hybrid = classical["gate1b_nominal_success"] * 100
ax4.scatter([macs_hybrid], [gate1b_hybrid], s=180, color="#2b5c8f", marker="*", label="Hybrid (Classical Bang-Bang)", zorder=5)

ws_macs = [x["weight_macs"] for x in warmstart]
ws_gate1b = [x["gate1b_nominal_success"] * 100 for x in warmstart]
ax4.scatter(ws_macs, ws_gate1b, s=120, color="#2ca02c", marker="o", label="NSGA2-WarmStart (Learned, 48 MACs)", zorder=5)

es_macs = [x["weight_macs"] for x in es_scratch]
es_gate1b = [x["gate1b_nominal_success"] * 100 for x in es_scratch]
ax4.scatter(es_macs, es_gate1b, s=100, color="#d62728", marker="x", label="ES-FromScratch (Learned, 96 MACs)", zorder=5)

nsga2_s_macs = [x["weight_macs"] for x in nsga2_scratch]
nsga2_s_gate1b = [x["gate1b_nominal_success"] * 100 for x in nsga2_scratch]
ax4.scatter(nsga2_s_macs, nsga2_s_gate1b, s=100, color="#ff7f0e", marker="^", label="NSGA2-FromScratch (Learned, 38-48 MACs)", zorder=5)

ax4.axvline(128, color="black", linestyle=":", linewidth=1.5, label="Complexity Limit (<= 128 MACs)")
ax4.axhline(90, color="green", linestyle="--", linewidth=1.5, label="Nominal Target (>= 90%)")
ax4.set_xlabel("Inference Complexity (Weight MACs)", fontsize=12, fontweight="bold")
ax4.set_ylabel("Gate 1B Nominal Success Rate (%)", fontsize=12, fontweight="bold")
ax4.set_title("Controller Complexity vs. Nominal Performance", fontsize=13, fontweight="bold")
ax4.set_xlim(-10, 140)
ax4.set_ylim(-10, 115)
ax4.legend(loc="center right")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUTPUT_CHART)
plt.close()

print(f"Comparison chart successfully generated at: {OUTPUT_CHART}")
