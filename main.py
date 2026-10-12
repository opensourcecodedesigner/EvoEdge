"""NEAT CartPole Swing-Up — Experimental Framework
====================================================

Evolve minimal neural controllers for the CartPole Swing-Up task
under three optimisation pressures:

    performance   Maximise raw episode reward.
    complexity    Maximise reward, penalise topology size.
    hardware      Maximise reward, penalise estimated inference cost.

Usage examples
--------------
    python main.py                                # Full sweep (3 modes × 5 seeds)
    python main.py --modes performance complexity # Subset of modes
    python main.py --seeds 42 123 --generations 50
    python main.py --compare                      # Re-generate report from CSV
    python main.py --stress-test                  # Evaluate robustness under shift
    python main.py --export-c                     # Export champion network to C
    python main.py --ui                           # Real-time web visualizer & sandbox
    python main.py --run-baselines                # Run frozen benchmark suite (LQR, MLP, NEAT)
"""

from __future__ import annotations

import argparse
import os
import sys

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

from config.experiment_config import ExperimentConfig
from experiments.runner import run_all_experiments
from analysis.compare import compare_results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NEAT CartPole Swing-Up - Experimental Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--modes",
        nargs="+",
        choices=["performance", "complexity", "hardware"],
        default=None,
        help="Experimental modes to run (default: all three).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds for independent runs (default: 42 123 456 789 1024).",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of NEAT generations per run (default: 100).",
    )
    p.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="NEAT population size (default: 150).",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for all outputs (default: results/).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Skip evolution; only generate the comparison report from existing CSV.",
    )
    p.add_argument(
        "--stress-test",
        action="store_true",
        help="Run robustness stress test across all modes and seeds.",
    )
    p.add_argument(
        "--export-c",
        nargs="?",
        const="auto",
        default=None,
        metavar="TOPOLOGY_JSON",
        help="Compile a topology.json file into a standalone C header (controller.h). "
             "If path is omitted, exports the champion topology from results/.",
    )
    p.add_argument(
        "--visualize",
        "--ui",
        action="store_true",
        dest="visualize",
        help="Launch the real-time web visualizer and sandbox server (ui_server.py).",
    )
    p.add_argument(
        "--run-baselines",
        action="store_true",
        help="Execute frozen deterministic benchmark suite (LQR, FixedMLP, and NEAT champions).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    config = ExperimentConfig()

    # ── Override defaults from CLI ───────────────────────────
    if args.modes is not None:
        config.modes = args.modes
    if args.seeds is not None:
        config.seeds = args.seeds
    if args.generations is not None:
        config.generations = args.generations
    if args.pop_size is not None:
        config.population_size = args.pop_size
    if args.results_dir is not None:
        config.results_dir = args.results_dir
        os.makedirs(config.results_dir, exist_ok=True)

    csv_path = os.path.join(config.results_dir, "results.csv")

    # ── Stress-test mode ─────────────────────────────────────
    if args.stress_test:
        from analysis.stress_test import run_stress_test
        run_stress_test(results_dir=config.results_dir)
        return

    # ── Export C controller mode ─────────────────────────────
    if args.export_c is not None:
        from utils.c_exporter import export_c
        topology_path = args.export_c
        if topology_path == "auto":
            found_topology = None
            for root, _, files in os.walk(config.results_dir):
                if "topology.json" in files:
                    found_topology = os.path.join(root, "topology.json")
                    break
            if not found_topology:
                print(
                    f"Error: No topology.json found in '{config.results_dir}'. "
                    "Run experiments first or specify a path."
                )
                sys.exit(1)
            topology_path = found_topology

        output_path = os.path.join(config.results_dir, "controller.h")
        export_c(topology_path, output_path=output_path)
        root_header = "controller.h"
        if os.path.abspath(output_path) != os.path.abspath(root_header):
            import shutil
            shutil.copyfile(output_path, root_header)
            print(f"     Root copy created:   {root_header}")
        return

    # ── Real-time Web Visualizer mode ────────────────────────
    if args.visualize:
        import subprocess
        python_exe = sys.executable or "python"
        ui_script = os.path.join(_PROJECT_ROOT if "_PROJECT_ROOT" in globals() else os.path.dirname(os.path.abspath(__file__)), "ui_server.py")
        subprocess.run([python_exe, ui_script])
        return

    # ── Frozen Baseline Benchmark Suite mode ─────────────────
    if args.run_baselines:
        from experiments.benchmark_suite import run_benchmark_suite
        run_benchmark_suite(results_dir=config.results_dir)
        return

    # ── Compare-only mode ────────────────────────────────────
    if args.compare:
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found. Run experiments first.")
            sys.exit(1)
        compare_results(csv_path)
        return

    # ── Print experiment plan ────────────────────────────────
    total_runs = len(config.modes) * len(config.seeds)
    print("\n" + "=" * 64)
    print("  NEAT CartPole Swing-Up - Experimental Framework")
    print("=" * 64)
    print(f"  Modes        : {', '.join(config.modes)}")
    print(f"  Seeds        : {config.seeds}")
    print(f"  Generations  : {config.generations}")
    print(f"  Pop size     : {config.population_size}")
    print(f"  Total runs   : {total_runs}")
    print(f"  Results dir  : {config.results_dir}")
    print(f"  Survival thr : {config.survival_threshold}")
    print("=" * 64 + "\n")

    # ── Run experiments ──────────────────────────────────────
    run_all_experiments(config)

    # ── Generate comparison report ───────────────────────────
    if os.path.exists(csv_path):
        print("\n" + "=" * 64)
        print("  Generating Statistical Comparison")
        print("=" * 64 + "\n")
        compare_results(csv_path)


if __name__ == "__main__":
    main()
