"""Validation tests for final modules: stress_test.py and c_exporter.py."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

# Ensure project root in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.stress_test import (
    PerturbedCartPoleEnv,
    compute_retention_pct,
    find_champion_genomes,
    run_stress_test,
)
from utils.c_exporter import (
    calculate_metrics,
    export_c,
    generate_c_header,
    load_topology,
    topological_sort_nodes,
)


class TestCExporter(unittest.TestCase):
    """Test suite for utils/c_exporter.py."""

    def setUp(self):
        self.sample_topology_path = os.path.join(
            "results", "performance", "seed_42", "topology.json"
        )

    def test_metrics_calculation(self):
        """Verify metric calculation on seed 42 champion topology."""
        topology = load_topology(self.sample_topology_path)
        metrics = calculate_metrics(topology)

        expected_conns = sum(1 for c in topology.get("connections", []) if c.get("enabled", True))
        expected_hidden = sum(1 for n in topology.get("nodes", []) if n.get("type") == "hidden")
        expected_params = expected_conns + expected_hidden + topology.get("num_outputs", 1)
        expected_macs = expected_conns + expected_hidden + topology.get("num_outputs", 1)

        self.assertEqual(metrics["num_inputs"], topology.get("num_inputs", 5))
        self.assertEqual(metrics["num_outputs"], topology.get("num_outputs", 1))
        self.assertEqual(metrics["hidden_neurons"], expected_hidden)
        self.assertEqual(metrics["active_synapses"], expected_conns)
        self.assertEqual(metrics["total_parameters"], expected_params)
        self.assertEqual(metrics["theoretical_macs"], expected_macs)

    def test_topological_sort_feedforward_dag(self):
        """Verify topological sorting on multi-layer DAG with hidden neurons."""
        # 5 inputs (-1 to -5)
        # Hidden node 1: depends on inputs -1 and -2
        # Hidden node 2: depends on Hidden node 1 and input -3
        # Output node 0: depends on Hidden node 2 and Hidden node 1
        custom_topology = {
            "num_inputs": 5,
            "num_outputs": 1,
            "num_hidden": 2,
            "num_active_connections": 6,
            "nodes": [
                {"id": -1, "type": "input", "activation": None, "bias": None, "response": None},
                {"id": -2, "type": "input", "activation": None, "bias": None, "response": None},
                {"id": -3, "type": "input", "activation": None, "bias": None, "response": None},
                {"id": -4, "type": "input", "activation": None, "bias": None, "response": None},
                {"id": -5, "type": "input", "activation": None, "bias": None, "response": None},
                {"id": 0, "type": "output", "activation": "tanh", "bias": 0.1, "response": 1.0},
                {"id": 1, "type": "hidden", "activation": "tanh", "bias": -0.2, "response": 1.0},
                {"id": 2, "type": "hidden", "activation": "tanh", "bias": 0.3, "response": 1.0},
            ],
            "connections": [
                {"from": -1, "to": 1, "weight": 0.5, "enabled": True},
                {"from": -2, "to": 1, "weight": -0.4, "enabled": True},
                {"from": 1, "to": 2, "weight": 0.8, "enabled": True},
                {"from": -3, "to": 2, "weight": 0.2, "enabled": True},
                {"from": 2, "to": 0, "weight": -0.7, "enabled": True},
                {"from": 1, "to": 0, "weight": 0.3, "enabled": True},
                {"from": -4, "to": 0, "weight": 9.9, "enabled": False},  # disabled
            ],
        }

        sorted_nodes = topological_sort_nodes(custom_topology)
        sorted_ids = [n["id"] for n in sorted_nodes]

        # Node 1 must be computed before Node 2, and Node 2 before Output Node 0
        idx_1 = sorted_ids.index(1)
        idx_2 = sorted_ids.index(2)
        idx_0 = sorted_ids.index(0)

        self.assertLess(idx_1, idx_2, "Node 1 must precede Node 2")
        self.assertLess(idx_2, idx_0, "Node 2 must precede Node 0")

        # Metrics
        metrics = calculate_metrics(custom_topology)
        self.assertEqual(metrics["hidden_neurons"], 2)
        self.assertEqual(metrics["active_synapses"], 6)
        self.assertEqual(metrics["total_parameters"], 6 + 3)  # 6 weights + 3 biases
        self.assertEqual(metrics["theoretical_macs"], 6 + 2 + 1)  # 6 conns + 2 hidden + 1 out

    def test_c_compilation_and_numerical_parity(self):
        """Export controller.h, compile with GCC, and verify numerical parity."""
        scratch_dir = os.path.join(_PROJECT_ROOT, ".test_scratch")
        os.makedirs(scratch_dir, exist_ok=True)

        header_path = os.path.join(scratch_dir, "test_controller.h")
        export_c(self.sample_topology_path, output_path=header_path)

        c_source = f"""#include <stdio.h>
#include <math.h>
#include "test_controller.h"

int main(void) {{
    const float s[5] = {{0.12f, -0.34f, 0.88f, -0.47f, 0.56f}};
    float out = run_controller(s);
    printf("%.9f\\n", out);
    return 0;
}}
"""
        c_file = os.path.join(scratch_dir, "test_runner.c")
        exe_file = os.path.join(scratch_dir, "test_runner.exe")

        with open(c_file, "w", encoding="utf-8") as fh:
            fh.write(c_source)

        try:
            # Compile with GCC
            comp_res = subprocess.run(
                ["gcc", "-Wall", "-Wextra", "-Wpedantic", "-I", scratch_dir, c_file, "-o", exe_file],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                comp_res.returncode, 0, f"GCC compilation failed: {comp_res.stderr}"
            )

            # Run C executable
            run_res = subprocess.run([exe_file], capture_output=True, text=True)
            self.assertEqual(run_res.returncode, 0)
            c_output = float(run_res.stdout.strip())

            # Evaluate exact NEAT math in Python
            # tanh(2.5 * (bias + sum(w * x)))
            topology = load_topology(self.sample_topology_path)
            node_0 = [n for n in topology["nodes"] if n["id"] == 0][0]
            bias = float(node_0["bias"])
            conns = [c for c in topology["connections"] if c["enabled"]]

            s = [0.12, -0.34, 0.88, -0.47, 0.56]
            weighted_sum = sum(
                c["weight"] * s[-(c["from"]) - 1] for c in conns
            )
            z = 2.5 * (bias + weighted_sum)
            z_clamped = max(-60.0, min(60.0, z))
            py_output = float(np.tanh(z_clamped))

            self.assertAlmostEqual(
                c_output, py_output, places=5,
                msg=f"C output ({c_output}) does not match Python ({py_output})"
            )
        finally:
            import shutil
            shutil.rmtree(scratch_dir, ignore_errors=True)


class TestStressTest(unittest.TestCase):
    """Test suite for analysis/stress_test.py."""

    def test_perturbed_env_dynamics(self):
        """Verify physics parameters scaling in PerturbedCartPoleEnv."""
        env_base = PerturbedCartPoleEnv(masscart_scale=1.0, length_scale=1.0)
        self.assertEqual(env_base.MASS_CART, 1.0)
        self.assertEqual(env_base.LENGTH, 0.5)
        self.assertAlmostEqual(env_base._total_mass, 1.1)

        env_heavy = PerturbedCartPoleEnv(masscart_scale=1.4, length_scale=1.0)
        self.assertAlmostEqual(env_heavy.MASS_CART, 1.4)
        self.assertAlmostEqual(env_heavy._total_mass, 1.5)

        env_long = PerturbedCartPoleEnv(masscart_scale=1.0, length_scale=1.3)
        self.assertAlmostEqual(env_long.LENGTH, 0.65)
        self.assertAlmostEqual(env_long._polemass_length, 0.1 * 0.65)

    def test_noisy_sensors(self):
        """Verify sensor noise perturbation is applied to observations."""
        env_clean = PerturbedCartPoleEnv(obs_noise_sigma=0.0)
        obs_clean, _ = env_clean.reset(seed=123)

        env_noisy = PerturbedCartPoleEnv(obs_noise_sigma=0.05)
        obs_noisy, _ = env_noisy.reset(seed=123)

        # Observations should differ due to noise
        self.assertFalse(np.allclose(obs_clean, obs_noisy))

    def test_retention_calculation(self):
        """Verify percentage performance retention math."""
        self.assertEqual(compute_retention_pct(100.0, 100.0), 100.0)
        self.assertEqual(compute_retention_pct(80.0, 100.0), 80.0)
        self.assertEqual(compute_retention_pct(-10.0, 100.0), 0.0)
        self.assertEqual(compute_retention_pct(-10.0, -20.0), 0.0)  # non-positive baseline

    def test_stress_test_run(self):
        """Run stress test and verify CSV and PNG outputs."""
        results = run_stress_test(results_dir="results", num_episodes=2)
        self.assertGreater(len(results), 0)

        csv_path = os.path.join("results", "stress_test_results.csv")
        self.assertTrue(os.path.exists(csv_path))

        png_path = os.path.join("results", "robustness_comparison.png")
        self.assertTrue(os.path.exists(png_path))
        self.assertGreater(os.path.getsize(png_path), 1000)


if __name__ == "__main__":
    unittest.main()
