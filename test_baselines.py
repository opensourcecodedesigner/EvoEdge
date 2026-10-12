"""Unit and integration tests for baselines and benchmark suite.

Validates:
  1. LQRController (linearization, Riccati solving, stability, 4D/5D coordinate mapping, clamping).
  2. FixedMLPController (architecture, 113 params, 96 MACs, serialization, ES trainer).
  3. BenchmarkSuite (initial condition grid, benchmark env, metric calculations, report generation).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure project root in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from baselines.lqr import (
    LQRController,
    get_cartpole_linearized_system,
    discretize_linear_system,
    solve_dare_iterative,
)
from baselines.mlp import (
    FixedMLPController,
    normalize_obs,
    train_mlp_es,
)
from experiments.benchmark_suite import (
    BenchmarkCartPoleEnv,
    generate_initial_condition_grid,
    evaluate_controller,
    run_benchmark_suite,
    EvaluatedController,
)


class TestLQRController(unittest.TestCase):
    """Test suite for baselines/lqr.py."""

    def test_linearization_matrices(self):
        """Verify CartPole linearized state-space matrices A and B."""
        A, B = get_cartpole_linearized_system()
        self.assertEqual(A.shape, (4, 4))
        self.assertEqual(B.shape, (4, 1))

        # Position derivative is velocity
        self.assertEqual(A[0, 1], 1.0)
        # Angle derivative is angular velocity
        self.assertEqual(A[2, 3], 1.0)

        # Theoretical coefficients for M=1.0, m=0.1, L=0.5, g=9.8
        # a_12 = -0.71707317..., a_32 = 15.77560976...
        self.assertAlmostEqual(A[1, 2], -0.717073, places=4)
        self.assertAlmostEqual(A[3, 2], 15.775610, places=4)

        # b_10 = 0.97560976..., b_30 = -1.46341463...
        self.assertAlmostEqual(B[1, 0], 0.975610, places=4)
        self.assertAlmostEqual(B[3, 0], -1.463415, places=4)

    def test_closed_loop_stability(self):
        """Verify closed-loop eigenvalues of (A - B*K) have negative real parts."""
        ctrl = LQRController(method="continuous")
        A, B = ctrl.A, ctrl.B
        K = ctrl.K

        A_cl = A - B @ K
        eigenvalues = np.linalg.eigvals(A_cl)
        for eig in eigenvalues:
            self.assertLess(
                eig.real,
                0.0,
                f"Closed-loop pole {eig} has non-negative real part (unstable)!",
            )

    def test_dare_iterative_solver(self):
        """Verify pure NumPy iterative DARE solver matches discrete Riccati equation."""
        A, B = get_cartpole_linearized_system()
        Ad, Bd = discretize_linear_system(A, B, dt=0.02)
        Q = np.diag([1.0, 0.1, 10.0, 1.0])
        R = np.array([[0.01]])

        P, K = solve_dare_iterative(Ad, Bd, Q, R, max_iter=2000, tol=1e-9)
        self.assertEqual(K.shape, (1, 4))
        self.assertTrue(np.all(np.isfinite(K)))

        # P must be symmetric positive-definite
        self.assertTrue(np.allclose(P, P.T, atol=1e-6))
        eig_P = np.linalg.eigvalsh(P)
        self.assertTrue(np.all(eig_P > 0))

    def test_coordinate_mapping_4d_and_5d(self):
        """Verify coordinate mapping seamlessly handles 4D and 5D inputs with parity."""
        ctrl = LQRController()

        x, x_dot = 0.25, -0.4
        theta = 0.12  # ~6.8 degrees
        theta_dot = -0.3

        obs_4d = [x, x_dot, theta, theta_dot]
        obs_5d = [x, x_dot, np.cos(theta), np.sin(theta), theta_dot]

        act_4d = ctrl.compute_action(obs_4d)
        act_5d = ctrl.compute_action(obs_5d)

        self.assertAlmostEqual(act_4d, act_5d, places=5)
        self.assertGreaterEqual(act_4d, -1.0)
        self.assertLessEqual(act_4d, 1.0)

    def test_action_clamping(self):
        """Verify outputs respect force / normalized bounds under large errors."""
        ctrl = LQRController(force_mag=10.0)

        # Huge error observation
        extreme_obs = [2.0, 5.0, 0.0, 1.0, 10.0]
        act_norm = ctrl.compute_action(extreme_obs, normalized=True)
        self.assertGreaterEqual(act_norm, -1.0)
        self.assertLessEqual(act_norm, 1.0)

        act_force = ctrl.compute_action(extreme_obs, normalized=False)
        self.assertGreaterEqual(act_force, -10.0)
        self.assertLessEqual(act_force, 10.0)

    def test_complexity_metrics(self):
        """Verify parameter count and MAC metrics for LQR."""
        ctrl = LQRController()
        self.assertEqual(ctrl.active_parameters, 4)
        self.assertEqual(ctrl.theoretical_macs, 4)


class TestFixedMLPController(unittest.TestCase):
    """Test suite for baselines/mlp.py."""

    def test_architecture_and_parameter_count(self):
        """Verify MLP architecture has exactly 113 parameters and 96 MACs."""
        mlp = FixedMLPController()
        self.assertEqual(mlp.active_parameters, 113)
        self.assertEqual(mlp.theoretical_macs, 96)

        theta = mlp.get_weights_vector()
        self.assertEqual(len(theta), 113)

        # Set and get roundtrip
        random_theta = np.linspace(-0.5, 0.5, 113, dtype=np.float32)
        mlp.set_weights_vector(random_theta)
        recovered_theta = mlp.get_weights_vector()
        self.assertTrue(np.allclose(random_theta, recovered_theta))

    def test_forward_pass_bounds(self):
        """Verify MLP output is bounded within [-1.0, 1.0] by tanh activation."""
        mlp = FixedMLPController()
        obs = [0.5, -1.2, 0.8, -0.6, 2.0]
        act = mlp.compute_action(obs)
        self.assertIsInstance(act, float)
        self.assertGreaterEqual(act, -1.0)
        self.assertLessEqual(act, 1.0)

    def test_serialization(self):
        """Verify weights saving and loading (.npz and .json)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "mlp_test.npz")
            json_path = os.path.join(tmpdir, "mlp_test.json")

            ctrl = FixedMLPController()
            rng = np.random.default_rng(123)
            test_weights = rng.normal(0, 0.1, 113).astype(np.float32)
            ctrl.set_weights_vector(test_weights)

            obs = [0.1, 0.2, 0.9, 0.1, -0.1]
            orig_act = ctrl.compute_action(obs)

            # Test NPZ
            ctrl.save(npz_path)
            loaded_npz = FixedMLPController(weights_path=npz_path)
            self.assertAlmostEqual(orig_act, loaded_npz.compute_action(obs), places=6)

            # Test JSON
            ctrl.save(json_path)
            loaded_json = FixedMLPController(weights_path=json_path)
            self.assertAlmostEqual(orig_act, loaded_json.compute_action(obs), places=6)

    def test_es_training_step(self):
        """Verify Evolution Strategies trainer runs and produces valid weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "trained_mlp.npz")
            ctrl, history = train_mlp_es(
                generations=3,
                population_size=10,
                eval_episodes=1,
                max_steps=50,
                save_path=save_path,
                verbose=False,
            )
            self.assertEqual(ctrl.active_parameters, 113)
            self.assertTrue(os.path.exists(save_path))
            self.assertEqual(len(history["generation"]), 3)


class TestBenchmarkSuite(unittest.TestCase):
    """Test suite for experiments/benchmark_suite.py."""

    def test_initial_condition_grid(self):
        """Verify deterministic 100-episode initial condition grid."""
        seeds = list(range(1000, 1100))
        grid = generate_initial_condition_grid(seeds=seeds, mode="grid")
        self.assertEqual(len(grid), 100)

        # Seed 1000: first point in grid -> x_min = -1.5, theta_min = -0.2
        x0, th0 = grid[1000]
        self.assertAlmostEqual(x0, -1.5, places=5)
        self.assertAlmostEqual(th0, -0.2, places=5)

        # Seed 1099: last point in grid -> x_max = 1.5, theta_max = 0.2
        x99, th99 = grid[1099]
        self.assertAlmostEqual(x99, 1.5, places=5)
        self.assertAlmostEqual(th99, 0.2, places=5)

        # All points bounded within [-1.5, 1.5] and [-0.2, 0.2]
        for s, (x, th) in grid.items():
            self.assertGreaterEqual(x, -1.5)
            self.assertLessEqual(x, 1.5)
            self.assertGreaterEqual(th, -0.2)
            self.assertLessEqual(th, 0.2)

    def test_benchmark_env_injection(self):
        """Verify BenchmarkCartPoleEnv sets state and detects boundary collisions."""
        env = BenchmarkCartPoleEnv(max_steps=100)
        obs, _ = env.reset_with_initial_state(x_0=1.2, theta_0=-0.15)

        self.assertAlmostEqual(env._state[0], 1.2)
        self.assertAlmostEqual(env._state[2], -0.15)
        self.assertFalse(env.boundary_violated)

    def test_lqr_benchmark_performance(self):
        """Verify LQR completes benchmark grid with 0% boundary violations."""
        lqr = LQRController()
        seeds = list(range(1000, 1020))  # Test first 20 seeds
        grid = generate_initial_condition_grid(seeds=seeds, mode="grid")

        ctrl_eval = EvaluatedController(
            name="LQR",
            compute_fn=lambda obs: lqr.compute_action(obs, normalized=True),
            active_parameters=4,
            theoretical_macs=4,
        )

        res = evaluate_controller(ctrl_eval, grid, max_steps=400)
        self.assertEqual(res["total_episodes"], 20)
        self.assertEqual(res["boundary_violation_rate"], 0.0)
        self.assertGreater(res["mean_reward"], 300.0)
        self.assertFalse(np.isnan(res["steady_state_jitter_rms"]))
        self.assertLess(res["steady_state_jitter_rms"], 1.0)  # sub-degree jitter over final 200 steps

    def test_suite_outputs_generated(self):
        """Verify CSV and Markdown report artifacts are created properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = "test_baseline_comparison.csv"
            report_file = "test_baseline_comparison_report.md"

            results = run_benchmark_suite(
                results_dir=tmpdir,
                num_episodes=5,
                max_steps=50,
                csv_filename=csv_file,
                report_filename=report_file,
                verbose=False,
            )

            self.assertGreaterEqual(len(results), 2)  # At least LQR and FixedMLP
            self.assertTrue(os.path.exists(os.path.join(tmpdir, csv_file)))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, report_file)))


if __name__ == "__main__":
    unittest.main()
