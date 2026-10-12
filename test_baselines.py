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


class TestEnergyShapingLinearFeedbackController(unittest.TestCase):
    """Test suite for baselines/hybrid_controller.py (Fix 1)."""

    def test_true_lqr_computation(self):
        """Verify continuous ARE Riccati solver produces K close to [-1.0, -1.21, -7.81, -1.87]."""
        from baselines.hybrid_controller import compute_true_lqr_gains
        K = compute_true_lqr_gains()
        self.assertEqual(len(K), 4)
        self.assertAlmostEqual(K[0], -1.00, places=2)
        self.assertAlmostEqual(K[1], -1.21, places=2)
        self.assertAlmostEqual(K[2], -7.81, places=2)
        self.assertAlmostEqual(K[3], -1.87, places=2)

    def test_pumping_sign(self):
        """Verify pumping sign accelerates cart in direction assisting inverted pendulum."""
        from baselines.hybrid_controller import EnergyShapingLinearFeedbackController
        ctrl = EnergyShapingLinearFeedbackController(k_E=0.6, k_x=0.0, k_v=0.0)

        # Inverted hanging (theta = pi, cos_th = -1.0, sin_th = 0.0), positive velocity
        obs_pos = [0.0, 0.0, -1.0, 0.0, 0.5]
        u_pos = ctrl.compute_action(obs_pos)
        # E < 0, th_dot * cos_th = 0.5 * (-1) = -0.5 < 0
        # sign_E = -1, sign_p = -1 -> sign_E * sign_p = +1 -> u = +0.6
        self.assertAlmostEqual(u_pos, 0.6, places=4)

        # Inverted hanging, negative velocity
        obs_neg = [0.0, 0.0, -1.0, 0.0, -0.5]
        u_neg = ctrl.compute_action(obs_neg)
        # sign_E = -1, sign_p = +1 -> sign_E * sign_p = -1 -> u = -0.6
        self.assertAlmostEqual(u_neg, -0.6, places=4)

    def test_energy_increasing_during_pumping(self):
        """Verify that mechanical energy strictly increases during pumping phase."""
        from env.cartpole_swingup import CartPoleSwingUpEnv
        from baselines.hybrid_controller import EnergyShapingLinearFeedbackController

        env = CartPoleSwingUpEnv(max_steps=100, continuous=True)
        obs, _ = env.reset(seed=1000)
        ctrl = EnergyShapingLinearFeedbackController(k_E=0.6, k_x=0.5, k_v=0.25)

        c_kin = (2.0 / 3.0) * env.MASS_POLE * (env.LENGTH ** 2)
        c_pot = env.MASS_POLE * env.GRAVITY * env.LENGTH

        def calc_energy(observation):
            _, _, cos_th, _, th_dot = observation
            return c_kin * (th_dot ** 2) + c_pot * (cos_th - 1.0)

        initial_energy = calc_energy(obs)

        energies = [initial_energy]
        for _ in range(50):
            a = ctrl.compute_action(obs)
            obs, _, term, trunc, _ = env.step(a)
            energies.append(calc_energy(obs))
            if term:
                break

        # Energy must have pumped up significantly above initial hanging energy (-0.98 J)
        max_energy = max(energies)
        final_energy = energies[-1]
        self.assertGreater(max_energy, initial_energy + 0.3, "Energy failed to pump up during pumping phase")
        self.assertGreater(final_energy, initial_energy, "Final energy is not greater than initial energy")

    def test_swingup_gate_100_percent(self):
        """GATE: Swing-up success must be 100% over seeds 1000-1099."""
        from env.cartpole_swingup import CartPoleSwingUpEnv
        from baselines.hybrid_controller import EnergyShapingLinearFeedbackController

        ctrl = EnergyShapingLinearFeedbackController(k_E=0.6, k_x=0.5, k_v=0.25)
        successes = 0
        total_seeds = 100

        for s in range(1000, 1000 + total_seeds):
            env = CartPoleSwingUpEnv(max_steps=500, continuous=True)
            obs, _ = env.reset(seed=s)
            cos_thetas = []
            rail_hit = False

            for _ in range(500):
                a = ctrl.compute_action(obs)
                obs, _, term, trunc, _ = env.step(a)
                cos_thetas.append(obs[2])
                if term:
                    rail_hit = True
                    break

            if not rail_hit and len(cos_thetas) == 500:
                final_100 = cos_thetas[-100:]
                if sum(1 for c in final_100 if c > 0.9) >= 95:
                    successes += 1

        self.assertEqual(
            successes,
            total_seeds,
            f"Fix 1 GATE failed: expected 100% success (100/{total_seeds}), got {successes}/{total_seeds}",
        )

    def test_closed_loop_compiled_c_neural(self):
        """Fix 4: Run compiled C neural controller via ctypes in closed-loop simulation."""
        import ctypes
        import subprocess
        from env.cartpole_swingup import CartPoleSwingUpEnv

        c_bridge_src = os.path.join(_PROJECT_ROOT, "scratch", "c_neural_bridge.c")
        dll_path = os.path.join(_PROJECT_ROOT, "scratch", "c_neural_bridge.dll")
        if not os.path.exists(dll_path):
            subprocess.run(["gcc", "-O3", "-shared", "-o", dll_path, c_bridge_src, f"-I{_PROJECT_ROOT}"], check=True)

        dll = ctypes.CDLL(os.path.abspath(dll_path))
        dll.c_act.argtypes = [ctypes.POINTER(ctypes.c_float)]
        dll.c_act.restype = ctypes.c_float

        successes = 0
        total_seeds = 100
        for s in range(1000, 1000 + total_seeds):
            env = CartPoleSwingUpEnv(max_steps=500, continuous=True)
            obs, _ = env.reset(seed=s)
            cos_thetas = []
            rail_hit = False

            for _ in range(500):
                obs_c = (ctypes.c_float * 5)(*obs.astype(np.float32))
                a = float(dll.c_act(obs_c))
                obs, _, term, trunc, _ = env.step(a)
                cos_thetas.append(obs[2])
                if term:
                    rail_hit = True
                    break

            if not rail_hit and len(cos_thetas) == 500:
                if sum(1 for c in cos_thetas[-100:] if c > 0.9) >= 95:
                    successes += 1

        self.assertEqual(
            successes,
            total_seeds,
            f"Compiled C neural controller closed-loop test failed: got {successes}/{total_seeds}",
        )


if __name__ == "__main__":
    unittest.main()


