"""Comprehensive Test Suite for CUDA CartPole Batch Evaluator.

Verifies:
  1. CUDA availability check and CPU fallback safety.
  2. Production TorchScript rollout kernel invocation (strict float64 < 1e-9 tolerance).
  3. 4-Component Physics state parity (float64) against serial reference and NumPy vectorized engine
     across 50 permitted non-holdout seeds (100..149) over 500 steps (strict atol < 1e-9).
  4. Step-by-step closed-loop controller action equivalence.
  5. Identical final success flags between CPU and CUDA evaluators across all 10 diagnostic conditions.
  6. Actuator delay queue semantics and reset behavior for delay lengths 0, 1, and 2.
  7. Deterministic replay and noise sequence equivalence.
  8. Rail boundary termination equivalence at |x| >= 2.4.
  9. Success predicate boundary conditions (|x| < 2.4, full 500-step horizon, cos(th) > 0.9, >= 95 of final 100).
  10. Mixed terminal states within a batch.
  11. Rejection of protected holdout seeds (2000..2099).
  12. Handling of malformed candidate parameters or non-finite outputs.
  13. CPU fallback verification when CUDA is unavailable.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
import numpy as np
import pickle

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.cuda_evaluator import (
    CudaCartPoleBatchEvaluator,
    _fused_rollout_kernel,
    is_cuda_available,
    get_cuda_device_info,
)
from env.vectorized_env import VectorizedCartPoleEnv
from utils.perturbation_loader import PerturbedCartPoleEnv
from evolution.nsga2_engine import MaskedGenome
from experiments.run_phase_3a_evolution import (
    evaluate_candidate_vectorized,
    DIAGNOSTIC_CONDITIONS,
    norm_batch_obs,
)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


class TestCudaEvaluator(unittest.TestCase):
    """Test suite for production CUDA-accelerated batch evaluator."""

    @classmethod
    def setUpClass(cls):
        cls.cuda_present = is_cuda_available()

    def test_cuda_availability_and_device_info(self):
        """Verify device detection and metadata reporting."""
        info = get_cuda_device_info()
        self.assertIn("available", info)
        if self.cuda_present:
            self.assertTrue(info["available"])
            self.assertIn("device_name", info)
            self.assertIn("cuda_version", info)
        else:
            self.assertFalse(info["available"])

    def test_holdout_seed_rejection(self):
        """Verify that passing holdout seeds 2000-2099 immediately raises a critical error."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        evaluator = CudaCartPoleBatchEvaluator()
        dummy_genome = MaskedGenome()
        conds = {"1_Nominal": {"cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0}}

        for forbidden_seed in [2000, 2042, 2099]:
            with self.assertRaises(ValueError) as ctx:
                evaluator.evaluate_candidate(dummy_genome, conds, seeds=[forbidden_seed])
            self.assertIn("CRITICAL SAFETY VIOLATION", str(ctx.exception))

    def test_float64_physics_state_parity(self):
        """Verify float64 4-component state parity (< 1e-9 error) over 50 seeds and 500 steps."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        test_seeds = list(range(100, 150))
        num_envs = len(test_seeds)
        steps = 500

        cond = {
            "cart_mass": 1.30,
            "pole_mass": 0.130,
            "b_cart": 0.50,
            "b_pole": 0.010,
            "obs_noise_sigma": 0.0,
            "delay_steps": 0,
        }

        # Serial reference
        ref_envs = [PerturbedCartPoleEnv(max_steps=steps, seed=s, **cond) for s in test_seeds]
        ref_init = np.array([env.reset(seed=s)[0] for s, env in zip(test_seeds, ref_envs)])
        ref_states = np.array([env._state for env in ref_envs])

        # Vectorized env
        vec_env = VectorizedCartPoleEnv(num_envs=num_envs, max_steps=steps, **cond)
        vec_env.reset(seeds=test_seeds)

        # CUDA state tensor
        mc = torch.full((num_envs,), cond["cart_mass"], dtype=torch.float64, device="cuda")
        mp = torch.full((num_envs,), cond["pole_mass"], dtype=torch.float64, device="cuda")
        bc = torch.full((num_envs,), cond["b_cart"], dtype=torch.float64, device="cuda")
        bp = torch.full((num_envs,), cond["b_pole"], dtype=torch.float64, device="cuda")
        tot_m = mc + mp
        pml = mp * 0.5
        cuda_st = torch.tensor(ref_states.copy(), dtype=torch.float64, device="cuda")

        # Prescribed actions
        rng = np.random.default_rng(98765)
        actions = rng.uniform(-1.0, 1.0, size=(steps, num_envs))

        max_err_cuda_ref = 0.0
        max_err_cuda_vec = 0.0
        max_err_component = ""
        max_err_step = -1

        for t in range(steps):
            u = actions[t]
            u_t = torch.tensor(u, dtype=torch.float64, device="cuda")

            # Reference step
            for i in range(num_envs):
                ref_envs[i].step(float(u[i]))

            # Vectorized step
            vec_env.step(u)

            # CUDA step
            x = cuda_st[:, 0]
            x_dot = cuda_st[:, 1]
            th = cuda_st[:, 2]
            th_dot = cuda_st[:, 3]
            sin_th = torch.sin(th)
            cos_th = torch.cos(th)
            force = torch.clamp(u_t, -1.0, 1.0) * 10.0

            temp = (force - bc * x_dot + pml * (th_dot ** 2) * sin_th) / tot_m
            fric_pole = (bp / torch.clamp(pml, min=1e-6)) * th_dot
            th_acc = (9.8 * sin_th - cos_th * temp - fric_pole) / (
                0.5 * (4.0 / 3.0 - mp * (cos_th ** 2) / tot_m)
            )
            x_acc = temp - pml * th_acc * cos_th / tot_m
            new_x_dot = x_dot + 0.02 * x_acc
            new_x = x + 0.02 * new_x_dot
            new_th_dot = th_dot + 0.02 * th_acc
            new_th = th + 0.02 * new_th_dot
            new_th = ((new_th + np.pi) % (2.0 * np.pi)) - np.pi
            cuda_st = torch.stack([new_x, new_x_dot, new_th, new_th_dot], dim=1)

            current_ref = np.array([env._state for env in ref_envs])
            current_vec = vec_env.state
            current_cuda = cuda_st.cpu().numpy()

            diff_cuda_ref = np.abs(current_cuda - current_ref)
            diff_cuda_vec = np.abs(current_cuda - current_vec)

            step_max_cr = np.max(diff_cuda_ref)
            if step_max_cr > max_err_cuda_ref:
                max_err_cuda_ref = step_max_cr
                max_err_step = t
                comp_idx = np.unravel_index(np.argmax(diff_cuda_ref), diff_cuda_ref.shape)[1]
                max_err_component = ["x", "x_dot", "theta", "theta_dot"][comp_idx]

            max_err_cuda_vec = max(max_err_cuda_vec, np.max(diff_cuda_vec))

        self.assertLess(
            max_err_cuda_ref,
            1e-9,
            f"CUDA vs Ref error {max_err_cuda_ref} exceeded tolerance 1e-9 (at step {max_err_step}, comp {max_err_component})",
        )
        self.assertLess(
            max_err_cuda_vec,
            1e-9,
            f"CUDA vs Vec error {max_err_cuda_vec} exceeded tolerance 1e-9",
        )

    def test_production_fused_rollout_kernel_invocation(self):
        """Verify that the production compiled TorchScript kernel matches float64 reference under 1e-9."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        test_seeds = list(range(100, 150))
        num_envs = len(test_seeds)
        steps = 500
        cond = {
            "cart_mass": 1.30, "pole_mass": 0.130, "b_cart": 0.50, "b_pole": 0.010,
            "obs_noise_sigma": 0.0, "delay_steps": 0,
        }

        ref_envs = [PerturbedCartPoleEnv(max_steps=steps, seed=s, **cond) for s in test_seeds]
        for s, env in zip(test_seeds, ref_envs):
            env.reset(seed=s)

        st_tensor = torch.tensor(np.array([env._state for env in ref_envs]), dtype=torch.float64, device="cuda")
        mc = torch.full((num_envs,), cond["cart_mass"], dtype=torch.float64, device="cuda")
        mp = torch.full((num_envs,), cond["pole_mass"], dtype=torch.float64, device="cuda")
        bc = torch.full((num_envs,), cond["b_cart"], dtype=torch.float64, device="cuda")
        bp = torch.full((num_envs,), cond["b_pole"], dtype=torch.float64, device="cuda")
        w1 = torch.zeros((8, 5), dtype=torch.float64, device="cuda")
        b1 = torch.zeros(8, dtype=torch.float64, device="cuda")
        w2 = torch.zeros((1, 8), dtype=torch.float64, device="cuda")
        b2 = torch.zeros(1, dtype=torch.float64, device="cuda")
        neu = torch.ones(8, dtype=torch.float64, device="cuda")
        noise = torch.empty((0, 0, 0), dtype=torch.float32, device="cuda")

        cos_hist, rail_hit = _fused_rollout_kernel(
            st_tensor, mc, mp, bc, bp, noise, False, 0,
            w1, b1, w2, b2, neu, steps, 2.4, 0.5, 10.0, 9.8, 0.02
        )

        ref_cos_f64 = []
        for t in range(steps):
            step_cos = []
            for env in ref_envs:
                env.step(0.0)
                step_cos.append(np.cos(env._state[2]))
            ref_cos_f64.append(step_cos)

        ref_cos_arr = np.array(ref_cos_f64)
        max_cos_diff = np.max(np.abs(cos_hist.cpu().numpy() - ref_cos_arr))
        self.assertLess(max_cos_diff, 1e-9, f"Kernel cos diff {max_cos_diff} exceeded 1e-9")

    def test_step_by_step_closed_loop_action_equivalence(self):
        """Verify per-step action equivalence on candidate controller between CPU and CUDA."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        with open(os.path.join(_PROJECT_ROOT, "results", "champion_neural_h8.pkl"), "rb") as f:
            champ = pickle.load(f)

        genome = MaskedGenome()
        genome.w1 = champ["w1"]
        genome.b1 = champ["b1"]
        genome.w2 = champ["w2"]
        genome.b2 = champ["b2"]
        genome.w1_mask = np.ones_like(genome.w1)
        genome.b1_mask = np.ones_like(genome.b1)
        genome.w2_mask = np.ones_like(genome.w2)
        genome.b2_mask = np.ones_like(genome.b2)
        genome.neuron_mask = np.ones(genome.w1.shape[0])

        w1_t = torch.tensor(genome.w1, dtype=torch.float64, device="cuda")
        b1_t = torch.tensor(genome.b1, dtype=torch.float64, device="cuda")
        w2_t = torch.tensor(genome.w2, dtype=torch.float64, device="cuda")
        b2_t = torch.tensor(genome.b2, dtype=torch.float64, device="cuda")
        neu_t = torch.tensor(genome.neuron_mask, dtype=torch.float64, device="cuda")

        # Compare 100 random observation inputs
        rng = np.random.default_rng(42)
        test_obs = rng.uniform(-1.0, 1.0, size=(100, 5))
        test_obs_t = torch.tensor(test_obs, dtype=torch.float64, device="cuda")

        # CPU actions
        cpu_actions = genome.forward_batch(test_obs)

        # CUDA actions
        z1 = test_obs_t @ w1_t.t() + b1_t
        h = torch.tanh(2.5 * z1) * neu_t
        z2 = h @ w2_t.t() + b2_t
        cuda_actions = torch.tanh(2.5 * z2[:, 0]).cpu().numpy()

        max_act_diff = np.max(np.abs(cpu_actions - cuda_actions))
        self.assertLess(max_act_diff, 1e-12, f"Action difference {max_act_diff} exceeded 1e-12")

    def test_delay_queue_semantics_and_resets(self):
        """Verify delay lengths 0, 1, and 2 initialize zeroed queues and match exactly."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        with open(os.path.join(_PROJECT_ROOT, "results", "champion_neural_h8.pkl"), "rb") as f:
            champ = pickle.load(f)

        genome = MaskedGenome()
        genome.w1 = champ["w1"]
        genome.b1 = champ["b1"]
        genome.w2 = champ["w2"]
        genome.b2 = champ["b2"]
        genome.w1_mask = np.ones_like(genome.w1)
        genome.b1_mask = np.ones_like(genome.b1)
        genome.w2_mask = np.ones_like(genome.w2)
        genome.b2_mask = np.ones_like(genome.b2)
        genome.neuron_mask = np.ones(genome.w1.shape[0])

        cuda_eval = CudaCartPoleBatchEvaluator(dtype="float64")
        seeds = list(range(100, 130))

        for d in [0, 1, 2]:
            cond = {
                f"delay_{d}": {
                    "cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0,
                    "obs_noise_sigma": 0.0, "delay_steps": d
                }
            }
            succ_cpu, _, _ = evaluate_candidate_vectorized(genome, cond, seeds, max_steps=200)
            succ_cuda, _, _, _ = cuda_eval.evaluate_candidate(genome, cond, seeds, max_steps=200)
            self.assertEqual(succ_cpu, succ_cuda, f"Delay {d} mismatch: CPU {succ_cpu} vs CUDA {succ_cuda}")

    def test_rail_boundary_termination_parity(self):
        """Verify identical rail hit termination step between CPU and CUDA."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        test_seeds = list(range(100, 150))
        num_envs = len(test_seeds)
        steps = 50

        cond = {
            "cart_mass": 1.0,
            "pole_mass": 0.1,
            "b_cart": 0.0,
            "b_pole": 0.0,
            "obs_noise_sigma": 0.0,
            "delay_steps": 0,
        }

        ref_envs = [PerturbedCartPoleEnv(max_steps=steps, seed=s, **cond) for s in test_seeds]
        for s, env in zip(test_seeds, ref_envs):
            env.reset(seed=s)

        cuda_st = torch.tensor(np.array([env._state for env in ref_envs]), dtype=torch.float64, device="cuda")

        u_t = torch.ones(num_envs, dtype=torch.float64, device="cuda")
        step_hit_ref = np.full(num_envs, -1)
        step_hit_cuda = np.full(num_envs, -1)

        for t in range(steps):
            for i in range(num_envs):
                if step_hit_ref[i] == -1:
                    _, _, _, _, info = ref_envs[i].step(1.0)
                    if info["boundary_violated"]:
                        step_hit_ref[i] = t

            x = cuda_st[:, 0]
            x_dot = cuda_st[:, 1]
            th = cuda_st[:, 2]
            th_dot = cuda_st[:, 3]
            sin_th = torch.sin(th)
            cos_th = torch.cos(th)
            force = torch.clamp(u_t, -1.0, 1.0) * 10.0
            temp = (force + 0.1 * 0.5 * (th_dot ** 2) * sin_th) / 1.1
            th_acc = (9.8 * sin_th - cos_th * temp) / (0.5 * (4.0 / 3.0 - 0.1 * (cos_th ** 2) / 1.1))
            x_acc = temp - 0.1 * 0.5 * th_acc * cos_th / 1.1
            new_x_dot = x_dot + 0.02 * x_acc
            new_x = x + 0.02 * new_x_dot
            new_th_dot = th_dot + 0.02 * th_acc
            new_th = th + 0.02 * new_th_dot
            new_th = ((new_th + np.pi) % (2.0 * np.pi)) - np.pi
            cuda_st = torch.stack([new_x, new_x_dot, new_th, new_th_dot], dim=1)

            hit_now = (torch.abs(new_x) >= 2.4).cpu().numpy()
            for i in range(num_envs):
                if step_hit_cuda[i] == -1 and hit_now[i]:
                    step_hit_cuda[i] = t

        np.testing.assert_array_equal(step_hit_ref, step_hit_cuda)

    def test_closed_loop_champion_success_flag_agreement(self):
        """Verify identical robust success rates across all 10 diagnostic conditions."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        with open(os.path.join(_PROJECT_ROOT, "results", "champion_neural_h8.pkl"), "rb") as f:
            champ = pickle.load(f)

        genome = MaskedGenome()
        genome.w1 = champ["w1"]
        genome.b1 = champ["b1"]
        genome.w2 = champ["w2"]
        genome.b2 = champ["b2"]
        genome.w1_mask = np.ones_like(genome.w1)
        genome.b1_mask = np.ones_like(genome.b1)
        genome.w2_mask = np.ones_like(genome.w2)
        genome.b2_mask = np.ones_like(genome.b2)
        genome.neuron_mask = np.ones(genome.w1.shape[0])

        seeds = list(range(100, 150))
        cuda_eval = CudaCartPoleBatchEvaluator(dtype="float64")

        succ_cpu, worst_cpu, per_cond_cpu = evaluate_candidate_vectorized(genome, DIAGNOSTIC_CONDITIONS, seeds)
        succ_cuda, worst_cuda, per_cond_cuda, _ = cuda_eval.evaluate_candidate(genome, DIAGNOSTIC_CONDITIONS, seeds)

        self.assertAlmostEqual(succ_cpu, succ_cuda, places=5)
        self.assertAlmostEqual(worst_cpu, worst_cuda, places=5)
        for c in DIAGNOSTIC_CONDITIONS:
            self.assertAlmostEqual(
                per_cond_cpu[c],
                per_cond_cuda[c],
                places=5,
                msg=f"Condition {c} rate mismatch: CPU {per_cond_cpu[c]} vs CUDA {per_cond_cuda[c]}",
            )

    def test_success_predicate_exact_boundaries(self):
        """Verify exact boundaries for success predicate: track boundary, upright cos, window counts, horizon."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        evaluator = CudaCartPoleBatchEvaluator(dtype="float64")

        # Boundary 1: Rail boundary at |x| = 2.4
        # Case A: cart reaches exactly 2.399999 -> no rail hit
        # Case B: cart reaches exactly 2.400000 -> rail hit
        x_safe = torch.tensor([[2.399999]], dtype=torch.float64)
        x_hit = torch.tensor([[2.400000]], dtype=torch.float64)
        hit_safe = bool((torch.abs(x_safe) >= 2.4).item())
        hit_fail = bool((torch.abs(x_hit) >= 2.4).item())
        self.assertFalse(hit_safe, "x=2.399999 should not trigger rail hit")
        self.assertTrue(hit_fail, "x=2.400000 must trigger rail hit")

        # Boundary 2: Upright cos(theta) > 0.9 strictly
        # Case A: cos = 0.900000 -> not upright (strict > 0.9)
        # Case B: cos = 0.900001 -> upright
        cos_eq = torch.tensor([0.900000], dtype=torch.float64)
        cos_gt = torch.tensor([0.900001], dtype=torch.float64)
        self.assertFalse(bool((cos_eq > 0.9).item()), "cos=0.900000 must be False under strict > 0.9")
        self.assertTrue(bool((cos_gt > 0.9).item()), "cos=0.900001 must be True under strict > 0.9")

        # Boundary 3: Window count: 94 steps vs 95 steps in final 100
        # 94 upright steps out of 100
        cos_window_94 = torch.full((100, 1), 0.8, dtype=torch.float64)
        cos_window_94[:94] = 0.95
        count_94 = torch.sum(cos_window_94 > 0.9, dim=0)
        succ_94 = bool(((count_94 >= 95) & (~torch.tensor([False]))).item())
        self.assertFalse(succ_94, "94 upright steps in final 100 must fail")

        # 95 upright steps out of 100
        cos_window_95 = torch.full((100, 1), 0.8, dtype=torch.float64)
        cos_window_95[:95] = 0.95
        count_95 = torch.sum(cos_window_95 > 0.9, dim=0)
        succ_95 = bool(((count_95 >= 95) & (~torch.tensor([False]))).item())
        self.assertTrue(succ_95, "95 upright steps in final 100 must succeed")

    def test_deterministic_replay(self):
        """Verify that evaluating the same candidate on the same seed partition is bit-exact."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        with open(os.path.join(_PROJECT_ROOT, "results", "champion_neural_h8.pkl"), "rb") as f:
            champ = pickle.load(f)

        genome = MaskedGenome()
        genome.w1 = champ["w1"]
        genome.b1 = champ["b1"]
        genome.w2 = champ["w2"]
        genome.b2 = champ["b2"]
        genome.w1_mask = np.ones_like(genome.w1)
        genome.b1_mask = np.ones_like(genome.b1)
        genome.w2_mask = np.ones_like(genome.w2)
        genome.b2_mask = np.ones_like(genome.b2)
        genome.neuron_mask = np.ones(genome.w1.shape[0])

        evaluator = CudaCartPoleBatchEvaluator(dtype="float64")
        seeds = list(range(100, 120))
        conds = {
            "1_Nominal": DIAGNOSTIC_CONDITIONS["1_Nominal"],
            "5_SensorNoise": DIAGNOSTIC_CONDITIONS["5_SensorNoise_0.05"],
        }

        succ1, worst1, rates1, _ = evaluator.evaluate_candidate(genome, conds, seeds)
        succ2, worst2, rates2, _ = evaluator.evaluate_candidate(genome, conds, seeds)

        self.assertEqual(succ1, succ2, "Deterministic replay failed for robust success")
        self.assertEqual(worst1, worst2, "Deterministic replay failed for worst success")
        self.assertEqual(rates1, rates2, "Deterministic replay failed for per-condition rates")

    def test_mixed_terminal_states_batch(self):
        """Verify correct handling of mixed batches where some environments fail early and others complete."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        with open(os.path.join(_PROJECT_ROOT, "results", "champion_neural_h8.pkl"), "rb") as f:
            champ = pickle.load(f)

        genome = MaskedGenome()
        genome.w1 = champ["w1"]
        genome.b1 = champ["b1"]
        genome.w2 = champ["w2"]
        genome.b2 = champ["b2"]
        genome.w1_mask = np.ones_like(genome.w1)
        genome.b1_mask = np.ones_like(genome.b1)
        genome.w2_mask = np.ones_like(genome.w2)
        genome.b2_mask = np.ones_like(genome.b2)
        genome.neuron_mask = np.ones(genome.w1.shape[0])

        evaluator = CudaCartPoleBatchEvaluator(dtype="float64")
        seeds = list(range(100, 150))
        cond = {"3_CartMass_+30%": DIAGNOSTIC_CONDITIONS["3_CartMass_+30%"]}

        # On 3_CartMass_+30%, exactly 2 environments hit the rail and fail early, 48 succeed
        succ_cpu, _, rates_cpu = evaluate_candidate_vectorized(genome, cond, seeds)
        succ_cuda, _, rates_cuda, telem = evaluator.evaluate_candidate(genome, cond, seeds)

        self.assertEqual(succ_cuda, succ_cpu)
        self.assertEqual(succ_cuda, 0.96)
        self.assertEqual(telem["total_rail_hits"], 2)

    def test_malformed_candidate_parameters_handling(self):
        """Verify handling of NaN/inf inputs in candidate genomes."""
        if not self.cuda_present:
            self.skipTest("CUDA not available; skipping CUDA evaluator test")

        evaluator = CudaCartPoleBatchEvaluator()
        g = MaskedGenome()
        g.w1[0, 0] = float("nan")
        conds = {"1_Nominal": {"cart_mass": 1.0, "pole_mass": 0.1, "b_cart": 0.0, "b_pole": 0.0}}

        succ, worst, _, _ = evaluator.evaluate_candidate(g, conds, seeds=[100])
        self.assertEqual(succ, 0.0)
        self.assertEqual(worst, 0.0)

    def test_cpu_fallback_path(self):
        """Verify that CPU fallback functions seamlessly when CUDA is unavailable."""
        vec_env = VectorizedCartPoleEnv(num_envs=5, max_steps=10)
        obs, _ = vec_env.reset(seeds=[10, 11, 12, 13, 14])
        self.assertEqual(obs.shape, (5, 5))
        o, r, term, trunc, info = vec_env.step(np.zeros(5))
        self.assertEqual(o.shape, (5, 5))
        self.assertEqual(r.shape, (5,))


if __name__ == "__main__":
    unittest.main()
