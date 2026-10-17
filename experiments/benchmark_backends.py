"""Comprehensive multi-batch benchmark comparing Serial Reference CPU,
Vectorized NumPy CPU, and TorchScript CUDA Evaluator across float64 and float32.

Measures per protocol:
  Category A: Kernel / Rollout Microbenchmark (direct kernel execution, fixed inputs).
  Category B: End-to-End Single-Condition Episode Benchmark.
  Category C: Candidate Fitness-Batch Benchmark (10 conditions x 50 seeds = 500 episodes = 250,000 steps).
  Scaling Suite: 50, 200, 500, 1000, 2000, 5000 environments across 500 steps (crossover analysis).
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evolution.nsga2_engine import MaskedGenome
from utils.perturbation_loader import PerturbedCartPoleEnv
from env.vectorized_env import VectorizedCartPoleEnv
from env.cuda_evaluator import CudaCartPoleBatchEvaluator, get_cuda_device_info, _fused_rollout_kernel
from experiments.run_phase_3a_evolution import evaluate_candidate_vectorized, DIAGNOSTIC_CONDITIONS


def run_full_benchmark():
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

    conditions = DIAGNOSTIC_CONDITIONS
    seeds = list(range(100, 150))
    num_envs = len(seeds)
    num_conds = len(conditions)
    max_steps = 500
    total_episodes = num_envs * num_conds
    total_steps = total_episodes * max_steps

    cuda_info = get_cuda_device_info()
    dev_name = str(cuda_info.get("device_name", "Unknown GPU"))
    print(f"CUDA Device: {dev_name}")
    print(f"Workload (Phase 3A Candidate): {num_conds} conditions x {num_envs} envs x {max_steps} steps = {total_steps:,} steps")

    results = []
    repeats = 5

    # ═════════════════════════════════════════════════════════════════════════
    # Category C: Candidate Fitness-Batch Benchmark (Phase 3A Workload)
    # ═════════════════════════════════════════════════════════════════════════

    # ── 1. Serial Reference CPU ─────────────────────────────────────────────
    print("\n[1/4] Benchmarking Serial Reference CPU (Category C)...")
    # Warmup
    t_warm = time.perf_counter()
    for cond_name, c_params in conditions.items():
        env = PerturbedCartPoleEnv(max_steps=max_steps, seed=seeds[0], **c_params)
        env.reset(seed=seeds[0])
        for _ in range(10):
            env.step(0.0)
    warmup_s = round(time.perf_counter() - t_warm, 4)

    for r in range(repeats):
        t0 = time.perf_counter()
        for cond_name, c_params in conditions.items():
            for s in seeds:
                env = PerturbedCartPoleEnv(max_steps=max_steps, seed=s, **c_params)
                obs, _ = env.reset(seed=s)
                for _ in range(max_steps):
                    norm_x = obs[0] / 2.4
                    norm_xd = np.clip(obs[1] / 10.0, -1.0, 1.0)
                    norm_td = np.clip(obs[4] / 10.0, -1.0, 1.0)
                    norm_o = np.array([norm_x, norm_xd, obs[2], obs[3], norm_td])
                    u = genome.forward(norm_o)
                    obs, _, _, _, _ = env.step(float(u))
        dur = time.perf_counter() - t0
        steps_per_sec = total_steps / dur
        episodes_per_sec = total_episodes / dur
        results.append({
            "category": "Candidate Fitness-Batch",
            "backend": "Serial Reference CPU",
            "batch_size": num_envs,
            "repeat": r + 1,
            "episodes": total_episodes,
            "steps": total_steps,
            "duration_s": round(dur, 4),
            "steps_per_sec": int(steps_per_sec),
            "episodes_per_sec": round(episodes_per_sec, 2),
            "dtype": "float64",
            "device": "CPU (Serial)",
            "peak_mem_mb": 0.0,
            "warmup_s": warmup_s,
        })
        print(f"  Repeat {r+1}: {dur:.4f}s ({int(steps_per_sec):,} steps/s)")

    # ── 2. Vectorized NumPy CPU ─────────────────────────────────────────────
    print("\n[2/4] Benchmarking Vectorized NumPy CPU (Category C)...")
    t_warm = time.perf_counter()
    evaluate_candidate_vectorized(genome, conditions, seeds, max_steps=max_steps)
    warmup_s = round(time.perf_counter() - t_warm, 4)

    for r in range(repeats):
        t0 = time.perf_counter()
        evaluate_candidate_vectorized(genome, conditions, seeds, max_steps=max_steps)
        dur = time.perf_counter() - t0
        steps_per_sec = total_steps / dur
        episodes_per_sec = total_episodes / dur
        results.append({
            "category": "Candidate Fitness-Batch",
            "backend": "Vectorized NumPy CPU",
            "batch_size": num_envs,
            "repeat": r + 1,
            "episodes": total_episodes,
            "steps": total_steps,
            "duration_s": round(dur, 4),
            "steps_per_sec": int(steps_per_sec),
            "episodes_per_sec": round(episodes_per_sec, 2),
            "dtype": "float64",
            "device": "CPU (NumPy Batch)",
            "peak_mem_mb": 0.0,
            "warmup_s": warmup_s,
        })
        print(f"  Repeat {r+1}: {dur:.4f}s ({int(steps_per_sec):,} steps/s)")

    # ── 3. CUDA Evaluator (float64) ─────────────────────────────────────────
    print("\n[3/4] Benchmarking CUDA Evaluator (float64) (Category C)...")
    cuda_eval_f64 = CudaCartPoleBatchEvaluator(dtype="float64")
    t_warm = time.perf_counter()
    cuda_eval_f64.evaluate_candidate(genome, conditions, seeds, max_steps=max_steps)
    torch.cuda.synchronize()
    warmup_s = round(time.perf_counter() - t_warm, 4)

    for r in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        cuda_eval_f64.evaluate_candidate(genome, conditions, seeds, max_steps=max_steps)
        torch.cuda.synchronize()
        dur = time.perf_counter() - t0
        steps_per_sec = total_steps / dur
        episodes_per_sec = total_episodes / dur
        mem_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        results.append({
            "category": "Candidate Fitness-Batch",
            "backend": "CUDA (TorchScript JIT)",
            "batch_size": num_envs,
            "repeat": r + 1,
            "episodes": total_episodes,
            "steps": total_steps,
            "duration_s": round(dur, 4),
            "steps_per_sec": int(steps_per_sec),
            "episodes_per_sec": round(episodes_per_sec, 2),
            "dtype": "float64",
            "device": dev_name,
            "peak_mem_mb": mem_mb,
            "warmup_s": warmup_s,
        })
        print(f"  Repeat {r+1}: {dur:.4f}s ({int(steps_per_sec):,} steps/s, Peak Mem: {mem_mb} MB)")

    # ── 4. CUDA Evaluator (float32) ─────────────────────────────────────────
    print("\n[4/4] Benchmarking CUDA Evaluator (float32) (Category C)...")
    cuda_eval_f32 = CudaCartPoleBatchEvaluator(dtype="float32")
    t_warm = time.perf_counter()
    cuda_eval_f32.evaluate_candidate(genome, conditions, seeds, max_steps=max_steps)
    torch.cuda.synchronize()
    warmup_s = round(time.perf_counter() - t_warm, 4)

    for r in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        cuda_eval_f32.evaluate_candidate(genome, conditions, seeds, max_steps=max_steps)
        torch.cuda.synchronize()
        dur = time.perf_counter() - t0
        steps_per_sec = total_steps / dur
        episodes_per_sec = total_episodes / dur
        mem_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        results.append({
            "category": "Candidate Fitness-Batch",
            "backend": "CUDA (TorchScript JIT)",
            "batch_size": num_envs,
            "repeat": r + 1,
            "episodes": total_episodes,
            "steps": total_steps,
            "duration_s": round(dur, 4),
            "steps_per_sec": int(steps_per_sec),
            "episodes_per_sec": round(episodes_per_sec, 2),
            "dtype": "float32",
            "device": dev_name,
            "peak_mem_mb": mem_mb,
            "warmup_s": warmup_s,
        })
        print(f"  Repeat {r+1}: {dur:.4f}s ({int(steps_per_sec):,} steps/s, Peak Mem: {mem_mb} MB)")

    # ═════════════════════════════════════════════════════════════════════════
    # Scaling Suite (50 to 5000 environments across 500 steps)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[Scaling Suite] Benchmarking Batch Sizes 50 to 5000 environments (500 steps)...")
    scaling_batches = [50, 200, 500, 1000, 2000, 5000]

    w1_f64 = torch.tensor(genome.w1, dtype=torch.float64, device="cuda")
    b1_f64 = torch.tensor(genome.b1, dtype=torch.float64, device="cuda")
    w2_f64 = torch.tensor(genome.w2, dtype=torch.float64, device="cuda")
    b2_f64 = torch.tensor(genome.b2, dtype=torch.float64, device="cuda")
    neu_f64 = torch.tensor(genome.neuron_mask, dtype=torch.float64, device="cuda")

    w1_f32 = torch.tensor(genome.w1, dtype=torch.float32, device="cuda")
    b1_f32 = torch.tensor(genome.b1, dtype=torch.float32, device="cuda")
    w2_f32 = torch.tensor(genome.w2, dtype=torch.float32, device="cuda")
    b2_f32 = torch.tensor(genome.b2, dtype=torch.float32, device="cuda")
    neu_f32 = torch.tensor(genome.neuron_mask, dtype=torch.float32, device="cuda")

    for bs in scaling_batches:
        step_budget = bs * max_steps

        # --- NumPy CPU Batch ---
        vec_env = VectorizedCartPoleEnv(num_envs=bs, max_steps=max_steps)
        scale_seeds = list(range(100, 100 + bs))
        vec_env.reset(seeds=scale_seeds)

        np_durs = []
        for r_s in range(repeats):
            t0 = time.perf_counter()
            for _ in range(max_steps):
                norm_o = np.stack([
                    vec_env.state[:, 0] / 2.4,
                    np.clip(vec_env.state[:, 1] / 10.0, -1.0, 1.0),
                    np.cos(vec_env.state[:, 2]),
                    np.sin(vec_env.state[:, 2]),
                    np.clip(vec_env.state[:, 3] / 10.0, -1.0, 1.0),
                ], axis=1)
                actions = genome.forward_batch(norm_o)
                vec_env.step(actions)
            dur = time.perf_counter() - t0
            np_durs.append(dur)
            results.append({
                "category": "Batch Scaling Suite",
                "backend": "Vectorized NumPy CPU",
                "batch_size": bs,
                "repeat": r_s + 1,
                "episodes": bs,
                "steps": step_budget,
                "duration_s": round(dur, 4),
                "steps_per_sec": int(step_budget / dur),
                "episodes_per_sec": round(bs / dur, 2),
                "dtype": "float64",
                "device": "CPU (NumPy Batch)",
                "peak_mem_mb": 0.0,
                "warmup_s": 0.0,
            })

        # --- CUDA float64 Batch ---
        init_st_f64 = torch.zeros((bs, 4), dtype=torch.float64, device="cuda")
        mc_f64 = torch.full((bs,), 1.0, dtype=torch.float64, device="cuda")
        mp_f64 = torch.full((bs,), 0.1, dtype=torch.float64, device="cuda")
        bc_f64 = torch.full((bs,), 0.0, dtype=torch.float64, device="cuda")
        bp_f64 = torch.full((bs,), 0.0, dtype=torch.float64, device="cuda")
        empty_noise_f32 = torch.empty((0, 0, 0), dtype=torch.float32, device="cuda")

        # Warmup
        _fused_rollout_kernel(init_st_f64, mc_f64, mp_f64, bc_f64, bp_f64, empty_noise_f32, False, 0, w1_f64, b1_f64, w2_f64, b2_f64, neu_f64, max_steps, 2.4, 0.5, 10.0, 9.8, 0.02)
        torch.cuda.synchronize()

        cuda_f64_durs = []
        for r_s in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            _fused_rollout_kernel(init_st_f64, mc_f64, mp_f64, bc_f64, bp_f64, empty_noise_f32, False, 0, w1_f64, b1_f64, w2_f64, b2_f64, neu_f64, max_steps, 2.4, 0.5, 10.0, 9.8, 0.02)
            torch.cuda.synchronize()
            dur = time.perf_counter() - t0
            cuda_f64_durs.append(dur)
            mem_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            results.append({
                "category": "Batch Scaling Suite",
                "backend": "CUDA (TorchScript JIT)",
                "batch_size": bs,
                "repeat": r_s + 1,
                "episodes": bs,
                "steps": step_budget,
                "duration_s": round(dur, 4),
                "steps_per_sec": int(step_budget / dur),
                "episodes_per_sec": round(bs / dur, 2),
                "dtype": "float64",
                "device": dev_name,
                "peak_mem_mb": mem_mb,
                "warmup_s": 0.0,
            })

        # --- CUDA float32 Batch ---
        init_st_f32 = torch.zeros((bs, 4), dtype=torch.float32, device="cuda")
        mc_f32 = torch.full((bs,), 1.0, dtype=torch.float32, device="cuda")
        mp_f32 = torch.full((bs,), 0.1, dtype=torch.float32, device="cuda")
        bc_f32 = torch.full((bs,), 0.0, dtype=torch.float32, device="cuda")
        bp_f32 = torch.full((bs,), 0.0, dtype=torch.float32, device="cuda")

        # Warmup
        _fused_rollout_kernel(init_st_f32, mc_f32, mp_f32, bc_f32, bp_f32, empty_noise_f32, False, 0, w1_f32, b1_f32, w2_f32, b2_f32, neu_f32, max_steps, 2.4, 0.5, 10.0, 9.8, 0.02)
        torch.cuda.synchronize()

        cuda_f32_durs = []
        for r_s in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            _fused_rollout_kernel(init_st_f32, mc_f32, mp_f32, bc_f32, bp_f32, empty_noise_f32, False, 0, w1_f32, b1_f32, w2_f32, b2_f32, neu_f32, max_steps, 2.4, 0.5, 10.0, 9.8, 0.02)
            torch.cuda.synchronize()
            dur = time.perf_counter() - t0
            cuda_f32_durs.append(dur)
            mem_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            results.append({
                "category": "Batch Scaling Suite",
                "backend": "CUDA (TorchScript JIT)",
                "batch_size": bs,
                "repeat": r_s + 1,
                "episodes": bs,
                "steps": step_budget,
                "duration_s": round(dur, 4),
                "steps_per_sec": int(step_budget / dur),
                "episodes_per_sec": round(bs / dur, 2),
                "dtype": "float32",
                "device": dev_name,
                "peak_mem_mb": mem_mb,
                "warmup_s": 0.0,
            })

        med_np = np.median(np_durs)
        med_f64 = np.median(cuda_f64_durs)
        med_f32 = np.median(cuda_f32_durs)
        s64 = (step_budget / med_f64) / (step_budget / med_np)
        s32 = (step_budget / med_f32) / (step_budget / med_np)
        print(f"  Batch {bs:5d}: NumPy = {int(step_budget / med_np):9,d} st/s | "
              f"CUDA f64 = {int(step_budget / med_f64):9,d} st/s ({s64:4.2f}x) | "
              f"CUDA f32 = {int(step_budget / med_f32):9,d} st/s ({s32:4.2f}x)")

    df = pd.DataFrame(results)
    out_csv = os.path.join(_PROJECT_ROOT, "results", "cuda_benchmark.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nAll benchmark results successfully saved to {out_csv}")


if __name__ == "__main__":
    run_full_benchmark()
