"""CUDA-Accelerated Batch Evaluator for CartPole Swing-Up.

Features:
  1. Fused TorchScript CUDA rollout kernel:
     - The entire 500-step trajectory rollout is compiled via TorchScript (`@torch.jit.script`)
       into a fused execution graph.
     - Maintains all environment states, parameters, delay circular buffers, and neural controller
       parameters directly on GPU memory without per-step Python loop dispatch.
  2. Precision modes:
     - float64 (strict correctness path, targeted for atol < 1e-9 physics error).
     - float32 (experimental high-throughput path).
  3. Safe fallback:
     - If CUDA is not available or initialization fails, provides clear diagnostics
       and allows seamless fallback to the verified NumPy CPU evaluator.
  4. Identical interface:
     - Evaluates candidates over condition suites and seed partitions with identical
       success predicates, boundary checking, and fitness aggregation.
"""

from __future__ import annotations

import collections
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evolution.nsga2_engine import MaskedGenome

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


def is_cuda_available() -> bool:
    """Check if PyTorch and a compatible CUDA device are available."""
    if not _TORCH_AVAILABLE:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def get_cuda_device_info() -> Dict[str, Any]:
    """Return device metadata if CUDA is available."""
    if not is_cuda_available():
        return {"available": False, "reason": "PyTorch not installed or torch.cuda.is_available() is False"}
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


# ── Compiled TorchScript Fused Rollout Kernel ──────────────────────────────

if _TORCH_AVAILABLE:
    @torch.jit.script
    def _fused_rollout_kernel(
        st: torch.Tensor,
        mc: torch.Tensor,
        mp: torch.Tensor,
        bc: torch.Tensor,
        bp: torch.Tensor,
        noise_tensor: torch.Tensor,
        has_noise: bool,
        delay: int,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        neu: torch.Tensor,
        max_steps: int,
        x_threshold: float,
        length: float,
        force_mag: float,
        gravity: float,
        tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Device-resident compiled rollout executing all timesteps on GPU.

        Category classification: TorchScript-compiled control flow launching
        multiple device kernels per timestep (Category 2).
        """
        num_envs = st.size(0)
        rail_hit = torch.zeros(num_envs, dtype=torch.bool, device=st.device)
        cos_history = torch.zeros((max_steps, num_envs), dtype=st.dtype, device=st.device)

        # Preallocate circular delay buffer
        buf_len = delay + 1 if delay > 0 else 1
        delay_buf = torch.zeros((buf_len, num_envs), dtype=st.dtype, device=st.device)
        delay_idx = 0

        tot_m = mc + mp
        pml = mp * length

        # Observation at reset (t=0 before step 0)
        x = st[:, 0]
        x_dot = st[:, 1]
        th = st[:, 2]
        th_dot = st[:, 3]
        sin_th = torch.sin(th)
        cos_th = torch.cos(th)

        if has_noise:
            n0 = noise_tensor[0]
            # Match VectorizedCartPoleEnv: noise added to raw obs before normalization
            raw_x = x.to(torch.float32) + n0[:, 0]
            raw_xd = x_dot.to(torch.float32) + n0[:, 1]
            raw_cos = cos_th.to(torch.float32) + n0[:, 2]
            raw_sin = sin_th.to(torch.float32) + n0[:, 3]
            raw_td = th_dot.to(torch.float32) + n0[:, 4]
            norm_x = (raw_x / x_threshold).to(st.dtype)
            norm_xd = torch.clamp(raw_xd / 10.0, -1.0, 1.0).to(st.dtype)
            norm_td = torch.clamp(raw_td / 10.0, -1.0, 1.0).to(st.dtype)
            curr_obs = torch.stack([norm_x, norm_xd, raw_cos.to(st.dtype), raw_sin.to(st.dtype), norm_td], dim=1)
        else:
            norm_x = x / x_threshold
            norm_xd = torch.clamp(x_dot / 10.0, -1.0, 1.0)
            norm_td = torch.clamp(th_dot / 10.0, -1.0, 1.0)
            curr_obs = torch.stack([norm_x, norm_xd, cos_th, sin_th, norm_td], dim=1)

        for t in range(max_steps):
            z1 = curr_obs @ w1.t() + b1
            h = torch.tanh(2.5 * z1) * neu
            z2 = h @ w2.t() + b2
            u = torch.tanh(2.5 * z2[:, 0])

            if delay > 0:
                delay_buf[delay_idx] = u
                app_u = delay_buf[(delay_idx + 1) % buf_len]
                delay_idx = (delay_idx + 1) % buf_len
            else:
                app_u = u

            force = torch.clamp(app_u, -1.0, 1.0) * force_mag

            temp = (force - bc * x_dot + pml * (th_dot ** 2) * sin_th) / tot_m
            fric_pole = (bp / torch.clamp(pml, min=1e-6)) * th_dot
            th_acc = (gravity * sin_th - cos_th * temp - fric_pole) / (
                length * (4.0 / 3.0 - mp * (cos_th ** 2) / tot_m)
            )
            x_acc = temp - pml * th_acc * cos_th / tot_m

            new_x_dot = x_dot + tau * x_acc
            new_x = x + tau * new_x_dot
            new_th_dot = th_dot + tau * th_acc
            new_th = th + tau * new_th_dot
            new_th = ((new_th + 3.141592653589793) % (2.0 * 3.141592653589793)) - 3.141592653589793

            x = new_x
            x_dot = new_x_dot
            th = new_th
            th_dot = new_th_dot
            sin_th = torch.sin(th)
            cos_th = torch.cos(th)

            rail_hit = rail_hit | (torch.abs(new_x) >= x_threshold)

            # Post-step observation
            if has_noise:
                nt = noise_tensor[t + 1]
                raw_x = x.to(torch.float32) + nt[:, 0]
                raw_xd = x_dot.to(torch.float32) + nt[:, 1]
                raw_cos = cos_th.to(torch.float32) + nt[:, 2]
                raw_sin = sin_th.to(torch.float32) + nt[:, 3]
                raw_td = th_dot.to(torch.float32) + nt[:, 4]
                norm_x = (raw_x / x_threshold).to(st.dtype)
                norm_xd = torch.clamp(raw_xd / 10.0, -1.0, 1.0).to(st.dtype)
                norm_td = torch.clamp(raw_td / 10.0, -1.0, 1.0).to(st.dtype)
                curr_obs = torch.stack([norm_x, norm_xd, raw_cos.to(st.dtype), raw_sin.to(st.dtype), norm_td], dim=1)
                cos_history[t] = raw_cos.to(st.dtype)
            else:
                norm_x = x / x_threshold
                norm_xd = torch.clamp(x_dot / 10.0, -1.0, 1.0)
                norm_td = torch.clamp(th_dot / 10.0, -1.0, 1.0)
                curr_obs = torch.stack([norm_x, norm_xd, cos_th, sin_th, norm_td], dim=1)
                cos_history[t] = cos_th

        return cos_history, rail_hit


class CudaCartPoleBatchEvaluator:
    """Device-resident batched CartPole evaluator running on CUDA with compiled TorchScript kernel."""

    GRAVITY: float = 9.8
    FORCE_MAG: float = 10.0
    TAU: float = 0.02
    X_THRESHOLD: float = 2.4
    LENGTH: float = 0.5

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: str = "float64",
    ):
        if not is_cuda_available():
            raise RuntimeError("CUDA is not available on this system.")

        self.device = torch.device(device if device is not None else "cuda:0")
        self.dtype_str = dtype
        self.torch_dtype = torch.float64 if dtype == "float64" else torch.float32

    def evaluate_candidate(
        self,
        genome: MaskedGenome,
        conditions: Dict[str, Dict[str, Any]],
        seeds: List[int],
        max_steps: int = 500,
        pregenerated_noise: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[float, float, Dict[str, float], Dict[str, Any]]:
        """Evaluate a candidate genome across multiple physical conditions on CUDA.

        Returns:
            (robust_success, worst_success, per_condition_rates, telemetry)
        """
        # Ensure holdout seeds are forbidden
        for s in seeds:
            if 2000 <= s <= 2099:
                raise ValueError(f"CRITICAL SAFETY VIOLATION: Access to holdout seed {s} forbidden!")

        t0 = time.perf_counter()
        num_envs = len(seeds)
        cond_rates: Dict[str, float] = {}
        total_steps_simulated = 0
        total_rail_hits = 0

        # Transfer genome weights to GPU tensors
        w1_eff = genome.w1 * genome.w1_mask
        b1_eff = genome.b1 * genome.b1_mask
        w2_eff = genome.w2 * genome.w2_mask
        b2_eff = genome.b2 * genome.b2_mask
        active_neu = genome.neuron_mask.astype(np.float64)

        w1_t = torch.tensor(w1_eff, dtype=self.torch_dtype, device=self.device)
        b1_t = torch.tensor(b1_eff, dtype=self.torch_dtype, device=self.device)
        w2_t = torch.tensor(w2_eff, dtype=self.torch_dtype, device=self.device)
        b2_t = torch.tensor(b2_eff, dtype=self.torch_dtype, device=self.device)
        neu_t = torch.tensor(active_neu, dtype=self.torch_dtype, device=self.device)

        # Initial states per seed: match CartPoleSwingUpEnv exact initialization
        init_states = np.zeros((num_envs, 4), dtype=np.float64)
        for i, s in enumerate(seeds):
            rng = np.random.default_rng(s)
            perturb = rng.uniform(-0.05, 0.05, size=4)
            init_states[i] = np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float64) + perturb

        init_st_tensor = torch.tensor(init_states, dtype=self.torch_dtype, device=self.device)

        for cond_name, c_params in conditions.items():
            cart_m = float(c_params.get("cart_mass", 1.0))
            pole_m = float(c_params.get("pole_mass", 0.1))
            b_c = float(c_params.get("b_cart", 0.0))
            b_p = float(c_params.get("b_pole", 0.0))
            noise_sigma = float(c_params.get("obs_noise_sigma", 0.0))
            delay = int(c_params.get("delay_steps", 0))

            mc = torch.full((num_envs,), cart_m, dtype=self.torch_dtype, device=self.device)
            mp = torch.full((num_envs,), pole_m, dtype=self.torch_dtype, device=self.device)
            bc = torch.full((num_envs,), b_c, dtype=self.torch_dtype, device=self.device)
            bp = torch.full((num_envs,), b_p, dtype=self.torch_dtype, device=self.device)

            # Pre-generate noise if specified or generate per environment on GPU
            has_noise = (noise_sigma > 0.0)
            if has_noise:
                if pregenerated_noise is not None and cond_name in pregenerated_noise:
                    noise_tensor = torch.tensor(pregenerated_noise[cond_name], dtype=self.torch_dtype, device=self.device)
                else:
                    noise_arr = np.zeros((max_steps + 1, num_envs, 5), dtype=np.float32)
                    for i, s in enumerate(seeds):
                        rng_noise = np.random.default_rng(s)
                        for t in range(max_steps + 1):
                            noise_arr[t, i, :] = rng_noise.normal(0.0, noise_sigma, size=5).astype(np.float32)
                    noise_tensor = torch.tensor(noise_arr, dtype=torch.float32, device=self.device)
            else:
                noise_tensor = torch.empty((0, 0, 0), dtype=torch.float32, device=self.device)

            # Invoke compiled fused rollout kernel on GPU
            cos_history, rail_hit = _fused_rollout_kernel(
                init_st_tensor.clone(),
                mc,
                mp,
                bc,
                bp,
                noise_tensor,
                has_noise,
                delay,
                w1_t,
                b1_t,
                w2_t,
                b2_t,
                neu_t,
                max_steps,
                self.X_THRESHOLD,
                self.LENGTH,
                self.FORCE_MAG,
                self.GRAVITY,
                self.TAU,
            )

            # Evaluate success predicate on GPU
            final_100_cos = cos_history[-100:]  # shape (100, num_envs)
            upright_counts = torch.sum(final_100_cos > 0.9, dim=0)
            success = (upright_counts >= 95) & (~rail_hit)

            rate = float(torch.mean(success.to(torch.float64)).item())
            cond_rates[cond_name] = rate
            total_steps_simulated += num_envs * max_steps
            total_rail_hits += int(torch.sum(rail_hit).item())

        robust_succ = float(np.mean(list(cond_rates.values())))
        worst_succ = float(np.min(list(cond_rates.values())))

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - t0

        telemetry = {
            "backend": "cuda",
            "kernel": "TorchScript_fused",
            "device": str(self.device),
            "dtype": self.dtype_str,
            "wall_clock_s": round(elapsed, 4),
            "steps_simulated": total_steps_simulated,
            "steps_per_second": int(total_steps_simulated / elapsed) if elapsed > 0 else 0,
            "total_rail_hits": total_rail_hits,
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 2) if is_cuda_available() else 0.0,
        }

        return robust_succ, worst_succ, cond_rates, telemetry
