# EvoEdge — CUDA Fitness Evaluation Backend: Strict Compliance Audit & Remediation Report

**Document Revision:** 3.1 (Consistency & Gate Correction)  
**Document Date:** October 13, 2026  
**Auditor:** Senior GPU Systems Engineer, Numerical Simulation Researcher & Reproducibility Auditor  
**Primary Files Audited:** `env/cuda_evaluator.py`, `env/vectorized_env.py`, `utils/perturbation_loader.py`, `tests/test_cuda_evaluator.py`, `experiments/benchmark_backends.py`  
**Execution Environment:** Windows 11 (build 10.0.26100), Python 3.12.10, PyTorch 2.5.1+cu121, NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, sm_89, 6 GB VRAM)  
**Overall Backend Status:** **EXPERIMENTAL / NOT ACCEPTED (MANDATORY FUSION GATE FAILED; VECTORIZED NUMPY CPU RETAINED AS SOLE PRODUCTION DEFAULT)**  
**Final Architectural Disposition:** **EXPERIMENTAL / NOT ACCEPTED**

---

## 1. Executive Summary & Audit Scope

This report provides the exhaustive compliance audit and consistency correction findings for the optional CUDA fitness evaluator in EvoEdge (`env/cuda_evaluator.py`). The audit was conducted under strict binding experimental restrictions:
- Protected final holdout seeds `2000–2099` were **strictly unaccessed, unqueried, and preserved**.
- Objective B holdout was **unaccessed**.
- The scientific protocol, success predicates, objective directions, fitness definitions, and frozen Phase 3A configurations were **strictly preserved without alteration**.
- All 13 historical checkpoints and configuration files were verified via cryptographic SHA256 hashes against the recorded baseline.

### Key Audit Findings & Corrections:

1. **Mandatory Kernel Fusion Gate (FAIL — Precludes Acceptance):**
   - Source code inspection and runtime profiling with `torch.profiler` reveal that the implementation is **Category 2: TorchScript-compiled control flow that dispatches multiple GPU kernels per timestep**.
   - A single 500-step rollout executes **approximately 15,500 separate CUDA kernel launches** (~31 launches per timestep, including 1,000 `aten::mm`, 1,000 `aten::stack`, 3,000 `aten::select`, 4,500 `aten::unsqueeze`, 1,000 `aten::sin`/`cos`, etc.).
   - Genuine single-kernel fusion is **BLOCKED** on this host due to missing toolchains (no `nvcc`, MSVC `cl.exe`, or `triton` compiler installed on the system).
   - Under the mandatory compliance contract, single-kernel fusion is an uncompromised acceptance requirement. Because the implementation is unfused and compiles multiple discrete kernels, the fusion gate status is **FAIL**.
   - Consequently, the final disposition of the CUDA backend is **EXPERIMENTAL / NOT ACCEPTED**. CUDA is **NOT** accepted for production, and is **NOT** accepted for large batches under the protocol.

2. **Separation of Open-Loop Physics Correctness from Production-Path Behavioral Equivalence:**
   - **Float64 Physics-State Parity (PASS):** Under prescribed open-loop actions across 50 non-holdout seeds (`100..149`) over 500 steps (25,000 state transitions), the maximum 4-component state error `[x, x_dot, theta, theta_dot]` against serial reference `PerturbedCartPoleEnv` and vectorized `VectorizedCartPoleEnv` is **`3.38e-14`** (< 1e-9). The historical 2.98e-8 defect was resolved as an artifact of comparing float64 state against Gymnasium's float32-cast observation vector (`obs[2]`), whereas comparing directly to `np.cos(env._state[2])` yields `1.11e-16`.
   - **Production-Path Closed-Loop Behavioral Equivalence (INCOMPLETE / INCONCLUSIVE):** An exhaustive audit of the test suite (`tests/test_cuda_evaluator.py`) reveals that while macro condition-level success rates match across all 10 diagnostic conditions (59.60% mean), the tests do **not** log or assert a step-by-step trace of per-step actions, applied actions (through circular delay buffers), states, termination decisions, and individual episode binary success flags in a closed loop. `test_step_by_step_closed_loop_action_equivalence` only tests isolated controller feedforward on 100 static observation vectors. Because granular per-step trace comparisons are absent, production closed-loop behavioral equivalence cannot be certified as fully proven, and is classified as **INCOMPLETE / INCONCLUSIVE**.

3. **Audited Benchmark Claims vs. Raw Measurements (`results/cuda_benchmark.csv`):**
   - **End-to-End Candidate Fitness-Batch Evaluation (1 candidate, 10 conditions, 50 seeds, 500 steps = 250,000 steps):**
     - **Serial Reference CPU (float64):** Median duration **13.2552 s**, median throughput **18,860 steps/s** (mean: 20,463 steps/s).
     - **Vectorized NumPy CPU (float64, Production Default):** Median duration **0.9248 s**, median throughput **270,319 steps/s** (mean: 269,525 steps/s) — **14.33× faster than Serial Reference**.
     - **TorchScript CUDA (float64):** Median duration **2.6146 s**, median throughput **95,618 steps/s** (mean: 95,647 steps/s, peak VRAM: 9.01 MB).
     - **TorchScript CUDA (float32):** Median duration **2.5206 s**, median throughput **99,182 steps/s** (mean: 99,210 steps/s, peak VRAM: 8.81 MB).
     - **Workload Conclusion:** For the production candidate evaluation workload, **Vectorized NumPy CPU is 2.83× FASTER than CUDA float64**. Small matrix operations ($50 \times 4$ and $50 \times 8$) fit inside CPU L1/L2 caches with zero GPU kernel dispatch overhead.
   - **Batch Scaling Microbenchmark (Synthetic Single-Condition Scaled Batches 50..5000):**
     - The multi-batch scaling test is a **synthetic microbenchmark** measuring raw GPU SM occupancy scaling on a single nominal condition, and is **not equivalent** to multi-condition fitness evaluation.
     - Crossover is observed between batch sizes 500 and 1,000 (empirically ~650 environments).
     - At batch size 5,000, CUDA throughput reaches **10,209,227 steps/s** (0.2449 s median duration) vs. NumPy's **1,597,398 steps/s** (1.5650 s median duration), an observed speedup of **6.39×**.
     - This measured large-batch scaling is preserved strictly as an **experimental performance observation**, **NOT** an acceptance decision or approval for production deployment.

4. **Repository Integrity & Frozen Baselines:**
   - All 13 frozen checkpoints and configuration files match baseline SHA256 hashes bit-for-bit.
   - Git tracked modifications are present in `README.md` and `results/phase_3a_report.md` (historical audit notes).
   - Untracked audit files are maintained without modifying frozen experimental configs or Phase 3B directories.

5. **Final Disposition:**
   - **EXPERIMENTAL / NOT ACCEPTED.**
   - **Vectorized NumPy CPU remains the sole accepted production default evaluator.**

---

## 2. Baseline Architecture & Host Environment

### 2.1 Hardware and Runtime Telemetry
- **Host Operating System:** Windows 11 Home / Pro (Build 10.0.26100)
- **Host CPU:** AMD / Intel x86_64
- **Host Python:** Python 3.12.10 (MSC v.1943 64-bit)
- **PyTorch Installation:** 2.5.1+cu121
- **CUDA Runtime:** CUDA 12.1
- **GPU Accelerator:** NVIDIA GeForce RTX 4050 Laptop GPU
  - Compute Capability: 8.9 (Ada Lovelace)
  - VRAM: 6,141 MiB GDDR6
  - Device Count: 1
- **Compiler Investigation:**
  - `nvcc`: None (not installed / not on PATH)
  - `cl.exe`: None (MSVC compiler not on PATH)
  - `gcc`: Present at `C:\msys64\ucrt64\bin\gcc.EXE` (MinGW)
  - `ninja`: None
  - `triton`: None (unsupported on Windows PyTorch runtime)

### 2.2 Production Call Graph
```
Candidate Genome (MaskedGenome)
   │
   ├── evaluate_candidate (CudaCartPoleBatchEvaluator)
   │     │
   │     ├── Pre-generate / format initial states per seed
   │     ├── Format condition tensors (mc, mp, bc, bp, delay, noise)
   │     ├── _fused_rollout_kernel (TorchScript JIT @torch.jit.script)
   │     │     │
   │     │     └── 500 Timesteps on GPU:
   │     │           ├── Raw observation formatting & unnormalized noise injection
   │     │           ├── Input normalization: [x/2.4, clip(xd/10), cos, sin, clip(td/10)]
   │     │           ├── Neural forward: z1 = obs @ w1.T + b1; h = tanh(2.5*z1)*mask; z2 = h @ w2.T + b2
   │     │           ├── Action u = tanh(2.5*z2[0]); circular delay buffer fetch
   │     │           ├── Force clipping: clip(app_u, -1, 1) * 10.0
   │     │           ├── Non-linear equations of motion (Semi-implicit Euler integration)
   │     │           ├── Coordinate angle wrap to [-pi, pi]
   │     │           └── Rail boundary violation tracking: |x| >= 2.4
   │     │
   │     ├── Success Predicate Reduction on GPU:
   │     │     └── upright_counts >= 95 in final 100 steps & ~rail_hit
   │     │
   │     └── Synchronization: torch.cuda.synchronize() & Telemetry Collection
```

---

## 3. Kernel Fusion Audit & Mandatory Gate Enforcement

### 3.1 Implementation Categorization
The protocol defines four distinct categories:
1. Python loop dispatching individual CUDA tensor operations.
2. TorchScript-compiled control flow that still launches multiple GPU kernels per timestep.
3. Graph-captured execution that replays multiple kernels with reduced host overhead.
4. A genuinely fused or compiled rollout in which simulation and controller computation are combined into a single device kernel.

**Classification:** The implementation is unequivocally **Category 2 (TorchScript-compiled control flow launching multiple GPU kernels per timestep)**.

### 3.2 Profiler Evidence
Profiling `CudaCartPoleBatchEvaluator.evaluate_candidate` over a 500-step rollout using `torch.profiler` yielded the following kernel breakdown:

| Kernel / Operation Name | Call Count (500 steps) | Self CPU Time | Self CUDA Time | CUDA Time Avg | Category Note |
|---|---|---|---|---|---|
| `_fused_rollout_kernel` (TorchScript Wrapper) | 1 | 258.38 ms | 179.37 ms | 787.68 ms total | JIT control flow |
| `aten::as_strided` | 9,508 | 6.49 ms | 85.78 ms | 9.02 $\mu$s | Tensor view/stride manipulation |
| `aten::unsqueeze` | 4,500 | 80.61 ms | 85.34 ms | 28.00 $\mu$s | Shape manipulation |
| `aten::select` | 3,000 | 61.64 ms | 64.37 ms | 30.78 $\mu$s | Index slicing |
| `aten::stack` | 1,000 | 105.28 ms | 72.93 ms | 227.47 $\mu$s | Tensor stacking |
| `aten::mm` (cuBLAS GEMM) | 1,000 | 34.58 ms | 39.04 ms | 39.04 $\mu$s | 2 GEMMs per timestep (Layer 1 + Layer 2) |
| `aten::cat` | 1,000 | 24.27 ms | 28.54 ms | 28.54 $\mu$s | Tensor concatenation |
| `aten::transpose` / `aten::t` | 2,000 | 37.49 ms | 37.22 ms | 37.22 $\mu$s | Weight transpositions |
| `fused_clamp_mul_mul` | 500 | 11.86 ms | 14.09 ms | 28.17 $\mu$s | JIT elementwise sub-fuser |
| `fused_add_mul_tanh_mul` | 500 | 9.80 ms | 13.11 ms | 26.21 $\mu$s | JIT elementwise sub-fuser |
| `fused_div_clamp_div_clamp` | 500 | 9.53 ms | 13.01 ms | 26.01 $\mu$s | JIT elementwise sub-fuser |
| `aten::sin` / `aten::cos` | 1,000 | 20.47 ms | 24.62 ms | 24.62 $\mu$s | Trigonometric evaluations |
| `aten::remainder` / `aten::div` | 1,000 | 41.42 ms | 25.74 ms | 25.74 $\mu$s | Angle wrapping & equations of motion |
| **Total Launched GPU Operations** | **~15,500 launches** | **789.85 ms** | **790.20 ms** | — | **~31 launches per step** |

### 3.3 Gate Decision: Mandatory Fusion Gate FAIL
- The rollout executes ~31 distinct GPU kernels per timestep (~15,500 kernel launches per 500 steps).
- Genuine single-kernel fusion cannot be built in this environment because `nvcc` and MSVC `cl.exe` are missing from the host machine, and Triton is unsupported on Windows.
- Because genuine fusion is a strict prerequisite for CUDA backend acceptance, the gate decision is **FAIL**.
- In accordance with the compliance contract, failure of the fusion gate mandates that the backend disposition is **EXPERIMENTAL / NOT ACCEPTED**. CUDA cannot be certified or accepted for large batches or production use.

---

## 4. Numerical Parity Protocol & Diagnosed Defects

### 4.1 Frozen Acceptance Criterion
$$\text{maximum absolute state error across all 4 state components } [x, \dot{x}, \theta, \dot{\theta}] < 10^{-9} \quad (\text{strict inequality})$$

### 4.2 Defect 1: Historical 2.98e-08 Cosine Discrepancy
- **Defect Analysis:** In the initial report, a cosine discrepancy of $2.98\text{e}-08$ was documented, which exceeded the declared $\text{atol} < 10^{-9}$ threshold.
- **Root Cause:** The original test compared the `float64` cosine history from `_fused_rollout_kernel` against `obs[2]` returned by Gymnasium's `PerturbedCartPoleEnv.step()`. Gymnasium explicitly casts its observation vector to `np.float32`:
  ```python
  obs = np.stack([x, x_dot, cos_th, sin_th, theta_dot], axis=1).astype(np.float32)
  ```
  Comparing a `float64` cosine calculation against a `float32`-truncated observation incurs $\sim 1.4\text{e}-08$ to $3.0\text{e}-08$ quantization error.
- **Remediation & Retest:** When compared against the true underlying `float64` simulator state `np.cos(env._state[2])`, the maximum absolute cosine error over 50 seeds and 500 steps is:
  $$\mathbf{1.11\times 10^{-16}} \ll 10^{-9} \quad (\text{machine epsilon for float64})$$
  The float64 cosine calculation is mathematically verified.

### 4.3 Defect 2: Sensor Noise Injection Point & Schedule
- **Defect Analysis:** Initial testing across the 10 diagnostic conditions showed divergence in noisy conditions:
  - `5_SensorNoise_0.05`: Vectorized CPU = $94.0\%$, CUDA = $60.0\%$ (Mismatch).
  - `7_CombinedHistoricalStress`: Vectorized CPU = $6.0\%$, CUDA = $28.0\%$ (Mismatch).
- **Root Cause:**
  1. *Normalization Order:* In `env/vectorized_env.py` and `PerturbedCartPoleEnv`, Gaussian noise is added to the **unnormalized raw observations** $[x, \dot{x}, \cos\theta, \sin\theta, \dot{\theta}]$, and only then clipped and scaled by `norm_batch_obs()`. In `env/cuda_evaluator.py`, noise was previously added directly to `norm_x_dot` and `norm_th_dot` after clipping and dividing by 10.0, resulting in 10× excessive velocity noise.
  2. *Schedule Length:* In Gymnasium, an observation is generated at `reset()` (draw 0) and at each of the 500 steps (draws 1..500), totaling 501 noise draws. `cuda_evaluator.py` previously generated only 500 draws.
- **Remediation:** `_fused_rollout_kernel` and `evaluate_candidate` generate 501 float32 noise vectors and apply them to raw states prior to normalization, matching `VectorizedCartPoleEnv`.
- **Retest:** Condition-level success rates match:
  - `5_SensorNoise_0.05`: Vectorized CPU = $94.0\%$, CUDA = $94.0\%$ (**MATCH**).
  - `7_CombinedHistoricalStress`: Vectorized CPU = $6.0\%$, CUDA = $6.0\%$ (**MATCH**).

### 4.4 4-Component Open-Loop State Parity (PASS)
Over 50 permitted non-holdout seeds (`100..149`) across 500 environment steps (25,000 state transitions) under prescribed open-loop actions:
- **Maximum Error (CUDA vs Serial Reference `PerturbedCartPoleEnv`):** **`3.38e-14`** (< 1e-9)
- **Maximum Error (CUDA vs Vectorized `VectorizedCartPoleEnv`):** **`3.38e-14`** (< 1e-9)
- **Status:** **PASS**.

---

## 5. Production-Path Behavioral Equivalence Audit

### 5.1 Analysis of Executed Test Assertions

A rigorous audit was conducted on the specific assertions executed in `tests/test_cuda_evaluator.py`:

| Test Name | What Was Actually Tested | What Was NOT Tested | Scientific Classification |
|---|---|---|---|
| `test_float64_physics_state_parity` | Open-loop state integration given predefined actions $u_t = \sin(t \cdot 0.1)$. Compares all 4 state components step-by-step against serial and vectorized references. | Closed-loop policy interaction, delay queue propagation, neural network forward pass. | **PASS (Open-Loop Physics Parity)** |
| `test_step_by_step_closed_loop_action_equivalence` | Isolated forward pass of controller on 100 synthetic random observation vectors ($u_{\text{cpu}} \text{ vs } u_{\text{cuda}}$). | Closed-loop dynamic rollout; does NOT record or compare step-by-step actions during an actual environment trajectory. | **INCOMPLETE (Static Forward Pass Only)** |
| `test_closed_loop_champion_success_flag_agreement` | Aggregate success percentages (`succ_cpu == succ_cuda`, `per_cond_cpu[c] == per_cond_cuda[c]`) over 50 seeds across 10 conditions. | Per-step actions, applied actions from delay queue, step-by-step states, step-by-step termination triggers, individual binary episode success flags. | **INCOMPLETE (Aggregate Rates Only)** |
| `test_delay_queue_semantics_and_resets` | Aggregate success rates on delays 0, 1, 2 across 30 seeds. | Per-step contents of the circular delay buffer. | **PASS (Aggregate Delay Agreement)** |
| `test_rail_boundary_termination_parity` | Discrete step index where rail boundary $|x| \ge 2.4$ is triggered under constant $u=1.0$. | Step-by-step intermediate velocity and angle trajectory. | **PASS (Boundary Trigger Parity)** |

### 5.2 Behavioral Equivalence Conclusion: INCOMPLETE / INCONCLUSIVE
While macro success rates match at 59.60% across all 10 diagnostic conditions and open-loop physics integration is verified to machine precision, **complete production-path closed-loop equivalence requires an unbroken per-step assertion trace** comparing applied actions, delay buffers, state transitions, termination decisions, and individual episode success flags. Because the test suite evaluates only static forward passes and macro aggregate percentages, full closed-loop trace equivalence is scientifically classified as **INCOMPLETE / INCONCLUSIVE**.

---

## 6. Comprehensive Workload Benchmarking & Scaling Audit

All benchmarks in `results/cuda_benchmark.csv` were executed under controlled conditions:
- **Warmup:** 1 untimed warmup run executed prior to timed iterations.
- **Repeats:** Exactly 5 timed steady-state repeats per configuration.
- **GPU Synchronization:** Explicit `torch.cuda.synchronize()` before timing start and stop.
- **Methodology Delineation:** Candidate fitness-batch evaluation is an end-to-end multi-condition workflow; multi-batch scaling is a synthetic homogeneous microbenchmark.

### 6.1 End-to-End Candidate Fitness-Batch Evaluation (Phase 3A Workload)
Workload: 1 candidate evaluated across 10 diagnostic conditions, 50 seeds per condition, 500 steps = **500 episodes = 250,000 environment steps**.

| Backend Engine | Repeat 1 | Repeat 2 | Repeat 3 | Repeat 4 | Repeat 5 | Median Duration | Median Throughput | Mean Throughput | Speedup vs Ref | Speedup vs NumPy |
|---|---|---|---|---|---|---|---|---|---|---|
| **Serial Reference CPU (float64)** | 17,395 | 17,372 | 18,860 | 22,723 | 25,967 | 13.2552 s | **18,860 steps/s** | 20,463 st/s | 1.00× (base) | 0.07× |
| **Vectorized NumPy CPU (float64, Default)** | 271,619 | 270,818 | 270,317 | 270,319 | 264,550 | **0.9248 s** | **270,319 steps/s** | 269,525 st/s | **14.33×** | **1.00×** (default) |
| **TorchScript CUDA (float64)** | 95,618 | 94,770 | 95,454 | 96,021 | 96,373 | 2.6146 s | **95,618 steps/s** | 95,647 st/s | 5.07× | **0.35×** (NumPy 2.83× faster) |
| **TorchScript CUDA (float32)** | 99,703 | 99,182 | 98,607 | 99,414 | 99,145 | 2.5206 s | **99,182 steps/s** | 99,210 st/s | 5.26× | **0.37×** (NumPy 2.73× faster) |

*Key Result:* For the candidate fitness-batch workload, **Vectorized NumPy CPU is 2.83× FASTER than CUDA float64**. Small batch matrix operations ($50 \times 4$ and $50 \times 8$) operate entirely within CPU cache with 0 kernel dispatch latency. CUDA executes ~15,500 GPU kernel launches with host-device orchestration overhead, causing a severe performance deficit.

---

### 6.2 Multi-Batch Scaling Suite (Synthetic Microbenchmark)
Workload: Synthetic scaling across concurrent environments from 50 to 5,000 on a single nominal condition over 500 steps (5 repeats each, medians reported):

| Batch Size (Envs) | Total Steps | NumPy CPU Duration | NumPy CPU Throughput | CUDA f64 Duration | CUDA f64 Throughput | CUDA f32 Throughput | CUDA Speedup (f64 vs NumPy) | CUDA Speedup (f32 vs NumPy) |
|---|---|---|---|---|---|---|---|---|
| **50** | 25,000 | 0.0645 s | 387,495 steps/s | 0.2128 s | 117,467 steps/s | 112,009 steps/s | **0.30×** (NumPy faster) | **0.29×** |
| **200** | 100,000 | 0.1104 s | 905,870 steps/s | 0.2466 s | 405,556 steps/s | 400,452 steps/s | **0.45×** (NumPy faster) | **0.44×** |
| **500** | 250,000 | 0.2050 s | 1,219,625 steps/s | 0.2462 s | 1,015,598 steps/s | 991,023 steps/s | **0.83×** (NumPy faster) | **0.81×** |
| **1,000** | 500,000 | 0.3575 s | 1,398,525 steps/s | 0.2448 s | **2,042,266 steps/s** | 1,971,867 steps/s | **1.46×** (Crossover) | **1.41×** |
| **2,000** | 1,000,000 | 0.6635 s | 1,507,127 steps/s | 0.2469 s | **4,050,235 steps/s** | 3,935,337 steps/s | **2.69×** | **2.61×** |
| **5,000** | 2,500,000 | 1.5650 s | 1,597,398 steps/s | 0.2449 s | **10,209,227 steps/s** | 9,786,421 steps/s | **6.39×** | **6.13×** |

### 6.3 Scientific Interpretation of Microbenchmark Results
1. **Delineation from Fitness Evaluation:** This microbenchmark measures raw GPU compute scaling on a single homogeneous condition. It does not sequence through 10 perturbed conditions or handle disparate condition parameterizations.
2. **GPU Fixed Latency Floor:** For any batch size $\le 5,000$, CUDA execution duration is dominated by a fixed kernel launch latency floor of $\approx 0.245\text{ s}$ (~15,500 kernel dispatches). As batch size scales from 50 to 5,000, GPU streaming multiprocessors become saturated, scaling throughput from $117\text{k}$ to **$10.2\text{M}$ steps/s**.
3. **Crossover Threshold:** Crossover occurs between batch sizes **500 and 1,000** (empirically ~650 environments).
4. **Experimental Nature:** High throughput at batch sizes $\ge 1,000$ represents a valid empirical scaling observation, but is **NOT** an acceptance decision. Acceptance remains blocked by the failed fusion gate.

---

## 7. Precision Policy & Float32 Findings

1. **Float64 Parity:** Preserved as the default correctness standard for all parity assertions.
2. **Float32 Performance:** Float32 achieved $99,182\text{ steps/s}$ vs $95,618\text{ steps/s}$ on float64 (+3.7% throughput improvement). The primary bottleneck is kernel dispatch latency, not floating-point ALU throughput.
3. **Float32 Ranking Divergence:** Due to chaotic divergence in inverted pendulum dynamics over 500 steps, float32 causes subtle ranking inversions on borderline seeds. Float32 is excluded from production consideration.

---

## 8. Test Suite Verification & Execution Evidence

### 8.1 Unit Test Suite Execution
```powershell
python -m unittest tests/test_cuda_evaluator.py -v
```
**Process Exit Code:** `0`  
**Test Results:** **13 passed, 0 failed, 0 skipped** (Duration: 13.344 s)

```
test_closed_loop_champion_success_flag_agreement ... ok
test_cpu_fallback_path ... ok
test_cuda_availability_and_device_info ... ok
test_delay_queue_semantics_and_resets ... ok
test_deterministic_replay ... ok
test_float64_physics_state_parity ... ok
test_holdout_seed_rejection ... ok
test_malformed_candidate_parameters_handling ... ok
test_mixed_terminal_states_batch ... ok
test_production_fused_rollout_kernel_invocation ... ok
test_rail_boundary_termination_parity ... ok
test_step_by_step_closed_loop_action_equivalence ... ok
test_success_predicate_exact_boundaries ... ok
```

### 8.2 Regression Test Suite Execution
```powershell
python -m unittest tests/test_phase_3a_evolution.py -v
```
**Process Exit Code:** `0`  
**Test Results:** **8 passed, 0 failed** (Duration: 2.836 s)

---

## 9. Final Compliance & Acceptance Matrix

| Requirement / Gate | Status | Executed Evidence | Remaining Blocker / Caveat |
|---|---|---|---|
| **Actual Kernel Fusion Verified** | **FAIL** | Profiler captured 15,500+ kernel launches over 500 steps (~31 launches/step, Category 2 TorchScript JIT). | Missing host toolchains (`nvcc`, `cl.exe`, `triton`); single-kernel compilation blocked. |
| **Float64 State Tolerance (< 1e-9)** | **PASS** | `test_float64_physics_state_parity` max error **`3.38e-14`** (< 1e-9) across 50 seeds, 500 steps, all 4 state components. | None. Historical 2.98e-8 defect resolved (was float32 Gym obs quantization). |
| **Production-Path Closed-Loop Equivalence** | **INCOMPLETE / INCONCLUSIVE** | Condition-level macro rates match (59.60% vs 59.60%), but test suite does not compare per-step actions, applied actions, states, or termination decisions. | Per-step closed-loop trace assertions are missing from the test suite. |
| **Noise and Delay Parity** | **PASS** | Delays 0, 1, 2 and noise $\sigma=0.05$ match CPU Vectorized at aggregate level. | None. |
| **Reset and Termination Parity** | **PASS** | Discrete timestep of rail violation at $|x| \ge 2.4$ matches bit-for-bit. | None. |
| **Success-Predicate Boundaries** | **PASS** | Strict boundaries tested at $|x|=2.4$, $\cos\theta=0.9$, 94 vs 95 steps, 499 vs 500 horizon. | None. |
| **Deterministic Replay** | **PASS** | `test_deterministic_replay` passes with bit-exact identical success rates. | None. |
| **Equivalent End-to-End Benchmarks** | **PASS** | 5 repeats per backend measured with explicit `torch.cuda.synchronize()`; end-to-end separated from microbenchmarks. | None. |
| **Repeatability & Performance Crossover** | **OBSERVED (EXPERIMENTAL)** | Crossover observed at ~650 envs; at 5,000 envs CUDA reaches 10.2M steps/s (6.39× speedup). | For batch size 50 (Phase 3A workload), CPU NumPy is 2.83× faster than CUDA. |
| **CPU Fallback Preserved** | **PASS** | `test_cpu_fallback_path` passes; Vectorized NumPy CPU confirmed as default. | None. |
| **Frozen Artifacts Preserved** | **PASS** | All 13 frozen checkpoint and config SHA256 hashes verified identical. | None. |
| **Protected Holdouts Untouched** | **PASS** | Seeds `2000–2099` and Objective B unaccessed; `test_holdout_seed_rejection` verified. | None. |
| **Overall Backend Acceptance** | **EXPERIMENTAL / NOT ACCEPTED** | Mandatory fusion gate failed; intended workload is 2.83× slower than NumPy CPU; closed-loop traces incomplete. | **NOT ACCEPTED FOR PRODUCTION. Vectorized NumPy CPU remains default.** |

---

## 10. Final Architectural Disposition & Operational Directives

### Final Disposition:
**EXPERIMENTAL / NOT ACCEPTED**

### Operational Directives:
1. **Mandatory Production Evaluator:** `VectorizedCartPoleEnv` on NumPy CPU is the **sole accepted production evaluator** for all evolutionary optimization, Phase 3A workloads, and validation pipelines. It provides 270,319 steps/s (2.83× faster than CUDA on single candidates) with zero GPU initialization, memory transfer, or kernel dispatch overhead.
2. **CUDA Backend Status:** The CUDA backend (`env/cuda_evaluator.py`) is categorized strictly as an **experimental prototype**. It is **NOT** accepted for production, and is **NOT** accepted for large batches under the compliance contract. Measured large-batch speedups remain experimental performance observations only.
3. **Integrity Confirmation:** No evolutionary runs were performed, Phase 3B remains untouched, and protected holdouts (`2000–2099` and Objective B) remain pristine and unqueried.
