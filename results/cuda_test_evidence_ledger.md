# EvoEdge CUDA — Test Evidence Ledger

**Audit Protocol:** EvoEdge CUDA Strict Final Compliance Audit and Remediation Protocol  
**Audit Date:** October 13, 2026  
**Host Environment:** Windows 11, Python 3.12.10, PyTorch 2.5.1+cu121, NVIDIA RTX 4050 Laptop GPU (sm_89)  

---

## 1. Compliance Gate Ledger

| Gate ID | Requirement Description | Status | Executed Evidence Reference | Remaining Blocker / Caveat |
|---|---|---|---|---|
| **GATE-01** | Actual Kernel Fusion Verified | **FAIL** | Profiler trace (`torch.profiler` in `env/cuda_evaluator.py`) captured 15,500+ kernel launches over 500 steps (~31 launches per step). Categorized as Category 2 (TorchScript JIT control flow), NOT single-kernel fused. | Single-kernel compilation blocked by missing host toolchain (`nvcc`, `cl.exe`, `triton` absent on Windows host). Precludes acceptance. |
| **GATE-02** | Float64 State Parity Below 1e-9 | **PASS** | `tests/test_cuda_evaluator.py::test_float64_physics_state_parity` executed over 50 non-holdout seeds (100..149) across 500 steps under open-loop actions. Max absolute 4-component state error is **3.38e-14** (< 1e-9). | Open-loop physics integration is verified to machine precision. Historical 2.98e-8 defect resolved as float32 Gym observation comparison artifact. |
| **GATE-03** | Production-Path Closed-Loop Equivalence | **INCOMPLETE / INCONCLUSIVE** | `tests/test_cuda_evaluator.py::test_closed_loop_champion_success_flag_agreement` matches aggregate rates across 10 diagnostic conditions (59.60% vs 59.60%), but test suite lacks per-step closed-loop assertions for applied actions, delay buffers, state trajectories, and individual episode termination decisions. | Per-step closed-loop trajectory tracing is unasserted in the test suite; classified as incomplete. |
| **GATE-04** | Noise and Delay Parity | **PASS** | `tests/test_cuda_evaluator.py::test_delay_queue_semantics_and_resets` verifies delays 0, 1, 2; condition `5_SensorNoise_0.05` matches at 94.00% across backends. | Unnormalized noise injection point and 501-vector schedule synchronized with Gymnasium semantics. |
| **GATE-05** | Reset and Termination Parity | **PASS** | `tests/test_cuda_evaluator.py::test_rail_boundary_termination_parity` confirms discrete rail-hit step parity across 50 seeds. | None. |
| **GATE-06** | Success-Predicate Boundaries | **PASS** | `tests/test_cuda_evaluator.py::test_success_predicate_exact_boundaries` verifies strict inequality for \|x\|=2.4, cos(th)=0.9, 94 vs 95 upright steps, and 499 vs 500 horizon. | None. |
| **GATE-07** | Deterministic Replay | **PASS** | `tests/test_cuda_evaluator.py::test_deterministic_replay` passes with bit-exact identical success and telemetry metrics. | None. |
| **GATE-08** | Equivalent End-to-End Benchmarks | **PASS** | `experiments/benchmark_backends.py` executed across 5 timed repeats with explicit synchronization and warmup tracking. Candidate evaluation (end-to-end) is clearly separated from batch scaling microbenchmarks. | None. |
| **GATE-09** | Repeatability & Performance Crossover | **OBSERVED (EXPERIMENTAL)** | Multi-batch scaling (50 to 5,000 envs) measured in `results/cuda_benchmark.csv`. Crossover observed at ~650 envs. At batch size 5,000, CUDA reaches 10.2M steps/s (6.39× speedup). | Measured large-batch speedup is an experimental observation, NOT an acceptance decision. For Phase 3A candidate workload (batch 50), NumPy CPU is 2.83× faster. |
| **GATE-10** | CPU Fallback Preserved | **PASS** | `tests/test_cuda_evaluator.py::test_cpu_fallback_path` passes; VectorizedCartPoleEnv confirmed as default evaluator. | None. |
| **GATE-11** | Frozen Artifacts Preserved | **PASS** | Cryptographic SHA256 hashes of all 13 checkpoints and configs verified identical before and after audit. | None. |
| **GATE-12** | Protected Holdouts Untouched | **PASS** | Seeds 2000–2099 and Objective B unaccessed. `tests/test_cuda_evaluator.py::test_holdout_seed_rejection` verifies rejection. | None. |

---

## 2. Overall Disposition

**EXPERIMENTAL / NOT ACCEPTED**

- **Mandatory Kernel Fusion:** **FAIL** (Implementation executes ~15,500 discrete GPU kernel launches; compilation of a single fused kernel is blocked by missing host toolchain). Under the compliance contract, this failure mandates non-acceptance.
- **Open-Loop Numerical Parity:** **PASS** (Float64 4-component state error is $3.38\times 10^{-14} < 10^{-9}$).
- **Closed-Loop Behavioral Equivalence:** **INCOMPLETE / INCONCLUSIVE** (Aggregate condition rates agree at 59.60%, but granular per-step trace comparisons of actions, applied delay buffer actions, states, and termination triggers are absent from the test suite).
- **Intended Candidate Workload Performance:** **FAIL** (Vectorized NumPy CPU is **2.83× faster** than CUDA float64 for the Phase 3A candidate evaluation workload: 0.9248 s vs 2.6146 s).
- **Production Default Evaluator:** **Vectorized NumPy CPU (`VectorizedCartPoleEnv`) is retained as the sole accepted production evaluator.**
- **CUDA Disposition:** Classified strictly as **EXPERIMENTAL / NOT ACCEPTED**. CUDA is **not accepted** for large batches or production use. Measured large-batch speedups remain experimental performance observations only.
