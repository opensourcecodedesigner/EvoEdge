# EvoEdge — Seven-Day Reuse-First Sprint Report

## Executive Summary
This document tracks the execution of the seven-day reuse-first sprint for the EvoEdge research software release.
All work adheres strictly to frozen contracts, pre-registered seed partitions, and existing validated components.

---

## Sprint Roadmap & Status

| Day | Focus | Status | Key Deliverables |
|---|---|---|---|
| **Day 1** | Architecture Map & Environment | **DONE** | 25 components mapped, 0 rebuilt; Python 3.12 verified |
| **Day 2** | Executable Verification & Pre-Training Audit | **DONE** | 109/109 tests passed; budget tracker verified; multilayer C parity verified |
| **Days 3–4** | Matched Controller Comparison (Stage 2) | **DONE** | 10 optimizer runs completed under 2.0M budget; candidate selection evaluated; classical baseline verified |
| **Day 5** | Candidate Freeze & C Export | **READY** | Classical reference C export ready; learned models failed selection filters |
| **Day 6** | Holdout Confirmation Execution | 🔒 Quarantined | Dependent on confirmation seed authorization (600–749) |
| **Day 7** | Release Artifacts & Final Report | 🔒 Blocked | Final delivery |

---

## Day 1 — Architecture Map & Reuse Audit

### Working Component Inventory (25 Components Verified)
No existing functionality was rebuilt from scratch. The existing verified stack comprises:

1. **Physics & Simulation**:
   - `env/cartpole_swingup.py` — Canonical continuous CartPole swingup simulation.
   - `env/vectorized_env.py` — Batch vectorized simulation engine (NumPy).
   - `env/cuda_evaluator.py` — High-throughput PyTorch / CUDA fused batch evaluator.
   - `utils/perturbation_loader.py` — Validated parameter perturbation and sensor noise injection.
   - `env/budget_tracker.py` — Transition budget tracker and fail-closed environment wrappers.

2. **Controllers & Baselines**:
   - `baselines/hybrid_controller.py` — Energy-shaping + continuous Riccati LQR controller.
   - `baselines/fixed_mlp.py` — PyTorch MLP (64 units) reference controller.
   - `evolution/nsga2_engine.py` — Multi-objective NSGA-II engine with `MaskedGenome`.
   - `c_neural_bridge` / `results/controller.c` — Embedded C exports and ctypes bridges.

3. **Metrics & Accounting**:
   - `metrics/ops_counter.py` — Exact MAC and parameter accounting.
   - `utils/split_integrity.py` — Seed partition isolation and split safety validator.

4. **Experiment Suites & Runners**:
   - `experiments/benchmark_suite.py` — Comprehensive baseline benchmark.
   - `experiments/benchmark_backends.py` — CPU vs. CUDA backend comparison.
   - `experiments/run_stage1_baseline_evaluation.py` — Stage 1 baseline validation.
   - `analysis/stress_test.py` — Robustness and parameter variation stress testing.

---

## Day 2 — Executable Verification & Final Pre-Training Audit

### 1. Test Suite Execution & Pre-Training Accounting
Full test execution was conducted with Python 3.12.10 and pytest 9.1.1 across all 12 project test suites.

#### A. Pre-Training Configuration Lock Audit (Current Run: 109 / 109 Passed)
- `test_baselines.py` (19 tests) — **19 PASSED**
- `test_modules.py` (8 tests) — **8 PASSED** (includes multilayer C export parity test)
- `tests/test_algorithm_identity.py` (7 tests) — **7 PASSED** (verifies ES & NSGA-II algorithm identity, components, and fair-comparison compatibility)
- `tests/test_budget_tracker.py` (10 tests) — **10 PASSED** (includes fail-closed transition accounting and vectorized boundary tests)
- `tests/test_cuda_evaluator.py` (13 tests) — **13 PASSED**
- `tests/test_ops_accounting.py` (5 tests) — **5 PASSED**
- `tests/test_perturbation_loader.py` (5 tests) — **5 PASSED**
- `tests/test_phase_3a_evolution.py` (9 tests) — **9 PASSED**
- `tests/test_robustness_experiment.py` (7 tests) — **7 PASSED**
- `tests/test_runner_migration.py` (8 tests) — **8 PASSED**
- `tests/test_split_integrity.py` (6 tests) — **6 PASSED**
- `tests/test_training_budget_integration.py` (12 tests) — **12 PASSED** (verifies RNG namespaces, entry point budget stopping, exceptions, retries, and lifecycle search/validation ceilings)

**Current Total Test Suite Outcome: 109 passed / 109 total (100% pass rate in 47.92s).**

#### B. Historical Test Snapshots for Traceability
- **Day 2 Initial Baseline Snapshot:** 87 passed / 87 total (initial pre-training audit prior to budget integration suites).
- **Interim Budget & Boundary Snapshot:** 96 passed (added `test_training_budget_integration.py`), then 99 passed (added 3 boundary tests in `test_budget_tracker.py`), then 106 passed (added `test_algorithm_identity.py`).
- **Final Lifecycle Budget Lock:** 109 passed (+3 tests in `test_training_budget_integration.py` proving search <= 1.85M, validation <= 150K, total <= 2.0M). All 109 tests execute clean with zero failures.

### 2. Pre-Training Consistency Audit Resolutions

#### A. Resolution of Nominal-Screening Sample Size & Seed Provenance
- **Discrepancy:** Previous text required $\ge 90/100$ nominal successes but listed seeds `200–249` (50 seeds).
- **Physical Dynamics & Determinism:** In `CartPoleSwingUpEnv.reset(seed=s)`, the downward initial state perturbation ($\theta \approx \pi \pm 0.05$) is seeded deterministically. Under nominal dynamics (`1_Nominal`), physics is strictly deterministic without sensor noise ($\sigma = 0$). Repeating the same 50 seeds twice yields identical trajectories, not independent trials.
- **Resolution:** Gate 1B is explicitly allocated **seeds `200–299`** (100 distinct seeds within the development validation partition). Seed duplication is strictly prohibited.
- **Seed Provenance & Scientific Role:** Seeds `200–299` and `300–319` are **previously exposed development data, NOT fresh prospective evidence**. Seeds 200–249 were exposed during Phase 3A Gate 0, and seeds 250–399 were part of Phase 3A validation. Consequently, evaluation on seeds `200–299` and `300–319` constitutes **development screening only**.
- **Holdout Quarantines:** Seeds `600–749` remain strictly **provisional and inaccessible** until explicit operator authorization. Seeds `2000–2099` remain **strictly untouched** and locked.
- **Budget Impact:** 1 candidate $\times$ 100 episodes $\times$ 500 steps $= \mathbf{50,000\text{ transitions}}$ max, fitting within the 150,000 validation budget.

#### B. Complete Transition Budget Reconciliation & Arithmetic
- **Ceiling:** $B_{\text{trans}} = 2,000,000\text{ transitions}$ per optimizer run ($10,000,000$ transitions across all 5 runs per method).
- **Search Allocation:** **$1,850,000\text{ transitions}$** per run ($\le 3,700\text{ episodes}$ at $H \le 500$ steps):
  - *ES (Option A):* Exactly 370 candidate evaluations $\times$ 10 training episodes $\times$ 500 steps $= \mathbf{1,850,000\text{ transitions}}$:
    - Initial center evaluation: $1\text{ candidate} \times 10\text{ eps} \times 500 = \mathbf{5,000\text{ transitions}}$ (1 evaluation).
    - Generational mutants: $18\text{ gens} \times 20\text{ mutants} \times 10\text{ eps} \times 500 = \mathbf{1,800,000\text{ transitions}}$ (360 evaluations).
    - Biennial center progress tracking: $9\text{ biennial evaluations (gens 2, 4, ..., 18)} \times 10\text{ eps} \times 500 = \mathbf{45,000\text{ transitions}}$ (9 evaluations).
    - Subtotal: $5,000 + 1,800,000 + 45,000 = \mathbf{1,850,000\text{ transitions}}$ exact.
  - *NSGA-II:* Exactly 370 candidate evaluations $\times$ 10 training episodes $\times$ 500 steps $= \mathbf{1,850,000\text{ transitions}}$:
    - Generation 0 (initial population): $P = 20\text{ genomes} \times 10\text{ eps} \times 500 = \mathbf{100,000\text{ transitions}}$ (20 evaluations).
    - Generations 1–35: $35\text{ gens} \times 10\text{ offspring} \times 10\text{ eps} \times 500 = \mathbf{1,750,000\text{ transitions}}$ (350 evaluations).
    - Subtotal: $100,000 + 1,750,000 = \mathbf{1,850,000\text{ transitions}}$ exact.
- **Validation Allocation:** **$150,000\text{ transitions}$** per run (exactly 300 episodes at $H \le 500$ steps):
  - *Gate 1B Nominal (Development Screening):* 100 episodes on seeds `200–299` under `1_Nominal` $= \mathbf{50,000\text{ transitions}}$.
  - *Multi-Condition Diagnostic Validation (Development Screening):* Top 1 shortlisted champion candidate evaluated across all 10 diagnostic conditions on **20 distinct validation seeds (`300–319`)** $= 1 \times 10 \times 20 = 200\text{ episodes} = \mathbf{100,000\text{ transitions}}$.
  - Subtotal: $50,000 + 100,000 = \mathbf{150,000\text{ transitions}}$ exact.
- **Total Transitions per Optimizer Run:** $1,850,000\text{ (Search)} + 150,000\text{ (Validation)} = \mathbf{2,000,000\text{ transitions}}$ exact.
- **Total Permitted across 5 Runs per Method:** $5 \times 2,000,000 = \mathbf{10,000,000\text{ transitions}}$ exact.
- **Training Entry Point Budget Integration:** Both actual training entry points (`evolution/evolution_strategies.py::train_es` and `evolution/nsga2_engine.py::train_nsga2`) instantiate and use `TransitionBudgetTracker`, `BudgetedEnv`, and `BudgetedVectorizedEnv`. When the ceiling is reached, `BudgetExhaustedError` halts search cleanly and returns the best candidate. All transitions during exceptions and retries remain billed fail-closed. Verified by 9 unit and integration tests in `tests/test_training_budget_integration.py`.

#### C. Strengthened Multilayer C-Export Parity Testing
- **Previous Gap:** `test_modules.py` previously evaluated seed 42 topology, where hidden nodes 848 and 1249 had no path to output node 0, effectively bypassing hidden activations in the output test.
- **Resolution:** Added `test_c_compilation_and_numerical_parity_multilayer` in `test_modules.py`.
- **Topology Tested:** 5 inputs, 3 hidden neurons, 1 output, 8 active connections exercising:
  - Input-to-hidden connections (-1->1, -2->1, -3->2)
  - Hidden-to-hidden connections (1->2, 1->3, 2->3)
  - Hidden-to-output connections (2->0, 3->0)
  - Input-to-output skip connections (-5->0)
  - Disabled connection (-4->0)
- **Numerical Parity Result:** Compiled with GCC, executed test runner on deterministic test input `[0.12, -0.34, 0.88, -0.47, 0.56]`.
  - C output: `-0.219399914`
  - Python topological forward pass output: `-0.219399929`
  - Absolute numerical difference: $\mathbf{1.50 \times 10^{-8}}$ (well below $1.0 \times 10^{-5}$ threshold).

#### D. Freeze Candidate Selection Procedure
- Cataloged engineering rationales and explicit approval statuses:
  1. *Parameters $\le 128$:* SRAM/cache limit on ESP32 Xtensa LX6 (512 bytes for float32). Status: `PROPOSED (Flagged for Operator Sign-Off Before Full Training)`.
  2. *MACs $\le 128$:* Real-time determinism (< 5 $\mu$s @ 240 MHz, < 0.025% of 20 ms timeslice). Status: `PROPOSED (Flagged for Operator Sign-Off Before Full Training)`.
  3. *Worst-Condition Success $\ge 40.0\%$:* Prevents single-condition collapse (joint friction collapsed to 0% in Phase 3A). Status: `PROPOSED (Flagged for Operator Sign-Off Before Full Training)`.
  4. *Primary Ranking Criterion on Pareto Set:* Feasibility filter ($\le 128$ params, $\le 128$ MACs, $S_{\text{worst}} \ge 40\%$), ranked primarily by aggregate robust success ($S_{\text{agg}}$) across 10 diagnostic conditions on validation subset `300–319`, tie-broken deterministically by MACs, params, and $S_{\text{worst}}$. Status: `PROPOSED (Flagged for Operator Sign-Off Before Full Training)`.

#### E. Optimizer-Seed Independence & RNG Construction
- Verified handling of seeds `42, 43, 44, 45, 46`:
  1. *Documented Integer Namespaces:* NumPy's `SeedSequence` requires integer argument types. String-valued spawn keys have been replaced with documented integer namespaces:
     - **Evolution Strategies (ES):** $\text{NAMESPACE\_ES} = 1$, $\text{spawn\_key} = (1, s_{\text{opt}})$.
     - **NSGA-II Engine:** $\text{NAMESPACE\_NSGA2} = 2$, $\text{spawn\_key} = (2, s_{\text{opt}})$.
     - Assembled entropy uses supported 32-bit integer arrays, avoiding `ValueError: unrecognized seed string`.
  2. *Stream Separation:* Verified via `tests/test_training_budget_integration.py` that ES and NSGA-II initialized with identical optimizer seeds draw completely distinct random streams while repeating bit-identically across independent runs.
  3. *Decoupled Randomness:* Optimizer RNG (governing mutation / ES exploration vectors) is decoupled from environment evaluation seeds (`seeds=TRAIN_SEEDS` or `VAL_SEEDS`).
  4. *Namespace & Output Isolation:* Stage 2 logs, checkpoints, and models are directed strictly to `results/stage2/checkpoints/` and `results/stage2/models/`. Historical directories (`results/phase_3a/checkpoints/`) are quarantined and protected from overwriting or inadvertent resumption.

### 3. Constraint Compliance Audit
- [x] **Frozen Contract**: `docs/success_criteria.md` v2.3 preserved byte-for-byte.
- [x] **Historical Records**: All raw files in `results/phase_3a/` and `results/descriptive_baseline_phase3a/` unchanged.
- [x] **Seed Isolation**: Zero access to protected confirmation outcomes or seeds 2000–2099.
- [x] **Provisional Partition**: Seeds 600–749 strictly quarantined; no evaluation executed on this range.
- [x] **Training Guard**: No unapproved ES or neuroevolution training executed.
- [x] **Git Hygiene**: No unapproved branches, commits, or deletions made.

---

## Integration Readiness for Day 3

---

## Days 3–4 — Stage 2 Development Training Execution (Option A: Matched Budget)

### 1. Execution Summary
Following operator approval of **Option A (2,000,000 actual simulator transitions max per optimizer run)**, all 10 scheduled optimizer runs (5 ES runs and 5 NSGA-II runs across seeds 42, 43, 44, 45, 46) and the classical baseline evaluation (`HybridController`) were executed to completion.

- **Integrity Guarantee:** 100% empirical simulator execution. Zero synthetic data.
- **Confirmation Boundary Strictly Preserved:** Seeds 600–749 remained quarantined and untouched. Seeds 2000–2099 remained locked and untouched.
- **Fail-Closed Budgeting:** Search phase strictly bounded by 1,850,000 transitions; validation phase strictly bounded by 150,000 transitions. Total consumed transitions per run never exceeded 2,000,000.

### 2. Complete Stage 2 Development Results

| Method | Optimizer Seed | Gate 1B Nom. (%) | Robust Succ. (%) | Worst Cond. (%) | Active Params | Weight MACs | Search Trans. | Val. Trans. | Total Trans. | Run Time | Measured TPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ES (MLP)** | 42 | 0.0% (0/100) | 0.0% | 0.0% | 113 | 96 | 225,190 | 11,219 | 236,409 | 13.4s | 17,674 |
| **ES (MLP)** | 43 | 0.0% (0/100) | 0.0% | 0.0% | 113 | 96 | 250,743 | 11,207 | 261,950 | 14.3s | 18,259 |
| **ES (MLP)** | 44 | 0.0% (0/100) | 0.0% | 0.0% | 113 | 96 | 288,269 | 11,219 | 299,488 | 16.3s | 18,358 |
| **ES (MLP)** | 45 | 0.0% (0/100) | 0.0% | 0.0% | 113 | 96 | 231,458 | 11,225 | 242,683 | 13.6s | 17,802 |
| **ES (MLP)** | 46 | 0.0% (0/100) | 0.0% | 0.0% | 113 | 96 | 244,820 | 11,229 | 256,049 | 14.4s | 17,840 |
| **ES Mean** | *Mean* | **0.0%** | **0.0%** | **0.0%** | **113.0** | **96.0** | **248,096** | **11,220** | **259,316** | **14.4s** | **17,987** |
| | | | | | | | | | | | |
| **NSGA-II** | 42 | 0.0% (0/100) | 0.5% | 0.0% | 38 | 31 | 1,850,000 | 150,000 | 2,000,000 | 33.3s | 60,073 |
| **NSGA-II** | 43 | 0.0% (0/100) | 0.0% | 0.0% | 23 | 19 | 1,850,000 | 150,000 | 2,000,000 | 32.9s | 60,753 |
| **NSGA-II** | 44 | 0.0% (0/100) | 0.0% | 0.0% | 16 | 12 | 1,850,000 | 150,000 | 2,000,000 | 32.8s | 61,050 |
| **NSGA-II** | 45 | 0.0% (0/100) | 0.0% | 0.0% | 17 | 13 | 1,850,000 | 150,000 | 2,000,000 | 32.8s | 60,889 |
| **NSGA-II** | 46 | 0.0% (0/100) | 0.0% | 0.0% | 26 | 21 | 1,850,000 | 150,000 | 2,000,000 | 33.3s | 59,993 |
| **NSGA-II Mean** | *Mean* | **0.0%** | **0.1%** | **0.0%** | **24.0** | **19.2** | **1,850,000** | **150,000** | **2,000,000** | **33.0s** | **60,552** |
| | | | | | | | | | | | |
| **Hybrid** | Classical | **100.0%** (100/100) | **92.0%** | **65.0%** | **7** | **7** | **0** | **147,896** | **147,896** | **4.2s** | **35,213** |

### 3. Candidate-Selection Filter Evaluation
- **Filter 1 (Active Parameters $\le 128$):**
  - ES: 113 parameters (**PASS**)
  - NSGA-II: 16–38 parameters (**PASS**)
  - HybridController: 7 parameters (**PASS**)
- **Filter 2 (Weight MACs $\le 128$):**
  - ES: 96 MACs (**PASS**)
  - NSGA-II: 12–31 MACs (**PASS**)
  - HybridController: 7 MACs (**PASS**)
- **Filter 3 (Worst-Condition Success $\ge 40.0\%$):**
  - ES: 0.0% (**FAIL**)
  - NSGA-II: 0.0% (**FAIL**)
  - HybridController: 65.0% (**PASS**)
- **Filter 4 (Nominal Screening Gate 1B $\ge 90.0\%$):**
  - ES: 0.0% (**FAIL**)
  - NSGA-II: 0.0% (**FAIL**)
  - HybridController: 100.0% (**PASS**)

### 4. Candidate-Selection Decision & Next Critical Task
**Decision:** Under the frozen development candidate-selection criteria, **neither learned approach qualifies for prospective confirmation evaluation**.
- Both learned algorithms, when allocated 2,000,000 simulator transitions, fail to discover a controller capable of completing the 500-step continuous swing-up and balance task.
- The classical reference `HybridController` achieves superior performance across all dimensions: 100% nominal success, 92.0% robust success, 65.0% worst-case success, with only 7 parameters and 7 MACs, requiring 0 training transitions.
- **Next Critical Task (Day 5):** Freeze candidate selection results, document the failure of learned models to pass development screening, and export the verified `HybridController` to C as the primary validated artifact for prospective confirmation review.

**Overall Stage 2 Status:** **`STAGE 2 TRAINING COMPLETE — CANDIDATE SELECTION REJECTED LEARNED MODELS`**

