# EvoEdge — Phase 2A: Experimental Infrastructure Integration Report

**Date & Time:** 2026-10-12  
**Commit Baseline:** `a36e352` ("fix(protocol): clarify seed-exposure combinations, metric proxy, and friction hypothesis")  
**Branch:** `master`  
**Phase Status:** **PHASE 2A COMPLETE — ALL GATES PASSED**

---

## Executive Summary

Phase 2A authorizes configuration integration, protocol formalization, and automated verification only. No controller training, hyperparameter tuning, or evaluation on prospective final test conditions was performed.

All four primary experimental runner scripts have been migrated to the unified perturbation loader ([`utils/perturbation_loader.py`](file:///d:/neat/utils/perturbation_loader.py)). The experimental protocol has been formalized under **Objective B** (generalization beyond all previously explored physical ranges in every designated dimension), with all test benchmarks strictly exceeding historical exposure ceilings. The supplementary untouched test seed partition (`2000..2099`) is strictly guarded against evaluation during development.

---

## 1. Finalized Perturbation Protocol & Scientific Objective

### Objective Formal Declaration
- **Declared Objective:** **Objective B** — *Generalization beyond all previously explored physical ranges in every designated dimension.*
- **Protocol Version:** `v2.3` in [`config/perturbations.yaml`](file:///d:/neat/config/perturbations.yaml).
- **Justification:** Historical diagnostic exposures (documented in [`results/perturbation_exposure_ledger.csv`](file:///d:/neat/results/perturbation_exposure_ledger.csv)) evaluated cart friction up to $0.50\text{ N}\cdot\text{s/m}$, joint friction up to $0.010\text{ N}\cdot\text{m}\cdot\text{s/rad}$, sensor noise $\sigma$ up to $0.050$, and delay up to 2 steps. The previous test values ($0.45, 0.009, 0.045$) fell below these explored points. Under Objective B, all test benchmarks are established strictly above prior exposure ceilings.

### Perturbation Dimensions & Boundary Accounting

| Physical Dimension | Variable | Nominal | Training Range (`0..199`) | Validation Range (`200..399`) | Historical Exposed Max | Final Test Benchmark (`1000..1099` & `2000..2099`) | OOD Status vs Prior Exposure |
|---|---|---|---|---|---|---|---|
| Pole Mass | `MASS_POLE` | $0.10\text{ kg}$ | $[0.08, 0.12]\text{ kg}$ | $[0.075, 0.125]\text{ kg}$ | $0.130\text{ kg}$ | **$0.140\text{ kg}$** (+40%) | **Strictly OOD** ($0.140 > 0.130$) |
| Cart Mass | `MASS_CART` | $1.00\text{ kg}$ | $[0.85, 1.20]\text{ kg}$ | $[0.80, 1.25]\text{ kg}$ | $1.30\text{ kg}$ | **$1.40\text{ kg}$** (+40%) | **Strictly OOD** ($1.40 > 1.30$) |
| Cart Rail Friction | `b_cart` | $0.00\text{ N}\cdot\text{s/m}$ | $[0.00, 0.25]\text{ N}\cdot\text{s/m}$ | $[0.00, 0.35]\text{ N}\cdot\text{s/m}$ | $0.50\text{ N}\cdot\text{s/m}$ | **$0.60\text{ N}\cdot\text{s/m}$** | **Strictly OOD** ($0.60 > 0.50$) |
| Joint Friction | `b_pole` | $0.000\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | $[0.00, 0.005]\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | $[0.00, 0.0075]\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | $0.010\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | **$0.015\text{ N}\cdot\text{m}\cdot\text{s/rad}$** | **Strictly OOD** ($0.015 > 0.010$) |
| Observation Noise | `obs_noise_sigma` | $0.000$ | $[0.00, 0.020]$ | $[0.00, 0.035]$ | $0.050$ | **$0.065$** | **Strictly OOD** ($0.065 > 0.050$) |
| Actuator Delay | `delay_steps` | $0\text{ steps}$ ($0\text{ ms}$) | $[0]\text{ steps}$ | $[0]\text{ steps}$ | $2\text{ steps}$ ($40\text{ ms}$) | **$3\text{ steps}$** ($60\text{ ms}$) | **Strictly OOD** ($3 > 2$) |

### Seed Partition Rules
- **Training Seeds:** `0..199` (controller training and DAgger rollouts).
- **Validation Seeds:** `200..399` (hyperparameter tuning, model selection, early stopping).
- **Historical Test Seeds:** `1000..1099` (frozen historical baseline comparability; preserved unmodified).
- **Supplementary Untouched Final Test Seeds:** `2000..2099` (reserved exclusively for prospective post-training evaluation; execution strictly blocked during Phase 2A).

---

## 2. Configuration as Source of Truth & Runner Migration

All four targeted execution paths were migrated to the unified perturbation loader:

1. **[`experiments/benchmark_suite.py`](file:///d:/neat/experiments/benchmark_suite.py)**:
   - Upgraded `BenchmarkCartPoleEnv` to inherit from `PerturbedCartPoleEnv`.
   - Supports all 6 physical dimensions (`pole_mass`, `cart_mass`, `b_cart`, `b_pole`, `obs_noise_sigma`, `delay_steps`).
   - `run_benchmark_suite()` validates configuration via `load_perturbation_suite()`.
   - Enforces seed partition safety: raises `PerturbationConfigError` if supplementary final test seeds (`2000..2099`) are passed.
   - Logs `config_hash` and per-episode provenance (`seed`, `sampled_params`, `steps`, `reward`, `boundary_violated`).
   - Appends `config_hash` to CSV schema.

2. **[`experiments/runner.py`](file:///d:/neat/experiments/runner.py)**:
   - Integrated with `load_perturbation_suite()`.
   - Enforces that training runs strictly use seeds in the declared split (`training` `0..199`).
   - Rejects supplementary final test seeds (`2000..2099`).
   - Logs `config_hash`, `model_artifact`, and `sampled_physical_params` in result dict and output CSV.

3. **[`experiments/run_baselines_pruning.py`](file:///d:/neat/experiments/run_baselines_pruning.py)**:
   - Replaced bespoke `PerturbedEnv` implementation with `PerturbedCartPoleEnv`.
   - `evaluate_model_on_env()` validates split isolation and logs provenance.
   - Attaches `suite.config_hash` to evaluation results summary.

4. **[`evolution/evolve_tinyml.py`](file:///d:/neat/evolution/evolve_tinyml.py)**:
   - Replaced unperturbed gym environment with `PerturbedCartPoleEnv`.
   - `train_and_evaluate()` validates training seeds against `suite.splits['training']` and defaults evaluation to `suite.splits['validation']`.
   - Strictly blocks access to supplementary final test seeds (`2000..2099`).
   - Records `config_hash` from the perturbation suite.

### Unmigrated Code Paths Audit
- **Status:** **NONE** (0 unmigrated execution paths).
- Verified via `get_unmigrated_runners()` in `tests/test_perturbation_loader.py`.

---

## 3. Automated Verification Results

### Unit Test Execution
Ran full test suite containing 43 unit tests covering split integrity, baseline logic, ops accounting, perturbation loading, and runner migration:

```powershell
& "C:\Users\cosmic\AppData\Local\Programs\Python\Python312\python.exe" -m unittest test_baselines.py tests/test_split_integrity.py tests/test_ops_accounting.py tests/test_perturbation_loader.py tests/test_runner_migration.py
```

**Raw Output:**
```
...........................................
----------------------------------------------------------------------
Ran 43 tests in 2.311s

OK
```
**Exit Code:** `0`

### Specific Verification Items in `tests/test_runner_migration.py`:
1. **`test_runner_applies_yaml_values`:** PASS (verified physics parameters match YAML sampling).
2. **`test_sampled_distribution_bounds`:** PASS (30 seeds sampled across training and validation; all bounds strictly respected).
3. **`test_delay_steps_applied_correctly`:** PASS (2-step action transport queue verified: actions held for 2 steps before actuation).
4. **`test_noise_affects_specified_channels`:** PASS (observation noise verified only on target channels while preserving clean channels).
5. **`test_seed_and_condition_partitions_frozen`:** PASS (training 0..199, validation 200..399, historical test 1000..1099, supplementary final test 2000..2099 are pairwise disjoint; access to 2000..2099 raises `PerturbationConfigError`).
6. **`test_provenance_reconstruction`:** PASS (result dictionary and episode provenance contain `seed`, `sampled_params`, `config_hash`).
7. **`test_invalid_configurations_fail_safely`:** PASS (unsupported physics keys, out-of-split seeds, and malformed YAML raise clear exceptions).
8. **`test_historical_results_remain_unchanged`:** PASS (historical baseline and benchmark files remain intact and non-empty).

---

## 4. Validation Smoke Test Execution

A dedicated smoke test script ([`scratch/smoke_test_validation.py`](file:///d:/neat/scratch/smoke_test_validation.py)) was executed on the validation partition (`seeds 200..209`):

```powershell
& "C:\Users\cosmic\AppData\Local\Programs\Python\Python312\python.exe" scratch/smoke_test_validation.py
```

**Raw Output:**
```
================================================================================
  EvoEdge Phase 2A: Smoke Test on Validation Partition (Seeds 200..209)
================================================================================
  Configuration Loaded : D:\neat\config\perturbations.yaml
  Configuration Hash   : a36571f215fe771e
  Protocol Version     : 2.3
  Objective            : B
  Validation Seeds     : [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]

  Sampled Validation Parameters (Seed 200):
    - MASS_POLE         : 0.1073417071839569
    - MASS_CART         : 1.0987639871207358
    - b_cart            : 0.01046557835480567
    - b_pole            : 0.0013337350675318126
    - obs_noise_sigma   : 0.02007238272372779
    - delay_steps       : 0.0

  Evaluating Controllers...
    [OK] LQR             | Reward: -199.86 | Rail:   0.0% | Time: 0.060s
    [OK] Hybrid ES-LQR   | Reward:    0.35 | Rail:   0.0% | Time: 0.043s

  Smoke Test Completed Successfully in 0.14 seconds.
================================================================================
  Summary saved to D:\neat\results\smoke_test_validation_summary.json
```
**Exit Code:** `0`  
**Duration:** `0.14 seconds` (Budget limit: 20 minutes)  
**Configuration SHA-256:** `a36571f215fe771e`

---

## 5. Phase 2A Gate Status

| Gate | Requirement | Status | Evidence / Notes |
|---|---|---|---|
| **Gate 1** | Perturbation protocol audit & Objective B declaration | **PASS** | Formalized Objective B in `config/perturbations.yaml` v2.3; all final test dimensions exceed historical maximums; documented in `perturbation_exposure_ledger.csv`. |
| **Gate 2** | Configuration as single source of truth across runners | **PASS** | Migrated `benchmark_suite.py`, `runner.py`, `run_baselines_pruning.py`, and `evolve_tinyml.py` to `utils/perturbation_loader.py`. |
| **Gate 3** | Automated verification of physics, delay, noise, and bounds | **PASS** | 43/43 unit tests passed (`test_runner_migration.py` + existing test suites). |
| **Gate 4** | Safe failure on invalid/unsupported configuration | **PASS** | Unsupported parameters, malformed YAML, and split leaks raise `PerturbationConfigError` or `SplitIntegrityError`. |
| **Gate 5** | Provenance recording per episode (seed, params, hash) | **PASS** | Recorded in `BenchmarkCartPoleEnv`, `runner.py`, and `evaluate_controller` outputs. |
| **Gate 6** | Validation smoke test $\le 20$ minutes on val/train seeds | **PASS** | Completed in 0.14s on seeds `200..209`; output logged in `smoke_test_validation_summary.json`. |
| **Gate 7** | Supplementary untouched final test suite protection | **PASS** | Execution paths raise error if seeds `2000..2099` are passed; no final OOD run executed. |
| **Gate 8** | Historical results and prior metrics immutability | **PASS** | `baseline_comparison.csv`, `final_evaluation_report.md`, and `baselines_pruning_results.pkl` unmodified. |

---

## 6. Exact Files Changed

```
M  config/perturbations.yaml
M  evolution/evolve_tinyml.py
M  experiments/benchmark_suite.py
M  experiments/run_baselines_pruning.py
M  experiments/runner.py
M  results/perturbation_exposure_ledger.csv
M  tests/test_perturbation_loader.py
M  tests/test_split_integrity.py
M  utils/perturbation_loader.py
A  tests/test_runner_migration.py
A  results/smoke_test_validation_summary.json
A  results/phase_2a_report.md
```

---

## 7. Proposed Next Phase (Phase 2B) & Prerequisites

### Scope of Phase 2B (Robust Training & Candidate Selection)
With infrastructure and protocol audit complete and validated, the experimental framework is ready for Phase 2B:
1. **Previous-Action Recurrent State Augmentation ($a_{t-1}$):**
   - Provide $a_{t-1}$ to neural controller inputs ($5 \to 6$ inputs) to enable internal estimation and compensation for actuator transport delay ($b_{\text{delay}} > 0$).
2. **Domain-Randomized Training (Robustness Training):**
   - Train candidates using the declared training distribution (`seeds 0..199` with uniform randomization over mass, rail drag, joint friction, and sensor noise).
3. **Multi-Objective Architecture Search:**
   - Pareto-optimal candidate discovery balancing Task B swing-up reward, boundary collision avoidance, parameter count, and theoretical MAC operations.
4. **Validation Model Selection:**
   - Evaluate candidates strictly on validation split (`seeds 200..399`) under validation perturbation bounds to select the final Pareto front.
   - **Prerequisite:** Do NOT touch or evaluate the prospective final test suite (`seeds 2000..2099`) during Phase 2B training or model selection.
