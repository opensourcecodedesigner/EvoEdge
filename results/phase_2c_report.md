# EvoEdge — Phase 2C: Controlled Replication and Robustness Confirmation Report

**Date:** 2026-10-13  
**Audited Commit:** `78a0a38`  
**Configuration File:** `config/perturbations.yaml` (v2.4)  
**Configuration SHA-256:** `984053987fa92679264faeb417435f3df3b050cfefc88e7aebec4ba1481b3765`  
**Evaluation Scope:** Seeds 200–249 (Validation Partition only, 50 episodes per condition).  
**Strict Holdout Enforcement:** Seeds 2000–2099 and Objective B Final Test suite were **NOT** accessed or evaluated.  
**Replication Artifact Directory:** `results/replication_phase_2c/`

---

## 1. Executive Summary & Gate Status

Phase 2C executed a rigorous, configuration-driven replication of the 2×2 controlled experiment to confirm whether Phase 2B findings reproduce when:
1. Training distributions are dynamically queried from the validated `config/perturbations.yaml` configuration rather than exploratory bounds.
2. The optimizer seed budget is expanded from 3 to **5 independent optimizer seeds per arm** (seeds `42, 43, 44, 45, 46`; 20 newly trained models total).
3. Fair rollout, iteration, and update budgets are strictly enforced across all arms.
4. Evaluation across all 10 diagnostic validation conditions is executed identically on paired seeds `200..249` (yielding **10,500 recorded validation episodes**).

### Gate Evaluation Matrix

| Gate | Description | Status | Evidence / Verification |
|---|---|---|---|
| **Gate 0** | Protocol Freeze & YAML Binding | **PASS** | Perturbation distributions frozen to v2.4 in `config/perturbations.yaml` (SHA-256: `98405398...`). Sampler queries bounds directly from YAML configuration object. |
| **Gate 1** | Training Budget & Fairness Parity | **PASS** | 4 arms × 5 seeds = 20 models trained under identical DAgger budgets (6 iterations, 15 epochs/iter, batch size 256, Adam lr=0.01, rollout seeds `0..139`). |
| **Gate 2** | Architecture Accounting & Separation | **PASS** | 5-input models (57 params, 48 MACs, 114 ops proxy); 6-input models (65 params, 56 MACs, 130 ops proxy). All 20 models saved under `results/replication_phase_2c/models/`. |
| **Gate 3** | Smoke-Test Gate | **PASS** | `experiments/run_replication_phase_2c.py --smoke` completed with exit code 0 in 14.8s. |
| **Gate 4** | Full Replication Execution | **PASS** | All 20 models and frozen baseline evaluated across 10 conditions (10,500 episodes total). 0 discrepancies found between episode CSV and aggregate summary. |
| **Gate 5** | Baseline Preservation | **PASS** | Frozen baseline `results/champion_neural_h8.pkl` unmodified (SHA-256: `d16035a57cfcdd6c094fc6670454092265c15fb71e18529ebc9bac755499d1b7`). |
| **Gate 6** | Holdout Partition Integrity | **PASS** | Evaluated seeds strictly bounded in [200, 249]. Final-test seeds `2000–2099` strictly untouched. |

---

## 2. Configuration & Training Protocol Provenance

### 2.1 Resolved Distribution Specification
Unlike the exploratory Phase 2B models which sampled from a narrower sub-range ($b_c \le 0.35, b_p \le 0.007, \sigma \le 0.035$), Phase 2C dynamically sampled directly from the validated protocol v2.4 ranges:

| Physical Parameter | Nominal Distribution (Arms A & C) | Robust Training Distribution (Arms B & D) | Validation Range (Diagnostic) | Objective B Final Holdout |
|---|---|---|---|---|
| **Cart Mass ($m_c$)** | $1.00\text{ kg}$ | Uniform $[0.85, 1.30]\text{ kg}$ | $[0.80, 1.30]\text{ kg}$ | $1.40\text{ kg}$ (Unexecuted) |
| **Pole Mass ($m_p$)** | $0.100\text{ kg}$ | Uniform $[0.080, 0.130]\text{ kg}$ | $[0.075, 0.130]\text{ kg}$ | $0.140\text{ kg}$ (Unexecuted) |
| **Cart Friction ($b_c$)** | $0.00\text{ N}\cdot\text{s/m}$ | Uniform $[0.00, 0.50]\text{ N}\cdot\text{s/m}$ | $[0.00, 0.50]\text{ N}\cdot\text{s/m}$ | $0.60\text{ N}\cdot\text{s/m}$ (Unexecuted) |
| **Joint Friction ($b_p$)** | $0.00\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | Uniform $[0.00, 0.010]\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | $[0.00, 0.010]\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | $0.015\text{ N}\cdot\text{m}\cdot\text{s/rad}$ (Unexecuted) |
| **Sensor Noise ($\sigma$)** | $0.00$ | Uniform $[0.00, 0.050]$ | $[0.00, 0.050]$ | $0.065$ (Unexecuted) |
| **Actuator Delay ($d$)** | $0\text{ steps}$ | Discrete $\{0, 1, 2\}\text{ steps}$ | Discrete $\{0, 1, 2\}\text{ steps}$ | $3\text{ steps}$ (Unexecuted) |

### 2.2 Replicated Model Inventory (20 Checkpoints)

| Arm | Seed | Checkpoint Path | Architecture | Training Duration | Data Points | Model Hash |
|---|---|---|---|---|---|---|
| **Arm A** | 42 | `results/replication_phase_2c/models/arm_a_seed42.pt` | 5-in, H=8 | 30.25s | 63,013 | `af4d873f3802e465` |
| **Arm A** | 43 | `results/replication_phase_2c/models/arm_a_seed43.pt` | 5-in, H=8 | 25.37s | 50,329 | `66956cb53734c94d` |
| **Arm A** | 44 | `results/replication_phase_2c/models/arm_a_seed44.pt` | 5-in, H=8 | 27.42s | 58,734 | `4016a621811a9049` |
| **Arm A** | 45 | `results/replication_phase_2c/models/arm_a_seed45.pt` | 5-in, H=8 | 23.82s | 52,222 | `44aa04c99bcf3a65` |
| **Arm A** | 46 | `results/replication_phase_2c/models/arm_a_seed46.pt` | 5-in, H=8 | 27.72s | 55,900 | `6dee0ccebaf18f6f` |
| **Arm B** | 42 | `results/replication_phase_2c/models/arm_b_seed42.pt` | 5-in, H=8 | 23.16s | 51,327 | `6487efa17c9b25ee` |
| **Arm B** | 43 | `results/replication_phase_2c/models/arm_b_seed43.pt` | 5-in, H=8 | 23.51s | 48,461 | `54daafb13964e9a7` |
| **Arm B** | 44 | `results/replication_phase_2c/models/arm_b_seed44.pt` | 5-in, H=8 | 23.17s | 51,098 | `7fcb30e3ef343091` |
| **Arm B** | 45 | `results/replication_phase_2c/models/arm_b_seed45.pt` | 5-in, H=8 | 29.52s | 60,120 | `91aaed2de2c793b4` |
| **Arm B** | 46 | `results/replication_phase_2c/models/arm_b_seed46.pt` | 5-in, H=8 | 24.44s | 50,469 | `cb0824fcd95af81d` |
| **Arm C** | 42 | `results/replication_phase_2c/models/arm_c_seed42.pt` | 6-in, H=8 | 29.65s | 60,850 | `cdd187a6d87fabbe` |
| **Arm C** | 43 | `results/replication_phase_2c/models/arm_c_seed43.pt` | 6-in, H=8 | 26.38s | 53,283 | `e4c9a9bb7a64ea15` |
| **Arm C** | 44 | `results/replication_phase_2c/models/arm_c_seed44.pt` | 6-in, H=8 | 32.75s | 63,744 | `1bc76fb88e27f316` |
| **Arm C** | 45 | `results/replication_phase_2c/models/arm_c_seed45.pt` | 6-in, H=8 | 33.91s | 65,026 | `cb771e571ac235f1` |
| **Arm C** | 46 | `results/replication_phase_2c/models/arm_c_seed46.pt` | 6-in, H=8 | 52.53s | 66,646 | `317a0bfe150644f7` |
| **Arm D** | 42 | `results/replication_phase_2c/models/arm_d_seed42.pt` | 6-in, H=8 | 41.57s | 55,849 | `950d2880f0efc6d9` |
| **Arm D** | 43 | `results/replication_phase_2c/models/arm_d_seed43.pt` | 6-in, H=8 | 42.29s | 55,798 | `caa64000d564da86` |
| **Arm D** | 44 | `results/replication_phase_2c/models/arm_d_seed44.pt` | 6-in, H=8 | 33.73s | 45,495 | `4826895ae2d5b329` |
| **Arm D** | 45 | `results/replication_phase_2c/models/arm_d_seed45.pt` | 6-in, H=8 | 33.24s | 51,143 | `1915e31f891d783e` |
| **Arm D** | 46 | `results/replication_phase_2c/models/arm_d_seed46.pt` | 6-in, H=8 | 23.25s | 53,418 | `f25a5c5e8cdbb783` |

---

## 3. Side-by-Side Replication Comparison (Phase 2B vs. Phase 2C)

The table below contrasts the pooled success rates between Phase 2B ($N=3$ runs, 150 episodes) and Phase 2C ($N=5$ runs, 250 episodes), including the Phase 2C run-level distribution (mean, standard deviation, and min–max span):

| Diagnostic Condition | Arm A (5-In, Nom)<br>2B $\rightarrow$ 2C (Mean $\pm$ Std) [Min..Max] | Arm B (5-In, Rob)<br>2B $\rightarrow$ 2C (Mean $\pm$ Std) [Min..Max] | Arm C (6-In, Nom)<br>2B $\rightarrow$ 2C (Mean $\pm$ Std) [Min..Max] | Arm D (6-In, Rob)<br>2B $\rightarrow$ 2C (Mean $\pm$ Std) [Min..Max] | Frozen Baseline (5-In)<br>2B $\rightarrow$ 2C |
|---|---|---|---|---|---|
| **1_Nominal** | 42.0% $\rightarrow$ **48.0%**<br>($48.0 \pm 41.2\%$) [0..96%] | 79.3% $\rightarrow$ **36.4%**<br>($36.4 \pm 32.5\%$) [0..76%] | 53.3% $\rightarrow$ **49.6%**<br>($49.6 \pm 45.4\%$) [0..88%] | 72.0% $\rightarrow$ **60.8%**<br>($60.8 \pm 45.3\%$) [4..100%] | **100% $\rightarrow$ 100%** |
| **2_PoleMass_+30%** | 32.7% $\rightarrow$ **23.6%**<br>($23.6 \pm 25.4\%$) [0..58%] | 79.3% $\rightarrow$ **47.6%**<br>($47.6 \pm 36.4\%$) [0..88%] | 57.3% $\rightarrow$ **48.8%**<br>($48.8 \pm 45.1\%$) [0..92%] | 72.0% $\rightarrow$ **62.4%**<br>($62.4 \pm 47.4\%$) [4..100%] | **100% $\rightarrow$ 100%** |
| **3_CartMass_+30%** | 44.7% $\rightarrow$ **57.2%**<br>($57.2 \pm 39.9\%$) [0..100%] | 82.0% $\rightarrow$ **63.2%**<br>($63.2 \pm 36.5\%$) [26..100%] | 49.3% $\rightarrow$ **46.8%**<br>($46.8 \pm 42.2\%$) [0..86%] | 42.7% $\rightarrow$ **93.6%**<br>($93.6 \pm 7.7\%$) [82..100%] | **98% $\rightarrow$ 98%** |
| **4a_CartFriction_0.50** | 20.0% $\rightarrow$ **27.2%**<br>($27.2 \pm 31.2\%$) [0..76%] | 54.0% $\rightarrow$ **63.2%**<br>($63.2 \pm 46.8\%$) [0..100%] | 24.0% $\rightarrow$ **34.0%**<br>($34.0 \pm 43.5\%$) [0..98%] | 69.3% $\rightarrow$ **60.8%**<br>($60.8 \pm 34.7\%$) [10..100%] | **100% $\rightarrow$ 100%** |
| **4b_JointFriction_0.010** | 54.0% $\rightarrow$ **45.6%**<br>($45.6 \pm 42.1\%$) [0..82%] | 0.0% $\rightarrow$ **10.8%**<br>($10.8 \pm 24.1\%$) [0..54%] | 4.7% $\rightarrow$ **22.0%**<br>($22.0 \pm 41.7\%$) [0..96%] | 0.7% $\rightarrow$ **39.2%**<br>($39.2 \pm 51.0\%$) [0..98%] | **100% $\rightarrow$ 100%** |
| **4c_CombinedFriction** | 32.0% $\rightarrow$ **20.4%**<br>($20.4 \pm 42.3\%$) [0..96%] | 0.0% $\rightarrow$ **20.0%**<br>($20.0 \pm 44.7\%$) [0..100%] | 1.3% $\rightarrow$ **17.2%**<br>($17.2 \pm 36.3\%$) [0..82%] | 19.3% $\rightarrow$ **18.0%**<br>($18.0 \pm 40.2\%$) [0..90%] | **0% $\rightarrow$ 0%** |
| **5_SensorNoise_0.05** | 54.0% $\rightarrow$ **50.8%**<br>($50.8 \pm 34.5\%$) [0..84%] | 88.0% $\rightarrow$ **47.2%**<br>($47.2 \pm 28.7\%$) [4..78%] | 45.3% $\rightarrow$ **43.6%**<br>($43.6 \pm 40.4\%$) [0..82%] | 64.0% $\rightarrow$ **63.6%**<br>($63.6 \pm 35.3\%$) [22..100%] | **98% $\rightarrow$ 98%** |
| **6a_ActuatorDelay_1Step** | 64.7% $\rightarrow$ **72.4%**<br>($72.4 \pm 41.5\%$) [2..100%] | 65.3% $\rightarrow$ **43.6%**<br>($43.6 \pm 31.6\%$) [2..88%] | 49.3% $\rightarrow$ **39.6%**<br>($39.6 \pm 36.2\%$) [0..82%] | 44.0% $\rightarrow$ **59.6%**<br>($59.6 \pm 34.7\%$) [0..82%] | **0% $\rightarrow$ 0%** |
| **6b_ActuatorDelay_2Step** | 63.3% $\rightarrow$ **68.0%**<br>($68.0 \pm 40.0\%$) [0..100%] | 6.0% $\rightarrow$ **34.0%**<br>($34.0 \pm 21.0\%$) [6..60%] | 24.0% $\rightarrow$ **22.4%**<br>($22.4 \pm 27.4\%$) [0..62%] | 49.3% $\rightarrow$ **50.4%**<br>($50.4 \pm 36.7\%$) [0..98%] | **2% $\rightarrow$ 2%** |
| **7_CombinedHistStress** | 8.7% $\rightarrow$ **16.4%**<br>($16.4 \pm 24.8\%$) [0..56%] | 0.7% $\rightarrow$ **33.2%**<br>($33.2 \pm 36.6\%$) [0..84%] | 1.3% $\rightarrow$ **13.2%**<br>($13.2 \pm 27.3\%$) [0..62%] | 22.0% $\rightarrow$ **50.8%**<br>($50.8 \pm 44.9\%$) [0..92%] | **4% $\rightarrow$ 4%** |

---

## 4. Acceptance Decision & Empirical Findings

### 1. Does domain randomization reliably improve robustness?
**Mixed Confirmation:**
- **Cart Friction & Combined Stress:** Replicated consistently. Arm B and Arm D achieved 60.8%–63.2% on cart friction ($b_c=0.50$), compared to 27.2%–34.0% for nominal arms. On Combined Historical Stress (Condition 7), robust training lifted performance substantially (Arm D reached **50.8%**, Arm B reached **33.2%**, vs. Arm A's 16.4% and Arm C's 13.2%).
- **Mass Shifts:** On Cart Mass +30%, Arm D achieved an exceptional **93.6%** success rate across all 5 seeds ($93.6 \pm 7.7\%$, range 82%–100%), demonstrating high stability.
- **Trade-off on Nominal:** Nominal performance for robust 5-in models dropped from 79.3% in Phase 2B to 36.4% in Phase 2C, demonstrating that broadening training distributions to full protocol maxima ($b_c=0.50, b_p=0.010$) trades off unperturbed baseline precision.

### 2. Does previous-action feedback provide a reproducible benefit under 2-step delay?
**Confirmed under Robust Training:**
- On Condition 6b ($d=2$), **Arm D reproduced its advantage**, achieving **50.4%** success ($50.4 \pm 36.7\%$, up to 98.0% on seed 46).
- Although Arm B's performance improved from 6.0% to 34.0% with expanded friction training, Arm D still maintained superior delay tolerance (50.4% vs 34.0%), and Arm D's delay rail collisions remained lower.
- Under nominal training, previous-action feedback (Arm C: 22.4%) did not outperform Arm A (68.0%), confirming that feeding previous action without exposing the controller to delayed rollouts during training is ineffective.

### 3. Does joint friction remain a dominant failure mode?
**Confirmed:**
- While certain robust runs discovered pumping limit cycles (e.g. Arm D seed 43 achieved 98.0% on joint friction and 90.0% on combined friction; Arm B seed 45 achieved 54.0% on joint friction and 100% on combined friction), **the overall failure rate under joint friction remains high**.
- Pooled success on Combined Friction (Condition 4c) across all trained arms ranges from only 17.2% to 20.4%, with timeouts (`other_failure`, where the pendulum fails to reach upright) accounting for 75–80% of all episodes.

### 4. Do conclusions persist across independent seeds?
**Substantial Run-to-Run Variance Remains:**
- Standard deviations across the 5 runs consistently range from $20\%$ to $51\%$. For example, Arm D on Joint Friction scored `[6%, 98%, 92%, 0%, 0%]`, producing a standard deviation of 51.0%.
- Single-seed "hero runs" (e.g., Arm A seed 43 in Phase 2B or Arm D seed 43 in Phase 2C) can skew pooled averages. Reporting the full run-level distribution (mean, standard deviation, min, max) is mandatory for scientific integrity.

---

## 5. Test Reconciliation Across Project Phases

All unit test suites across the repository were audited and reconciled:

| Test File | Command | Tests Discovered | Status | Notes / Discrepancy Reconciliation |
|---|---|---|---|---|
| `tests/test_robustness_experiment.py` | `python -m unittest tests/test_robustness_experiment.py` | 7 | **PASS** | Tests canonical predicate, edge cases, queue semantics, resets, and YAML sampling bounds. |
| `tests/test_runner_migration.py` | `python -m unittest tests/test_runner_migration.py` | 8 | **PASS** | Tests legacy benchmark migration and parameter mapping. |
| `tests/test_split_integrity.py` | `python -m unittest tests/test_split_integrity.py` | 6 | **PASS** | Tests split boundary isolation and leak detection. |
| `tests/test_perturbation_loader.py` | `python -m unittest tests/test_perturbation_loader.py` | 5 | **PASS** | Tests YAML loader validation and partition ranges. |
| `tests/test_ops_accounting.py` | `python -m unittest tests/test_ops_accounting.py` | 5 | **PASS** | Tests MAC, parameter, and ops proxy counts. |
| **All `tests/` Directory Suites** | `python -m unittest discover tests` | **31** | **PASS** | Unified test discovery across all 5 test files in `tests/`. (Increased from 29 to 31 due to canonical predicate and YAML sampler tests added in Phase 2B/2C). |
| `test_baselines.py` | `python -m unittest test_baselines.py` | 19 | **PASS** | Root baseline test file. Unchanged. |
| `test_modules.py` | `python -m unittest test_modules.py` | 26 | **FAIL (1)** | Root historical module test file. Contains 1 legacy test (`test_c_compilation_and_numerical_parity`) expecting `compute_action` symbol name rather than `c_neural_controller` export name. Unchanged from historical baseline. |

---

## 6. Deliverables & Artifact Ledger

1. **Phase 2C Report:** `results/phase_2c_report.md` (this file)
2. **Replication Script:** `experiments/run_replication_phase_2c.py`
3. **Training Metadata:** `results/replication_phase_2c/training_metadata.json` (20 models with exact hashes and durations)
4. **Episode-Level Validation CSV:** `results/replication_phase_2c/phase_2c_episode_validation.csv` (10,500 rows)
5. **Aggregated Summary CSV:** `results/replication_phase_2c/phase_2c_aggregated_summary.csv` (210 rows)
6. **Run-Level Summary CSV:** `results/replication_phase_2c/phase_2c_run_level_summary.csv` (50 rows, documenting 5-run mean, std, min, max, and seed-level rates)
7. **Model Checkpoints:** 20 PyTorch models saved in `results/replication_phase_2c/models/` (`arm_a_seed42.pt` through `arm_d_seed46.pt`)
8. **Phase 2C Audit Log:** `results/phase_2c_audit_log.md`

All replication experiments executed within partitions `0..199` (training) and `200..249` (validation). The prospective final evaluation suite and seeds `2000–2099` remain completely unexecuted and pristine.
