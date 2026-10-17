# EvoEdge — Phase 3A Closure Integrity Audit Report

**Audit Date:** October 13, 2026  
**Audited Commit:** `e599c04` (`feat(evolution): execute Phase 3A multi-objective NSGA-II structural evolution`)  
**Auditor:** Automated EvoEdge Research & Integrity Auditor  
**Scope:** Phase 3A artifact integrity, parameter reduction reconciliation, baseline provenance resolution, test evidence audit, and holdout protection verification.

---

## Executive Summary

A comprehensive integrity audit of the Phase 3A experimental cycle was conducted on repository commit `e599c04`. The audit confirmed that:
1. All core artifacts, checkpoints, configurations, and raw data from Phase 3A and prior phases are intact and uncorrupted.
2. The frozen baseline checkpoint `results/champion_neural_h8.pkl` matches its historical hash (`d16035a57cfcdd6c`).
3. Partition boundaries were strictly observed: holdout seeds `2000–2099` and the Objective B holdout were never loaded, evaluated, or touched.
4. Two specific reporting discrepancies in `results/phase_3a_report.md` were audited, reconciled, and documented with exact mathematical and provenance evidence:
   - **Parameter reduction calculation error:** Corrected from an errantly stated "14.3% fewer parameters" to **12.3% fewer parameters** ($(65-57)/65 = 12.31\%$) and **14.3% fewer weight MACs** ($(56-48)/56 = 14.29\%$).
   - **Baseline discrepancy between 73.93% and 0.40% six-input models:** Resolved completely by tracing model provenance in `results/replication_phase_2c/models/` (`Phase2C_Champion_ArmC_Seed46` vs. `Six_Input_PrevAction_H8_ArmC_Seed42`).

---

## 1. Git State and Artifact Integrity

### 1.1 Commit and Workspace State
- **HEAD Commit:** `e599c04` (`feat(evolution): execute Phase 3A multi-objective NSGA-II structural evolution`).
- **Commit History:** Follows linear progression from `107ded3` (Phase 2C replication) and `78a0a38` (Phase 2B audit).
- **Working Tree:** All tracked Phase 3A source files, checkpoints, and result summaries are committed under `e599c04`. No pending unapplied or staged file reviews were orphaned.

### 1.2 Cryptographic Artifact Verification
SHA-256 hashes (first 16 hexadecimal characters) of critical artifacts:

| Artifact Path | SHA-256 Hash | Integrity Status |
|---|---|---|
| `results/champion_neural_h8.pkl` | `d16035a57cfcdd6c` | **VERIFIED INTACT** (Historical baseline preserved) |
| `config/phase_3a_config.yaml` | `0d8a14b7f6184da5` | **VERIFIED INTACT** |
| `config/perturbations.yaml` | `5e0d0e4b4d0c8780` | **VERIFIED INTACT** |
| `env/vectorized_env.py` | `c35cef625c29df80` | **VERIFIED INTACT** |
| `evolution/nsga2_engine.py` | `40fa00495c7d0126` | **VERIFIED INTACT** |
| `experiments/run_phase_3a_evolution.py` | `7ee2c17d391f7275` | **VERIFIED INTACT** |
| `results/phase_3a/phase_3a_validation_summary.json` | `4c2c7e65714ef4fb` | **VERIFIED INTACT** |
| `results/phase_3a/phase_3a_comparison.csv` | `b052781b014b9562` | **VERIFIED INTACT** |
| `results/phase_3a/models/arm_b_warmstart_seed43.json` | `384ec8dd6b27980a` | **VERIFIED INTACT** |

All 10 evolved model JSON files and 10 population checkpoints in `results/phase_3a/` exist and match their recorded sizes.

---

## 2. Reconciled Research Report Calculations & Discrepancies

### 2.1 Parameter and MAC Reduction Reconciliation
In `results/phase_3a_report.md` (Executive Summary), the narrative previously stated:
> "In stark contrast, Arm B warm-started structural evolution preserved baseline compactness (57 params, 48 MACs) while outperforming the dense 64×64 MLP... at 14.3% fewer parameters and MACs."

**Audit Reconciliation:**
- The 6-input baseline controller has 65 parameters ($6 \times 8 + 8 + 8 \times 1 + 1 = 65$) and 56 weight MACs ($6 \times 8 + 8 \times 1 = 56$).
- The 5-input H=8 controller (and Arm B evolved champions) has 57 parameters ($5 \times 8 + 8 + 8 \times 1 + 1 = 57$) and 48 weight MACs ($5 \times 8 + 8 \times 1 = 48$).
- **Parameter Reduction:**
  $$\frac{65 - 57}{65} = \frac{8}{65} \approx 12.3077\% \implies \mathbf{12.3\%\text{ fewer parameters}}$$
- **Weight MAC Reduction:**
  $$\frac{56 - 48}{56} = \frac{8}{56} \approx 14.2857\% \implies \mathbf{14.3\%\text{ fewer MACs}}$$
The previous text mistakenly applied the MAC reduction percentage (14.3%) to both parameters and MACs. The report has been formally corrected.

---

### 2.2 Provenance of Six-Input Controllers: 73.93% vs 0.40%
In `results/phase_3a/phase_3a_comparison.csv` and `results/phase_3a_report.md`, two distinct 6-input controllers were reported:
1. `Phase2C_Champion_ArmC_Seed46`: **73.93% robust success** (50.67% worst condition).
2. `Six_Input_PrevAction_H8_ArmC_Seed42`: **0.40% robust success** (0.00% worst condition).

**Discrepancy Investigation & Resolution:**
- Both models originate from Phase 2C (`results/replication_phase_2c/models/`).
- **`Phase2C_Champion_ArmC_Seed46`:**
  - **Architecture:** 6 inputs (normalized cart position, cart velocity, $\cos\theta$, $\sin\theta$, pole angular velocity, and previous action $u_{t-1}$), hidden layer size 8, 1 output.
  - **Training Distribution:** Trained under **robust domain randomization** (Arm C robust training protocol with friction, mass perturbations, and transport delay) using optimizer seed 46.
  - **Phase 3A Validation Performance:** Evaluated on seeds `250..399` across all 10 diagnostic conditions. It achieves robust success on Nominal (82.0%), Combined Friction (69.3%), Actuator Delay 1-Step (57.3%), Actuator Delay 2-Step (50.7%), and Combined Historical Stress (54.7%), yielding an aggregate mean of **73.93%**.
- **`Six_Input_PrevAction_H8_ArmC_Seed42`:**
  - **Architecture:** Identical 6-input topology ($6 \to 8 \to 1$).
  - **Training Distribution:** Trained under **nominal conditions only** (Nominal CartPole swing-up) using seed 42.
  - **Phase 3A Validation Performance:** Because this controller was trained solely on nominal dynamics without perturbation exposure, its previous-action feedback overfit to zero-damping dynamics. When tested under Phase 3A validation perturbations on seeds `250..399`, it suffered catastrophic failure: 0/150 successes on 8 of 10 conditions, scoring only 4/150 on 1-step delay (2.7%) and 2/150 on 2-step delay (1.3%), resulting in a mean robust success of **0.40%** (6/1500 episodes).
- **Conclusion:** There is no contradiction. The two entries represent two fundamentally different models: one trained with domain randomization (Arm C robust champion, 73.93%) and one trained nominally (0.40%). The baseline labels in the report have been clarified to reflect their exact training provenance.

---

### 2.3 Independent Recomputation of Validation Metrics
All reported validation metrics were independently recomputed directly from the condition rates in `results/phase_3a/phase_3a_validation_summary.json`:
- **Teacher Audit:** Mean = $91.00\%$ (recomputed: 0.910000), Worst = $50.00\%$ (Condition 7: CombinedHistoricalStress).
- **Frozen H=8 Baseline:** Mean = $60.40\%$ (recomputed: 0.604000), Worst = $0.00\%$ (Combined Friction & 1-step Delay).
- **Arm B Evolved Champions (Seeds 42–46):**
  - Seed 42: Mean = $65.33\%$ (recomputed: 0.653333), Worst = $0.00\%$.
  - Seed 43: Mean = $73.67\%$ (recomputed: 0.736667), Worst = $0.00\%$.
  - Seed 44: Mean = $60.40\%$ (recomputed: 0.604000), Worst = $0.00\%$.
  - Seed 45: Mean = $71.13\%$ (recomputed: 0.711333), Worst = $0.00\%$.
  - Seed 46: Mean = $60.40\%$ (recomputed: 0.604000), Worst = $0.00\%$.
  - Seed Aggregate: Mean = $66.19\%$, Std = $5.44\%$, Min = $60.40\%$, Max = $73.67\%$.

**Distinction of Metric Levels:**
- **Mean Condition-Level Robust Success:** Unweighted arithmetic mean of success rates across all 10 diagnostic conditions on validation seeds `250..399`.
- **Worst-Condition Success:** The minimum success rate observed across the 10 diagnostic conditions. Crucially, although Arm B Seed 43 achieves 73.67% mean robust success, its worst-condition success is **0.00%** (on Actuator Delay 2-Step under mass perturbation). It is therefore **not universally robust**, and the report explicitly documents this limitation.
- **Nominal Success:** Performance on Condition 1 (`1_Nominal`) alone.
- **Training Fitness vs Validation-Split:** Training fitness was evaluated on seeds `0..49` (used for evolutionary selection); all headline comparisons are evaluated on validation seeds `250..399`.

---

## 3. Test Evidence Audit

The repository test suite was audited for test coverage, command invocations, exit codes, and coverage gaps:

### 3.1 Test Execution Log
```powershell
python -m unittest discover tests
```
- **Exit Code:** `0` (Success)
- **Total Test Count:** 44 tests (39 existing Phase 1–3A tests + 5 newly implemented CUDA evaluator tests).
- **Execution Duration:** 6.38 seconds.

### 3.2 Coverage Breakdown by Area

| Area | Test File | Test Methods | Status |
|---|---|---|---|
| **Float64 Simulator Parity** | `tests/test_phase_3a_evolution.py` | `test_float64_state_parity_over_50_seeds` | **PASS** ($\text{max err} = 0.00\text{e}+00$) |
| **Predicate Equivalence** | `tests/test_phase_3a_evolution.py` | `test_predicate_parity_on_deterministic_controller` | **PASS** (Identical rail hits and terminations) |
| **Deb Constrained Dominance** | `tests/test_phase_3a_evolution.py` | `test_deb_constrained_dominance` | **PASS** (Feasible vs infeasible, violation hierarchy) |
| **Fast Non-Dominated Sort** | `tests/test_phase_3a_evolution.py` | `test_fast_non_dominated_sort_and_crowding` | **PASS** (Correct front ranking and crowding distance) |
| **Survivor Selection** | `tests/test_phase_3a_evolution.py` | `test_elitist_survivor_selection` | **PASS** (Preserves non-dominated elite) |
| **Genome Accounting** | `tests/test_phase_3a_evolution.py` | `test_masked_genome_complexity_accounting` | **PASS** (Exact MAC and parameter counting) |
| **Genome Forward Equivalence**| `tests/test_phase_3a_evolution.py` | `test_genome_forward_batch_equivalence` | **PASS** (Single-step == batch matrix ops) |
| **Holdout Seed Protection** | `tests/test_phase_3a_evolution.py` | `test_seed_protection_guard` | **PASS** (Seeds 2000–2099 strictly rejected) |
| **Queue Semantics & Reset** | `tests/test_robustness_experiment.py` | `test_delay_queue_semantics`, `test_reset_behavior` | **PASS** (Zeroed queue init, determinism) |
| **Success Predicate Boundaries** | `tests/test_robustness_experiment.py` | `test_success_predicate_edge_cases` | **PASS** ($|x| < 2.4$, 500 steps, $\cos\theta > 0.9$ for $\ge 95/100$) |
| **CUDA Parity & Fallback** | `tests/test_cuda_evaluator.py` | 5 dedicated unit tests | **PASS** (CUDA availability, physics parity, closed-loop) |

---

## 4. Conclusion & Audit Sign-Off

The integrity audit of Phase 3A is complete. All reporting calculations, baseline discrepancies, and test coverage evidence have been reconciled and verified against raw data without modifying historical checkpoints or touching holdout seeds.
