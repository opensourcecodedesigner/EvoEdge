# Phase 2B Audit Correction Log

**Date:** 2026-10-12  
**Audit Target:** Commit `2da1aef` (Phase 2B Experiment)  
**Status:** Audit Completed; All Discrepancies Resolved.

---

## 1. Discrepancy & Correction Ledger

| # | Component | Original Defect / Ambiguity | Corrective Action Taken | Status |
|---|---|---|---|---|
| **1** | Narrative Claims (`phase_2b_report.md`) | Section 5.2 and Section 6 incorrectly described Arm D as uniquely capable of solving Combined Friction (4c) and Combined Historical Stress (7). | Reconciled against raw data: Disclosed that Arm A seed 43 achieved 48/50 on 4c (pooled 48/150 for Arm A vs 29/150 for Arm D) and 13/50 on 7 (pooled 13/150 for Arm A vs 33/150 for Arm D). Replaced claim of uniqueness with run-to-run sensitivity analysis. | **RESOLVED** |
| **2** | Training Distribution Sampler (`run_robustness_experiment.py`) | `sample_training_condition()` hardcoded slightly narrower upper bounds ($b_c \le 0.35, b_p \le 0.007, \sigma \le 0.035$) than declared protocol v2.4 maxima ($0.50, 0.010, 0.050$). | Refactored sampler to accept `suite: Optional[PerturbationSuite]` and query bounds dynamically from `config/perturbations.yaml`. Added automated test verifying exact match against YAML. Classified original trained models as exploratory reconstructed models. | **RESOLVED** |
| **3** | Success Predicate Implementation (`run_robustness_experiment.py`) | Success logic was inlined in the evaluator loop rather than exposed as a shared production helper. | Extracted canonical `evaluate_success_predicate()` in `run_robustness_experiment.py`. Verified mutual exclusivity and exhaustiveness of outcomes (Success + Rail Hit + Other Failure == 1). | **RESOLVED** |
| **4** | Automated Unit Tests (`test_robustness_experiment.py`) | Tests did not verify exact 95 vs 94 cutoff, strict inequality at 0.900, short episodes, inference path $a_{t-1}$ feeding, or YAML sampler fidelity. | Expanded test suite from 5 to 7 comprehensive unit tests testing production code directly. All 7 tests passing in 0.25s. Ran full suite: all 31 tests in repository pass. | **RESOLVED** |
| **5** | Statistical Reporting | Only pooled success rates and per-episode Wilson intervals were presented, obscuring run-to-run training variance. | Created `results/phase_2b_run_level_summary.csv` reporting individual optimizer seed rates (seeds 42, 43, 44), run mean, run standard deviation, min, and max. Explicitly disclosed $N=3$ runs as exploratory. | **RESOLVED** |
| **6** | Previous-Action Mechanism Audit | Narrative described $a_{t-1}$ as a lead compensator without documenting weight sensitivity or partial observability limitations. | Conducted offline audit of all 6-input models: confirmed active weights and output sensitivity (0.06 to 0.87 modulation range). Documented that under 2-step delay, $a_{t-1}$ provides partial observability rather than full pipeline state. Clarified that teacher does not condition on $a_{t-1}$. | **RESOLVED** |

---

## 2. Artifacts Produced / Updated

1. **`results/phase_2b_audit_report.md`**: Comprehensive audit report detailing Gate statuses, data reconciliation, distribution provenance, hypothesis audit, and statistical interpretation.
2. **`results/phase_2b_run_level_summary.csv`**: Run-level summary table (50 rows) providing mean, std, min, and max across optimizer seeds.
3. **`results/audit_correction_log.md`**: This discrepancy resolution log.
4. **`experiments/run_robustness_experiment.py`**: Canonical `evaluate_success_predicate()` and YAML-driven `sample_training_condition()`.
5. **`tests/test_robustness_experiment.py`**: 7 strengthened unit tests covering production functions and edge cases.
6. **`results/phase_2b_report.md`**: Narrative text corrected to reconcile with raw episode data and disclose statistical limits.
