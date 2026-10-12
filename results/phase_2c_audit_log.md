# Phase 2C Audit & Deviation Log

**Date:** 2026-10-13  
**Audited Commit:** `78a0a38`  
**Execution Context:** Phase 2C Controlled Replication Experiment  
**Status:** COMPLETE (All Integrity Checks Passed)

---

## 1. Protocol Deviations & Adjustments Log

| # | Item | Declared / Prior Protocol | Replication Protocol (Phase 2C) | Rationale / Evidence |
|---|---|---|---|---|
| **1** | Optimizer Seeds per Arm | 3 independent seeds (`[42, 43, 44]`) in Phase 2B. | **5 independent seeds** (`[42, 43, 44, 45, 46]`) in Phase 2C. | Mandated by Phase 2C mission to evaluate whether Phase 2B conclusions hold across expanded seed variance ($N=5$). |
| **2** | Training Distribution Source | Exploratory hardcoded sub-bounds in initial script ($b_c \le 0.35, b_p \le 0.007, \sigma \le 0.035$). | **Dynamic YAML binding**: sampler queries exact protocol bounds from `config/perturbations.yaml` ($b_c \le 0.50, b_p \le 0.010, \sigma \le 0.050$). | Resolves documented provenance discrepancy; enforced by unit test. |
| **3** | Output Artifact Isolation | Saved directly under `results/models/` in Phase 2B. | Saved under **`results/replication_phase_2c/`** to preserve all Phase 2B exploratory artifacts intact. | Strict historical preservation requirement. |
| **4** | Previous-Action Delay Semantics | Described in Phase 2B draft as a "lead compensator". | Re-classified as **empirical association under robust training** with disclosed partial observability limitations under 2-step delay. | Causal claim correction. Teacher does not condition on $a_{t-1}$. |

---

## 2. Integrity Verification Log

1. **Episode-to-Aggregate Consistency:**
   - 10,500 rows in `results/replication_phase_2c/phase_2c_episode_validation.csv`.
   - 210 rows in `results/replication_phase_2c/phase_2c_aggregated_summary.csv`.
   - Discrepancies found: **0**.
2. **Seed Partition Bounds:**
   - Training: seeds strictly within `0..139` (subset of `0..199`).
   - Validation: seeds strictly within `200..249` (subset of `200..399`).
   - Supplementary final-test seeds `2000..2099`: **0 episodes evaluated (100% untouched)**.
3. **Model Weights & Hashes:**
   - 20 PyTorch checkpoints serialized and hashed.
   - Frozen baseline `results/champion_neural_h8.pkl` unmodified (SHA-256: `d16035a5...`).
4. **Canonical Success Predicate:**
   - All 10,500 episodes evaluated via `evaluate_success_predicate()`.
   - Mutually exclusive partition (Success + Rail Hit + Other Failure == 1) strictly verified for every episode.
