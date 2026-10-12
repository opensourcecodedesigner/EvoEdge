# Phase 1 Correction Log: Final Protocol Clarification

**Date:** October 12, 2026  
**Auditor:** Lead Research Engineer  
**Scope:** Phase 1 Final Protocol Clarification — Seed-Exposure Disambiguation, Metric Terminology Correction, and Friction Hypothesis Reclassification  
**Commit Baseline:** `3bfeb03`  

---

## 1. Summary of Protocol Corrections & Amendments

| Area / Task | What Changed | Rationale / Problem Corrected |
|---|---|---|
| **Protocol 1: Seed-Exposure Disambiguation** | Updated `results/perturbation_exposure_ledger.csv` to explicitly distinguish individual seed IDs from evaluated seed-condition combinations. Added a dedicated `evaluated_seed_condition_combinations` column. Declared a supplementary untouched final test seed range `2000..2099` in `config/perturbations.yaml` v2.2. Updated `utils/split_integrity.py` to enforce disjointness for this partition. | Resolves any potential ambiguity regarding whether seed IDs `1000..1099` were tainted by prior exposure under historical conditions. Establishes a pure untouched seed reserve (`2000..2099`) guaranteed never exposed to any model selection, training, or runner. |
| **Protocol 2: Metric Terminology Correction** | Updated `metrics/ops_counter.py`, `tests/test_ops_accounting.py`, and `results/phase_1_report.md`. Formally retained verified dense MAC counts (48 weight MACs / 57 params for $H=8$; 4,480 weight MACs / 4,609 params for $64\times 64$ MLP). Relabeled the composite metric `2 * MACs + bias additions + activation evaluations` as an **operation-count proxy**, explicitly stating it is NOT a standard FLOP count. | Standard deep learning and embedded computing conventions define MACs as multiply-accumulate operations and report Weight MACs and Parameter counts. Prevents confusing an ad-hoc proxy formula with standard FLOP metrics. |
| **Protocol 3: Causal Reclassification of Friction Mechanism** | Reclassified the energy starvation explanation for friction failure in `results/phase_1_report.md` from a proven causal mechanism to a **supported hypothesis**. Added exact mechanical energy balance ($\Delta E_{\text{mech}} = \int F dx - \int P_{\text{loss}} dt$) to the "Remaining NOT VERIFIED Items" list. | Preserves empirical observations ($\cos\theta \le 0.4135$, 0% rail collisions, 0% success) while maintaining scientific rigor, acknowledging that continuous numerical energy balance integration was not computed in Phase 1. |
| **Protocol 4: Runner & Test Suite Integrity** | Ran all 35 unit tests across the test suite. All tests pass with zero errors. Updated tests in `tests/test_ops_accounting.py` and `tests/test_perturbation_loader.py` to validate `operation_count_proxy` and YAML v2.2. | Ensures zero regression across split integrity, baselines, operation accounting, and perturbation loading interfaces. |

---

## 2. Quantitative Verification & Test Execution Log

All automated tests executed locally on Python 3.12:

```powershell
python -m unittest test_baselines.py tests/test_split_integrity.py tests/test_ops_accounting.py tests/test_perturbation_loader.py
```
```
...................................
----------------------------------------------------------------------
Ran 35 tests in 2.768s

OK
```

Total tests passing: **35 / 35 (100%)**.

---

## 3. Explicit Gate Status Summary

- **Seed-Exposure Disambiguation:** Complete (`results/perturbation_exposure_ledger.csv`, `config/perturbations.yaml` v2.2, supplementary untouched seeds `2000..2099` declared and reserved).
- **Metric Terminology:** Complete (`metrics/ops_counter.py`, `tests/test_ops_accounting.py`, labeled as operation-count proxy).
- **Causal Claims & Hypotheses:** Complete (friction reclassified as supported hypothesis; energy balance tracked as NOT VERIFIED).
- **Test Integrity:** Complete (35/35 passing unit tests).
- **Execution Boundary:** Frozen at Phase 1. No controller training, robust optimization, or final OOD suite evaluation performed.
