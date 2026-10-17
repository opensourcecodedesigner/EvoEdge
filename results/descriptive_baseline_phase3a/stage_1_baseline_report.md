# EvoEdge — Stage 1 Classical Baseline Descriptive Evaluation Report

**Document Status:** Empirical Experimental Record (Descriptive Validation Comparison Only)  
**Date:** 2026-10-13 14:50:36 UTC  
**Git Commit:** `e599c04299bcd859423cafacd52473c9ac3f8616`  
**Controller:** `energy-shaping + linear feedback` (SHA256: `d13fbbee79fcc98fde74b8dc0ee01a862a5f8366e328d5115474640c8f864e2c`)  
**Evaluation Partition:** Seeds `250–399` inclusive (150 seeds per condition)  
**Total Episodes Completed:** 1500 / 1,500 planned (100.0%)  
**Total Transitions Stepped:** 750,000 transitions  

## 1. Executive Summary & Protocol Status

> [!IMPORTANT]
> **Strict Scientific Protocol Notice:** This experiment establishes a descriptive classical-reference score on an already-utilized validation partition (seeds `250–399`). It does **NOT** constitute fresh confirmatory evidence, does not satisfy `GATE-02B1` or `GATE-02B2`, and must not be presented as proof that robust control has been solved.

### Headline Metrics
- **Aggregate Robust Success Rate (Equally Weighted Across 10 Conditions):** **`89.67%`** (1,345 / 1,500 episodes)
- **Aggregate Uncertainty via Seed-Cluster Bootstrap (10,000 resamples over 150 seeds, preserving all 10 condition outcomes within each resampled seed):**
  - Standard Error: `0.74%`
  - Two-Sided 95% Bootstrap Confidence Interval: **`[88.20%, 91.07%]`**
  - One-Sided 95% Lower Confidence Bound: **`88.47%`**
  - *Methodological Distinction:* This single-model uncertainty is derived via **seed-cluster bootstrap**. The term **paired cluster bootstrap** is strictly reserved for subsequent candidate-versus-reference comparative tests where paired differences are computed on identical seeds.
- **Worst-Condition Performance:** **`56.67%`** on condition `7_CombinedHistoricalStress` (95% Wilson CI: [48.67%, 64.33%])
- **Execution Duration:** `3.22 s` on NumPy CPU vectorized evaluator
- **Parity Evidence Status:** **Sampled Outcome Agreement** (verified on 5-seed-per-condition sample). Does *not* constitute production-path bit-for-bit trajectory equivalence across states, applied actions, and delay queues.

## 2. Per-Condition Performance Matrix

| Condition ID & Name | Episodes | Successes | Success Rate | 95% Wilson CI | Rail Violations | Other Failures |
|---|---|---|---|---|---|---|
| **`1_Nominal`** | 150 | 150 | **100.00%** | [97.50%, 100.00%] | 0 | 0 |
| **`2_PoleMass_+30%`** | 150 | 150 | **100.00%** | [97.50%, 100.00%] | 0 | 0 |
| **`3_CartMass_+30%`** | 150 | 150 | **100.00%** | [97.50%, 100.00%] | 0 | 0 |
| **`4a_CartFriction_0.50`** | 150 | 150 | **100.00%** | [97.50%, 100.00%] | 0 | 0 |
| **`4b_JointFriction_0.010`** | 150 | 98 | **65.33%** | [57.42%, 72.48%] | 51 | 1 |
| **`4c_CombinedFriction`** | 150 | 117 | **78.00%** | [70.72%, 83.88%] | 19 | 14 |
| **`5_SensorNoise_0.05`** | 150 | 146 | **97.33%** | [93.34%, 98.96%] | 0 | 4 |
| **`6a_ActuatorDelay_1Step`** | 150 | 149 | **99.33%** | [96.32%, 99.88%] | 1 | 0 |
| **`6b_ActuatorDelay_2Step`** | 150 | 150 | **100.00%** | [97.50%, 100.00%] | 0 | 0 |
| **`7_CombinedHistoricalStress`** | 150 | 85 | **56.67%** | [48.67%, 64.33%] | 24 | 41 |

## 3. Comparative Analysis with Historical Phase 3A Results

This table compares the newly measured classical reference against the historical frozen baseline and evolved neuroevolutionary candidates evaluated on the **exact same validation seed partition (250–399)**:

| Controller Architecture | Provenance | Aggregate Robust Success | Worst-Condition Success | Tunable Parameters | Computational Complexity (MACs / Cost) |
|---|---|---|---|---|---|
| **Energy-Shaping + Linear Feedback (`HybridController`)** | Classical Control | **`89.67%`** | **`56.67%`** (`7_Combined`) | 7 tunable gains | **`NOT ESTABLISHED`** (7 linear-gain MACs + trigonometric & nonlinear energy logic) |
| **Frozen Compact Baseline (`H=8`)** | Teacher-Assisted | 60.40% | 0.00% (`4c_Combined` & `6a_Delay`) | 57 params | 48 weight MACs |
| **Phase 3A Peak Warm-Started NEAT (Seed 43)** | Teacher-Assisted | 73.67% | **0.00%** (`4b_JointFriction`) | 57 params | 48 weight MACs |
| **Phase 3A Arm B Mean Across 5 Seeds** | Teacher-Assisted | 66.19% ± 5.44% | 0.00% (all 5 seeds suffer $\ge 1$ 0% collapse) | 57 params | 48 weight MACs |
| **Phase 3A From-Scratch NEAT (Arm A)** | From-Scratch | 0.00% | 0.00% (all 10 conditions 0%) | — | — |

### Reconciled Integrity Findings
1. **Resolution of Seed 43 Discrepancy:**
   - Authoritative historical raw data in `results/phase_3a/phase_3a_validation_summary.json` and `results/phase_3a/phase_3a_comparison.csv` record Seed 43 at **`73.67%` aggregate robust success** and **`0.00%` worst-condition success** (failing completely with 0/150 on `4b_JointFriction_0.010`).
   - The erroneous `44.00%` figure in preliminary summary text was a transcription error. Seed 43 is **not** universally robust; it suffers catastrophic collapse on joint friction.
2. **Classical Controller Complexity Accounting:**
   - The classical controller possesses **7 tunable gains** (4 linear balance gains $[K_x, K_{\dot{x}}, K_\theta, K_{\dot{\theta}}]$ and 3 pumping gains $[k_E, k_x, k_v]$).
   - In the balancing regime, it performs 4 MACs. In the pumping regime, it computes nonlinear mechanical energy $E = \frac{2}{3} m L^2 \dot{\theta}^2 + m g L (\cos\theta - 1)$, state extraction with `math.atan2`, sign checks, and 3 MACs.
   - Because operation accounting between hybrid branching logic and feedforward neural tensor contraction cannot be directly equated by linear multiply-accumulate count alone, the comparative computational cost is classified as **`NOT ESTABLISHED`**.
3. **Parity Evidence Boundary:**
   - Pre-execution smoke tests established **sampled outcome agreement** across 5 seeds per condition.
   - Production-path trajectory equivalence remains governed by unit test `tests/test_phase_3a_evolution.py` (which verified float64 state difference $= 0.00\text{e}+00$ on 50 seeds over 500 steps).

## 4. File Hashes & Reproducibility Ledger

- `raw_episodes.csv` SHA256: `2b39a9e476ec60c64da633b2c5eb019d1b405f466785490b175c943272986b36`
- `condition_summary.csv` SHA256: `ed7d2e493f2c9863b9bc678b3c4a4cb8f0ea29238f45403f2cf27e31936129aa`
- `aggregate_summary.json` SHA256: `5c6e1a3198d6bea8acaaaad9a1862e5cbad62c7bf83418bab96e8d8247a80f72`
