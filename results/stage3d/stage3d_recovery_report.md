# EvoEdge Phase 3D Recovery Report: Robustness-First Neuroevolution

**Document Version:** 1.0  
**Date:** 2026-10-17  
**Status:** COMPLETE — FROZEN DEVELOPMENT EVALUATION  
**Classification:** **SUCCESS — ROBUST LEARNED CONTROLLER FOUND**  

---

## 1. Executive Summary & Verdict

### Final Official Classification

$$\mathbf{SUCCESS\ —\ ROBUST\ LEARNED\ CONTROLLER\ FOUND}$$

At least one neural candidate passes all four frozen development acceptance criteria simultaneously. In fact, **four out of five optimizer seeds** (Seeds 43, 44, 45, and 46) satisfied every acceptance criterion, with **Seed 46** emerging as the Phase 3D champion.

### Central Research Question & Answer

> **Research Question:** Did robustness-first, domain-randomized neuroevolution produce a compact neural controller that handles the disturbance conditions which previously defeated Stage 3C?
>
> **Definitive Answer:** **YES.** In Stage 3C, despite achieving 99.0% nominal success and 69.5% aggregate robust success, all evolved candidates suffered from a **0.0% worst-condition success rate** (failing completely on Cart Mass +30% and Combined Friction). In Phase 3D, by re-architecting the selection mechanism to prioritize worst-condition performance (constrained dominance with robust-first lexicography) and incorporating targeted physical variations in training, the worst-condition bottleneck was eliminated:
> - **Worst-Condition Success:** **90.0%** (Screening acceptance floor $\ge 40.0\%$; Stage 3C achieved $0.0\%$; Classical baseline achieved $65.0\%$).
> - **Aggregate Robust Success:** **98.0%** across all 10 diagnostic conditions (Stage 3C achieved $69.5\%$; Classical baseline achieved $92.0\%$).
> - **Gate 1B Nominal Success:** **100.0%** (Floor $\ge 90.0\%$).
> - **Complexity:** **69 active parameters** (limit $\le 128$) and **60 weight MACs** (limit $\le 128$).

---

## 2. Root-Cause Diagnosis of Stage 3C Failures

Prior to altering optimization dynamics, offline diagnosis of the Stage 3C champion (`results/stage3c/models/stage3c_nsga2_delayaware_seed44.pkl`) was conducted across the 10 diagnostic conditions using validation seeds 300..319.

### Diagnostic Findings

| Diagnostic Condition | Stage 3C Success | Stage 3C Failure Rate | Primary Recorded Failure Mechanism |
| :--- | :---: | :---: | :--- |
| **3: Cart Mass +30%** | **5.0%** | **95.0%** | **Rail Boundary Violation ($x > 2.4$ / $x < -2.4$) during Upright Balance** |
| **4c: Combined Friction** | **0.0%** | **100.0%** | **Rail Boundary Violation ($x > 2.4$ / $x < -2.4$) during Upright Balance** |
| **4b: Joint Friction 0.010** | **70.0%** | **30.0%** | Rail boundary collision ($x$ drift) |
| **4a: Cart Friction 0.50** | **65.0%** | **35.0%** | Rail boundary collision ($x$ drift) |
| **5: Sensor Noise 0.05** | **60.0%** | **40.0%** | Rail boundary collision and upright loss |

### Physical Mechanism vs. Historical Misconceptions

1. **Not an Energy / Swing-up Deficit:**  
   Earlier reports hypothesized that failures in Cart Mass +30% and Combined Friction were due to "insufficient kinetic energy transfer during swing-up". Detailed inspection of episode telemetry disproved this: in **100% of failed episodes**, the pole successfully achieved vertical inverted alignment ($\cos\theta > 0.99$, $|\dot{\theta}| < 0.3$).
2. **Delayed-Action Braking Overshoot on Rail Limits:**  
   Under 30% increased cart mass ($M_c = 1.30\text{ kg}$) or combined friction ($b_c = 0.50\text{ N}\cdot\text{s/m}, b_p = 0.010\text{ N}\cdot\text{m}\cdot\text{s/rad}$), cart momentum and damping altered the mechanical impedance of the carriage. The Stage 3C controller swung up aggressively, catching the pole upright at $|x| \approx 1.2\text{ m}$ to $1.8\text{ m}$. However, under a 2-step actuator delay ($40\text{ ms}$ latency), the controller’s stabilizing corrections were delayed in phase. With higher inertia, the cart could not arrest its drift before colliding with the rail boundary ($|x| \ge 2.4\text{ m}$).
3. **Absence from Training Distribution:**  
   Stage 3C’s training distribution only varied pole mass ($m_p$), pole length ($l$), actuator delays, and light combined stress. **Pure Cart Mass +30% and pure Combined Friction were completely absent from Stage 3C training.**
4. **Deficiency in Objective Formulation:**  
   Stage 3C’s NSGA-II comparator used aggregate robust success $\bar{S}_{\text{rob}}$ as an objective, but omitted $S_{\text{worst}}$. Consequently, candidates scoring 85% average across 8 easy conditions while scoring 0% on Condition 3 and 4c dominated balanced candidates.

---

## 3. Robustness-First NSGA-II Formulation

To directly enforce robustness over the weakest conditions, Phase 3D modified the multi-objective optimization engine in `evolution/nsga2_engine.py`.

### Constrained Dominance & Lexicographic Ordering (`constrained_dominates_stage3d`)

Candidate $A$ dominates candidate $B$ ($A \prec B$) according to the following strict hierarchy:

1. **Feasibility / Viability Filter (Anti-Collapse Barrier):**  
   A candidate that achieves active control ($\text{robust\_success} > 0$) strictly dominates a passive or non-functional candidate ($\text{robust\_success} = 0$). This mathematically prevents "passive collapse" (where an inert network that leaves the pole hanging downward scores $+500$ points while an active swing-up controller that clips a rail boundary scores $-40,000$ points).
2. **Worst-Condition Success ($S_{\text{worst}}$):**  
   If $S_{\text{worst}}(A) > S_{\text{worst}}(B)$, $A$ dominates $B$.
3. **Aggregate Robust Success ($\bar{S}_{\text{rob}}$):**  
   If $S_{\text{worst}}$ is tied and $\bar{S}_{\text{rob}}(A) > \bar{S}_{\text{rob}}(B)$, $A$ dominates $B$.
4. **Dense Worst-Condition Progress ($R_{\text{worst}}$):**  
   If both candidates have zero binary success on their worst condition ($S_{\text{worst}} = 0$) and equal aggregate success, the candidate with higher shaped return on that failing condition $R_{\text{worst}}(A) > R_{\text{worst}}(B)$ dominates, driving continuous evolutionary progress.
5. **Pareto Dominance on Performance vs. Complexity:**  
   When functional performance is equivalent, $A$ dominates $B$ if $A$ has strictly fewer active parameters ($P(A) < P(B)$) or weight MACs ($M(A) < M(B)$).

### Deterministic Unit Test Verification

The selection rule was implemented and rigorously verified in `tests/test_stage3d_preflight.py` (`test_robustness_first_comparator_fixtures`):
- Candidate with higher worst-condition success strictly dominates a candidate with higher aggregate average but lower worst-condition success.
- Functional candidate strictly dominates a passive/collapsed candidate.
- Smaller network dominates equal-performance larger network.

---

## 4. Controlled Domain Randomization Distribution

Phase 3D established `STAGE3D_TRAIN_CONDITIONS` comprising 8 balanced training conditions covering every physical factor identified in the failure diagnosis:

```python
STAGE3D_TRAIN_CONDITIONS = {
    "1_Nominal":                  PerturbationConfig(delay_steps=0),
    "2_CartMass_+30%":            PerturbationConfig(cart_mass=1.30, delay_steps=0),
    "3_CartFriction_0.50":        PerturbationConfig(cart_friction=0.50, delay_steps=0),
    "4_JointFriction_0.010":      PerturbationConfig(joint_friction=0.010, delay_steps=0),
    "5_CombinedFriction":         PerturbationConfig(cart_friction=0.50, joint_friction=0.010, delay_steps=0),
    "6_SensorNoise_0.05":         PerturbationConfig(noise_std=0.05, delay_steps=0),
    "7_ActuatorDelay_2Step":      PerturbationConfig(delay_steps=2),
    "8_CombinedStress_MassDelay": PerturbationConfig(cart_mass=1.30, cart_friction=0.30, delay_steps=2),
}
```

- **Evaluation Partition:** Training seeds 0–199 only.
- **Episodes per Generation:** 8 conditions $\times$ 2 seeds = 16 episodes per genome evaluation (1000 steps per episode).

---

## 5. Development Acceptance Criteria (Frozen v2.3)

| Criterion | Requirement / Threshold | Evaluation Partition |
| :--- | :---: | :--- |
| **Complexity Gate 1** | $\text{Active Parameters} \le 128$ | Architectural inspection |
| **Complexity Gate 2** | $\text{Weight MACs} \le 128$ | Architectural inspection |
| **Gate 1B Nominal** | $\text{Nominal Success Rate} \ge 90.0\%$ | Seeds 200..299 (100 episodes, 1_Nominal) |
| **Robust Screening** | $\text{Worst-Condition Success Rate} \ge 40.0\%$ | Seeds 300..319 (20 seeds $\times$ 10 conditions) |

---

## 6. Phase 3D Campaign Results (Seeds 42–46)

The campaign evaluated five optimizer seeds (42, 43, 44, 45, 46) starting from the Stage 3C seed-44 warm-start checkpoint. All transition ceilings and partition safety rules were strictly enforced.

### Summary of Campaign Runs

| Method / Seed | Active Params | Weight MACs | Search Trans. | Valid. Trans. | Gate 1B Nominal | Aggregate Robust | Worst-Condition Success | All 4 Gates Passed? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Reference** | 0 | 0 | — | — | 100.0% | 92.0% | 65.0% | **YES** |
| **Stage 3B Champion (s44)** | 67 | 58 | 1,850,000 | 150,000 | 100.0% | 39.5% | 0.0% | NO |
| **Stage 3C Champion (s44)** | 69 | 60 | 1,850,000 | 150,000 | 99.0% | 69.5% | 0.0% | NO |
| **Stage 3D Seed 42** | 68 | 59 | 1,850,000 | 150,000 | 92.0% | 76.0% | 0.0% (Cond 3) | NO |
| **Stage 3D Seed 43** | 65 | 56 | 1,850,000 | 150,000 | 100.0% | 93.0% | 45.0% (Cond 6a) | **PASSED** |
| **Stage 3D Seed 44** | 65 | 56 | 1,850,000 | 150,000 | 100.0% | 93.5% | 60.0% (Cond 7) | **PASSED** |
| **Stage 3D Seed 45** | 69 | 60 | 1,850,000 | 150,000 | 100.0% | 88.5% | 40.0% (Cond 3) | **PASSED** |
| **Stage 3D Seed 46 (Champion)** | **69** | **60** | **1,850,000** | **150,000** | **100.0%** | **98.0%** | **90.0% (Cond 7)** | **PASSED** |

---

## 7. Condition-Wise Breakdown: Phase 3D Champion (Seed 46)

The table below details the performance of the Phase 3D Champion (`stage3d_nsga2_robust_seed46.pkl`) across all 10 diagnostic conditions evaluated on development seeds 300..319 ($N = 20$ per condition):

| Condition ID | Description | Denominator | Success Count | Success Rate | Recorded Failure Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1_Nominal** | Standard dynamics, zero delay | 20 | 20 | **100.0%** | None (All 20 balanced) |
| **2_PoleMass_+30%** | Pole mass $m_p = 0.13\text{ kg}$ | 20 | 20 | **100.0%** | None (All 20 balanced) |
| **3_CartMass_+30%** | Cart mass $M_c = 1.30\text{ kg}$ | 20 | 19 | **95.0%** | Rail boundary touch at $t=492$ (1 episode) |
| **4a_CartFriction_0.50** | Cart damping $b_c = 0.50\text{ N}\cdot\text{s/m}$ | 20 | 20 | **100.0%** | None (All 20 balanced) |
| **4b_JointFriction_0.010** | Joint damping $b_p = 0.010\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | 20 | 20 | **100.0%** | None (All 20 balanced) |
| **4c_CombinedFriction** | $b_c = 0.50$, $b_p = 0.010$ | 20 | 20 | **100.0%** | None (All 20 balanced) |
| **5_SensorNoise_0.05** | Gaussian noise $\sigma = 0.05$ | 20 | 19 | **95.0%** | Rail touch at $t=380$ (1 episode) |
| **6a_ActuatorDelay_1Step** | Actuator delay 1-step ($20\text{ ms}$) | 20 | 20 | **100.0%** | None (All 20 balanced) |
| **6b_ActuatorDelay_2Step** | Actuator delay 2-step ($40\text{ ms}$) | 20 | 20 | **100.0%** | None (All 20 balanced) |
| **7_CombinedHistoricalStress**| $M_c=1.2, m_p=0.12, \text{delay}=1, \sigma=0.02$ | 20 | 18 | **90.0%** | Rail drift under combined noise/delay (2 eps) |

### Milestone Comparisons

- **Cart Mass +30%:** Stage 3C: $5.0\%$ $\rightarrow$ **Phase 3D Champion: 95.0%**
- **Combined Friction:** Stage 3C: $0.0\%$ $\rightarrow$ **Phase 3D Champion: 100.0%**
- **Aggregate Robust Success:** Classical: $92.0\%$ | Stage 3C: $69.5\%$ $\rightarrow$ **Phase 3D Champion: 98.0%**
- **Worst-Condition Success:** Classical: $65.0\%$ | Stage 3C: $0.0\%$ $\rightarrow$ **Phase 3D Champion: 90.0%**

---

## 8. Budget Accounting & Partition Quarantine Declaration

### Transition Budget Verification

All runs complied strictly with the Transition Budget Tracker (`env/budget_tracker.py`):
- **Search Budget Allocation:** $1,850,000$ transitions.
- **Search Transitions Consumed:** Exactly $1,850,000$ transitions per run.
- **Validation Budget Allocation:** $150,000$ transitions ($100 \times 500$ nominal Gate 1B + $20 \times 10 \times 500$ diagnostics).
- **Validation Transitions Consumed:** Exactly $150,000$ transitions per run.
- **Total Transitions Consumed:** Exactly $2,000,000$ transitions (Ceiling $\le 2,000,000$).

### Partition Quarantine Declaration

The research protocol’s strict seed partitions were programmatically enforced via `assert_partition_safety` in `experiments/run_stage3d_recovery.py`:
- **Evolutionary Training:** Strictly restricted to training seeds **0–199**.
- **Development Validation:** Exposed seeds **200–299** (Gate 1B) and **300–319** (Diagnostic suite).
- **Quarantined Evaluation Partition:** Seeds **600–749** were untouched and **never accessed**.
- **Protected Confirmation Partition:** Seeds **2000–2099** were untouched and **never accessed**.

---

## 9. Deliverables and Reproducibility Registry

All Phase 3D artifacts have been deposited exclusively under `results/stage3d/`:

1. **Machine-Readable Summary:**  
   `results/stage3d/stage3d_summary.json` (Full metrics for all 5 seeds + classical reference).
2. **Visual Comparison Artifact:**  
   `results/stage3d/stage3d_comparison.png` (High-resolution 4-way bar chart across all 10 diagnostic conditions).
3. **Model Checkpoints and JSON Descriptions:**  
   - `results/stage3d/models/stage3d_nsga2_robust_seed42.pkl` & `.json`
   - `results/stage3d/models/stage3d_nsga2_robust_seed43.pkl` & `.json`
   - `results/stage3d/models/stage3d_nsga2_robust_seed44.pkl` & `.json`
   - `results/stage3d/models/stage3d_nsga2_robust_seed45.pkl` & `.json`
   - `results/stage3d/models/stage3d_nsga2_robust_seed46.pkl` & `.json` (Champion)
4. **Unit and Preflight Tests:**  
   - `tests/test_stage3d_preflight.py` (7 tests covering comparator, distribution, budget, checkpointing, and determinism).
   - Entire project test suite validated: **28/28 tests passed** (Exit code 0).

---

## 10. Conclusion

Phase 3D successfully solved the worst-condition robustness bottleneck that prevented previous neural candidates from passing the EvoEdge development acceptance criteria. By pairing delay-aware inputs with a robustness-first multi-objective ranking rule and controlled domain randomization, the resulting neural controller (`stage3d_nsga2_robust_seed46`) achieves **98.0% aggregate robust success** and **90.0% worst-condition success** while consuming only **69 parameters and 60 MACs**, comfortably beating both previous learned baselines and the classical analytical hybrid controller on development benchmarks.
