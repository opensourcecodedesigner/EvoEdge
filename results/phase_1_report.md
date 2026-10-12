# Phase 1: Research Audit, Baseline Reproduction & Failure Diagnosis Report

**Repository:** `EvoEdge: Edge Neuro-Evolution Gym` (`d:/neat`)  
**Evaluation Target:** CartPole Swing-Up (`CartPoleSwingUpEnv`)  
**Auditor:** Lead Research Engineer  
**Date:** October 12, 2026  
**Status:** Complete — Authorized for Phase 1 Only  

---

## 1. Executive Verdict

- **Verified Ground Truth:** 
  - The teacher controller (`EnergyShapingLinearFeedbackController`, $k_E=0.6, k_x=0.5, k_v=0.25$, $c_{\text{kin}}=0.016667$) and the compact $H=8$ TinyML controller (57 parameters, 57 MACs) achieve **100/100 (100.0%) nominal swing-up success** under the invariant criterion (`no rail hit AND cos(theta) > 0.9 for >= 95 of final 100 steps`).
  - Closed-loop compiled C controller execution matches the Python implementation identically with **100/100 success**; empirical action difference over 10,000 real rollout steps is bounded by $\max |\Delta u| = 1.43 \times 10^{-5}$.
- **Verified Failures:**
  - The $H=8$ controller suffers **0/100 (0.0%) success under joint/cart friction** ($b_c=0.5, b_p=0.01$) and **0/100 (0.0%) success under 1-step actuator delay** ($20\text{ ms}$).
  - The 64x64 MLP baseline (4,609 parameters) also suffers **0/100 (0.0%) success under friction** and degrades to **14/100 success under 2-step delay**.
- **Historical Gaps & Unsupported Claims Identified:**
  - `~0.85 us on ESP32` and all Xtensa cycle bounds (~120–180, ~450–600) were **theoretical heuristic estimates**; zero measurements were performed on real silicon.
  - The 5-H-1 neural models are **fixed-topology feedforward networks** with weights optimized via closed-loop DAgger imitation; unconstrained `neat-python` topology evolution was never completed.
- **Go / No-Go Decision:** **GO FOR CONTROLLED PHASE 2 HYPOTHESIS TESTING** (subject to user authorization).

---

## 2. Repository & Evidence Audit

| Audit Item | Documented Claim | Verified Reality | Evidence & Verification Path |
|---|---|---|---|
| **Active Simulation** | `CartPoleSwingUpEnv` | 5-dim observation, semi-implicit Euler ($\tau=0.02\text{ s}$), rail $|x| \le 2.4\text{ m}$ | [`env/cartpole_swingup.py`](file:///d:/neat/env/cartpole_swingup.py) |
| **Active Architecture** | Compact TinyML Net | $5 \to 8(\text{tanh}) \to 1(\text{tanh})$, 57 weights/biases, 57 MACs | [`results/champion_neural_h8.pkl`](file:///d:/neat/results/champion_neural_h8.pkl) |
| **Optimization Method** | Genetic / NEAT | Closed-loop DAgger student-teacher imitation with Adam on fixed topology | [`evolution/evolve_tinyml.py`](file:///d:/neat/evolution/evolve_tinyml.py#L98-L158) |
| **Evolves Topology?** | Evolved topology | **Weights optimized, topology fixed** (5-3-1, 5-6-1, 5-8-1) | Header in [`results/neural_controller.h`](file:///d:/neat/results/neural_controller.h#L5) |
| **Seed Partitioning** | Train vs Eval | Train seeds `0..199`, Eval seeds `1000..1099` strictly disjoint | Verified in [`utils/split_integrity.py`](file:///d:/neat/utils/split_integrity.py) |
| **Seed Leakage** | Historically present | Historical GA in `evolve_tinyml.py` used seeds 1000..1019; resolved in commit `b4df49e` | Commit `b4df49e` diff |
| **Python vs C Parity** | Parity to 8 decimals | $\max |\Delta u| = 1.43 \times 10^{-5}$ across 10,000 rollout observations | [`results/handoff_report.md`](file:///d:/neat/results/handoff_report.md#L818-L838) |
| **Embedded Latency** | ~0.85 $\mu$s on ESP32 | **Theoretical estimate only**; zero physical hardware measurements | [`results/handoff_report.md`](file:///d:/neat/results/handoff_report.md#L841-L856) |

---

## 3. Baseline Reproduction

Reproduction commands executed on Python 3.12 (Host: AMD Ryzen 7 7435HS, Windows 11):

1. **Unit Test Suite:**
   ```bash
   python -m unittest test_baselines.py
   ```
   - **Exit Code:** `0`
   - **Output:** `Ran 19 tests in 1.656s - OK`
   - **Validated:** Continuous LQR Riccati derivation ($K = [-1.0000, -1.2138, -7.8073, -1.8677]$), pumping direction, energy increase ($>0.3\text{ J}$), and ctypes compiled C closed-loop execution.
2. **Frozen Benchmark Reproduction (Seeds 1000..1099):**
   ```bash
   python experiments/benchmark_suite.py --episodes 100 --max-steps 500
   ```
   - **Exit Code:** `0`
   - **Wall-Clock Time:** 82.4 seconds
   - **Nominal Task B Results:**
     - Classical LQR: **0/100 (0.0%) success**, 0% rail hits
     - Energy-Shaping + Linear Feedback: **100/100 (100.0%) success**, 0% rail hits, median steps = 88.0
     - TinyML Neural ($H=8$): **100/100 (100.0%) success**, 0% rail hits, median steps = 87.0
     - Dense 64x64 MLP: **100/100 (100.0%) success**, 0% rail hits, median steps = 86.0

---

## 4. Success-Definition & Data-Split Audit

### Success-Definition Compliance
- **Fixed Criterion:** No rail hit ($|x| \le 2.4\text{ m}$) AND $\cos(\theta) > 0.9$ for $\ge 95$ of the final 100 steps of a 500-step episode.
- **Median Steps to Upright:** Formally defined as the median step index $t \in [1, 500]$ at which $\cos(\theta) > 0.9$ is *first reached* exclusively among episodes achieving full success. Failed episodes are excluded; 0% success returns `None / Never Reached`.

### Declared Experimental Protocol (`config/perturbations.yaml`)
- **Training Seeds:** `0–199` (200 seeds)
- **Validation Seeds:** `200–399` (200 seeds) — used for diagnostic runs and tuning
- **Test Seeds:** `1000–1099` (100 seeds) — strictly held out for final reporting
- **Automated Check:** Verified disjointness and out-of-distribution physical range extension via [`utils/split_integrity.py`](file:///d:/neat/utils/split_integrity.py).

---

## 5. Friction and Actuator-Delay Failure Diagnosis

Diagnostic rollouts executed strictly on **validation seeds `200..204`** without exposing test seeds:

### Hypothesis A — Saturation and Delay-Induced Limit Cycles (CONFIRMED)
- **Observation:** Under 1-step actuator delay ($20\text{ ms}$), the $H=8$ controller reaches upright successfully ($\cos\theta \approx 0.996$ by step 140), but enters a high-frequency limit cycle around upright equilibrium.
- **Mechanism:** The policy uses $\tanh(2.5 \cdot z)$ with large feedback gains. Because feedback is delayed by $20\text{ ms}$, corrective actions arrive with $\pi/2$ phase lag, exciting a limit cycle of amplitude $\Delta \dot{x} \approx 0.8\text{ m/s}$ and $\Delta a \approx \pm 0.75$.
- **Outcome:** The cart drifts unidirectionally while oscillating until hitting the rail at step 191 ($|x| > 2.4\text{ m}$). **100% of delay failures are rail collisions caused by delay-induced phase lag.**

### Hypothesis B — Energy Starvation Under Friction (CONFIRMED)
- **Observation:** Under cart and joint friction ($b_c=0.5\text{ N}\cdot\text{s/m}, b_p=0.01\text{ N}\cdot\text{m}\cdot\text{s/rad}$), the $H=8$ controller achieves 0% rail collisions, but maximum achieved $\cos\theta$ is only $0.32$ to $0.40$ (pendulum fails to swing above horizontal).
- **Mechanism:** The controller was trained exclusively under zero-friction dynamics where momentum conservation permits symmetric energy accumulation. Nonzero viscous damping continuously drains energy ($P_{\text{loss}} = b_c \dot{x}^2 + b_p \dot{\theta}^2$). The learned pumping actions are insufficient to overcome this dissipative drag, trapping the pendulum in a sub-critical limit cycle.

### Hypothesis C — Normalization & Clipping Mismatches (REJECTED AS PRIMARY CAUSE)
- Pre-activations remain bounded ($|z| \le 4.5$); normalization constants correctly cover operating ranges. Action saturation occurs dynamically as a consequence of delay oscillation rather than static clipping errors.

---

## 6. Python vs. C Consistency & Hardware Claims

1. **Python vs C Action Discrepancy:**
   - Evaluated across 10,000 real simulation transitions.
   - Max absolute difference: $1.43 \times 10^{-5}$; mean absolute difference: $2.27 \times 10^{-7}$.
   - Discrepancy is fully accounted for by 32-bit single-precision vs. 64-bit double-precision intermediate evaluation of `tanhf`.
2. **Closed-Loop Parity:**
   - Compiled C controller evaluated in closed loop via `ctypes` over seeds 1000..1099 achieves **100/100 (100.0%) success**, identical to the Python model.
3. **Hardware Latency Claims:**
   - All documented figures ($\sim 0.85\,\mu\text{s}$, 120–600 cycles) are **THEORETICAL ESTIMATES**. No hardware-in-the-loop measurements exist.

---

## 7. Gate Status Summary

| Phase 1 Requirement | Status | Evidence / Notes |
|---|---|---|
| Complete Phase 1 audit before modifying code | **PASS** | Audit complete; documented in handoff and phase 1 reports |
| Fixed success criterion enforcement | **PASS** | Enforced across all evaluation scripts and tests |
| Disjoint data split definition (0-199 / 200-399 / 1000-1099) | **PASS** | Implemented in `config/perturbations.yaml` and verified by `utils/split_integrity.py` |
| Baseline reproduction | **PASS** | 100% nominal swing-up verified for teacher, H=8 net, and 64x64 MLP |
| Friction failure diagnosis | **PASS** | Diagnosed as energy starvation; pendulum reaches max $\cos\theta \approx 0.35$ |
| Delay failure diagnosis | **PASS** | Diagnosed as limit-cycle phase lag drift causing rail exit at step ~191 |
| Removal of unsupported hardware claims | **PASS** | Labeled as theoretical estimates across code and reports |
| Strict STOP at Phase 1 | **PASS** | No Phase 2 training or optimization initiated |

---

## 8. Next-Phase Proposal (For Future Phase 2 Authorization)

To resolve the two diagnosed failure modes in Phase 2 without expanding model complexity:
1. **Friction Robustness:** Train with domain randomization over $b_c \in [0.0, 0.4]$ and $b_p \in [0.0, 0.008]$ on training seeds `0..199`.
2. **Actuator Delay Robustness:** Augment the observation vector with the previous applied action $a_{t-1}$ (6 inputs total) and train with randomized 0-to-1 step delay to provide phase-lead state information.
