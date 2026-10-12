# Phase 1: Research Audit, Baseline Reproduction & Failure Diagnosis Report

**Repository:** `EvoEdge: Edge Neuro-Evolution Gym` (`d:/neat`)  
**Evaluation Target:** CartPole Swing-Up (`CartPoleSwingUpEnv`)  
**Auditor:** Lead Research Engineer  
**Date:** October 12, 2026 (Updated with Integrity Corrections)  
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
   ```powershell
   python -m unittest test_baselines.py
   ```
   - **Exit Code:** `0`
   - **Output:** `Ran 19 tests in 1.656s - OK`
   - **Validated:** Continuous LQR Riccati derivation ($K = [-1.0000, -1.2138, -7.8073, -1.8677]$), pumping direction, energy increase ($>0.3\text{ J}$), and ctypes compiled C closed-loop execution.
2. **Split Integrity Test Suite:**
   ```powershell
   python -m unittest tests/test_split_integrity.py
   ```
   - **Exit Code:** `0`
   - **Output:** `Ran 6 tests in 0.167s - OK`
   - **Validated:** Scheme verification, pairwise seed disjointness, OOD test extension, validation benchmark isolation, syntax safety, and AST/regex static code leakage inspection.
3. **Frozen Benchmark Reproduction (Seeds 1000..1099):**
   ```powershell
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

### Declared Experimental Protocol (`config/perturbations.yaml` v2.0)
- **Training Seeds:** `0–199` (200 seeds)
- **Validation Seeds:** `200–399` (200 seeds) — strictly isolated from test benchmark points:
  - Pole mass val range: $[0.075, 0.125]\text{ kg}$ (strictly excludes test benchmark $0.13\text{ kg}$)
  - Cart mass val range: $[0.80, 1.25]\text{ kg}$ (strictly excludes test benchmark $1.30\text{ kg}$)
  - Cart friction val range: $[0.00, 0.35]\text{ N}\cdot\text{s/m}$ (excludes test benchmark $0.50\text{ N}\cdot\text{s/m}$)
  - Joint friction val range: $[0.00, 0.0075]\text{ N}\cdot\text{m}\cdot\text{s/rad}$ (excludes test benchmark $0.010$)
  - Sensor noise val range: $[0.00, 0.035]$ (excludes test benchmark $\sigma = 0.05$)
  - Actuator delay val values: $[0]$ (excludes test benchmark delay steps $\{1, 2\}$)
- **Test Seeds:** `1000–1099` (100 seeds) — strictly held out for final reporting.
- **Automated Validation:** Verified via [`utils/split_integrity.py`](file:///d:/neat/utils/split_integrity.py) and [`tests/test_split_integrity.py`](file:///d:/neat/tests/test_split_integrity.py).

---

## 5. Friction and Actuator-Delay Failure Diagnosis

Diagnostic rollouts executed strictly on **validation seeds `200..299`** without exposing test seeds:

### Hypothesis A — High-Gain Balancing and Transport Delay Destabilization (SUPPORTED)
- **Empirical Evidence (100 Validation Episodes, Seeds 200..299):**
  - Success Rate: **0/100 (0.0%)**; Rail Collision Rate: **100/100 (100.0%)**.
  - Median crash step: **196.0 steps** (range: $[189, 215]$).
  - The controller reaches upright successfully ($\cos\theta > 0.99$ by step $\sim 140$).
- **Analytical Cause:**
  - A discrete transport delay of 1 control step ($\tau = 20\text{ ms}$) introduces a frequency-dependent phase lag $\Delta \phi = -\omega \tau$.
  - At linearized equilibrium around upright, the network exhibits high effective physical feedback gains ($K_\theta \approx 26.3$, $K_{\dot{\theta}} \approx 5.05$).
  - For rapid corrective oscillations at $\omega \approx 15\text{ rad/s}$, the phase lag erodes phase margin by $\sim 17^\circ$.
  - Combined with $\tanh(2.5 \cdot z)$ saturation, the closed-loop system undergoes a supercritical bifurcation into an aggressive limit cycle ($\Delta a \approx \pm 0.75$, $\Delta \dot{x} \approx 0.8\text{ m/s}$). The cart drifts across the track and collides with the boundary at step $\sim 196$.
- **Required Further Evidence for Phase 2:** Nyquist/Bode frequency response analysis of the linearized policy and closed-loop phase-lead compensation experiments.

### Hypothesis B — Energy Starvation Under Dissipative Friction (SUPPORTED)
- **Empirical Evidence (100 Validation Episodes, Seeds 200..299):**
  - Success Rate: **0/100 (0.0%)**; Rail Collision Rate: **0/100 (0.0%)**.
  - Mean maximum $\cos\theta$: **0.368** (range: $[0.322, 0.413]$); pendulum never rises above horizontal.
- **Analytical Cause:**
  - The policy was optimized on zero-friction rollouts where mechanical energy $E$ is strictly conserved outside control inputs.
  - In the presence of rail friction $b_c = 0.5\text{ N}\cdot\text{s/m}$ and joint friction $b_p = 0.01\text{ N}\cdot\text{m}\cdot\text{s/rad}$, energy dissipation rate is $P_{\text{loss}} = b_c \dot{x}^2 + b_p \dot{\theta}^2$.
  - Because pumping action amplitudes decay as cart velocity reverses, the net energy injected per half-cycle is insufficient to reach the capture threshold ($E \approx 0\text{ J}$), trapping the system in a sub-critical limit cycle.
- **Required Further Evidence for Phase 2:** Parameterized friction sweeps during training to identify the critical dissipation threshold where pumping stalls.

### Hypothesis C — Normalization & Clipping Mismatches (REJECTED AS PRIMARY ROOT CAUSE)
- Pre-activations remain well within floating-point ranges ($|z| \le 4.5$); normalization constants correctly bound features into $[-1, 1]$. Saturation is a consequence of delay limit cycles rather than static clipping errors.

---

## 6. Python vs. C Consistency & Hardware Claims

1. **Python vs C Action Discrepancy:**
   - Evaluated across 10,000 real simulation transitions.
   - Max absolute difference: $1.43 \times 10^{-5}$; mean absolute difference: $2.27 \times 10^{-7}$.
   - Discrepancy is fully accounted for by 32-bit single-precision vs. 64-bit double-precision intermediate evaluation of `tanhf`.
2. **Closed-Loop Parity:**
   - Compiled C controller evaluated in closed loop via `ctypes` over seeds 1000..1099 achieves **100/100 (100.0%) success**, identical to the Python model.
3. **Hardware Latency Claims:**
   - All documented figures ($\sim 0.85\,\mu\text{s}$, 120–600 cycles) are **THEORETICAL ESTIMATES**. Zero hardware-in-the-loop measurements exist.

---

## 7. Gate Status Summary

| Phase 1 Gate Requirement | Status | Justification & Traceable Evidence |
|---|---|---|
| Complete Phase 1 audit before modifying code | **PASS** | Audit complete; documented in handoff, Phase 1, and correction reports. |
| Invariant success criterion enforcement | **PASS** | `no rail hit AND cos(theta) > 0.9 for >= 95 of final 100 steps` verified across all active scripts. |
| Disjoint data split definition (0..199 / 200..399 / 1000..1099) | **PASS** | Defined in `config/perturbations.yaml` v2.0; verified pairwise disjoint and validation-isolated via `tests/test_split_integrity.py`. |
| Baseline reproduction | **PASS** | 100% nominal swing-up verified for teacher, H=8 net, and 64x64 MLP. |
| Friction failure diagnosis | **PASS** | 100-episode validation sweep (`200..299`) confirms energy starvation (mean max $\cos\theta = 0.368$, 0% rail hits). |
| Actuator-delay failure diagnosis | **PASS** | 100-episode validation sweep (`200..299`) confirms delay-induced limit cycles (100% rail hits, median crash step 196.0). |
| Static split-integrity validator | **PASS** | Implemented in `utils/split_integrity.py` with 6 passing unit tests in `tests/test_split_integrity.py`. |
| Automated consumption of `perturbations.yaml` by runners | **NOT VERIFIED** | Active benchmark runners use hardcoded dictionaries; dynamic consumption pipeline reserved for Phase 2. |
| Physical hardware ESP32 latency measurement | **NOT VERIFIED** | No physical ESP32 connected; numbers are labeled as theoretical estimates. |
| Strict STOP at Phase 1 | **PASS** | No Phase 2 training, tuning, or optimization initiated. |

---

## 8. Next-Phase Proposal (For Future Phase 2 Authorization)

When authorized, execute the following controlled experiments in Phase 2:
1. **Friction Adaptation:** Train with domain randomization over $b_c \in [0.0, 0.35]$ and $b_p \in [0.0, 0.0075]$ on training seeds `0..199`.
2. **Delay Compensation:** Augment the observation vector with previous applied action $a_{t-1}$ (6 inputs total) and train with randomized 0-to-1 step delay to evaluate whether phase-lead state information stabilizes the upright balance loop.
3. **Dynamic Config Pipeline:** Refactor benchmark runners to dynamically load and evaluate distributions from `config/perturbations.yaml`.
