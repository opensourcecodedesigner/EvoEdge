# Phase 1: Research Audit, Baseline Reproduction & Failure Diagnosis Report

**Repository:** `EvoEdge: Edge Neuro-Evolution Gym` (`d:/neat`)  
**Evaluation Target:** CartPole Swing-Up (`CartPoleSwingUpEnv`)  
**Auditor:** Lead Research Engineer  
**Date:** October 12, 2026 (Updated with Final Phase 1 Closure Gate)  
**Status:** Complete — Authorized for Phase 1 Only  

---

## 1. Executive Verdict

- **Verified Ground Truth:** 
  - The teacher controller (`EnergyShapingLinearFeedbackController`, $k_E=0.6, k_x=0.5, k_v=0.25$, $c_{\text{kin}}=0.016667$) and the compact $H=8$ TinyML controller (57 parameters, 48 weight MACs) achieve **100/100 (100.0%) nominal swing-up success** under the invariant criterion (`no rail hit AND cos(theta) > 0.9 for >= 95 of final 100 steps`).
  - Closed-loop compiled C controller execution matches the Python implementation identically with **100/100 success**; empirical action difference over 10,000 real rollout steps is bounded by $\max |\Delta u| = 1.43 \times 10^{-5}$.
- **Verified Failures & Evidence Grounding:**
  - The $H=8$ controller suffers **0/100 (0.0%) success under joint/cart friction** ($b_c=0.5, b_p=0.01$) and **0/100 (0.0%) success under 1-step actuator delay** ($20\text{ ms}$).
  - The 64x64 MLP baseline (4,609 parameters) also suffers **0/100 (0.0%) success under friction** and degrades to **14/100 success under 2-step delay**.
  - All empirical failure claims are grounded in saved episode-level logs ([`results/diagnostics/validation_delay_1step_100ep.csv`](file:///d:/neat/results/diagnostics/validation_delay_1step_100ep.csv) and [`results/diagnostics/validation_friction_100ep.csv`](file:///d:/neat/results/diagnostics/validation_friction_100ep.csv)).
- **Historical Gaps & Unsupported Claims Resolved:**
  - `~0.85 us on ESP32` and all Xtensa cycle bounds (~120–180, ~450–600) were theoretical heuristic estimates; zero measurements were performed on real silicon.
  - Operation accounting has been audited against conventional dense implementation standards: $H=8$ has **48 weight MACs**, 9 bias additions, 9 activations, and 57 total parameters; Dense 64x64 MLP has **4,480 weight MACs**, 129 bias additions, 129 activations, and 4,609 total parameters.
  - Prior perturbation exposures are formally cataloged in [`results/perturbation_exposure_ledger.csv`](file:///d:/neat/results/perturbation_exposure_ledger.csv), and an unexposed final test suite is established in [`config/perturbations.yaml`](file:///d:/neat/config/perturbations.yaml) v2.1.
- **Go / No-Go Decision:** **PHASE 1 CLOSURE GATE COMPLETE — AWAITING AUTHORIZATION FOR PHASE 2**.

---

## 2. Repository & Computational Operation Accounting

### Standard Mathematical Operation Accounting (Conventional Dense Standard)

In standard embedded and deep learning hardware literature, a multiply-accumulate operation (MAC) computes $w \cdot x$ accumulated into a register. Dense linear layer operation accounting separates weight multiplications, bias additions, activation evaluations, and parameter storage:

| Model / Architecture | Layer Dimensions | Weights (Weight MACs) | Biases (Bias Additions) | Activation Evaluations | Total Trainable Parameters | Total Arithmetic FLOPs ($2 \cdot \text{MACs} + B + \text{Act}$) |
|---|---|---|---|---|---|---|
| **TinyML Neural ($H=3$)** | $5 \to 3 \to 1$ | **18** | 4 | 4 | **22** | **44** |
| **TinyML Neural ($H=6$)** | $5 \to 6 \to 1$ | **36** | 7 | 7 | **43** | **86** |
| **TinyML Neural ($H=8$)** | $5 \to 8 \to 1$ | **48** | 9 | 9 | **57** | **114** |
| **Dense MLP ($64\times 64$)** | $5 \to 64 \to 64 \to 1$ | **4,480** | 129 | 129 | **4,609** | **9,218** |

*Verification:* Executable tests in [`tests/test_ops_accounting.py`](file:///d:/neat/tests/test_ops_accounting.py) inspect the saved weight tensors directly and verify these figures to 100% precision.

---

## 3. Prior Perturbation Exposure Audit & Seed Isolation

A rigorous audit of all historical reports, stress-test logs, and diagnostic evaluations was conducted to establish true out-of-distribution (OOD) isolation.

### Exposure Summary ([`results/perturbation_exposure_ledger.csv`](file:///d:/neat/results/perturbation_exposure_ledger.csv))
- **Historically Exposed Test Benchmarks (Ineligible for Unseen Final Evaluation):**
  - Cart mass: $1.30\text{ kg}$ ($+30\%$)
  - Pole mass: $0.130\text{ kg}$ ($+30\%$)
  - Friction: $b_c = 0.50\text{ N}\cdot\text{s/m}, b_p = 0.010\text{ N}\cdot\text{m}\cdot\text{s/rad}$
  - Sensor noise: $\sigma = 0.05$
  - Actuator delay: 1-step ($20\text{ ms}$) and 2-step ($40\text{ ms}$)
  - Seeds: `42`, `200..299` (validation diagnostics), and `1000..1099` (handoff benchmark)
- **Declared Genuinely Unexposed Final Benchmark Suite (`config/perturbations.yaml` v2.1):**
  - **Unexposed Cart Mass:** $1.35\text{ kg}$ ($+35\%$) — strictly outside train $[0.85, 1.20]$ and val $[0.80, 1.25]$
  - **Unexposed Pole Mass:** $0.135\text{ kg}$ ($+35\%$) — strictly outside train $[0.08, 0.12]$ and val $[0.075, 0.125]$
  - **Unexposed Friction:** $b_c = 0.45\text{ N}\cdot\text{s/m}, b_p = 0.009\text{ N}\cdot\text{m}\cdot\text{s/rad}$ — strictly outside train and val bounds
  - **Unexposed Sensor Noise:** $\sigma = 0.045$ — strictly outside train $[0.0, 0.02]$ and val $[0.0, 0.035]$
  - **Unexposed Actuator Delay:** 3-step ($60\text{ ms}$) — strictly unseen discrete latency
  - **Status:** **NOT RUN** in Phase 1; reserved exclusively for post-Phase-2 final model verification.

---

## 4. Baseline Reproduction

Reproduction commands executed on Python 3.12 (AMD Ryzen 7 7435HS, Windows 11):

1. **Unit Test Suite:** `python -m unittest test_baselines.py`  
   - Output: `Ran 19 tests - OK`
2. **Split Integrity Test Suite:** `python -m unittest tests/test_split_integrity.py`  
   - Output: `Ran 6 tests - OK`
3. **Operation Accounting Test Suite:** `python -m unittest tests/test_ops_accounting.py`  
   - Output: `Ran 5 tests - OK`
4. **Perturbation Configuration Loader Suite:** `python -m unittest tests/test_perturbation_loader.py`  
   - Output: `Ran 5 tests - OK`
5. **Frozen Benchmark Reproduction (Seeds 1000..1099):** `python experiments/benchmark_suite.py --episodes 100 --max-steps 500`  
   - Output: Nominal Task B Success: LQR 0%, Hybrid Teacher 100%, TinyML $H=8$ 100%, Dense MLP64 100%.

---

## 5. Audit of Causal Claims in Failure Diagnosis

To maintain strict scientific integrity, claims regarding friction and actuator delay have been audited against actual trajectory evidence, separating measured observations from theoretical interpretations:

### Diagnosis A — Actuator Transport Delay
- **Measured Observations:**
  - 100 validation episodes on seeds `200..299` ([`results/diagnostics/validation_delay_1step_100ep.csv`](file:///d:/neat/results/diagnostics/validation_delay_1step_100ep.csv)) show **0/100 (0.0%) success** and **100/100 (100.0%) rail collisions**, with a median collision step of $196.0$ (range: $[189, 215]$).
  - In step-level trajectory data ([`results/diagnostics/trajectory_delay_seed200.csv`](file:///d:/neat/results/diagnostics/trajectory_delay_seed200.csv)), the controller successfully brings the pole upright ($\cos\theta > 0.995$ by step $140$).
  - Once upright, cart velocity and control actions oscillate with growing amplitude until rail impact ($|x| > 2.4\text{ m}$) at step $191$.
- **Supported Causal Interpretation:**
  - A 1-step discrete transport delay ($\tau = 20\text{ ms}$) introduces negative phase shift in the feedback loop. In the upright balancing regime, high-gain linear feedback without delay compensation produces growing closed-loop oscillations and track drift.
- **Removed / Qualified Unverified Claims:**
  - Claims of an exact $\pi/2$ phase shift have been removed (phase lag is frequency-dependent: $\Delta \phi = -\omega \tau$).
  - Claims of formal mathematical bifurcation or precise numerical stability margins have been qualified as speculative hypotheses, as no formal Nyquist/Bode or Lyapunov analysis has been conducted.

### Diagnosis B — Joint and Rail Friction
- **Measured Observations:**
  - 100 validation episodes on seeds `200..299` ([`results/diagnostics/validation_friction_100ep.csv`](file:///d:/neat/results/diagnostics/validation_friction_100ep.csv)) show **0/100 (0.0%) success** and **0/100 (0.0%) rail collisions**.
  - Across all 100 episodes, $\cos\theta$ has a mean maximum of $0.368$ (range: $[0.3219, 0.4135]$); the pole never rises above horizontal.
  - In step-level trajectory data ([`results/diagnostics/trajectory_friction_seed200.csv`](file:///d:/neat/results/diagnostics/trajectory_friction_seed200.csv)), the cart oscillates within $|x| \le 1.205\text{ m}$ for all 500 steps.
- **Supported Causal Interpretation (Energy Starvation Hypothesis):**
  - Dissipative forces ($P_{\text{loss}} = b_c \dot{x}^2 + b_p \dot{\theta}^2$) continuously drain kinetic energy. Because the controller was trained exclusively under zero-friction conditions, its fixed control amplitude cannot inject energy faster than friction dissipates it, trapping the system in sub-horizontal limit-cycle oscillations.
- **Removed / Qualified Unverified Claims:**
  - Claims stating exact critical damping thresholds or bifurcation points are qualified as requiring systematic parametric friction sweeps in Phase 2.

---

## 6. Perturbation Configuration Interface & Runner Audit

### Unified Configuration Interface ([`utils/perturbation_loader.py`](file:///d:/neat/utils/perturbation_loader.py))
- Provides `load_perturbation_suite()`, `sample_parameters()`, and `apply_to_env()`.
- Validates split bounds, checks seed containment, and ensures sampled parameters are recorded into rollout metadata.
- Validated via automated unit tests in [`tests/test_perturbation_loader.py`](file:///d:/neat/tests/test_perturbation_loader.py).

### Audit of Benchmark Runners Consuming `config/perturbations.yaml`
- **Current Status:** Benchmark runners have **NOT YET BEEN MIGRATED** to consume `config/perturbations.yaml` dynamically.
- **Unmigrated Runner Registry:**
  - `experiments/benchmark_suite.py`: **UNMIGRATED** (uses hardcoded dictionaries for cart mass 1.3 kg and noise 0.05).
  - `experiments/runner.py`: **UNMIGRATED** (does not consume YAML).
  - `experiments/run_baselines_pruning.py`: **UNMIGRATED** (uses hardcoded parameter loops).
  - `evolution/evolve_tinyml.py`: **UNMIGRATED** (uses default nominal environment).
- **Migration Plan:** Full migration of runners to dynamic configuration loading is scheduled for Phase 2.

---

## 7. Remaining NOT VERIFIED Items

The following items are explicitly tracked as **NOT VERIFIED** as of the Phase 1 closure gate:

1. **Hardware Microcontroller Latency on Physical Silicon:**
   - *Status:* **NOT VERIFIED**.
   - *Reason:* No physical ESP32 / Xtensa LX6 hardware bench is connected to the execution environment. All figures ($\sim 0.85\,\mu\text{s}$, 120–600 cycles) remain ungrounded theoretical estimates.
2. **Automated Dynamic YAML Consumption by Benchmark Runners:**
   - *Status:* **NOT VERIFIED**.
   - *Reason:* While the unified loader ([`utils/perturbation_loader.py`](file:///d:/neat/utils/perturbation_loader.py)) has been built and tested, the active execution scripts (`experiments/benchmark_suite.py`, `experiments/runner.py`) currently retain hardcoded parameter paths.
3. **Formal Nyquist / Bode Stability Margins for Delay Policy:**
   - *Status:* **NOT VERIFIED**.
   - *Reason:* Analytical phase margin erosion ($\sim 17^\circ$) is derived from linearized approximations; rigorous frequency response modeling of the nonlinear tanh policy has not been performed.
4. **Generalization Performance on Unexposed Final Benchmark Suite:**
   - *Status:* **NOT VERIFIED (INTENTIONALLY RESERVED)**.
   - *Reason:* Conditions in the unexposed final test suite (pole mass 0.135, cart mass 1.35, friction 0.45/0.009, noise 0.045, delay 3) have not been evaluated, preserving strict held-out integrity.

---

## 8. Gate Status Summary

| Phase 1 Gate Requirement | Status | Justification & Traceable Evidence |
|---|---|---|
| Complete Phase 1 audit before modifying code | **PASS** | Audit complete; documented in handoff, Phase 1, and correction reports. |
| Invariant success criterion enforcement | **PASS** | `no rail hit AND cos(theta) > 0.9 for >= 95 of final 100 steps` verified across all active scripts. |
| Disjoint data split definition (0..199 / 200..399 / 1000..1099) | **PASS** | Defined in `config/perturbations.yaml` v2.1; verified pairwise disjoint and validation-isolated via `tests/test_split_integrity.py`. |
| Audit of prior perturbation exposure | **PASS** | Created `results/perturbation_exposure_ledger.csv`; cataloged all historical and diagnostic conditions. |
| Establishment of unexposed final test suite | **PASS** | Defined in `config/perturbations.yaml` v2.1; isolated from all prior training, validation, and diagnosis. |
| Correct operation and parameter accounting | **PASS** | Conventional dense MAC accounting implemented in `metrics/ops_counter.py` and verified by `tests/test_ops_accounting.py`. |
| Grounding and qualification of causal claims | **PASS** | Grounded in saved CSV trajectories (`trajectory_delay_seed200.csv`, `trajectory_friction_seed200.csv`); unverified claims qualified. |
| Unified perturbation loading and validation interface | **PASS** | Implemented in `utils/perturbation_loader.py` with 5 passing tests in `tests/test_perturbation_loader.py`. |
| Automated consumption of `perturbations.yaml` by runners | **NOT VERIFIED** | Registry created; runners identified as UNMIGRATED. |
| Physical hardware ESP32 latency measurement | **NOT VERIFIED** | Explicitly classified as theoretical estimates. |
| Strict STOP at Phase 1 | **PASS** | No controller optimization or Phase 2 training initiated. |
