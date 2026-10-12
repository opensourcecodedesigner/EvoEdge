# Phase 1 Correction Log: Evidence Correction and Integrity Gate

**Date:** October 12, 2026  
**Auditor:** Lead Research Engineer  
**Scope:** Phase 1 Evidence Audit, Split Repair, Validator Upgrade, and Gate Review  
**Commit Baseline:** `19439c9`  

---

## 1. Summary of Changes & Rationale

| Area / Task | What Changed | Rationale / Problem Corrected |
|---|---|---|
| **Task 1: Split Validator** | Completely rewrote `utils/split_integrity.py` with schema verification, variable/unit enforcement, OOD extension, validation isolation, and AST/regex static code leakage inspection. Created test suite `tests/test_split_integrity.py`. | Original validator only checked basic range numbers without verifying YAML syntax safety, physical units, or validation isolation, and falsely claimed code leakage verification without inspecting code. |
| **Task 2: Split Design** | Updated `config/perturbations.yaml` to v2.0: tightened validation bounds for pole mass to $[0.075, 0.125]\text{ kg}$ (excluding $0.13\text{ kg}$) and cart mass to $[0.80, 1.25]\text{ kg}$ (excluding $1.30\text{ kg}$). Excluded $0.50\text{ N}\cdot\text{s/m}$, $0.010\text{ N}\cdot\text{m}\cdot\text{s/rad}$, and $\sigma=0.05$ from validation bounds. | Original v1.0 validation ranges contained the exact final benchmark test points (e.g. 0.13 kg and 1.30 kg), violating split isolation and risking benchmark-overfitting during model selection/tuning. |
| **Task 3: Configuration Consumption** | Audited active evaluation scripts. Identified that `experiments/benchmark_suite.py` and `experiments/run_baselines_pruning.py` use **hardcoded dictionary parameters** rather than dynamically parsing `config/perturbations.yaml`. | Documented this discrepancy honestly. Physical parameter values match the YAML declarations, but automated dynamic pipeline consumption remains NOT IMPLEMENTED in Phase 1 (reserved for Phase 2). |
| **Task 4: Causal Language in Failure Diagnosis** | Corrected claim that 1-step delay induces a fixed $\pi/2$ phase shift. Added frequency-dependent linear control analysis ($e^{-j\omega \tau}$ phase lag) and evaluated a 100-episode diagnostic sweep on validation seeds `200..299`. | A $20\text{ ms}$ delay produces frequency-dependent phase lag $\Delta \phi = -\omega \cdot 0.02\text{ rad}$. At the system natural frequency ($\omega_n \approx 4.4\text{ rad/s}$), phase lag is $\sim 5.1^\circ$; at limit cycle oscillation frequency ($\sim 15\text{ rad/s}$), phase lag reaches $\sim 17^\circ$. Generalizing 5 episodes to 100 was unfounded; 100-episode validation data now provides rigorous empirical support. |
| **Task 5: Gate Statuses Audited** | Re-audited all Phase 1 gates into strictly supported categories: **PASS**, **FAIL**, or **NOT VERIFIED**. | Identified 2 items as **NOT VERIFIED** (Hardware ESP32 physical latency measurements, and automated consumption of `perturbations.yaml` by execution scripts), preventing ungrounded claims of completeness. |

---

## 2. Quantitative Verification of Diagnostic Sweeps (Validation Seeds 200..299)

Diagnostic runs were strictly confined to **validation seeds `200..299`** under the 20-minute smoke test budget (actual run time: 4.2 seconds):

- **1-Step Delay ($20\text{ ms}$):**
  - **Success Rate:** $0/100$ ($0.0\%$).
  - **Rail Collision Rate:** $100/100$ ($100.0\%$).
  - **Median Crash Step:** $196.0$ steps (range: $[189, 215]$).
  - **Observed Behavior:** Upright balance achieved initially ($\cos\theta > 0.99$ by step $\sim 140$); unmodeled transport lag destabilizes high-gain balancing loop, exciting violent oscillations ($\Delta a \approx \pm 0.75$) that drive the cart into the rail.
- **Friction ($b_c = 0.5\text{ N}\cdot\text{s/m}, b_p = 0.01\text{ N}\cdot\text{m}\cdot\text{s/rad}$):**
  - **Success Rate:** $0/100$ ($0.0\%$).
  - **Rail Collision Rate:** $0/100$ ($0.0\%$).
  - **Mean Max $\cos\theta$:** $0.368$ (range: $[0.322, 0.413]$).
  - **Observed Behavior:** Zero boundary violations; pendulum fails to reach upright equilibrium due to dissipative energy starvation ($P_{\text{loss}} = b_c \dot{x}^2 + b_p \dot{\theta}^2$).
