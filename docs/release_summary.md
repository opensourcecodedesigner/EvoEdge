# EvoEdge — Research & Engineering Release Summary
## Seven-Day Sprint Final Release Candidate

**Date:** 2026-10-14  
**Project:** EvoEdge / NEAT CartPole Swing-Up  
**Milestone:** Day 7 — Final Release Candidate, Reproducibility & Technical Demonstration  
**Governing Standard:** `docs/success_criteria.md` v2.3  
**Final Status Classification:** `RELEASE CANDIDATE — VERIFIED WITH LIMITATIONS`  

---

## 1. Research Question and Frozen Success Criteria

### 1.1 Core Scientific Objective
Can compact, evolutionary, or learning-derived neural control policies achieve comparable robustness and control authority to classical control architectures on non-pre-stabilized nonlinear dynamical systems (CartPole Swing-Up), while complying with severe embedded microcontroller constraints ($\le 128$ parameters, $\le 128$ MACs, zero dynamic allocation, and $\le 20$ ms loop latency)?

### 1.2 Frozen Canonical Success Predicate (v2.3)
Governed by `docs/success_criteria.md` v2.3 (reproduced in `experiments/run_robustness_experiment.py`):
An episode is classified as successful **if and only if**:
1. **No Rail Collision:** The cart does not exceed the rail boundaries ($|x| \le 2.4\text{ m}$) at any point during the $T = 500$ timestep episode.
2. **Upright Inversion:** The pole reaches an inverted configuration ($\cos(\theta) \ge 0.8$, corresponding to $|\theta| \le 0.6435\text{ rad} \approx 36.87^\circ$).
3. **Sustained Stabilization:** The pole maintains $\cos(\theta) \ge 0.8$ continuously for **at least 100 consecutive timesteps** during the episode.
4. **Final Upright Termination:** The episode terminates with the pole in the upright basin ($\cos(\theta_T) \ge 0.8$).

*Constraint:* This success predicate is frozen and non-negotiable. No relaxed proxy (e.g. cumulative reward alone, momentary upright passing, or truncated 200-step windows) may be substituted for official qualification.

---

## 2. Experimental Protocol & Partitioning Policy

The framework enforces strict data isolation across evaluation seed partitions to prevent data leakage and premature claim contamination:

```
Seed Partitions:
├── Seeds 000–199:   Optimizer Training Seeds (Internal ES / NSGA-II fitness evaluation)
├── Seeds 200–299:   Development Validation Partition (Regression screening & SIL verification)
├── Seeds 300–399:   Development Stress-Test Partition (System parameter perturbations)
├── Seeds 400–599:   Exploratory / Unbudgeted Analysis Partition
├── Seeds 600–749:   [QUARANTINED] Official Confirmation Partition (NEVER EVALUATED)
└── Seeds 2000–2099: [QUARANTINED] Holdout Generalization Partition (NEVER EVALUATED)
```

**Policy Adherence:**
- Protected partitions 600–749 and 2000–2099 were **never accessed, read, or evaluated** during Days 5, 6, or 7.
- Development seeds 200–299 were utilized solely as established regression data for numerical-parity and SIL verification, not as fresh generalization claims.

---

## 3. Stage 2 Experimental Findings (ES vs. NSGA-II)

In Stage 2, five independent runs per method were executed under a registered budget ceiling of $1,850,000$ search transitions and $150,000$ validation transitions ($2,000,000$ total transitions per run):

### 3.1 Empirical Results

| Metric | Evolution Strategies (ES) | NSGA-II Neuroevolution | Project Selection Gate | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Search Algorithm** | Natural ES on Fixed MLP | Multi-objective Pareto Neuroevolution | Common search ceiling | Completed |
| **Architectural Complexity** | 80 parameters, 64 MACs | 5–18 parameters, 5–18 MACs | $\le 128$ params, $\le 128$ MACs | **PASSED** |
| **Peak Training Reward** | $298.4 \pm 41.2$ | $182.1 \pm 38.6$ | — | ES achieved higher reward |
| **Realized Transitions** | $\sim 1,848,000$ transitions/run | $\sim 412,000$ transitions/run | $\le 1,850,000$ search limit | **Unequal consumption** |
| **Nominal Success Rate** | **0.0%** (0 / 100 on seeds 200–299) | **0.0%** (0 / 100 on seeds 200–299) | $\ge 90.0\%$ nominal gate | **FAILED** |
| **Worst-Condition Success** | **0.0%** | **0.0%** | $\ge 40.0\%$ robustness gate | **FAILED** |
| **Screening Verdict** | Rejected for deployment | Rejected for deployment | Both gated out | **Both Failed Screening** |

### 3.2 Root-Cause Forensic Analysis
As established in `results/stage2_integrity_audit.md`:
1. **The Reward Deception Problem:**
   The training reward function rewarded $\cos(\theta)$ directly. Optimizers discovered a localized policy that rapidly swings the pole near horizontal ($\cos(\theta) \approx 0.0$ to $0.4$) to accumulate reward without mastering the highly sensitive non-linear transition into the upright basin ($\cos(\theta) \ge 0.8$). As a result, agents scored moderate rewards ($\sim 200–300$) while completely failing the 100-step sustained balance predicate.
2. **Unequal Realized Transition Consumption:**
   Because NSGA-II candidate networks frequently destabilized the cart into early track boundary collisions ($|x| > 2.4$), episodes terminated in 15–40 steps. Consequently, NSGA-II exhausted its fixed generational limit while consuming only $\sim 22\%$ of its allocated $1.85\text{M}$ transition budget.
3. **Scientific Conclusion:**
   The learned controllers failed the development screening gates. This does **not** prove neuroevolution is fundamentally incapable of swing-up control; rather, it demonstrates that unshaped reward optimization without curriculum or balance-basin credit assignment fails to converge under a $2\text{M}$ transition budget.

---

## 4. Classical Baseline (`HybridController`) Performance

To establish an authoritative engineering reference and unblock embedded deployment, an analytical two-regime `HybridController` was implemented in `baselines/hybrid_controller.py`:
- **Regime 1: Balancing (Linear LQR):** Active when $|\theta| < 0.35\text{ rad}$ AND $|\dot{\theta}| < 2.50\text{ rad/s}$.
  $$u = -(K_x x + K_{\dot{x}} \dot{x} + K_\theta \theta + K_{\dot{\theta}} \dot{\theta})$$
- **Regime 2: Pumping (Nonlinear Energy Shaping):** Active outside the balance basin.
  $$E = \frac{2}{3} m_p L^2 \dot{\theta}^2 + m_p g L (\cos\theta - 1.0)$$
  $$u = k_E \operatorname{sgn}(E - E_0) \operatorname{sgn}(\dot{\theta} \cos\theta) - k_{\text{cent},x} x - k_{\text{cent},v} \dot{x}$$

### 4.1 Classical Performance Metrics
- **Active Parameters:** 7 scalar gains ($\le 128$ constraint satisfied).
- **Inference Complexity:** 7 MACs per step ($\le 128$ constraint satisfied).
- **Nominal Success Rate:** **100.0%** (100 / 100 episodes on seeds 200–299).
- **Robustness Screening:** **100.0%** worst-case success across all 10 system perturbation conditions (mass, length, gravity, track limits, actuator noise).

---

## 5. Portable C Export, Numerical Parity & Float32 Discontinuity

In Day 5, the Python `HybridController` was exported into pure ANSI C99 (`results/deployment/hybrid_controller.c` and `hybrid_controller.h`).

### 5.1 Parity Results
- **Double-Precision C (`hybrid_compute_action_double`):**
  - Continuous state parity: Max action difference = **$0.00\text{e}+00$** (bit-identical to Python 64-bit float).
- **Single-Precision C (`hybrid_compute_action`):**
  - Continuous state parity: Max action difference = **$7.15 \times 10^{-6}$** ($\le 1.0 \times 10^{-5}$ tolerance).
  - Closed-loop action difference: Max per-step difference = **$1.07 \times 10^{-5}$**.
- **Energy-Sign Discontinuity:**
  - Float32 precision limits cause an unavoidable sign divergence when $E - E_0 \approx 0.0$ ($|E - E_0| \le 1.19 \times 10^{-7}$).
  - In a discrete neighborhood of $\pm 2.0 \times 10^{-7}\text{ rad/s}$ around the zero-energy homoclinic orbit, $\operatorname{sgn}(E - E_0)$ may flip sign between float32 and float64, yielding an action delta of $\Delta u \approx 1.20$.
  - In closed-loop dynamics, this occurs on a set of measure zero and has **zero impact** on swing-up stability (100% success maintained).

---

## 6. Embedded Architecture & Software-in-the-Loop (SIL) Validation

In Day 6, a platform-adapter boundary was integrated to decouple algorithmic control logic from future hardware peripherals:

```
[ Sensor Acquisition ] ──(platform_sensor_data_t)──► [ Observation Conversion ]
                                                              │
                                                     (platform_observation_t)
                                                              ▼
[ Actuator Conversion ] ◄──(normalized force u)─── [ Controller Step ]
```

### 6.1 SIL Closed-Loop Validation (Seeds 200–299)
- Evaluated 100 episodes (500 steps/episode, 50,000 transitions total):
  - **Python Reference:** 100.0% success
  - **C Double Precision:** 100.0% success ($\Delta u = 0.00$, $\Delta \text{traj} = 0.00$)
  - **C Single Precision:** 100.0% success ($\Delta u_{\max} = 1.07 \times 10^{-5}$, $\Delta \text{traj}_{\max} = 2.38 \times 10^{-6}$)
  - **Platform Adapter:** 100.0% success ($\Delta u_{\max} = 1.17 \times 10^{-5}$, $\Delta \text{traj}_{\max} = 2.62 \times 10^{-6}$)
- Memory resilience: 100,000 cycles executed with 0 bytes dynamic allocation and zero state accumulation.

### 6.2 Host Desktop Timing Contract (50 Hz / 20.0 ms budget)
- Evaluated 1,000,000 continuous forward inferences in standalone native runner (`hybrid_controller_runner.exe`):
  - Throughput: **79.06 million inferences/sec**
  - Median latency ($p_{50}$): **12.10 ns**
  - 99th percentile ($p_{99}$): **19.20 ns**
  - Max observed latency: **242.20 ns** ($0.000242$ ms)
  - Calls within 20 ms budget: **1,000,000 / 1,000,000 (100.0%)**

---

## 7. Categorized Final Status Matrix

To ensure scientific honesty and prevent overstating claims, the final release is categorized into four distinct levels of verification:

| Category | Components Included | Status | Evidence Base |
| :--- | :--- | :--- | :--- |
| **1. Verified Software** | • Python `HybridController`<br>• C99 `hybrid_controller.c` & `platform_adapter.c`<br>• Static library `libhybrid_controller.a`<br>• Native runner `hybrid_controller_runner.exe`<br>• Closed-loop SIL harness `run_sil_cartpole.py`<br>• Automated test suites (126 passing tests) | **COMPLETE & VERIFIED** | Recorded GCC builds (-Wall -Wextra -Wpedantic, 0 warnings); pytest suite pass (126/126); SHA-256 manifest. |
| **2. Reported Research Findings** | • Stage 2 ES vs. NSGA-II benchmark<br>• Reward deception analysis<br>• Realized transition disparity ($1.85\text{M}$ vs $0.41\text{M}$)<br>• Classical baseline robustness (100% on dev partition) | **DOCUMENTED WITH PROTOCOL LIMITATIONS** | `results/stage2_experiment_report.md` and `results/stage2_integrity_audit.md`. Gated out by development filters. |
| **3. Open Scientific Questions** | • Dense shaped reward vs sparse goal predicate<br>• Multi-stage curriculum learning for neuroevolution<br>• Budget normalization for variable-length episodes<br>• True Pareto frontier comparison under matched compute | **UNRESOLVED / FUTURE RESEARCH** | Identified in Stage 2 forensic audit; requires new experimental protocol beyond Sprint scope. |
| **4. Hardware-Dependent Future Work** | • Target cross-compilation (e.g. `arm-none-eabi-gcc`)<br>• Bare-metal Flash/SRAM footprint on MCU<br>• Peripheral HAL drivers (Encoder timer, PWM H-bridge)<br>• On-device execution latency (DWT cycle counter)<br>• Physical cart-pole test rig closed-loop validation | **DEFERRED (NO HARDWARE PRESENT)** | No physical microcontroller or test bench available. All timing labeled as host desktop only. |

---

## 8. Summary of Release Artifacts

The complete software release candidate is organized as follows:
- **Core Library & Adapter:** [`results/deployment/`](file:///d:/neat/results/deployment/)
  - `hybrid_controller.h`, `hybrid_controller.c`
  - `platform_adapter.h`, `platform_adapter.c`
  - `libhybrid_controller.a`, `hybrid_controller.dll`, `platform_adapter.dll`
  - `hybrid_controller_runner.exe`
  - `build_embedded_artifacts.ps1`, `build_embedded_artifacts.sh`
  - `DEPLOYMENT_GUIDE.md`
- **Demonstration & SIL:**
  - [`run_demo.py`](file:///d:/neat/run_demo.py) (Unified technical demonstration entry point)
  - [`experiments/run_sil_cartpole.py`](file:///d:/neat/experiments/run_sil_cartpole.py) (SIL 100-episode harness)
- **Documentation & Reports:**
  - [`README.md`](file:///d:/neat/README.md) (Authoritative project overview)
  - [`docs/release_summary.md`](file:///d:/neat/docs/release_summary.md) (This document)
  - [`results/day6_sil_deployment_report.md`](file:///d:/neat/results/day6_sil_deployment_report.md) (Detailed Day 6 engineering report)
  - [`results/hybrid_controller_c_export_report.md`](file:///d:/neat/results/hybrid_controller_c_export_report.md) (Day 5 parity report)
  - [`results/stage2_experiment_report.md`](file:///d:/neat/results/stage2_experiment_report.md) (Stage 2 empirical report)
  - [`results/stage2_integrity_audit.md`](file:///d:/neat/results/stage2_integrity_audit.md) (Stage 2 forensic audit)

**Final Verdict:** `RELEASE CANDIDATE READY — SOFTWARE COMPLETE WITH HARDWARE DEFERRED`
