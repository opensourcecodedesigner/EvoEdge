# EvoEdge — Compact CartPole Control & Embedded Deployment Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C99 Portable Core](https://img.shields.io/badge/Language-ANSI_C99-green.svg)](#embedded-architecture)
[![Test Suite Status](https://img.shields.io/badge/Tests-126%20Passed-brightgreen.svg)](#testing--verification)
[![SIL Validation](https://img.shields.io/badge/SIL%20Validation-100%25%20Success-success.svg)](#software-in-the-loop-sil-demonstration)

A reproducible research and embedded-software framework for evaluating compact control architectures on the nonlinear **CartPole Swing-Up** problem under strict embedded deployment constraints ($\le 128$ parameters, $\le 128$ MACs, zero dynamic heap allocation, and $\le 20\text{ ms}$ control loop period).

EvoEdge bridges evolutionary machine learning, classical hybrid nonlinear control, pure ANSI C99 deployment, and Software-in-the-Loop (SIL) simulation.

---

## 1. Problem & Motivation

Standard CartPole benchmarks initialize the pole nearly upright, requiring only local linear stabilization. The **Swing-Up** variant initializes the pole hanging downward ($\theta = \pi$), demanding an energy-pumping phase to swing the pole into the inverted basin followed by linear stabilization.

In edge and embedded cyber-physical systems (e.g. low-power Cortex-M microcontrollers, wearable devices, and micro-robotics), controllers face severe operational constraints:
- **Tight Footprint:** Tiny Flash and SRAM limits prohibit multi-megabyte neural networks.
- **Low Compute & Energy:** Strict Multiply-Accumulate (MAC) ceilings per forward step.
- **Deterministic Latency:** Zero dynamic heap allocation (`malloc`/`free`) and bounded worst-case execution time (WCET) to meet periodic control contracts ($50\text{ Hz} \iff 20\text{ ms}$).
- **Safety & Auditability:** Verifiable behavior, explicit saturation bounds, and fail-safe shutdown upon sensor faults or non-finite inputs.

---

## 2. Research Approach & Evolution

The project progressed through a structured seven-day engineering cycle:

1. **Neuroevolution Exploration (NEAT & NSGA-II):**
   Investigated whether topology-evolving algorithms could discover minimal neural networks under multi-objective Pareto optimization (reward vs. parameter count vs. MACs).
2. **Evolution Strategies (ES MLP Benchmark):**
   Evaluated Natural Evolution Strategies on fixed MLP architectures under an identical transition-budget ceiling ($1,850,000$ search transitions).
3. **Stage 2 Forensic Audit:**
   Established that both learned policies failed development screening gates due to reward deception (agents learned to hold the pole horizontal for reward rather than inverting) and unequal realized transition consumption.
4. **Classical Analytical Reference (`HybridController`):**
   Implemented an authoritative reference combining linear LQR balancing ($|\theta| < 0.35\text{ rad}$) and nonlinear energy shaping pumping ($|\theta| \ge 0.35\text{ rad}$) with cart centering, achieving 100% success on nominal and perturbed development sets in 7 parameters and 7 MACs.
5. **Portable Embedded C Export & Parity:**
   Transpiled the verified Python reference into pure ANSI C99 (`hybrid_controller.c`), verifying exact bit-level double-precision parity and bounded float32 rounding.
6. **Platform-Adapter Architecture & SIL Validation:**
   Designed an embedded platform-adapter boundary that isolates physical peripherals from control logic, verified via native test execution and 100-episode closed-loop simulation.

---

## 3. System Architecture

The framework enforces strict separation between algorithmic control logic, simulation environments, and physical peripheral interfaces:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 HIGH-LEVEL WORKFLOW                     │
                  │                                                         │
                  │   [ Sensor Acquisition ] (Raw encoder & timing data)    │
                  │             │                                           │
                  │             ▼                                           │
                  │   [ Platform Adapter ] (Observation conversion & checks)│
                  │             │                                           │
                  │             ▼                                           │
                  │   [ C Controller Core ] (7 MACs, 0B heap, pure C99)     │
                  │             │                                           │
                  │             ▼                                           │
                  │   [ Actuator Conversion ] (Force, PWM duty, Direction)  │
                  └─────────────────────────────────────────────────────────┘
                                   ▲                       ▲
                                   │                       │
                     Desktop SIL Harness         Future Microcontroller
                     (CartPoleSwingUpEnv)        (Bare-Metal BSP / HAL)
```

### Module Organization
```
neat/
├── run_demo.py                    # Unified reproducible technical demo entry point
├── requirements.txt               # Python dependencies (numpy, pytest, gym)
├── baselines/
│   ├── hybrid_controller.py       # Authoritative Python reference controller
│   └── mlp.py                     # ES MLP baseline
├── env/
│   ├── cartpole_swingup.py        # Canonical CartPole simulation environment
│   ├── vectorized_env.py          # High-speed vectorized batch simulator
│   └── budget_tracker.py          # Strict transition accounting wrapper
├── evolution/
│   ├── nsga2_engine.py            # NSGA-II multi-objective engine
│   └── evolution_strategies.py    # Natural Evolution Strategies engine
├── results/
│   ├── deployment/                # Embedded C deployment package
│   │   ├── hybrid_controller.h    # Controller public API and gains
│   │   ├── hybrid_controller.c    # Pure C99 controller implementation
│   │   ├── platform_adapter.h     # Platform adapter interface and structures
│   │   ├── platform_adapter.c     # Sensor and actuator conversion pipeline
│   │   ├── hybrid_controller_runner.c  # Native C unit tests and benchmark
│   │   ├── hybrid_controller_runner.exe # Standalone compiled native runner
│   │   ├── libhybrid_controller.a # Reusable embedded static library
│   │   ├── hybrid_controller.dll  # Controller dynamic shared library
│   │   ├── platform_adapter.dll   # Adapter dynamic shared library
│   │   ├── build_embedded_artifacts.ps1 # PowerShell build script
│   │   ├── build_embedded_artifacts.sh  # POSIX Bash build script
│   │   └── DEPLOYMENT_GUIDE.md    # Firmware integration and hardware checklist
│   ├── day6_sil_deployment_report.md # Day 6 embedded engineering report
│   └── hybrid_controller_c_export_report.md # Day 5 parity report
├── docs/
│   ├── release_summary.md         # Final research & engineering summary
│   ├── success_criteria.md        # Frozen official qualification standard v2.3
│   └── success_criteria_v2.4_PROPOSED.md # Proposed Stage 2 protocol revisions
├── experiments/
│   ├── run_sil_cartpole.py        # 100-episode closed-loop SIL evaluation
│   └── run_robustness_experiment.py # Canonical success predicate evaluator
└── tests/
    ├── test_embedded_sil.py       # Embedded C & SIL automated tests
    ├── test_hybrid_c_parity.py    # C-to-Python numerical parity tests
    └── test_budget_tracker.py     # Transition budget enforcement tests
```

---

## 4. Key Results & Findings

### 4.1 Stage 2 Learning Benchmark (ES vs. NSGA-II)
- **Constraint Compliance:** Both algorithms satisfied complexity constraints (ES: 80 params, 64 MACs; NSGA-II: 5–18 params, 5–18 MACs; ceiling $\le 128$).
- **Success Rate:** Both methods scored **0.0% nominal success** (0/100 on development seeds 200–299) against the official 100-step sustained balance predicate.
- **Root Cause:** Dense reward functions encouraged hanging horizontal ($\cos\theta \approx 0.0$ to $0.4$) rather than crossing the saddle into upright balance ($\cos\theta \ge 0.8$).
- **Status:** Gated out from deployment; preserved in `results/stage2_experiment_report.md`.

### 4.2 Analytical `HybridController` Performance
- **Complexity:** 7 parameters, 7 MACs, 0 bytes dynamic allocation.
- **Nominal Success:** **100.0%** (100 / 100 episodes on seeds 200–299).
- **Robustness:** **100.0%** worst-condition success across all 10 system perturbation scenarios (pole mass, length, rail bounds, actuator noise).

### 4.3 C Export & Numerical Parity
- **Double Precision (`hybrid_compute_action_double`):** Bit-identical parity with Python reference ($\Delta u = 0.00\text{e}+00$).
- **Single Precision (`hybrid_compute_action`):** Continuous state parity $\Delta u \le 7.15 \times 10^{-6}$; closed-loop difference $\Delta u \le 1.07 \times 10^{-5}$.
- **Energy-Sign Discontinuity:** Documented float32 sign flip when $|E - E_0| \le 1.19 \times 10^{-7}$ in an infinitesimal neighborhood of the homoclinic orbit; zero closed-loop impact (100% success maintained).

### 4.4 Software-in-the-Loop (SIL) Validation
- **100 Episodes (Seeds 200–299, 50,000 steps):**
  - Python Reference: **100.0% success**
  - C Double Precision: **100.0% success** ($\Delta u = 0.00$, $\Delta \text{traj} = 0.00$)
  - C Single Precision: **100.0% success** ($\Delta u_{\max} = 1.07 \times 10^{-5}$, $\Delta \text{traj}_{\max} = 2.38 \times 10^{-6}$)
  - Platform Adapter: **100.0% success** ($\Delta u_{\max} = 1.17 \times 10^{-5}$, $\Delta \text{traj}_{\max} = 2.62 \times 10^{-6}$)
- **Memory Resilience:** Zero memory leakage, zero state accumulation over 100,000 consecutive steps.

### 4.5 50 Hz Timing Contract (Host Desktop Measurements)
- **Sample:** 1,000,000 forward evaluations on host x86_64 desktop workstation.
- **Throughput:** ~79 million inferences/sec.
- **Median Latency ($p_{50}$):** 12.10 ns (0.00006% of 20 ms loop).
- **99th Percentile ($p_{99}$):** 19.20 ns.
- **Maximum Observed:** 242.20 ns.
- **Budget Compliance:** **100.0000% within 20 ms**.

---

## 5. Quick Start & Reproducibility

### Prerequisites
- Python 3.10+ (tested on Python 3.12.10)
- GCC C99 compiler (tested with GCC 16.1.0 on MSYS2 UCRT64)
- Windows 11 (tested platform) or POSIX Linux/macOS

### 5.1 One-Step Technical Demonstration
Execute the unified release demonstration:
```powershell
python run_demo.py
```
This single command:
1. Validates the host toolchain and dependencies.
2. Compiles the portable C artifacts (`.a`, `.dll`, `.exe`) with `-Wall -Wextra -Wpedantic -O3`.
3. Runs the standalone native C test runner (`hybrid_controller_runner.exe`).
4. Executes the 100-episode closed-loop SIL simulation.
5. Runs the automated regression test suite.

### 5.2 Manual Step-by-Step Execution

#### Step 1: Build the Embedded C Artifacts
```powershell
# PowerShell (Windows)
powershell -ExecutionPolicy Bypass -File results/deployment/build_embedded_artifacts.ps1

# Or Bash (Linux / macOS)
bash results/deployment/build_embedded_artifacts.sh
```

#### Step 2: Execute Standalone Native C Runner
```powershell
.\results\deployment\hybrid_controller_runner.exe
```

#### Step 3: Run Closed-Loop SIL Simulation
```powershell
python experiments/run_sil_cartpole.py
```

#### Step 4: Run Automated Test Suites
```powershell
# Run embedded architecture & SIL tests
python -m pytest tests/test_embedded_sil.py -v

# Run C-to-Python numerical parity tests
python -m pytest tests/test_hybrid_c_parity.py -v

# Run entire repository test suite (126 tests)
python -m pytest -q
```

---

## 6. Critical Limitations & Scope Boundary

In accordance with scientific and engineering rigor, the following boundaries are explicitly established:

1. **Host-Only Software Verification:**
   All execution timing (~12–25 ns) was recorded on a high-frequency desktop x86_64 CPU. It demonstrates minimal algorithmic overhead (7 MACs, branch-free arithmetic), but **does NOT constitute microcontroller hardware timing or a hard real-time guarantee**.
2. **No Physical Development Board:**
   No physical MCU (e.g. STM32, ESP32) or hardware test rig was present during this sprint. All peripheral interactions (optical encoder, H-bridge PWM, ADC) are simulated via software test doubles.
3. **Protected Data Isolation:**
   Official confirmation seed partitions (seeds 600–749 and 2000–2099) remain quarantined and unread. Development results (seeds 200–299) are presented strictly as regression checks.
4. **Learning vs. Classical Scope:**
   The failure of ES and NSGA-II under the Stage 2 protocol reflects the current reward formulation and transition budget, not an intrinsic impossibility of neural swing-up control.

---

## 7. Roadmap to Physical Hardware

When a physical development board becomes available, follow [`results/deployment/DEPLOYMENT_GUIDE.md`](results/deployment/DEPLOYMENT_GUIDE.md):

1. **Toolchain Setup:** Cross-compile `libhybrid_controller.a` with `arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -O3`.
2. **Implement Platform Adapter HAL Hooks:**
   - `bsp_sample_sensors()`: Read encoder hardware timer registers and differentiate velocity.
   - `bsp_write_actuator()`: Set H-bridge direction GPIO and PWM timer compare register.
   - `bsp_emergency_shutdown()`: Disable motor gate driver on fault.
3. **Timer ISR Scheduling:** Call `platform_execute_pipeline_step` inside a 50 Hz periodic timer interrupt (20.0 ms period).
4. **On-Target Latency Measurement:** Instrument execution time using ARM DWT cycle counter (`DWT->CYCCNT`) or an oscilloscope toggle pin.

---

## 8. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
