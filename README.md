# EvoEdge: Compact Neural Controllers & Embedded Deployment Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C99 Portable Core](https://img.shields.io/badge/Language-ANSI_C99-green.svg)](docs/embedded-deployment.md)
[![Test Suite Status](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](docs/reproducibility.md)
[![Research Status](https://img.shields.io/badge/Status-Research_Prototype-orange.svg)](docs/results-and-limitations.md)
[![Gate Matrix](https://img.shields.io/badge/Audited_Gates-12_Passed__2_Blocked-blue.svg)](docs/results-and-limitations.md#4-preserved-governance-gates--boundaries)

An open-source research and engineering framework investigating whether compact, topology-pruned neural networks can control underactuated, chaotic dynamical systems under strict micro-embedded constraints: **$\le 100$ parameters, $\le 100$ multiply-accumulate (MAC) operations, zero dynamic heap allocation, and deterministic execution**.

EvoEdge bridges multi-objective evolutionary optimization (NSGA-II), teacher-student imitation distillation (DAgger), classical nonlinear control, pure ANSI C99 code generation, and Software-in-the-Loop (SIL) simulation across two distinct benchmarks: **CartPole Swing-Up** and the **Acrobot-v1 Double Pendulum**.

---

## Navigation & Documentation Index

Comprehensive documentation is organized across specialized technical guides:

- [**System Architecture & Design**](docs/architecture.md): Observation pipelines, state-delay registers, multi-stage compression, C export pipeline, and Mermaid diagrams.
- [**Methodology & Experimental Protocol**](docs/methodology.md): 7D feature engineering, DAgger bootstrapping, NSGA-II operators, step-reward formulation, perturbation matrices, and bootstrap statistics.
- [**CartPole Swing-Up Experiments**](docs/experiments/cartpole.md): Task B swing-up, Task A balancing regime sweep, 10-condition robustness evaluation, failure mode analysis, and baseline comparison.
- [**Acrobot Double Pendulum Experiments**](docs/experiments/acrobot.md): Dynamics formulation, transition-budget reconciliation, full 3-seed distribution, and Spong V2 classical audit.
- [**Embedded Deployment & SIL Verification**](docs/embedded-deployment.md): Zero-heap C99 export, 64-byte stack footprint, ctypes numerical parity ($<1.25 \times 10^{-6}$), and desktop SIL benchmarking.
- [**Reproducibility & Replication Guide**](docs/reproducibility.md): Step-by-step setup, test commands, development evaluation scripts, cryptographic SHA-256 ledger, and seed governance.
- [**Verified Results & Scope Limitations**](docs/results-and-limitations.md): Core findings, 14-gate audit matrix, seed variability disclosures, and explicit anti-claims.
- [**Contributing Guidelines**](CONTRIBUTING.md): Code standards, seed protection rules, and pull request procedures.
- [**Citation Metadata**](CITATION.cff): BibTeX and CFF citation formats.

---

## Who Is EvoEdge For?

EvoEdge is specifically structured for researchers, engineers, and students exploring the intersection of lightweight machine learning, dynamical systems, and resource-constrained edge computing:

### 1. Control-Systems & Robotics Researchers
- **Nonlinear Manifold Exploration:** Study how sub-100-parameter neural networks discover and maintain resonant energy-pumping trajectories to cross homoclinic orbits on underactuated chaotic plants.
- **Comparison with Classical Baselines:** Benchmark learned policies directly against analytical energy shaping and Linear Quadratic Regulators (LQR).
- **Latency & Delay Compensation:** Examine the role of finite-history state delay registers (discrete Smith predictors) in stabilizing control loops subject to sensor and actuation delays.
- **Operating-Regime Boundary Analysis:** Investigate how local linear stabilization compares with global nonlinear recovery across continuous angle-basin sweeps.

### 2. Embedded-Systems Engineers
- **Deterministic Zero-Heap Architecture:** Inspect an end-to-end neural deployment pipeline written in pure ANSI C99 (`neural_controller.c` / `hybrid_controller.c`) requiring **0 bytes of dynamic heap allocation** (`malloc`/`free`) and a bounded **64-byte stack footprint**.
- **Platform-Adapter Abstraction:** Study a modular hardware abstraction layer (HAL) architecture that isolates physical peripheral drivers (encoders, PWM timers, motor drivers) from the mathematical control core.
- **Bit-Level Numerical Parity:** Review automated ctypes harnesses and native C unit tests verifying $< 1.25 \times 10^{-6}$ closed-loop parity between desktop Python models and compiled C libraries.
- **SIL Verification:** Run desktop Software-in-the-Loop simulations validating memory stability and control period contracts over tens of thousands of consecutive steps.

### 3. AI / ML Students & Practitioners
- **Multi-Objective Evolutionary Search:** Study practical NSGA-II implementations optimizing non-differentiable objectives (dwell time, worst-condition robustness, parameter counts, and MAC ceilings).
- **Imitation Distillation (DAgger):** Understand how teacher-assisted bootstrapping overcomes sparse-reward exploration bottlenecks on underactuated systems.
- **Experimental Rigor & Budget Accounting:** Learn how to implement fail-closed transition budget trackers, stratified multi-condition evaluation schedules, and clustered bootstrap confidence intervals.
- **Audit & Defect Remediation:** Review transparent engineering case studies detailing how reward deception, transition accounting discrepancies, and analytical controller defects were discovered and systematically resolved.

### 4. Developers Researching Compact Neural Controllers
- **Extreme Model Compression:** Analyze empirical trade-offs when compressing dense neural policies ($64 \times 64$ MLPs, 4,737 parameters) down to sparse, masked topologies (69 parameters, 60 MACs)—achieving equal or superior performance with **$68.6\times$ fewer weights** and **$76.8\times$ fewer operations**.
- **Edge Deployment Templates:** Utilize the provided ANSI C99 export templates and build scripts (`build_embedded_artifacts.ps1` / `.sh`) as starting points for deploying custom neural controllers to microcontrollers.
- **Hardware-Aware Design:** Evaluate model viability against micro-edge constraints where standard deep learning inference engines (e.g., TensorFlow Lite for Microcontrollers, ONNX Runtime) introduce excessive memory or runtime overhead.

---

## Current Readiness and Limitations

To maintain scientific integrity and prevent industrial misapplication, the project's current maturity and operational boundaries are explicitly stated:

### Research Prototype Status
EvoEdge is a **scientific research prototype and exploratory framework**. It is **NOT** production-ready, mission-critical, or certified for commercial, industrial, automotive, or flight control systems. It is intended solely for laboratory, educational, and research evaluation.

### Incomplete Physical Hardware Validation (Gate 10 BLOCKED)
While the ANSI C99 controller and platform adapter compile cleanly under strict compiler flags (`-Wall -Wextra -Wpedantic`) and pass extensive Software-in-the-Loop (SIL) desktop simulations:
- **Physical microcontroller testing (e.g., ESP32, STM32 Cortex-M) has not been performed.** No physical development board or test bench was connected during this project phase.
- **Desktop execution measurements ($p_{50} = 2.50\ \mu\text{s}$) represent high-performance desktop CPU execution (x86_64).** They demonstrate minimal algorithmic complexity (60 MACs, zero branches in real-time loops) but **do NOT constitute on-target microcontroller latency measurements or hard real-time guarantees**.
- Physical deployment requires implementing hardware timer ISRs, ADC/encoder sampling, and H-bridge PWM drivers on a physical development board, as outlined in [`docs/embedded-deployment.md`](docs/embedded-deployment.md).

### Prospective Holdout Confirmation Pending Authorization (Gate 11 BLOCKED)
All numerical claims and statistical intervals reported in this repository are derived from **development evaluation partitions** (CartPole seeds 200–219, Acrobot seeds 500–549):
- **Prospective out-of-sample confirmatory evaluation (seeds 3000–3099) remains strictly BLOCKED** pending formal human authorization.
- Seeds 600–749 and 2000–2099 remain permanently quarantined to prevent retrospective evaluation contamination.
- Until prospective confirmation is formally authorized and unblinded, all results must be interpreted as development-set empirical evidence.

### Significant Seed Optimization Variability
Across independent random training seeds ($N=3$), evolutionary neural search exhibits significant optimization variance:
- On CartPole Task B, learned MLP baseline seeds achieved 91.0%, 95.0%, and 98.0% success.
- On Acrobot-v1, the compact architecture converged to 100% on **only 1 of 3 seeds** (Seed 44), yielding an aggregate success of 33.3% (250/750). The MLP baseline converged on 2 of 3 seeds (aggregate 46.9%).
- By contrast, the analytical classical controller (Spong V2) achieved **100.0% (250/250)** deterministically across all conditions.

### Second-Task Learning vs. Zero-Shot Transfer
The Acrobot experiments demonstrate that the **compact topology and evolutionary training pipeline generalize to a second distinct control benchmark** when trained from tabula rasa. They do **NOT** demonstrate zero-shot transfer of the CartPole controller weights.

For full disclosures and anti-claims, see [`docs/results-and-limitations.md`](docs/results-and-limitations.md).

---

## Key Empirical Findings

### 1. CartPole Task B Swing-Up & Stabilization (10 Conditions, $N=100$)
Evaluated on development seeds 200–209 across 10 system perturbation conditions (mass, length, rail limits, sensor noise, delay):

| Controller | Architecture | Params | MACs | Success Rate | 95% Wilson CI | Worst-Cond Success | Rail Crashes | Dwell Failures |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EvoEdge Candidate (Seed 46)** | Masked MLP ($H=8$) | **69** | **60** | **99.0%** (99/100) | [94.6%, 99.8%] | **90.0%** | **0** | **1** |
| **OpenAI-ES Baseline (Seed 43 - Best)** | Dense MLP ($64\times 64$) | 4,737 | 4,608 | **98.0%** (98/100) | [93.0%, 99.4%] | 80.0% | 0 | 2 |
| **OpenAI-ES Baseline (Seed 44 - Mid)** | Dense MLP ($64\times 64$) | 4,737 | 4,608 | **95.0%** (95/100) | [88.8%, 97.8%] | 70.0% | 0 | 5 |
| **OpenAI-ES Baseline (Seed 42 - Min)** | Dense MLP ($64\times 64$) | 4,737 | 4,608 | **91.0%** (91/100) | [83.8%, 95.2%] | 50.0% | 0 | 9 |
| **Classical Reference (`HybridController`)** | Analytical Energy/LQR | **7** | **7** | **84.0%** (84/100) | [75.6%, 89.9%] | 20.0% | 4 | 12 |

### 2. Acrobot-v1 Second Task Benchmark (5 Conditions, $N=250$)
Evaluated on development seeds 500–549 under a strict transition budget ceiling ($\le 150,000$ search transitions):

| Architecture / Controller | Search Budget | Convergence Yield | Per-Seed Success (42 / 43 / 44) | Aggregate Success ($N=750$) | Best Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical Spong V2 (Corrected)** | 0 (Analytical) | N/A | N/A | **250 / 250 (100.0%)** | 100.0% |
| **EvoEdge Compact ($H=8$, 65 params)** | $\le 150,000$ | **1 of 3 (33.3%)** | 0.0% / 0.0% / 100.0% | **250 / 750 (33.3%)** | **100.0%** (Seed 44) |
| **Learned MLP 64×64 (4,673 params)** | $\le 150,000$ | **2 of 3 (66.7%)** | 0.0% / 88.0% / 52.8% | **352 / 750 (46.9%)** | **88.0%** (Seed 43) |

### 3. Embedded C99 Deployment Metrics
- **Dynamic Heap Memory:** 0 bytes (no `malloc`, `calloc`, or dynamic containers).
- **Stack Footprint:** 64 bytes worst-case.
- **Numerical Parity (ctypes harness):** Single-precision maximum closed-loop difference $\Delta u \le 1.07 \times 10^{-5}$; double-precision $\Delta u = 0.00$.
- **Host SIL Timing:** Median latency $p_{50} = 2.50\ \mu\text{s}$ on host desktop CPU.

---

## Quick Start & Verification

### Prerequisites
- Python 3.10+ (tested on Python 3.12.10)
- GCC / Clang / MSVC C99 compiler
- Windows 11, Linux, or macOS

### 1. Installation
```bash
git clone https://github.com/opensourcecodedesigner/EvoEdge.git
cd EvoEdge
pip install -r requirements.txt
```

### 2. Run Automated Regression Tests
Verify seed partition integrity, baseline implementations, and embedded parity:
```bash
# Verify seed governance boundaries
pytest tests/test_split_integrity.py -v

# Verify OpenAI-ES baseline implementation and budget enforcement
pytest work/phase3e_followup_6h/sprint3/project_closeout/v4_baseline/test_v4.py -v

# Verify embedded C numerical parity and SIL harnesses
pytest tests/test_hybrid_c_parity.py -v
pytest tests/test_embedded_sil.py -v
```

### 3. Reproduce Development Evaluations
Run the independent development evaluation scripts against frozen checkpoints:
```bash
# Evaluate CartPole Task B swing-up (100 episodes per controller)
python work/phase3e_followup_6h/sprint3/project_closeout/v4_baseline/evaluate_v4_development.py

# Evaluate Acrobot-v1 second task (250 episodes per controller)
python work/phase3e_followup_6h/sprint3/research_upgrade/second_task/evaluate_acrobot_v2.py
```

### 4. Compile & Run Native C Runner
Compile the portable ANSI C99 controller and run the standalone verification test suite:
```bash
gcc -std=c99 -Wall -Wextra -Wpedantic -O3 \
    results/deployment/hybrid_controller_runner.c \
    results/deployment/hybrid_controller.c \
    results/deployment/platform_adapter.c \
    -o results/deployment/hybrid_controller_runner.exe

./results/deployment/hybrid_controller_runner.exe
```

For full reproduction commands and SHA-256 hash verification, see [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Repository Structure

```
EvoEdge/
├── README.md                      # Project landing page and executive overview
├── CONTRIBUTING.md                # Development standards and seed protection rules
├── CITATION.cff                   # Citation metadata
├── LICENSE                        # MIT open-source license
├── requirements.txt               # Python package dependencies
├── docs/                          # Comprehensive technical documentation
│   ├── architecture.md            # System design, dataflow, and interfaces
│   ├── methodology.md             # Training protocols, DAgger, and NSGA-II
│   ├── embedded-deployment.md     # C99 export, stack analysis, and SIL parity
│   ├── reproducibility.md         # Replication guide and cryptographic ledger
│   ├── results-and-limitations.md # Findings, 14-gate audit, and anti-claims
│   ├── diagrams/                  # Mermaid system and dataflow diagrams
│   └── experiments/
│       ├── cartpole.md            # CartPole Task B & Task A empirical analysis
│       └── acrobot.md             # Acrobot-v1 second-task empirical analysis
├── baselines/                     # Classical reference controllers
│   ├── hybrid_controller.py       # Analytical Energy-LQR reference
│   └── mlp.py                     # PyTorch MLP baseline architecture
├── env/                           # Simulation environments and accounting
│   ├── cartpole_swingup.py        # Canonical continuous CartPole environment
│   └── budget_tracker.py          # Fail-closed transition budget wrapper
├── results/                       # Deployment artifacts and firmware guides
│   └── deployment/
│       ├── hybrid_controller.c    # Pure ANSI C99 controller core
│       ├── hybrid_controller.h    # Controller API and parameter definitions
│       ├── platform_adapter.c     # Sensor and actuator conversion HAL
│       ├── platform_adapter.h     # HAL structures and interface
│       ├── neural_controller.c    # Exported compact neural controller core
│       ├── neural_controller.h    # Neural controller API
│       └── DEPLOYMENT_GUIDE.md    # Firmware integration checklist
├── tests/                         # Automated test suites
│   ├── test_split_integrity.py    # Seed partition isolation tests
│   ├── test_hybrid_c_parity.py    # Numerical parity tests
│   └── test_embedded_sil.py       # Closed-loop SIL tests
└── work/                          # Preserved research packages & frozen models
    └── phase3e_followup_6h/sprint3/
        ├── project_closeout/      # Final CartPole closeout package & models
        └── research_upgrade/      # Second-task Acrobot package & models
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
