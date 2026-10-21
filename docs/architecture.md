# EvoEdge System Architecture & Technical Design

**Document Classification:** Technical Architecture Specification  
**Primary Repository:** `D:\neat`  
**Target Environments:** CartPole Swing-Up (`CartPoleSwingUpEnv`), Gymnasium `Acrobot-v1`  
**Deployment Target:** ANSI C99 Portable Library (`neural_controller.c` / `hybrid_controller.c`)

---

## 1. High-Level System Architecture

EvoEdge is an end-to-end framework for synthesizing, compressing, verifying, and deploying ultra-compact neural and hybrid controllers for nonlinear dynamical systems under strict edge computing constraints ($\le 128$ parameters, $\le 128$ multiply-accumulate operations (MACs), zero dynamic heap memory allocation, and deterministic latency $\le 20\text{ ms}$).

The framework decouples physical plant simulation, feature extraction, evolutionary optimization, controller representation, and embedded deployment:

```mermaid
graph TD
    subgraph Environment ["Simulation & Hardware Environment"]
        CP["CartPole-v1 Swing-Up Plant<br/>(Gymnasium / Custom ODE)"]
        AC["Acrobot-v1 Plant<br/>(Gymnasium Underactuated Double Pendulum)"]
    end

    subgraph Observations ["Observation & Feature Normalization"]
        OB1["CartPole State: [x, x_dot, cos_theta, sin_theta, theta_dot]"]
        OB2["Delay Registers: [u_{t-1}, u_{t-2}]"]
        OB_VEC["7D Normalized Input Vector"]
        OB1 --> OB_VEC
        OB2 --> OB_VEC
    end

    subgraph Controllers ["Controller Implementations"]
        subgraph NeuralCandidate ["EvoEdge Compact Neural Controller"]
            NC["Masked 1-Layer MLP (tanh)<br/>Input: 7D | Hidden: 8 | Output: 1D<br/>Active Params: 69 | MACs: 60"]
        end
        subgraph LearnedBaseline ["Full-Scale Learned Baseline"]
            BL["Dense 2-Layer MLP (tanh)<br/>Input: 7D | Hidden: 64x64 | Output: 1D<br/>Active Params: 4,737 | MACs: 4,608"]
        end
        subgraph ClassicalBaseline ["Analytical Reference"]
            HYB["Hybrid Nonlinear / LQR<br/>Energy Pumping (|theta| >= 0.35 rad)<br/>LQR Balancing (|theta| < 0.35 rad)<br/>Params: 7 | MACs: 7"]
        end
    end

    subgraph Actuation ["Actuation Pipeline"]
        ACT_RAW["Continuous Output: y in [-1.0, 1.0]"]
        ACT_FORCE["Scaled Action: F = 20.0 * y [N]"]
        ACT_DELAY["Delay FIFO Queue (0 to 2 Steps, tau = 20-40 ms)"]
        ACT_RAW --> ACT_FORCE --> ACT_DELAY
    end

    subgraph Deployment ["Deployment & SIL Validation"]
        C_EXP["Pure ANSI C99 Export<br/>neural_controller.c / .h<br/>Stack: 64 B | Dynamic Heap: 0 B"]
        SIL["Software-in-the-Loop (SIL) DLL<br/>Host Latency: p50 = 2.50 us"]
        MCU["Physical MCU Target (ESP32 / STM32)<br/>Status: BLOCKED (Hardware Unavailable)"]
        C_EXP --> SIL
        C_EXP -.-> MCU
    end

    CP --> OB1
    OB_VEC --> NC
    OB_VEC --> BL
    OB_VEC --> HYB
    NC --> ACT_RAW
    BL --> ACT_RAW
    HYB --> ACT_RAW
    ACT_DELAY --> CP
    NC -.-> C_EXP
```

---

## 2. CartPole Environment and Observation Pipeline

### 2.1 Mechanical Plant Dynamics
The primary benchmark is the **CartPole Swing-Up and Stabilization** problem (`env/cartpole_swingup.py`):
- **Cart Mass ($M_c$):** $1.0\text{ kg}$ (nominal; perturbed up to $1.30\text{ kg}$)
- **Pole Mass ($m_p$):** $0.1\text{ kg}$ (nominal; perturbed up to $0.13\text{ kg}$)
- **Pole Half-Length ($l$):** $0.5\text{ m}$ (total length $1.0\text{ m}$)
- **Rail Bounds:** Cart displacement must remain within $|x| < 2.4\text{ m}$. Any excursion beyond $\pm 2.4\text{ m}$ triggers an instant episode failure.
- **Initial State:** The pole begins hanging straight downwards ($\theta = \pi\text{ rad}$) with zero angular and linear velocities ($\dot\theta = 0, \dot{x} = 0, x = 0$).

### 2.2 Feature Normalization and Delay Registers
Controllers observe a continuous 7-dimensional normalized vector $\mathbf{x}_t \in \mathbb{R}^7$:
$$\mathbf{x}_t = \left[ \frac{x_t}{2.4},\, \text{clip}\left(\frac{\dot{x}_t}{10.0}, -1, 1\right),\, \cos\theta_t,\, \sin\theta_t,\, \text{clip}\left(\frac{\dot{\theta}_t}{10.0}, -1, 1\right),\, u_{t-1},\, u_{t-2} \right]$$

1. **Trigonometric Decomposition:** Continuous $\cos\theta$ and $\sin\theta$ eliminate phase-wrap branch discontinuities around $\pm\pi$, avoiding catastrophic gradient/evolutionary instability.
2. **Delay-Aware Memory Registers:** $u_{t-1}$ and $u_{t-2}$ represent the controller's own outputs from the previous two timesteps ($t-1$ and $t-2$). This internal state memory enables compensation for up to 2 steps ($40\text{ ms}$) of transport latency without requiring a recurrent neural network (RNN).

---

## 3. Controller Architectures & Complexity

| Controller | Architecture Type | Hidden Layers | Active Parameters | Weight MACs | Memory (Heap / Stack) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EvoEdge Candidate (`stage3d_nsga2_robust_seed46`)** | Masked MLP | 1 layer ($H=8$) | **69** | **60** | **0 B / 64 B** |
| **OpenAI-ES Baseline (`v4_seed42/43/44`)** | Dense MLP | 2 layers ($64 \times 64$) | **4,737** | **4,608** | **0 B / ~1 KB** |
| **Classical Reference (`HybridController`)** | Piecewise Energy/LQR | N/A (Analytical) | **7** | **7** | **0 B / 32 B** |

### Mathematical Formulation of the Compact Candidate
For an input vector $\mathbf{x} \in \mathbb{R}^7$:
$$\mathbf{h} = \tanh\left( \mathbf{M}_1 \odot (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \right) \in \mathbb{R}^8$$
$$y = \tanh\left( \mathbf{m}_2 \odot (\mathbf{W}_2 \mathbf{h} + b_2) \right) \in [-1.0, 1.0]$$
$$\text{Force } F = 20.0 \cdot y \quad [\text{N}]$$

Where $\mathbf{M}_1 \in \{0, 1\}^{8 \times 7}$ and $\mathbf{m}_2 \in \{0, 1\}^8$ are binary structural masks evolved via NSGA-II to prune inactive synaptic connections.

---

## 4. Multi-Stage Optimization and Pruning Pipeline

EvoEdge uses a **teacher-assisted evolutionary compression pipeline** rather than discovering networks from scratch:

```mermaid
graph TD
    subgraph Stage3A ["Stage 3A: Teacher Demonstration & Imitation"]
        T_EXP["Expert Controller Demonstrations<br/>(200,000 transitions)"]
        T_DAG["DAgger Imitation Training<br/>(200,000 transitions)"]
        T_MODEL["Trained PyTorch Teacher<br/>mlp64_baseline.pt (Dense 64x64)"]
        T_EXP --> T_DAG --> T_MODEL
    end

    subgraph Stage3B ["Stage 3B & 3C: Evolutionary Search & Delay Adaptation"]
        S3B_INIT["Warm-Start / Topology Seeding"]
        S3B_EVO["NSGA-II Structural Evolution<br/>Multi-Objective Pareto Optimization<br/>(~3,800,000 cumulative lineage transitions)"]
        S3C_DELAY["Delay Compensation Adaptation<br/>(Internal u_{t-1}, u_{t-2} state registers)"]
        T_MODEL --> S3B_INIT --> S3B_EVO --> S3C_DELAY
    end

    subgraph Stage3D ["Stage 3D: Robustness-First Pruning (Candidate Selection)"]
        S3D_SELECT["Constrained Lexicographic Selection<br/>Objectives: S_worst >= 40%, S_rob >= 90%, Min MACs, Min Params"]
        S3D_FROZEN["Frozen EvoEdge Candidate<br/>stage3d_nsga2_robust_seed46.json<br/>(69 params, 60 MACs)"]
        S3C_DELAY --> S3D_SELECT --> S3D_FROZEN
    end

    subgraph BaselineTraining ["Sprint 3 v4 Learned Baseline Retraining"]
        BL_LOAD["Teacher Weight Transfer<br/>(Factor 2.5 absorbed into W and b)"]
        BL_ES["OpenAI-ES Training Engine<br/>Antithetic Sampling & Stratified 5-Condition Schedule"]
        BL_BUDGET["Fail-Closed Step Budget Tracker<br/>Ceiling: Exactly 1,850,000 transitions"]
        BL_MODELS["Trained 64x64 Baselines<br/>Seeds 42, 43, 44 (4,737 params, 4,608 MACs)"]
        T_MODEL --> BL_LOAD --> BL_ES
        BL_BUDGET --> BL_ES --> BL_MODELS
    end

    subgraph SecondTask ["Second Control Task: Acrobot-v1 Generalization"]
        AC_TABULA["Tabula Rasa Random Xavier Initialization<br/>(No teacher, no CartPole weight transfer)"]
        AC_TRAIN["OpenAI-ES with Fail-Closed StepBudgetTracker<br/>Ceiling: <= 150,000 transitions"]
        AC_MODELS["Acrobot Controllers (Seeds 42, 43, 44)<br/>Compact (65 params, 56 MACs) vs MLP64 (4,673 params)"]
        AC_TABULA --> AC_TRAIN --> AC_MODELS
    end
```

### Stage Breakdown:
1. **Stage 3A (Teacher Pretraining):** A 64×64 dense MLP teacher (`results/mlp64_baseline.pt`) was trained via DAgger on 200,000 transitions of expert classical control demonstrations.
2. **Stage 3B (Structural Evolution):** Topology-evolving NSGA-II pruned redundant hidden neurons and connections while optimizing nominal reward.
3. **Stage 3C (Delay Adaptation):** Delay registers ($u_{t-1}, u_{t-2}$) were integrated into the genome state, evolving weights capable of stabilizing the cart under 20–40 ms actuator transport latency.
4. **Stage 3D (Robustness-First Selection):** Evolved candidates under domain randomization across 8 physical conditions. Champion selection enforced **constrained lexicographic dominance**:
   - Primary: Feasibility filter ($\text{robust\_success} > 0$).
   - Secondary: Worst-condition success rate ($S_{\text{worst}} \ge 40\%$).
   - Tertiary: Aggregate robust success ($\bar{S}_{\text{rob}} \ge 90\%$).
   - Quaternary: Minimal active parameters ($P \le 128$) and MACs ($M \le 128$).
   - **Champion:** Seed 46 (`stage3d_nsga2_robust_seed46.json`), achieving 90.0% worst-condition success, 98.0% aggregate robust success, 69 active parameters, and 60 weight MACs.

---

## 5. Embedded C Export & Execution Pipeline

The frozen Python candidate is transpiled to pure ANSI C99 (`results/deployment/neural_controller.c` and `neural_controller.h`):

```mermaid
graph TD
    subgraph Sensors ["Sensors & Raw State"]
        RAW_X["Cart Position x [m]"]
        RAW_XDOT["Cart Velocity x_dot [m/s]"]
        RAW_THETA["Pole Angle theta [rad]"]
        RAW_THETADOT["Pole Angular Velocity theta_dot [rad/s]"]
    end

    subgraph FeaturePipeline ["Feature Extraction & Delay History"]
        NORM_X["x / 2.4"]
        NORM_XDOT["clip(x_dot / 10.0, -1, 1)"]
        FEAT_COS["cos(theta)"]
        FEAT_SIN["sin(theta)"]
        NORM_THDOT["clip(theta_dot / 10.0, -1, 1)"]
        REG_U1["u_{t-1} (Previous Action)"]
        REG_U2["u_{t-2} (Two-Step Action)"]

        RAW_X --> NORM_X
        RAW_XDOT --> NORM_XDOT
        RAW_THETA --> FEAT_COS
        RAW_THETA --> FEAT_SIN
        RAW_THETADOT --> NORM_THDOT
    end

    subgraph ControllerCore ["ANSI C99 Forward Pass (neural_controller_step)"]
        IN_VEC["Input Vector: x[0..6] (7 floats)"]
        HID_LAYER["Hidden Layer (8 Neurons with tanh activation)<br/>h_j = tanh(sum_i W1_{ij} * x_i + b1_j)"]
        OUT_NEURON["Output Neuron (1 Neuron with tanh activation)<br/>y = tanh(sum_j W2_j * h_j + b2)"]
        MEM["Memory Contract: Stack = 64 Bytes | Heap = 0 Bytes"]

        NORM_X --> IN_VEC
        NORM_XDOT --> IN_VEC
        FEAT_COS --> IN_VEC
        FEAT_SIN --> IN_VEC
        NORM_THDOT --> IN_VEC
        REG_U1 --> IN_VEC
        REG_U2 --> IN_VEC

        IN_VEC --> HID_LAYER --> OUT_NEURON
    end

    subgraph ActuatorPipeline ["Actuation & Register Shift"]
        FORCE["Force Output: F = 20.0 * y [N] (Clamped to [-20, 20])"]
        UPDATE_REGS["Update History Registers:<br/>u_{t-2} <- u_{t-1}<br/>u_{t-1} <- y"]
        OUT_NEURON --> FORCE
        OUT_NEURON --> UPDATE_REGS
        UPDATE_REGS -.-> REG_U1
        UPDATE_REGS -.-> REG_U2
    end
```

### Key Embedded Contracts:
1. **Zero Dynamic Allocation:** No calls to `malloc`, `free`, or runtime heap managers.
2. **Static Bounded Stack:** Exactly 64 bytes of stack space required for local activation buffers.
3. **Deterministic Numerical Parity:** Max action discrepancy between the Python reference and native compiled C99 library is **$< 1.25 \times 10^{-6}$** across 50,000 closed-loop steps.
4. **Desktop Host Latency:** Median execution time of **$2.50\ \mu\text{s}$** measured via OS high-resolution timers on an x86_64 host (0.0125% of a 50 Hz control period).
5. **Hardware Scope Limitation:** Actual physical MCU execution (ESP32/STM32) has not been measured; physical deployment remains an open gate.

---

## 6. Cryptographic Artifact Integrity Ledger

Every published architecture specification, model checkpoint, and evaluation report maps to exact raw file bytes:

| Artifact Description | Canonical File Path | Size (Bytes) | Exact SHA-256 Digest |
| :--- | :--- | :---: | :--- |
| **Frozen CartPole Candidate** | `results/stage3d/models/stage3d_nsga2_robust_seed46.json` | 4,076 | `0C9A57692131972D52F75BD633A64BAE2A34A7B4EC2FB94C56041065C58A2B0E` |
| **DAgger Teacher Weights** | `results/mlp64_baseline.pt` | 21,070 | `2CD704F0064171190DAAC270DE317A37FCDF8A55CB884DA6EDB6DC744F409842` |
| **v4 Learned Baseline Seed 42** | `work/phase3e_followup_6h/sprint3/project_closeout/v4_baseline/checkpoints/baseline_teacher_init_seed42.json` | 134,705 | `B86C454CD82404FA88C00B5B11B8BF7A9CF50608F7969EAE289356396FFDCEB3` |
| **v4 Learned Baseline Seed 43** | `work/phase3e_followup_6h/sprint3/project_closeout/v4_baseline/checkpoints/baseline_teacher_init_seed43.json` | 134,699 | `4B8D149B9057EE5169EBD4F7250DE1DE867E2596BA9CD757E4EB2DF82EE8ECBC` |
| **v4 Learned Baseline Seed 44** | `work/phase3e_followup_6h/sprint3/project_closeout/v4_baseline/checkpoints/baseline_teacher_init_seed44.json` | 134,710 | `6AE128BE1176BF39FBF51D0455B771F5BC4AE6D6497B0ECF9B5BFEC282A81180` |
| **Acrobot Compact Seed 44 v2** | `work/phase3e_followup_6h/sprint3/research_upgrade/second_task/checkpoints/acrobot_compact_seed44.json` | 2,162 | `DB282929979BB141C06A5FDFECEBF1B237CF399434D45115EB0B8904857E9387` |
| **Acrobot MLP64 Seed 43 v2** | `work/phase3e_followup_6h/sprint3/research_upgrade/second_task/checkpoints/acrobot_mlp64_seed43.json` | 124,157 | `7F3A66F14178AA466C17F2587DF9523CFD867E19E2CA4ECCC83D732510A3A83B` |
| **ANSI C99 Export Source** | `results/deployment/neural_controller.c` | 3,923 | `21B21ACF0C962888D79EEB1DE2DFFC62F8A437C0E4BFA069502C66A96E7C418F` |
| **CartPole Raw Telemetry CSV** | `work/phase3e_followup_6h/sprint3/project_closeout/v4_baseline/reports/paired_episodes.csv` | 50,079 | `9944CDD92C451D447833C1488B27357F2DC0CD28B8413284B9F11394B1B85D83` |
| **Acrobot Raw Telemetry CSV** | `work/phase3e_followup_6h/sprint3/research_upgrade/second_task/results/acrobot_episodes_v2.csv` | 163,269 | `423DBAD6A86D497313DCFF2942990B6FDBD27A694480BA98E3410956AA0A2638` |
