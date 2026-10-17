# EvoEdge — Acceptance Contract & Phase-Gate Specification

**Document Version:** 2.4-PROPOSED (Protocol Reconciliation, Freeze, and Fair Controller Comparison)  
**Effective Date:** October 13, 2026  
**Document Status:** Proposed Governance Specification (Awaiting Operator Sign-Off)  
**Authority:** Senior GPU Systems Engineer, Control Systems Researcher & Reproducibility Auditor  
**Scope:** Formal criteria governing controller acceptance, baseline comparisons, edge deployment, budget accounting, statistical inference, and scientific integrity in EvoEdge. Reconciles Stage 2 proposal (`docs/proposals/stage2_spec.md`) with existing codebase and historical ledgers.

---

## 1. Executive Summary & Evidence-Based Starting Position

### 1.1 Historical Context & Empirical Ledger
The EvoEdge project investigates whether neuroevolutionary algorithms can synthesize compact, edge-deployable feedback controllers for the nonlinear CartPole swing-up problem that match or exceed classical control and dense deep reinforcement learning / evolutionary strategies baselines under physical perturbations.

Auditing the completed project artifacts (including Phase 1, Phase 2A/B/C, Phase 3A, Stage 1 baseline, and backend audits) establishes the following empirical starting position:

1. **Nominal Evolution Gate (Historical Gate 0) [`HISTORICAL (Phase 3A)`]:**
   - In Phase 3A, the nominal evolution gate passed on validation seeds `200–249` with a non-identical warm-started descendant scoring **50 / 50 (100.0%)** successes.
   - The exact 95% Wilson score confidence interval for this historical result is **$[92.89\%, 100.00\%]$** (approximately $[92.9\%, 100\%]$).
   - *Scientific Invariant:* **Nominal swing-up success is NOT evidence of robust control.** A controller capable of swinging up an unperturbed CartPole frequently collapses under minor viscous friction or sensor noise.
   - *Partition Non-Reusability:* This 50/50 nominal result was an isolated historical check on seeds `200–249`. It **cannot** be reused or cited to satisfy the separate prospective nominal acceptance criterion ($\ge 90/100$) on a newly authorized evaluation partition.

2. **From-Scratch Structural Neuroevolution (`Arm A`) [`HISTORICAL (Phase 3A)`]:**
   - Evolving topology and weights from scratch without teacher guidance or imitation seeding achieved **0.00% robust success** across all five evaluated optimizer seeds (Seeds 42–46).
   - Under Deb-style constrained tournament selection, randomly initialized topologies failed to discover the narrow upright basin of attraction under combined physical friction within the allocated generation budget. All candidates remained strictly infeasible ($f_{\text{robust}} = 0.0\% < 48.32\%$).

3. **Warm-Started Structural Neuroevolution (`Arm B`) [`HISTORICAL (Phase 3A)`]:**
   - Initializing structural mutation from a frozen compact $H=8$ teacher checkpoint achieved a run-level mean robust success of **$66.19\% \pm 5.44\%$** across five optimizer seeds on Phase 3A validation partition seeds `250–399`.
   - Seed-level outcomes on seeds `250–399`: Seed 42 = 65.33%, Seed 43 = **73.67%** (peak), Seed 44 = 60.40%, Seed 45 = **71.13%**, Seed 46 = 60.40%.
   - Evolved controllers maintained the exact baseline physical footprint: **57 active parameters and 48 active weight MACs**.

4. **Stage 1 Classical Baseline Evaluation [`EXECUTED TEST & DESCRIPTIVE BASELINE`]:**
   - In Stage 1, the analytical Energy-Shaping + Linear Feedback controller (`HybridController`) was evaluated across the 10 diagnostic conditions on Phase 3A validation partition seeds `250–399` (150 seeds per condition = 1,500 episodes).
   - `HybridController` achieved an aggregate robust success rate of **89.67%** (95% bootstrap CI $[88.20\%, 91.07\%]$), with a worst condition of **56.67%** on `7_CombinedHistoricalStress`.
   - This provides the descriptive reference against historical Arm B neuroevolution (73.67% peak, but 0.00% worst condition on joint friction).

5. **Frozen Baselines and Historical Champions [`VERIFIED (Documentation Audit)`]:**
   - The frozen compact neural baseline (`results/champion_neural_h8.pkl`, 60.40% robust success, SHA256: `d16035a5...`) and historical checkpoints must remain strictly unchanged.
   - These historical outcomes do **not** establish that the full problem statement has been solved. Evolved improvements remain validation-split observations that have not been confirmed on an independent holdout.

### 1.2 Strict Provenance Labeling Framework
To prevent misleading claims regarding what evolution "independently discovered," every model, checkpoint, and headline result reported in EvoEdge must explicitly carry one of three mandatory provenance labels:

- **`FROM-SCRATCH`:** Initialization contains zero teacher-derived controller weights, topology templates, or teacher-generated policy outputs. Search begins from random connections/weights or minimal perceptron topologies.
- **`TEACHER-ASSISTED`:** Architecture or parameterization utilizes teacher weights, imitation pre-training, DAgger data collection, teacher-generated targets, or warm-started seed checkpoints.
- **`OTHER-INITIALIZATION`:** Employs an alternative documented initialization scheme (e.g., analytical LQR initialization, system-identification warm-starts). Must be explicitly detailed.

*Binding Rule:* **Do not describe a `TEACHER-ASSISTED` outcome as evidence that evolution independently discovered robust swing-up control.** Each experimental arm must be evaluated, labeled, and reported separately.

---

## 2. Gate 1 — The "Works" Gate (Nominal Baseline & Dynamics Parity)

Before any optimization or perturbation evaluation is considered valid, the candidate controller and execution engines must pass the nominal functionality gate.

### 2.1 Physics & Baseline Pre-Condition Gate (Gate 1A)
Before running evolutionary search, the simulation engines must be verified against analytical standards:
1. **Float64 Simulator Parity [`VERIFIED (Executed Tests)`]:** Serial reference dynamics (`PerturbedCartPoleEnv`) and vectorized batch engine (`VectorizedCartPoleEnv`) must achieve a maximum absolute float64 state error below **$10^{-9}$** ($|x_{\text{vec}} - x_{\text{ref}}| < 10^{-9}$) across all 4 state components $[x, \dot{x}, \theta, \dot{\theta}]$ over 500 steps. Verified in `tests/test_phase_3a_evolution.py` (measured max diff $= 0.00\text{e}+00$).
2. **Trusted Controller Verification [`VERIFIED (Executed Tests)`]:** A known trusted controller (the frozen $H=8$ baseline or the analytical Energy-Shaping + Linear Feedback controller) must be evaluated in the environment to confirm that nominal swing-up and balance are achievable. (Frozen $H=8$ achieves 100% nominal success).
3. **Circuit Breaker:** If the trusted controller fails under nominal dynamics, execution must **STOP IMMEDIATELY**. The simulation physics, integrator step, coordinate wrap, and action application must be diagnosed before any evolutionary experiments proceed.

### 2.2 Task Specification & Exact Success Predicate
The official task definition and boundary conditions are frozen as follows:
- **Episode Horizon:** Exactly 500 discrete control timesteps ($\Delta t = 0.02\text{ s}$, 10.0 seconds total).
- **Cart Rail Boundary Violation:** $|x| < 2.4\text{ m}$ strictly enforced at every timestep. Any violation $|x| \ge 2.4\text{ m}$ immediately terminates the episode with failure (`boundary_violated = True`).
- **Official Upright Balance Predicate:** $\cos(\theta) > 0.9$ strictly (corresponding to $|\theta| < 0.451\text{ rad} \approx 25.84^\circ$) for at least **95 out of the final 100 steps** ($t \in [400, 499]$).
- **Secondary Metric (Diagnostic Only):** Sustained precision balance within $|\theta| \le 12^\circ$ ($\approx 0.209\text{ rad}$) for at least 3.0 consecutive seconds (150 steps). This measure is recorded strictly as diagnostic telemetry; it must **never** replace or silently alter the official 500-step predicate.
- **Physical Constants & Equations of Motion:**
  - Cart mass $m_c = 1.0\text{ kg}$, pole mass $m_p = 0.1\text{ kg}$, half-pole length $l = 0.5\text{ m}$, gravity $g = 9.8\text{ m/s}^2$.
  - Numerical integration: Semi-implicit Euler integration ($\dot{x}_{t+1} = \dot{x}_t + \Delta t \ddot{x}_t$; $x_{t+1} = x_t + \Delta t \dot{x}_{t+1}$).
  - Angle wrapping: $\theta \in [-\pi, \pi]$ via $((\theta + \pi) \pmod{2\pi}) - \pi$.
- **Observation Normalization:** Exactly 5 input channels:
  $$\text{obs} = \left[ \frac{x}{2.4}, \, \text{clip}\left(\frac{\dot{x}}{10.0}, -1.0, 1.0\right), \, \cos\theta, \, \sin\theta, \, \text{clip}\left(\frac{\dot{\theta}}{10.0}, -1.0, 1.0\right) \right]$$
- **Actuation Limits:** Continuous control signal $u \in [-1.0, 1.0]$, mapped to horizontal force $F = \text{clip}(u, -1.0, 1.0) \times 10.0\text{ N}$.
- **Reset Distribution:** Initial state sampled uniformly from $x \in [-0.05, 0.05]\text{ m}$, $\dot{x} \in [-0.05, 0.05]\text{ m/s}$, $\theta \in [\pi - 0.05, \pi + 0.05]\text{ rad}$ (downward hanging), $\dot{\theta} \in [-0.05, 0.05]\text{ rad/s}$.

### 2.3 Nominal Acceptance Criterion (Gate 1B)
- **Criterion:** At least **90 successes in 100 preregistered nominal evaluation episodes** ($\ge 90.0\%$) for at least one EvoEdge-generated controller on an authorized development nominal split.
- **Sample Size & Distinct Seed Protocol:**
  - *Determinism and Independence Requirement:* In `CartPoleSwingUpEnv.reset(seed=s)`, the initial state perturbation is sampled deterministically by Gymnasium's PRNG (`np_random.uniform(low=-0.05, high=0.05, size=(4,))`). Because nominal dynamics (`1_Nominal`) possess zero physical noise ($\sigma = 0$) and deterministic semi-implicit Euler integration, evaluating a deterministic policy on the same seed $s$ yields identical initial conditions and bit-exact trajectories. Repeated evaluations of identical seeds do **NOT** constitute independent samples. Presenting repeated deterministic trials as independent samples is scientifically invalid.
  - *Authorized Gate 1B Seed Partition:* To obtain 100 genuinely independent nominal episodes, Gate 1B is explicitly allocated **seeds `200–299`** (exactly 100 distinct seed values within the development validation partition).
  - *Seed Provenance & Scientific Role:* Seeds `200–299` are **previously exposed development data, NOT fresh prospective evidence**. Seeds 200–249 were historically exposed during Phase 3A Gate 0, and seeds 250–299 were part of the historical Phase 3A validation partition. Consequently, Gate 1B functions strictly as a **development screening gate** to verify basic swing-up competence before proceeding. It does NOT provide confirmatory evidence.
  - *Separation from Historical Gate 0:* Historical Gate 0 evaluated seeds `200–249` (only 50 distinct seeds, 50/50 successes). This historical observation cannot be reused or doubled to satisfy Gate 1B.
  - *Interaction Budget Reconciliation:* Evaluating 1 shortlisted candidate across 100 distinct nominal seeds (`200–299`) for 500 steps/episode consumes at most **$50,000\text{ transitions}$**, fitting strictly within the allocated $150,000\text{ transition}$ validation budget.
- **Reporting Requirements:** The report must provide:
  - Exact success count and success rate ($k / 100$).
  - 95% Wilson score confidence interval.
  - Exact breakdown of failures: rail boundary collisions ($|x| \ge 2.4$), failure to reach upright within 400 steps, or failure to maintain balance for $\ge 95$ steps.
- **Strict Partition Role:**
  - Historical Gate 0 (50/50 on seeds 200–249) does not satisfy Gate 1B.
  - Gate 1B must **NOT** access, pollute, or burn the locked confirmation partitions (`600–749` or `2000–2099`). Gate 1B is strictly a development screening gate executed on development seeds `200–299` prior to freezing the candidate.
- **Current Gate Status:** **`NOT RUN`** (awaiting prospective execution on authorized development partition seeds `200–299`).

---

## 3. Gate 2 — The "Robustness & Fair-Comparison" Gate

Robustness claims are scientifically meaningless in the absence of rigorous classical baselines, budget-matched neural comparators, and preregistered physical stress distributions.

### 3.1 Classical Control Reference Architecture
1. **Principal Classical Reference: Energy-Shaping + Linear Feedback (`HybridController`)**
   - The standard classical benchmark for CartPole swing-up consists of nonlinear mechanical energy-shaping for swing-up coupled with linear state feedback for local stabilization:
     $$u_{\text{pump}} = \text{clip}\left( k_E \cdot (E - E_0) \cdot \text{sign}(\dot{\theta} \cos\theta) - k_x x - k_v \dot{x}, -1.0, 1.0 \right)$$
     $$u_{\text{balance}} = \text{clip}\left( -(K_x x + K_{\dot{x}} \dot{x} + K_\theta \theta + K_{\dot{\theta}} \dot{\theta}), -1.0, 1.0 \right)$$
   - Switching condition: Hand-off occurs when $|\theta| < 0.35\text{ rad}$ and $|\dot{\theta}| < 2.50\text{ rad/s}$.
   - Complexity: 7 float gains, 7 MACs (in addition to mechanical energy and trigonometric operations).
2. **Diagnostic Linear Comparator: Classical LQR**
   - Pure LQR linearized around the upright equilibrium ($\theta = 0$).
   - *Diagnostic Role:* LQR provides a baseline for local linear regulatory stiffness and jitter, but has **0% swing-up capability** from the downward hanging position ($\theta \approx \pi$). It must be included for local balance comparisons, but **cannot** serve as the reference for defining a non-inferiority margin or success-rate threshold.

### 3.2 Historical Inventory & Legacy Partition Boundaries
1. **Historical Task B Benchmark (Seeds 1000–1099) [`HISTORICAL (Task B)`]:**
   - Evaluated under Protocol v2.4 across 8 conditions.
   - Energy-Shaping + Linear Feedback achieved **90.88% mean success**.
2. **Partition Inviolability [`VERIFIED (Documentation & Test Audit)`]:**
   - Seeds `1000–1099` represent a frozen legacy test partition. They must **NOT** be rerun, tuned against, or repurposed for exploratory development.
   - Seeds `2000–2099` represent the protected holdout partition. Code-level guards actively abort if accessed.
3. **Condition Suite Discrepancy:**
   - The historical 8-condition Task B suite combined rail and joint friction into a single joint perturbation, whereas Phase 3A established 10 distinct diagnostic conditions separating isolated rail friction (`4a`), isolated joint friction (`4b`), and combined friction (`4c`).
   - Therefore, historical 8-condition measurements **cannot** be directly substituted into the 10-condition Phase 3A diagnostic matrix.

### 3.3 Authoritative Phase 3A Perturbation Contract
The authoritative physical perturbation suite is transcribed verbatim from `config/perturbations.yaml` and `config/phase_3a_config.yaml`:

| Condition ID & Name | Physical Parameters & Values | Units | Noise Distribution & Scale | Actuator Delay Semantics | Horizon & Steps |
|---|---|---|---|---|---|
| **1_Nominal** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.00$ | $m$ in kg, $b$ in $\text{N}\cdot\text{s/m}$, $\text{N}\cdot\text{m}\cdot\text{s/rad}$ | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) |
| **2_PoleMass_+30%** | $m_c = 1.00$, $m_p = 0.130$, $b_c = 0.00$, $b_p = 0.00$ | $m_p$ in kg (+30%) | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) |
| **3_CartMass_+30%** | $m_c = 1.30$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.00$ | $m_c$ in kg (+30%) | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) |
| **4a_CartFriction_0.50** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.50$, $b_p = 0.00$ | $b_c$ in $\text{N}\cdot\text{s/m}$ | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) |
| **4b_JointFriction_0.010** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.010$ | $b_p$ in $\text{N}\cdot\text{m}\cdot\text{s/rad}$ | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) |
| **4c_CombinedFriction** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.50$, $b_p = 0.010$ | $b_c$ and $b_p$ combined | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) |
| **5_SensorNoise_0.05** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.00$ | Nominal mass/friction | Gaussian $\mathcal{N}(0, 0.05^2)$ on raw unnormalized obs | 0 steps (0 ms) | 500 steps (10 s) |
| **6a_ActuatorDelay_1Step** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.00$ | Nominal mass/friction | $\sigma = 0.00$ (None) | 1 step (20 ms FIFO queue) | 500 steps (10 s) |
| **6b_ActuatorDelay_2Step** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.00$ | Nominal mass/friction | $\sigma = 0.00$ (None) | 2 steps (40 ms FIFO queue) | 500 steps (10 s) |
| **7_CombinedHistoricalStress** | $m_c = 1.30$, $m_p = 0.130$, $b_c = 0.50$, $b_p = 0.010$ | All perturbations combined | Gaussian $\mathcal{N}(0, 0.05^2)$ on raw unnormalized obs | 1 step (20 ms FIFO queue) | 500 steps (10 s) |

---

## 4. Development-Confirmation Separation & Freeze Barrier

### 4.1 Strict Partition Isolation
To eliminate data leakage, overfitting, and post hoc candidate selection, the project lifecycle enforces complete separation between development and confirmatory activities:

```
┌────────────────────────────────────────────────────────────────────────┐
│ DEVELOPMENT PARTITION (Open Access)                                    │
│ • Seeds 0–199: Training rollouts & fitness evaluations.                │
│ • Seeds 200–399: Gate 1B nominal screening, validation tuning,         │
│   hyperparameter selection, and candidate Pareto front extraction.     │
│ • Invariant: All model tuning, selection, and debugging occur here.    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ CRYPTOGRAPHIC FREEZE BARRIER (Mandatory Gate)                          │
│ • Candidate controller weights, architecture, and topology serialized. │
│ • Execution scripts, evaluation protocol, and configs frozen.          │
│ • SHA-256 digests committed to audit ledger.                           │
│ • Invariant: ZERO changes permitted after crossing barrier.            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ CONFIRMATION PARTITION (Locked Single-Pass Access)                     │
│ • Provisional Seeds 600–749 (150 seeds × 10 conditions = 1,500 runs).  │
│ • Evaluated ONCE for the locked candidate, ES, and HybridController.   │
│ • Invariant: NEVER used for tuning, selection, screening, or retries.  │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Prohibited Confirmation Activities
The confirmation partition must **NEVER** be accessed or used for:
1. Gate 1B nominal functionality screening.
2. Candidate selection from a Pareto front or population.
3. Hyperparameter tuning, population sizing, or mutation rate adjustments.
4. Outcome inspection to guide debugging, architecture tweaks, or seed re-selection.
5. Post hoc selection of non-inferiority margins or statistical thresholds.

---

## 5. Exact Simulator-Transition Budget Accounting

### 5.1 Common 2,000,000 Transition Ceiling Contract
The interaction ceiling of $B_{\text{trans}} = 2,000,000\text{ actual simulator transitions}$ ($2.0 \times 10^6$) is formalized with unambiguous accounting rules:

1. **Ceiling Scope:** The budget ceiling applies **per optimizer run (per optimizer seed)**.
2. **Operations That Consume Budget:**
   - Every single physics integration step ($\Delta t = 0.02\text{ s}$) executed during training rollouts, fitness evaluations, population screening, candidate validation checks, and early-stopping rollouts increments the budget counter.
   - For an ES population of $P$ individuals evaluated over $H$ horizon steps per generation $g$, and an action repeat / sub-stepping factor $k_{\text{substeps}}$:
     $$\sum_{\text{evaluations}} \sum_{t=1}^{H} k_{\text{substeps}} \le B_{\text{trans}} = 2,000,000$$
3. **Operations That Do NOT Consume Budget:**
   - Static parameter counting, tensor weight initialization, gradient calculation / optimizer state updates, C code generation, and wall-clock time do not consume physics transitions.
4. **Counting Semantics:**
   - **Early Termination:** If an episode terminates early due to rail boundary violation ($|x| \ge 2.4\text{ m}$), the environment halts immediately, and only the actual executed integration steps prior to termination are billed to the counter.
   - **Vectorized Batch Rollouts:** In batch simulators that step all array slots without masking, every simulated slot is counted as an actual physics call unless early termination is masked out.
   - **Exceptions and Retries:** If an exception occurs, all transitions executed prior to the exception remain billed. Retries do **not** reset the counter.
5. **Fail-Closed Enforcement:**
   - Implemented in `env/budget_tracker.py` (`TransitionBudgetTracker`, `BudgetExhaustedError`, `BudgetedEnv`, `BudgetedVectorizedEnv`).
   - The environment step hook checks the atomic transition counter on every step. When the counter reaches $B_{\text{trans}}$, it raises a catchable `BudgetExhaustedError`.
   - The training orchestrator catches `BudgetExhaustedError`, immediately halts search, and serializes the current best candidate using development data only.
   - Any step attempted after exhaustion is blocked fail-closed without executing physics transitions.
6. **Multi-Seed Protocol & Aggregate Budget:**
   - Each method is evaluated across $S_{\text{opt}} = 5$ independent optimizer seeds (Seeds 42, 43, 44, 45, 46).
   - Total permitted training budget across all 5 runs: $5 \times 2,000,000 = \mathbf{10,000,000\text{ actual transitions}}$ per method.
7. **Comparable Candidate Selection Opportunities:**
   - Both algorithms evaluate candidates exclusively on the development partition (training seeds `0–199`, validation seeds `200–399`). Candidate selection is finalized before crossing the freeze barrier.

### 5.2 Exact Transition Budget Breakdown & Arithmetic Reconciliation
To resolve the discrepancy between proposed validation across seeds `250–399` (150 seeds $\times$ 10 conditions = 1,500 episodes = 750,000 transitions per candidate) and the allocated 150,000 transition validation ceiling, the budget is reconciled into closed-form arithmetic:

| Budget Component | Allocation per Run | Episode Horizon ($H$) | Max Episodes | Arithmetic & Evaluation Configuration |
|---|---|---|---|---|
| **Search & Fitness Evaluation** | **1,850,000 transitions** | $H \le 500$ steps | Exactly 3,700 episodes (370 evals) | **ES (Option A):** Exactly 370 candidate evaluations $\times$ 10 training episodes $\times$ 500 steps = **1,850,000 transitions**:<br>• Initial center evaluation: $1\text{ candidate} \times 10\text{ eps} \times 500 = \mathbf{5,000\text{ transitions}}$.<br>• Generational mutants: $18\text{ gens} \times 20\text{ mutants} \times 10\text{ eps} \times 500 = \mathbf{1,800,000\text{ transitions}}$ (360 evals).<br>• Biennial center tracking: $9\text{ biennial evaluations (gens 2, 4, ..., 18)} \times 10\text{ eps} \times 500 = \mathbf{45,000\text{ transitions}}$ (9 evals).<br>• Subtotal: $5,000 + 1,800,000 + 45,000 = \mathbf{1,850,000\text{ transitions}}$.<br>**NSGA-II:** Exactly 370 candidate evaluations $\times$ 10 training episodes $\times$ 500 steps = **1,850,000 transitions**:<br>• Generation 0 (initial pop): $P = 20\text{ genomes} \times 10\text{ eps} \times 500 = \mathbf{100,000\text{ transitions}}$ (20 evals).<br>• Generations 1–35: $35\text{ gens} \times 10\text{ offspring} \times 10\text{ eps} \times 500 = \mathbf{1,750,000\text{ transitions}}$ (350 evals).<br>• Subtotal: $100,000 + 1,750,000 = \mathbf{1,850,000\text{ transitions}}$. |
| **Gate 1B Nominal Screening** | **50,000 transitions** | $H \le 500$ steps | Exactly 100 episodes | 1 shortlisted candidate $\times$ 1 condition (`1_Nominal`) $\times$ **100 distinct seeds (`200–299`)** = $100 \times 500 = \mathbf{50,000\text{ transitions}}$.<br>*(Development screening only; previously exposed development data, not fresh prospective evidence).* |
| **Multi-Condition Diagnostic Validation** | **100,000 transitions** | $H \le 500$ steps | Exactly 200 episodes | **Exact Validation Subset:** Top 1 primary champion candidate evaluated across all 10 diagnostic conditions on **20 distinct validation seeds (`300–319`)**:<br>$1\text{ candidate} \times 10\text{ conditions} \times 20\text{ seeds} = 200\text{ episodes} = \mathbf{100,000\text{ transitions}}$.<br>*(Development screening only; previously exposed development data, not fresh prospective evidence).* |
| **Total Validation Budget** | **150,000 transitions** | $H \le 500$ steps | Exactly 300 episodes | $50,000\text{ (Nominal)} + 100,000\text{ (Multi-Condition)} = \mathbf{150,000\text{ transitions}}$. |
| **Total Budget Per Run** | **2,000,000 transitions** | $H \le 500$ steps | $\le 4,000$ episodes | $1,850,000\text{ (Search)} + 150,000\text{ (Validation)} = \mathbf{2,000,000\text{ transitions}}$. |
| **Total Across 5 Optimizer Runs** | **10,000,000 transitions** | $H \le 500$ steps | $\le 20,000$ episodes | $5\text{ runs} \times 2,000,000\text{ transitions} = \mathbf{10,000,000\text{ transitions}}$ per method. |

- **Maximum vs. Expected Transitions per Episode:**
  - *Maximum Transitions:* 500 steps (fixed horizon, $10.0\text{ s}$).
  - *Expected Transitions:* ~150–350 steps in early generations (due to early boundary violations $|x| \ge 2.4\text{ m}$), increasing toward 500 steps in late generations as balance capability stabilizes. Unused transitions remain in reserve or terminate early; the 2,000,000 ceiling is a strict upper bound.
- **Budget Wrapper Integration:** Both training entry points (`evolution/evolution_strategies.py::train_es` and `evolution/nsga2_engine.py::train_nsga2`) instantiate `TransitionBudgetTracker`, `BudgetedEnv`, and `BudgetedVectorizedEnv`. When the ceiling is reached, a catchable `BudgetExhaustedError` halts search cleanly and returns the best candidate found. Executions attempted after exhaustion are blocked fail-closed.

### 5.3 Candidate Selection Procedure & Proposed Constraint Status
To freeze the candidate selection procedure without silently promoting unapproved thresholds to formal gates, the engineering rationales and approval statuses are formally cataloged:

1. **Parameter Count $\le 128$:**
   - *Engineering Rationale:* Microcontroller SRAM / cache constraint (target: ESP32 Xtensa LX6 @ 240 MHz). In IEEE 754 float32, 128 parameters occupy 512 bytes, ensuring a negligible memory footprint (< 0.2% of usable DRAM). Matches TinyML architectures (compact $H=8$ has 57 parameters; $H=16$ has 113 parameters) while preventing parameter bloat.
   - *Approval Status:* **`PROPOSED (Flagged for Operator Sign-Off Before Full Training)`**.
2. **Weight MAC Count $\le 128$:**
   - *Engineering Rationale:* Real-time execution latency constraint. On ESP32 Xtensa LX6 without SIMD vector extensions, 128 MACs execute in ~500–1,000 CPU cycles (< 5 $\mu$s @ 240 MHz), consuming less than 0.025% of a 50 Hz (20 ms) control loop. Dense MLPs (e.g. 64×64 MLP with 4,480 MACs) require $35\times$ more compute.
   - *Approval Status:* **`PROPOSED (Flagged for Operator Sign-Off Before Full Training)`**.
3. **Worst-Condition Success Rate $S_c^{\min} \ge 40.0\%$:**
   - *Engineering Rationale:* Prevents aggregate robustness from concealing catastrophic failure on individual physical stress regimes. In Phase 3A, Arm B achieved 73.67% aggregate success but collapsed to **0.00% on joint friction (`4b_JointFriction_0.010`)**. The classical `HybridController` achieves 56.67% on its worst condition (`7_CombinedHistoricalStress`). Setting a floor of $\ge 40\%$ guarantees minimum stability under joint friction, rail friction, delays, and sensor noise.
   - *Approval Status:* **`PROPOSED (Flagged for Operator Sign-Off Before Full Training)`**.
4. **Primary Ranking Criterion on the Development Pareto Set:**
   - *Engineering Rationale:* Development multi-objective optimization (NSGA-II) produces a non-dominated Pareto front. To select a single candidate to freeze prior to confirmatory evaluation without post hoc cherry-picking:
     - *Step 1 (Feasibility Filter):* Retain only candidates satisfying active parameters $\le 128$, weight MACs $\le 128$, and validation worst-condition success $S_{\text{worst}} \ge 40.0\%$.
     - *Step 2 (Primary Objective):* Rank feasible candidates primarily by **aggregate robust success ($S_{\text{agg}}$) across the 10 diagnostic conditions on the validation subset (`300–319`)**.
     - *Step 3 (Deterministic Tie-Breakers):* If candidates are tied on $S_{\text{agg}}$ within 0.1%: (1) fewest weight MACs, (2) fewest active parameters, (3) highest worst-condition success ($S_{\text{worst}}$).
   - *Approval Status:* **`PROPOSED (Flagged for Operator Sign-Off Before Full Training)`**.

### 5.4 Optimizer-Seed Independence & Isolation Plan
The plan evaluates 5 optimizer seeds: `42, 43, 44, 45, 46`. Because these numeric values also appeared in Phase 3A, the following isolation protocol is registered:

1. **Documented Integer Namespaces & Effective RNG Construction:**
   - NumPy's `SeedSequence` requires integer argument types. To guarantee complete mathematical stream isolation without invalid string conversions, integer namespaces are formally registered:
     - **Evolution Strategies (ES):** $\text{NAMESPACE\_ES} = 1$. Effective construction:
       $$\text{RNG}_{\text{ES}} = \text{np.random.default\_rng}(\text{SeedSequence}(\text{entropy}=s_{\text{opt}}, \text{spawn\_key}=(1, s_{\text{opt}})))$$
     - **NSGA-II Engine:** $\text{NAMESPACE\_NSGA2} = 2$. Effective construction:
       $$\text{RNG}_{\text{NSGA2}} = \text{np.random.default\_rng}(\text{SeedSequence}(\text{entropy}=s_{\text{opt}}, \text{spawn\_key}=(2, s_{\text{opt}})))$$
   - *Mathematical Guarantees:* For the same run seed $s_{\text{opt}}$, each algorithm initializes deterministically and repeats bit-identically across independent calls. Because spawn keys $(1, s_{\text{opt}})$ and $(2, s_{\text{opt}})$ differ in their primary coordinate, their assembled initial hashing states are mathematically distinct, completely preventing stream collisions between algorithms and legacy Phase 3A runs. Verified by 9/9 passing tests in `tests/test_training_budget_integration.py`.
2. **Decoupling of Optimizer and Environment Evaluation Seeds:**
   - Optimizer randomness governs only policy mutations, initial weight generation, and ES exploration vectors.
   - Environment rollout initial conditions and observation noises are strictly governed by the registered evaluation seed arrays (`seeds=TRAIN_SEEDS` or `seeds=VAL_SEEDS`).
   - The optimizer RNG is never advanced by environment dynamics, and environment perturbations never draw from the optimizer RNG.
3. **Directory and Checkpoint Isolation:**
   - All Stage 2 checkpoints, logs, and models are directed strictly to `results/stage2/checkpoints/` and `results/stage2/models/`.
   - Historical directories (`results/phase_3a/checkpoints/`) are strictly read-isolated. Stage 2 checkpoint files include mandatory header metadata `{"stage": 2, "protocol_version": "2.4"}`; loaders reject any checkpoint missing this tag.

### 5.5 Two Formal Comparison Options
- **Option A (Recommended — Fresh Matched-Budget Comparison):**
  - Both Evolution Strategies (ES) and compact neuroevolution are trained under the exact same $2,000,000$ actual transition budget per run.
  - Designated: `EQUAL-INTERACTION-BUDGET`.
- **Option B (Descriptive Unequal-Budget Comparison):**
  - The newly trained ES baseline ($2.0 \times 10^6$ transitions) is compared against historical Phase 3A neuroevolution ($112.5 \times 10^6$ scheduled loop slots).
  - Designated: **`UNEQUAL-BUDGET DESCRIPTIVE EVIDENCE`** (must never be described as equal-compute).

---

## 6. Minimal Executable Statistical Decision Protocol

### 6.1 Statistical Objectives & Boundaries
To maintain scientific rigor without unnecessary complexity, the statistical protocol focuses strictly on binary official task success:
- **Primary Estimands:**
  - Official success rate for each method, aggregated with equal weight across the 10 diagnostic conditions:
    $$S_{\text{agg}} = \frac{1}{10} \sum_{c=1}^{10} S_c$$
  - Condition-level numerators, denominators, and success rates ($k_c / n_c$).
  - Paired candidate-minus-reference aggregate difference: $\bar{D}_{\text{agg}} = S_{\text{agg}}(\text{cand}) - S_{\text{agg}}(\text{ref})$.
  - Paired difference between ES and existing neuroevolution.
- **Uncertainty Quantification:**
  - **Individual Condition Rates:** Evaluated via **95% Wilson score confidence intervals** on binomial episode counts ($k_c / n_c$).
  - **Aggregate Success Rate & Differences:** Because $S_{\text{agg}}$ is an equally weighted average across 10 correlated physical conditions sharing the same underlying seed dynamic perturbations, **it cannot be treated as an ordinary binomial proportion for Wilson-interval purposes**.
  - All aggregate uncertainty intervals and candidate-minus-reference differences are evaluated exclusively via a **paired seed-cluster bootstrap** ($B = 10,000$ resamples over evaluation seeds, fixed PRNG seed 42) that preserves all 10 condition outcomes within each sampled seed.

### 6.2 Classical Non-Inferiority Testing Rule
- **Statistical Framework:** One-sided non-inferiority testing (distinguished strictly from two-one-sided-tests [TOST] equivalence testing).
- **Preregistered Engineering Margin:** $\delta_{\text{agg}} = -0.05$ ($-5.0$ percentage points).
  - *Methodological Basis:* This margin is an a priori engineering tolerance specifying the maximum acceptable performance degradation relative to classical control. It is **NOT** derived from observed baseline sample variance.
- **Pass Rule:** The candidate passes aggregate classical non-inferiority if and only if the lower one-sided 95% bootstrap confidence bound on the paired difference satisfies:
  $$L_{\text{agg}, 95} \ge -0.05 \quad (\text{lower 95\% bound on candidate } - \text{ reference is } \ge -5.0\%)$$
- **Condition-Level Reporting:** Aggregate non-inferiority does **not** certify robustness across all conditions. All 10 individual condition differences, uncertainty bounds, and failure modes must be tabulated explicitly. Near-total collapse on any individual condition (e.g. 0% success on joint friction) must be highlighted as a disqualifying limitation.

---

## 7. Edge Deployment Target Specification

### 7.1 MCU Architecture Disambiguation
The research proposal specifies an "ESP32-class MCU (Xtensa LX7 / RISC-V, 512 KB SRAM)". To eliminate architectural ambiguity across disparate hardware platforms:
- **Original ESP32 (Recommended Target):**
  - Architecture: Dual-core Xtensa LX6 @ 240 MHz.
  - Hardware FPU: Single-precision IEEE 754 floating-point unit.
  - Internal SRAM: 520 KB (typically ~320 KB usable contiguous DRAM).
  - ROM: 448 KB.
- **ESP32-S3:**
  - Architecture: Dual-core Xtensa LX7 @ 240 MHz with custom vector instructions (AI/DSP extensions).
- **ESP32-C3 / ESP32-C6:**
  - Architecture: Single-core 32-bit RISC-V (RV32IMC / RV32IMAC) @ 160 MHz. (ESP32-C3 lacks a hardware FPU; floating-point operations run via software emulation).

### 7.2 Evidence Hierarchy
1. **Analytical / Static Complexity:** Parameter count, active weight MACs, static Flash bytes, and static BSS/stack memory.
2. **Desktop Emulation:** Cycle counts and timing simulated in QEMU or Wokwi virtual environments.
3. **Physical Bench Measurements:** True oscilloscope-measured or GPIO-toggled control loop latency, current draw, and power consumption on a physical dev board.
*Binding Rule:* All reported numbers must state their exact position in this hierarchy. Do not claim physical hardware deployment or measured latency without actual physical device logs.

---

## 8. Confirmation Partition Eligibility & Seed Ledger Audit

### 8.1 Evidence-Backed Seed Ledger
Based on the repository-wide audit, the allocation of pseudo-random seeds is codified:

| Seed Range | Sample Count | Historical / Operational Status | Eligibility for Confirmation | Audited Sources & Known Limitations |
|---|---|---|---|---|
| **`0–199`** | 200 seeds | Training & Fitness Rollouts | **BURNED (Ineligible)** | `config/perturbations.yaml`, `config/phase_3a_config.yaml`, `experiments/run_phase_3a_evolution.py` |
| **`200–299`** | 100 seeds | **Gate 1B Nominal Screening Partition** | **DEVELOPMENT SCREENING ONLY (Previously Exposed Development Data, Ineligible for Confirmation)** | Seeds 200–249 historically exposed in Phase 3A Gate 0 (50/50); seeds 250–299 exposed in Phase 3A validation. Development screening only; NOT fresh prospective evidence. |
| **`300–319`** | 20 seeds | **Multi-Condition Diagnostic Validation Subset** | **DEVELOPMENT SCREENING ONLY (Previously Exposed Development Data, Ineligible for Confirmation)** | Part of historical Phase 3A validation partition (250–399). Evaluates shortlisted candidate on 10 conditions $\times$ 20 seeds = 200 episodes ($100,000$ transitions). NOT fresh prospective evidence. |
| **`320–399`** | 80 seeds | Development Validation Buffer | **DEVELOPMENT SCREENING ONLY (Ineligible for Confirmation)** | Part of historical Phase 3A validation partition. Reserved for tie-breaking or development hyperparameter diagnostics. |
| **`400–599`** | 200 seeds | Unused buffer (isolated references to 400/500 in scripts) | **BUFFER (Avoid)** | Literal hits in `scratch/check_seeds_400_999.py`, `evolution/evolve_tinyml.py` |
| **`600–749`** | 150 seeds | **Proposed Confirmation Partition** | **PROVISIONAL & INACCESSIBLE (Pending Operator Ratification)** | Zero literal matches in active code, configs, logs, and CSVs. *Limitation:* Does not prove unlogged runs were never executed out-of-band. Strictly locked until pre-training authorization. |
| **`1000–1099`** | 100 seeds | Frozen Legacy Task B Benchmark | **BURNED / LEGACY (Ineligible)** | `config/perturbations.yaml`, historical Task B reports |
| **`2000–2099`** | 100 seeds | Frozen Protected Holdout Partition | **STRICTLY RESERVED & UNTOUCHABLE** | Code-level assertion in `utils/split_integrity.py`, `test_cuda_evaluator.py`. Primary locked evaluation holdout. |

### 8.2 Scope of Seed Audit & Known Uncertainties
- **Sources Examined:** All active `.py`, `.yaml`, `.md`, `.json`, `.csv`, `.ini` files in `config/`, `experiments/`, `evolution/`, `baselines/`, `env/`, `results/`, `tests/`, and `scratch/`.
- **Sources That Cannot Be Examined:** Ephemeral terminal executions, interactive REPL sessions, uncommitted Git branches/stashes (none currently stashed), and private external development notes.
- **Scientific Disposition:** A zero-match scan establishes absence of recorded references in the repository, but cannot guarantee that seeds were never tested out-of-band. Therefore, seeds `600–749` are classified as **`PROVISIONAL`** pending explicit operator ratification.

---

## 9. Master Phase-Gate Compliance Matrix

| Gate ID | Gate Name | Acceptance Standard | Current Status | Executed Evidence & Technical Rationale |
|---|---|---|---|---|
| **GATE-01A** | Physics & Parity Gate | Vectorized NumPy simulator achieves max float64 state error $< 10^{-9}$ vs serial reference; trusted controller performs task. | **`PASS`** | Verified in `tests/test_phase_3a_evolution.py` (state diff $= 0.00\text{e}+00$). Frozen $H=8$ achieves 100% nominal success. |
| **GATE-01B** | Works Gate (Nominal) | $\ge 90/100$ nominal successes for at least one EvoEdge controller on authorized development seeds `200–299`. | **`NOT RUN`** | Historical 50/50 result on seeds 200–249 passed historical Gate 0; prospective evaluation on 100 distinct seeds `200–299` awaits execution. |
| **GATE-02A** | Perturbation Contract | Explicit numerical parameters, units, distributions, and delay semantics defined for all 10 diagnostic conditions. | **`PASS`** | Fully specified and verified in `config/perturbations.yaml` and Section 3.3 of this document. |
| **GATE-02B1** | Aggregate Classical Non-Inferiority | Candidate controller non-inferior to Energy-Shaping + Linear Feedback on aggregate success ($L_{\text{agg}, 95} \ge -0.05$). | **`BLOCKED`** | Stage 1 established classical baseline at 89.67%. Candidate neural evaluation on provisional confirmation partition is not yet authorized. |
| **GATE-02B2** | Condition-Wise Classical Non-Inferiority | Candidate controller non-inferior across all 10 conditions ($L_{c, \text{adj}} \ge -\delta_c$) with FWER $\le 0.05$ and preregistered floors $S_c^{\text{min}}$. | **`BLOCKED PENDING PREREGISTRATION APPROVAL`** | Condition-wise margins $\delta_c$ and absolute floors $S_c^{\text{min}}$ must be justified by engineering significance before results are observed. |
| **GATE-02C** | Compute & Baseline Parity | Equal-interaction-budget comparison against ES-trained MLP ($2.0 \times 10^6$ transitions); from-scratch and teacher-assisted arms reported separately. | **`BLOCKED (Awaiting Budget Sign-Off)`** | Transition budget tracker implemented in `env/budget_tracker.py` and verified by 7/7 tests in `tests/test_budget_tracker.py`. Reconciled under Option A. Awaiting operator sign-off. |
| **GATE-03A** | Numerical Edge Parity | Compiled float32 C inference matches Python float32 within $\max \|\Delta u\| \le 1.0 \times 10^{-5}$ on multilayer DAG. | **`PASS`** | Verified in `test_modules.py::test_c_compilation_and_numerical_parity_multilayer` across hidden-to-hidden and hidden-to-output connections (measured error $= 1.50 \times 10^{-8}$). |
| **GATE-03B** | Closed-Loop C Parity | C controller dynamic library achieves identical trajectory, rail collision, and success flag parity in closed loop. | **`NOT RUN`** | Awaiting selection of final candidate controller and closed-loop dynamic library execution. |
| **GATE-04** | Protocol Integrity | Checkpoints, configs, and hashes intact; seeds 2000–2099 and Objective B strictly unaccessed; NumPy CPU default. | **`PASS`** | 13/13 SHA256 hashes bit-exact; code-level seed fences actively verified; NumPy CPU confirmed as default evaluator. |

---

## 10. Protocol Reconciliation Decision Table

| Decision Item | Stage 2 Proposal Specification | Historical EvoEdge Reality | Reconciled Protocol Recommendation | Status |
|---|---|---|---|---|
| **Lifecycle Budget** | $2.0 \times 10^6$ actual simulator transitions | $112.5 \times 10^6$ scheduled loop iterations ($450 \times 500 \times 500$) | Offer Option A (2.0M matched ceiling: 1.85M search, 150K validation) vs Option B (descriptive comparison vs historical 112.5M). | **`AWAITING OPERATOR SIGN-OFF`** |
| **Nominal Sample Size** | $\ge 90/100$ nominal successes | Seeds 200–249 (only 50 distinct seeds) | Allocate 100 distinct seeds `200–299`. Disallow duplicate seed reuse as independent samples. | **`PROPOSED SPECIFICATION`** |
| **Validation Subset** | Unspecified subset within seeds 250–399 (1,500 episodes) | Exceeds 150K transition budget if fully evaluated (750K transitions/candidate) | Cap validation at 300 episodes: 100 nominal episodes (`200–299`) + 200 multi-condition episodes (`300–319`, 10 conditions $\times$ 20 seeds). | **`PROPOSED SPECIFICATION`** |
| **Budget Enforcement Code** | Unimplemented conceptual hook | No transition counter in codebase | Implemented `env/budget_tracker.py` and validated with 7 tests in `tests/test_budget_tracker.py`. | **`VERIFIED IN CODE`** |
| **Multilayer C Parity** | Only evaluated seed 42 (no hidden connections to output) | Skipped hidden activations in parity math | Implemented full topological multilayer test in `test_modules.py` (measured error $1.50 \times 10^{-8}$). | **`PASS (Executed Test)`** |
| **Proposed Selection Thresholds** | Params $\le 128$, MACs $\le 128$, $S_{\text{worst}} \ge 40\%$ | Historical Arm B had 57 params, 48 MACs, but 0.0% on joint friction | Document engineering rationale; flag for operator sign-off before full training without silent promotion. | **`PROPOSED (Flagged for Sign-Off)`** |
| **Optimizer Seed Independence** | Numeric seeds 42–46 reused from Phase 3A | Risk of checkpoint collision if paths shared | Quarantined Stage 2 namespace to `results/stage2/`; decoupled optimizer RNG from environment evaluation seeds. | **`PROPOSED SPECIFICATION`** |
| **Official Success Predicate** | $\le \pm 12^\circ$ for $\ge 3.0$ s across $\ge 90\%$ seeds | 500 steps, $|x| < 2.4$, $\cos\theta > 0.9$ for $\ge 95/100$ final steps | Retain frozen 500-step predicate as official. Document $\pm 12^\circ$ strictly as secondary diagnostic. | **`FROZEN (Contract Preserved)`** |
| **Statistical Test** | Formal TOST equivalence ($\Delta = 0.05 \cdot \bar{R}_{\text{LQR}}$) | One-sided non-inferiority ($L_{\text{agg}, 95} \ge -0.05$) | One-sided non-inferiority via paired seed-cluster bootstrap ($B=10,000$). LQR rejected as reference. | **`PROPOSED SPECIFICATION`** |
| **Confirmation Partition** | KDF generator seeds `0000–7499` split | Seeds `0–399` burned, `1000–1099` burned, `2000–2099` protected | Authorize unexposed seeds `600–749` (150 seeds). Reject arbitrary unverified ranges. Mark provisional. | **`PROVISIONAL SPECIFICATION`** |
| **Classical Reference** | Energy-Shaping or LQR | `HybridController` (Energy-Shaping + Linear Feedback) | `HybridController` primary; LQR diagnostic only. | **`FROZEN (Stage 1 Measured)`** |
| **Edge Hardware Target** | ESP32-class (Xtensa LX7 / RISC-V, 512 KB SRAM) | ESP32 Xtensa LX6 @ 240 MHz, single-precision FPU | Target ESP32 Xtensa LX6. Separate estimates, emulation, and physical bench testing. | **`PROPOSED SPECIFICATION`** |

---

## 11. Recommended Next Action

The reconciled protocol specification is internally coherent, grounded in empirical evidence, verified by executable code, and ready for operator review.
- **Current Gate Disposition:** **`READY FOR SIGN-OFF`**
- **Required Action:** Operator selection and approval of the budget option (Option A vs. Option B), ratification of provisional confirmation partition seeds `600–749`, sign-off on proposed candidate selection thresholds, and approval of proposed contract v2.4 prior to launching Stage 2 ES training.

