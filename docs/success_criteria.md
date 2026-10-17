# EvoEdge — Acceptance Contract & Phase-Gate Specification

**Document Version:** 2.3 (Final Contract Freeze: Budget Terminology, Execution Evidence & Stage 1 Classical Evaluation)  
**Effective Date:** October 13, 2026  
**Document Status:** Binding Governance Specification  
**Authority:** Senior GPU Systems Engineer, Control Systems Researcher & Reproducibility Auditor  
**Scope:** Formal criteria governing controller acceptance, baseline comparisons, edge deployment, budget accounting, statistical inference, and scientific integrity in EvoEdge.

---

## 1. Executive Summary & Evidence-Based Starting Position

### 1.1 Historical Context & Empirical Ledger
The EvoEdge project investigates whether neuroevolutionary algorithms can synthesize compact, edge-deployable feedback controllers for the nonlinear CartPole swing-up problem that match or exceed classical control and dense deep reinforcement learning / evolutionary strategies baselines under physical perturbations.

Auditing the completed project artifacts (including Phase 1, Phase 2A/B/C, Phase 3A, and backend audits) establishes the following empirical starting position:

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

4. **Frozen Baselines and Historical Champions [`VERIFIED (Documentation Audit)`]:**
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
1. **Float64 Simulator Parity [`VERIFIED (Documentation Audit)`]:** Serial reference dynamics (`PerturbedCartPoleEnv`) and vectorized batch engine (`VectorizedCartPoleEnv`) must achieve a maximum absolute float64 state error below **$10^{-9}$** ($|x_{\text{vec}} - x_{\text{ref}}| < 10^{-9}$) across all 4 state components $[x, \dot{x}, \theta, \dot{\theta}]$ over 500 steps. Verified in `tests/test_phase_3a_evolution.py` (measured max diff $= 0.00\text{e}+00$).
2. **Trusted Controller Verification [`VERIFIED (Documentation Audit)`]:** A known trusted controller (the frozen $H=8$ baseline or the analytical Energy-Shaping + Linear Feedback controller) must be evaluated in the environment to confirm that nominal swing-up and balance are achievable. (Frozen $H=8$ achieves 100% nominal success).
3. **Circuit Breaker:** If the trusted controller fails under nominal dynamics, execution must **STOP IMMEDIATELY**. The simulation physics, integrator step, coordinate wrap, and action application must be diagnosed before any evolutionary experiments proceed.

### 2.2 Task Specification & Exact Success Predicate
The nominal task definition and boundary conditions are frozen as follows:
- **Episode Horizon:** Exactly 500 discrete control timesteps ($\Delta t = 0.02\text{ s}$, 10.0 seconds total).
- **Cart Rail Boundary Violation:** $|x| < 2.4\text{ m}$ strictly enforced at every timestep. Any violation $|x| \ge 2.4\text{ m}$ immediately terminates the episode with failure (`boundary_violated = True`).
- **Upright Balance Predicate:** $\cos(\theta) > 0.9$ strictly (corresponding to $|\theta| < 0.451\text{ rad} \approx 25.84^\circ$) for at least **95 out of the final 100 steps** ($t \in [400, 499]$).
- **Physical Constants & Equations of Motion:**
  - Cart mass $m_c = 1.0\text{ kg}$, pole mass $m_p = 0.1\text{ kg}$, half-pole length $l = 0.5\text{ m}$, gravity $g = 9.8\text{ m/s}^2$.
  - Numerical integration: Semi-implicit Euler integration ($\dot{x}_{t+1} = \dot{x}_t + \Delta t \ddot{x}_t$; $x_{t+1} = x_t + \Delta t \dot{x}_{t+1}$).
  - Angle wrapping: $\theta \in [-\pi, \pi]$ via $((\theta + \pi) \pmod{2\pi}) - \pi$.
- **Observation Normalization:** Exactly 5 input channels:
  $$\text{obs} = \left[ \frac{x}{2.4}, \, \text{clip}\left(\frac{\dot{x}}{10.0}, -1.0, 1.0\right), \, \cos\theta, \, \sin\theta, \, \text{clip}\left(\frac{\dot{\theta}}{10.0}, -1.0, 1.0\right) \right]$$
- **Actuation Limits:** Continuous control signal $u \in [-1.0, 1.0]$, mapped to horizontal force $F = \text{clip}(u, -1.0, 1.0) \times 10.0\text{ N}$.
- **Reset Distribution:** Initial state sampled uniformly from $x \in [-0.05, 0.05]\text{ m}$, $\dot{x} \in [-0.05, 0.05]\text{ m/s}$, $\theta \in [\pi - 0.05, \pi + 0.05]\text{ rad}$ (downward hanging), $\dot{\theta} \in [-0.05, 0.05]\text{ rad/s}$.

### 2.3 Nominal Acceptance Criterion (Gate 1B)
- **Criterion:** At least **90 successes in 100 preregistered nominal evaluation episodes** ($\ge 90.0\%$) for at least one EvoEdge-generated controller on a newly authorized, non-reused nominal evaluation partition.
- **Reporting Requirements:** The report must provide:
  - Exact success count and success rate ($k / 100$).
  - 95% Wilson score confidence interval:
    $$w = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \quad z = 1.96$$
  - Exact breakdown of failures: rail boundary collisions ($|x| \ge 2.4$), failure to reach upright within 400 steps, or failure to maintain balance for $\ge 95$ steps.
- **Partition Role & Confirmation Invariant:**
  - Historical Gate 0 (50/50 on seeds 200–249) does not satisfy Gate 1B.
  - Gate 1B must **NOT** access, pollute, or burn the locked confirmation partition reserved for final confirmatory robustness evaluation.
  - **Two Authorized Architectural Protocols:**
    - *Protocol Alpha (Separate Development Split):* Gate 1B is evaluated on an authorized development nominal split (e.g., 100 prospective seeds dedicated to nominal screening). Passing Gate 1B qualifies a controller for subsequent Stage 2 selection, after which the chosen candidate is frozen.
    - *Protocol Beta (Single Locked Confirmation Evaluation):* The candidate controller is completely selected and frozen *prior* to touching the locked confirmation partition. Once frozen, all preregistered acceptance criteria (nominal Gate 1B and robust Gates 02B1/02B2) are evaluated concurrently in a single locked evaluation pass on the confirmation partition.
  - *Strict Non-Iteration Rule:* If a frozen candidate fails on the confirmation partition under either protocol, the run is recorded as an unrecoverable confirmatory failure. **Confirmation outcomes must NEVER be reused or fed back to select, tune, or screen another candidate.**
- **Current Gate Status:** **`NOT RUN`** (awaiting prospective execution under authorized partition protocol).

---

## 3. Gate 2 — The "Robustness & Fair-Comparison" Gate

Robustness claims are scientifically meaningless in the absence of rigorous classical baselines, budget-matched neural comparators, and preregistered physical stress distributions.

### 3.1 Classical Control Reference Architecture
1. **Principal Classical Reference: Energy-Shaping + Linear Feedback (`HybridController`)**
   - The standard classical benchmark for CartPole swing-up consists of nonlinear mechanical energy-shaping for swing-up coupled with linear state feedback for local stabilization:
     $$u_{\text{pump}} = \text{clip}\left( k_E \cdot (E - E_0) \cdot \text{sign}(\dot{\theta} \cos\theta) - k_x x - k_v \dot{x}, -1.0, 1.0 \right)$$
     $$u_{\text{balance}} = \text{clip}\left( -(k_1 x + k_2 \dot{x} + k_3 \theta + k_4 \dot{\theta}), -1.0, 1.0 \right)$$
   - Switching condition: Hand-off occurs when $|\theta| < 0.35\text{ rad}$ and $|\dot{\theta}| < 2.0\text{ rad/s}$.
   - Complexity: 7 float gains, 7 MACs.
2. **Diagnostic Linear Comparator: Classical LQR**
   - Pure LQR linearized around the upright equilibrium ($\theta = 0$).
   - *Diagnostic Role:* LQR provides a baseline for local linear regulatory stiffness and jitter, but has **0% swing-up capability** from the downward hanging position ($\theta \approx \pi$). It must be included for local balance comparisons, but **cannot** serve as the required swing-up performance threshold.

### 3.2 Historical Inventory & Legacy Partition Boundaries
1. **Historical Task B Benchmark (Seeds 1000–1099) [`HISTORICAL (Task B)`]:**
   - Evaluated under Protocol v2.4 across 8 conditions (Nominal, Pole +30%, Cart +30%, Friction, Sensor Noise, 1-Step Delay, 2-Step Delay, Combined).
   - In that benchmark, Energy-Shaping + Linear Feedback achieved **90.88% mean success** (100% on Nominal, 100% on Pole Mass, 100% on Cart Mass, 85% on Friction, 95% on Noise, 99% on 1-Step Delay, 100% on 2-Step Delay, 48% on Combined).
2. **Partition Inviolability [`VERIFIED (Documentation Audit)`]:**
   - Seeds `1000–1099` represent a frozen legacy test partition. They must **NOT** be rerun, tuned against, or repurposed for exploratory development.
3. **Condition Suite Discrepancy:**
   - The historical 8-condition Task B suite combined rail and joint friction into a single joint perturbation, whereas Phase 3A established 10 distinct diagnostic conditions separating isolated rail friction (`4a`), isolated joint friction (`4b`), and combined friction (`4c`).
   - Therefore, historical 8-condition measurements **cannot** be directly substituted into the 10-condition Phase 3A diagnostic matrix.

### 3.3 Authoritative Phase 3A Perturbation Contract & Partition Roles
The authoritative physical perturbation suite is transcribed verbatim from `config/perturbations.yaml` and `config/phase_3a_config.yaml`. For each condition, the exact parameters, units, distributions, horizon, and partitioned roles are codified:

| Condition ID & Name | Physical Parameters & Values | Units | Noise Distribution & Scale | Actuator Delay Semantics | Horizon & Steps | Training Role (Partition) | Model Selection Role (Partition) | Confirmation Role (Partition) |
|---|---|---|---|---|---|---|---|---|
| **1_Nominal** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.00$ | $m$ in kg, $b$ in $\text{N}\cdot\text{s/m}$, $\text{N}\cdot\text{m}\cdot\text{s/rad}$ | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) | Active Fitness (`0..49`) | Validation Front (`250..399`) | Confirmatory Evaluation (Locked Split) |
| **2_PoleMass_+30%** | $m_c = 1.00$, $m_p = 0.130$, $b_c = 0.00$, $b_p = 0.00$ | $m_p$ in kg (+30%) | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) | Active Fitness (`0..49`) | Validation Front (`250..399`) | Confirmatory Evaluation (Locked Split) |
| **3_CartMass_+30%** | $m_c = 1.30$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.00$ | $m_c$ in kg (+30%) | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) | Active Fitness (`0..49`) | Validation Front (`250..399`) | Confirmatory Evaluation (Locked Split) |
| **4a_CartFriction_0.50** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.50$, $b_p = 0.00$ | $b_c$ in $\text{N}\cdot\text{s/m}$ | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) | Active Fitness (`0..49`) | Validation Front (`250..399`) | Confirmatory Evaluation (Locked Split) |
| **4b_JointFriction_0.010** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.010$ | $b_p$ in $\text{N}\cdot\text{m}\cdot\text{s/rad}$ | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) | Active Fitness (`0..49`) | Validation Front (`250..399`) | Confirmatory Evaluation (Locked Split) |
| **4c_CombinedFriction** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.50$, $b_p = 0.010$ | $b_c$ and $b_p$ combined | $\sigma = 0.00$ (None) | 0 steps (0 ms) | 500 steps (10 s) | Active Fitness (`0..49`) | Validation Front (`250..399`) | Confirmatory Evaluation (Locked Split) |
| **5_SensorNoise_0.05** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.00$ | Nominal mass/friction | Gaussian $\mathcal{N}(0, 0.05^2)$ on raw unnormalized obs | 0 steps (0 ms) | 500 steps (10 s) | Active Fitness (`0..49`) | Validation Front (`250..399`) | Confirmatory Evaluation (Locked Split) |
| **6a_ActuatorDelay_1Step** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.00$ | Nominal mass/friction | $\sigma = 0.00$ (None) | 1 step (20 ms FIFO queue) | 500 steps (10 s) | Active Fitness (`0..49`) | Validation Front (`250..399`) | Confirmatory Evaluation (Locked Split) |
| **6b_ActuatorDelay_2Step** | $m_c = 1.00$, $m_p = 0.100$, $b_c = 0.00$, $b_p = 0.00$ | Nominal mass/friction | $\sigma = 0.00$ (None) | 2 steps (40 ms FIFO queue) | 500 steps (10 s) | Active Fitness (`0..49`) | Validation Front (`250..399`) | Confirmatory Evaluation (Locked Split) |
| **7_CombinedHistoricalStress** | $m_c = 1.30$, $m_p = 0.130$, $b_c = 0.50$, $b_p = 0.010$ | All perturbations combined | Gaussian $\mathcal{N}(0, 0.05^2)$ on raw unnormalized obs | 1 step (20 ms FIFO queue) | 500 steps (10 s) | Active Fitness (`0..49`) | Validation Front (`250..399`) | Confirmatory Evaluation (Locked Split) |

*Noise Injection Semantics:* Observation noise is drawn as i.i.d. Gaussian zero-mean noise and added directly to raw physical state channels $[x, \dot{x}, \cos\theta, \sin\theta, \dot{\theta}]$ prior to normalization and clipping, with an initial draw at `reset()` and subsequent draws at each step (501 total draws per 500-step episode).

### 3.4 Dual Mandatory Robustness Criteria Structure
Robustness claims require satisfying two separate, conjunctive, mandatory criteria relative to the classical reference (`HybridController`):
1. **Requirement A (Aggregate Non-Inferiority):** The candidate controller must demonstrate aggregate non-inferiority across all conditions combined ($L_{\text{agg}, 95} \ge -0.05$).
2. **Requirement B (Condition-Wise Non-Inferiority):** The candidate controller must satisfy simultaneous, multiplicity-adjusted non-inferiority across all $C = 10$ individual diagnostic conditions relative to classical control ($L_{c, \text{adj}} \ge -\delta_c$). High performance on easier nominal or mass-perturbed conditions is strictly forbidden from masking near-total collapse on friction, delay, or noise conditions.
3. **Mandatory Reporting:** Evaluations must report all condition-level sample differences $\bar{D}_c$, simultaneous uncertainty intervals, and identify any condition-level failures. Both requirements must pass independently for overall gate certification.

---

## 4. Experimental Sequence & Three-Stage Partition Protocol

To prevent validation leakage and ensure confirmatory integrity, project activities are strictly decoupled into three sequential stages:

```
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Descriptive Baseline Comparisons                              │
│ • Evaluate Classical Energy-Shaping & baselines on seeds 250..399.     │
│ • Purpose: Establish descriptive context for Phase 3A historical data. │
│ • Status: DESCRIPTIVE ONLY — STRICTLY NON-CONFIRMATORY.                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Budget-Matched ES MLP Training & Candidate Selection          │
│ • Train ES 64×64 MLP on training seeds 0..199 under matched budget.    │
│ • Compare against Arm B Pareto front on validation seeds 250..399.     │
│ • Select and FREEZE final candidate controller.                        │
│ • Status: DEVELOPMENT & MODEL SELECTION.                               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Independent Confirmation on Locked Partition                  │
│ • Evaluate frozen candidate and baselines ONCE on prospective split.   │
│ • Apply preregistered statistical non-inferiority test.                │
│ • Invariant: NEVER use confirmation data to tune, select, or revise.   │
│ • Status: FORMAL CONFIRMATORY TESTING.                                 │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Stage 1: Descriptive Baseline Comparisons on Existing Validation Data
- Evaluates `HybridController` across the 10 diagnostic conditions on Phase 3A validation partition seeds `250–399` (150 seeds per condition = 1,500 episodes).
- *Strict Scientific Classification:* These measurements provide descriptive comparison against historical Arm B models (which were evaluated on seeds `250–399`). They are **explicitly labeled as descriptive validation results** and **cannot be cited as fresh confirmatory evidence**.

### 4.2 Stage 2: Budget-Matched ES MLP Training & Candidate Selection
- Trains the dense 64×64 MLP baseline under evolutionary strategies (ES) using training partition seeds `0–199`.
- Evaluates the trained MLP on validation partition seeds `250–399`.
- Final candidate controller is selected from the Pareto front based on validation trade-offs and **frozen**.

### 4.3 Stage 3: Independent Confirmation on Locked Confirmation Partition
- **Mandatory Pre-Condition:** Candidate selection, hyperparameter choices, architecture pruning, and all optimization/tuning must be completed and cryptographically frozen *prior* to accessing the locked confirmation partition.
- **Evaluation Isolation:** Confirmatory evaluation is executed exactly once on a newly authorized, preregistered confirmation partition (distinct from training seeds `0–199` and validation seeds `250–399`).
- **All-In-One Evaluation Pass:** All preregistered confirmatory criteria (nominal Gate 1B if not previously cleared on a separate development split, Gate 02B1 aggregate non-inferiority, Gate 02B2 condition-wise non-inferiority, and Gate 02C baseline comparisons) are evaluated concurrently in this single confirmation pass.
- *Strict Scientific Invariant:* **Confirmation data must NEVER be inspected, queried, or utilized to tune, select, screen, or revise candidate controllers.** If a controller fails to satisfy the preregistered criteria on the confirmation partition, the outcome is recorded as a definitive failure. Re-tuning, altering hyperparameters, or selecting an alternate candidate from the Pareto front using confirmation feedback is strictly prohibited.

---

## 5. Fair Budget Accounting & Training Compute Contracts

### 5.1 Historical Phase 3A Compute & Rollout Capacity Audit [`VERIFIED (Code & Logic Audit)`]
Inspecting `experiments/run_phase_3a_evolution.py` (specifically `evaluate_candidate_vectorized`) and `env/vectorized_env.py` establishes the exact accounting structure for Phase 3A:
- Population size: $N = 30$, Generations: $G = 15$.
- Candidate evaluations per seed run: $15 \times 30 = \mathbf{450\text{ candidate evaluations}}$ (evaluated without survivor caching).
- Evaluation suite: 10 diagnostic conditions $\times$ 50 training seeds (`0..49`) $= 500$ rollout episodes per candidate evaluation.
- Scheduled episode horizon: Fixed loop of 500 steps (`for step in range(max_steps)` with `max_steps = 500`).
- **Maximum Scheduled Rollout Iterations:**
  $$450 \times 500 \times 500 = \mathbf{112,500,000\text{ scheduled rollout iterations}} \quad (\mathbf{1.125 \times 10^8\text{ iterations}})$$

**Rigorous Tripartite Distinction: Scheduled Iterations, Executed Transitions, and Computational Cost:**
1. **Scheduled Rollout Iterations:** The planned upper-bound schedule of discrete timesteps across all candidate episodes ($1.125 \times 10^8$).
2. **Actual Executed Environment Transitions:**
   - In `VectorizedCartPoleEnv.step()`, when a cart violates the rail boundary ($|x| \ge 2.4$), `self.boundary_violated` and `self.terminated` are set to `True`. However, in `evaluate_candidate_vectorized()`, the loop continues to step the physics simulation for all 500 steps regardless of early boundary violations (to compute trajectories without masking array shapes).
   - In serial environments with early-stopping rollouts, boundary violations terminate execution immediately, reducing actual environment transitions.
   - Because historical Phase 3A runs stepped the vectorized tensor for the full 500 steps, $112.5 \times 10^6$ represents **scheduled rollout iterations and executed physics integration cycles**, but does **NOT** represent useful transitions before termination, nor does it log the exact count of effective non-terminated steps.
3. **Total Computational Cost (Active Operations & Energy):**
   - Environment transitions alone do not define computational cost.
   - Forward-pass inference compute depends on active parameter and MAC counts (e.g., 48 MACs for compact $H=8$ vs. 4,480 MACs for dense 64×64 MLP).
   - Stepping an identical $1.125 \times 10^8$ transitions costs $93.3\times$ more floating-point operations for the dense MLP than for the compact network.
4. **Limitation & Baseline Parity Requirement:** Historical runs did not record step-level termination timestamps per episode. Therefore, to ensure strictly fair baseline comparisons under `GATE-02C`, the comparative ES MLP baseline must either:
   - Match the exact $1.125 \times 10^8$ fixed-horizon vectorized rollout capacity using the identical `evaluate_candidate_vectorized` protocol, OR
   - Report actual non-terminated transitions under an approved step-level accounting method.
5. **Gate Status:** `GATE-02C` remains **`BLOCKED`** pending operator approval of the specific baseline budget accounting protocol.

*(Note on Earlier Documents: Informal citations of $1.25 \times 10^7$ steps in preliminary notes corresponded to a hypothetical 50-evaluation budget. The actual executed Phase 3A run scheduled $1.125 \times 10^8$ iterations per seed run).*

### 5.2 Fair Budget Accounting Contract
1. **Reporting Dimensions:** Every comparative training experiment must log and report:
   - Total scheduled rollout iterations.
   - Actual executed environment transitions (and non-terminated transitions where early stopping applies).
   - Total candidate/policy evaluations.
   - Total optimizer parameter updates.
   - Total computational cost measured by inference floating-point operations executed during training ($\text{transitions} \times \text{active MACs}$).
   - Wall-clock training duration, compute device (CPU cores vs GPU model), and memory consumption.
2. **The "Equal Compute" vs "Equal Interaction" Distinction:**
   - Parameter count alone is **NOT** a valid measure of compute.
   - Matching environment step counts does **NOT** constitute equal compute:
     - The dense 64×64 MLP executes **4,480 active weight MACs** per step.
     - The compact $H=8$ network executes **48 active weight MACs** per step.
     - For an identical $1.125 \times 10^8$ environment steps, the 64×64 MLP performs **$93.3\times$ more floating-point operations** during rollout forward passes ($5.04 \times 10^{11}$ MACs vs $5.40 \times 10^9$ MACs).
   - *Binding Nomenclature:* When two methods are trained with the same number of environment interaction steps, the experiment must be explicitly designated an **`EQUAL-INTERACTION-BUDGET`** comparison, **NOT** an "equal-compute" comparison.

---

## 6. Executable Statistical Decision Protocol

### 6.1 Paired Non-Inferiority Testing Protocol Against Classical Reference
The comparison between candidate controller and Energy-Shaping + Linear Feedback (`HybridController`) must follow an exact, preregistered, paired statistical procedure. To prevent aggregate performance from concealing catastrophic failure on individual physical stress regimes, the evaluation enforces two separate, conjunctive, mandatory requirements:

#### 6.1.1 Statistical Foundations & Paired Structure
1. **Unit of Randomization & Clustering:** The **evaluation seed** $s \in \mathcal{S}$ is the independent cluster unit.
2. **Paired Observations:** Each seed $s$ is evaluated across all $C = 10$ diagnostic conditions for both the candidate controller and the classical reference under bit-exact initial conditions and disturbance trajectories:
   $$Y_{\text{cand}}(c, s) = \mathbb{I}(\text{cand succeeds on condition } c, \text{ seed } s)$$
   $$Y_{\text{ref}}(c, s) = \mathbb{I}(\text{ref succeeds on condition } c, \text{ seed } s)$$
   $$D(c, s) = Y_{\text{cand}}(c, s) - Y_{\text{ref}}(c, s) \in \{-1, 0, 1\}$$
3. **Condition-Level Sample Differences:**
   $$\bar{D}_c = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} D(c, s) = S_c(\text{cand}) - S_c(\text{ref})$$
4. **Aggregate Sample Mean Difference:**
   $$\bar{D}_{\text{agg}} = \frac{1}{10} \sum_{c=1}^{10} \bar{D}_c = S_{\text{agg}}(\text{cand}) - S_{\text{agg}}(\text{ref})$$
5. **Paired Cluster Bootstrap Over Seeds:**
   - Because outcomes across the 10 conditions for a given seed share the initial dynamic state perturbation $s$, observations within a seed are correlated across conditions.
   - We execute a **paired cluster bootstrap over seeds** ($B = 10,000$ resamples of seeds with replacement), resampling the entire 10-condition difference vector $\mathbf{D}(s) = [D(1, s), \dots, D(10, s)]^T$ for each resampled seed.
   - For each bootstrap replicate $b \in \{1, \dots, B\}$, compute the bootstrap replicate aggregate difference $\bar{D}_{\text{agg}}^{*(b)}$ and the 10-dimensional condition vector $\bar{\mathbf{D}}^{*(b)} = [\bar{D}_1^{*(b)}, \dots, \bar{D}_{10}^{*(b)}]^T$.

#### 6.1.2 Requirement A: Aggregate Non-Inferiority Test (Mandatory Gate Requirement)
1. **Confidence Bound:** From the bootstrap distribution of $\bar{D}_{\text{agg}}^*$, compute the lower one-sided 95% confidence bound $L_{\text{agg}, 95}$ (the 5th percentile of the bootstrap distribution).
2. **Preregistered Aggregate Margin:** $\delta_{\text{agg}} = -0.05$ ($-5.0$ percentage points).
3. **Pass Predicate:** Requirement A is declared **PASSED** if and only if:
   $$L_{\text{agg}, 95} \ge -0.05 \quad (\text{lower 95\% bound on aggregate candidate-minus-reference difference is } \ge -5.0\%)$$

#### 6.1.3 Requirement B: Condition-Wise Non-Inferiority Test (Mandatory Gate Requirement)
*Deficiency of Historical Standalone Floor:* The previous requirement $\min_{c} S_c > 0\%$ was structurally inadequate; a controller scoring 1% on friction or delay technically cleared it despite suffering near-total operational failure. A standalone $> 0\%$ floor is rejected as an adequate robustness criterion.

1. **Family of Inferences Across Conditions:** The 10 individual diagnostic conditions constitute a simultaneous inference family:
   $$\mathcal{F}_{\text{conditions}} = \{H_{0,c}: S_c(\text{cand}) - S_c(\text{ref}) < -\delta_c\} \quad \text{for } c \in \{1, \dots, 10\}$$
2. **Multiplicity Adjustment:**
   - Simultaneous lower confidence bounds $L_{c, \text{adj}}$ across all 10 conditions must be constructed to control the family-wise error rate across conditions at FWER $\le 0.05$.
   - Bound construction must use either:
     - **Bonferroni Adjustment:** Lower one-sided $(1 - 0.05 / 10) = 99.5\%$ bootstrap percentile bound ($0.5$th percentile) for each condition $c$.
     - **Simultaneous Studentized / Percentile Bootstrap Vector Confidence Region:** The lower $(1 - \alpha)$ simultaneous joint confidence band accounting for empirical correlation across conditions.
3. **Methodological Basis for Margins ($\delta_c$) vs Reference Variance:**
   - **Crucial Methodological Distinction:** The non-inferiority margin $\delta_c > 0$ defines the maximum clinically and practically acceptable degradation on condition $c$ relative to classical control. **Margins must be justified on the basis of engineering significance and application tolerance BEFORE observing experimental outcomes.**
   - Baseline reference variance informs prospective power calculations and required sample size; it must **never** be used to define acceptable performance degradation post hoc.
   - **Current Status:** **`BLOCKED PENDING PREREGISTRATION APPROVAL`**. Because condition-wise margins $\delta_c$ have not yet been formally established and approved based on engineering specifications, Gate 02B2 is strictly blocked. No arbitrary or convenient post hoc threshold may be substituted.
4. **Pass Predicate (When Preregistered & Unblocked):** Requirement B will be declared **PASSED** if and only if:
   $$L_{c, \text{adj}} \ge -\delta_c \quad \forall c \in \{1, \dots, 10\}$$
   $$\text{AND} \quad S_c(\text{cand}) \ge S_c^{\text{min}} \quad \forall c \in \{1, \dots, 10\}$$
   where $S_c^{\text{min}}$ represents a separately justified, preregistered absolute performance floor for condition $c$ (also `BLOCKED PENDING PREREGISTRATION APPROVAL`).

#### 6.1.4 Mandatory Comprehensive Reporting Protocol
Evaluations must explicitly tabulate and report:
- Condition-level empirical success rates $S_c(\text{cand})$ and $S_c(\text{ref})$.
- Paired differences $\bar{D}_c$ and simultaneous adjusted uncertainty intervals $[L_{c, \text{adj}}, U_{c, \text{adj}}]$ for all 10 conditions.
- Aggregate success rates $S_{\text{agg}}(\text{cand})$, $S_{\text{agg}}(\text{ref})$, difference $\bar{D}_{\text{agg}}$, and aggregate lower bound $L_{\text{agg}, 95}$.
- Any condition-level failure or boundary violation breakdown. Both Requirement A and Requirement B are mandatory; failure on either requirement constitutes an overall failure of Gate 2B.

### 6.2 Preregistered Claims Relative to ES 64×64 MLP
1. **Claim 1 (Robustness Superiority at Matched Complexity):**
   $$\Delta S_{\text{agg}} = S_{\text{agg}}(\text{Candidate}) - S_{\text{agg}}(\text{MLP64}) \ge +0.05 \quad (\text{paired bootstrap } p < 0.05)$$
2. **Claim 2 (Efficiency Superiority at Parity Performance):**
   $$\text{Params}_{\text{Candidate}} \le 0.75 \times \text{Params}_{\text{MLP64}}, \quad \text{MACs}_{\text{Candidate}} \le 0.75 \times \text{MACs}_{\text{MLP64}}$$
   $$\Delta S_{\text{agg}} = S_{\text{agg}}(\text{Candidate}) - S_{\text{agg}}(\text{MLP64}) \ge -0.02 \quad (\text{lower 95\% bound } \ge -0.02)$$

### 6.3 Family-Wise Error Rate (FWER) Control via Holm–Bonferroni
When $K$ candidate controllers from the Pareto front are evaluated against baselines:
- The comparisons form a family $\mathcal{F} = \{H_{0,1}, \dots, H_{0,K}\}$.
- Sort $p$-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(K)}$.
- Reject $H_{0,(i)}$ if $p_{(i)} \le \frac{\alpha}{K - i + 1}$ with nominal $\alpha = 0.05$.
- Report negative and inconclusive outcomes without adjusting thresholds post hoc.

---

## 7. Gate 3 — The "Ships" Gate (Edge Compilation & Verification)

The Ships gate certifies that an evolved neural network can be exported to zero-dependency ANSI C, compiled for resource-constrained microcontrollers, and executed with numerical and behavioral fidelity.

### 7.1 C Export & Memory Allocation Contract
- Selected neural controllers must export to standalone ANSI C (`controller.h` / `controller.c`) via `c_exporter.py`.
- **Zero Dynamic Allocation:** `malloc()`, `free()`, and heap allocations are strictly forbidden in the inference path. All state arrays and circular delay buffers must reside in static memory or local stack frames.
- **Precision:** Float32 arithmetic matching microcontroller hardware floating-point units (FPUs, e.g., ESP32 Xtensa LX6 single-precision FPU).

### 7.2 Fixed Versioned Edge Test Corpus
To prevent edge compiler regression, a fixed versioned corpus of test observations (`tests/corpus/edge_verification_corpus.json`) must be maintained:
1. **In-Distribution States:** Typical swing-up trajectory observations.
2. **Rail Boundary Extremes:** States with position near $|x| = 2.4\text{ m}$ (e.g., $x = \pm 2.3999\text{ m}$, $x = \pm 2.4001\text{ m}$).
3. **Upright Predicate Extremes:** States with angle near the upright boundary $\cos\theta = 0.9000$ (e.g., $\cos\theta = 0.8999$, $\cos\theta = 0.9001$).
4. **Saturation Limits:** Extreme angular and linear velocities exercising observation normalization clipping limits ($\dot{x} = \pm 12.0\text{ m/s}$, $\dot{\theta} = \pm 15.0\text{ rad/s}$).
5. **Actuator Boundary States:** States producing outputs near zero ($u \approx 0$) and near force saturation limits ($u \approx \pm 1.0$).
6. **Deterministic Episode Traces:** At least 5 complete recorded closed-loop episode traces covering reset, delay queue propagation, rail boundary triggering, and successful balance.

### 7.3 Dual-Stage Verification Criteria
1. **Stage A: Numerical Output Parity (Tolerance $\le 10^{-5}$):**
   $$\max |\Delta u| = \max |u_{\text{C}} - u_{\text{Python}}| \le 1.0 \times 10^{-5}$$
2. **Stage B: Closed-Loop Trajectory Parity:**
   - **Do not infer episode equivalence from forward-pass tolerance alone.**
   - The compiled C controller must be executed in closed-loop simulation via a dynamic C library wrapper (`ctypes`) across recorded test episodes.
   - Applied actions, state trajectories, discrete rail collision timesteps, and final success flags must match Python bit-for-bit.

### 7.4 Release Deliverable Standards
The first production release package must contain:
- A clean, functional CLI (`main.py` / `c_exporter.py`).
- Documented configuration examples (`config/`).
- Automated self-contained unit and regression tests (`tests/`).
- A comprehensive reproducibility report documenting exact seeds, environments, and hashes.
- Controller metadata (parameter count, active MACs, theoretical cycle counts, memory footprint).
- Explicitly documented limitations and failure modes.
- *Non-Requirement:* An interactive web dashboard is a stretch goal and is **NOT** a required release blocker.
- **Current Gate Status:** **`NOT RUN`** (awaiting candidate selection).

---

## 8. Invariants & Scientific Integrity Guarantees

### 8.1 Methodological Distinction: Evidence Types & Code Execution
To prevent ambiguity regarding experimental reproducibility and audit claims:
- **Historical Evidence (`HISTORICAL`):** Data, tables, and metrics extracted from completed historical run logs and reports (e.g., Phase 1, Phase 2A/B/C, Phase 3A reports). These reflect past runs and must not be conflated with newly executed live tests.
- **Static Inspection & Auditing (`INSPECTION`):** Read-only verification of source code, configuration files, schema consistency, mathematical derivations, and cryptographic checksums.
- **Executed Test Evidence (`EXECUTED TEST`):** Discrete verification scripts, unit tests, or test suites physically executed in the runtime environment (e.g., `tests/test_phase_3a_evolution.py`, `tests/test_cuda_evaluator.py`). Running verification commands constitutes active code execution; reports must accurately state what was run rather than claiming no code was executed.
- **Experimental Rollouts & Training (`EXPERIMENT`):** Full multi-seed evolutionary runs, comparative baseline evaluations, or physical stress experiments. These modify or populate experimental result ledgers and require explicit operator authorization.

### 8.2 Invariants & Scientific Integrity Guarantees
To preserve scientific reproducibility and experimental integrity, the following invariants are permanently established:

1. **Immutability of Frozen Artifacts [`VERIFIED (Checksum Audit)`]:**
   - Historical configurations (`config/phase_3a_config.yaml`, `config/perturbations.yaml`), historical checkpoints (`results/champion_neural_h8.pkl`), and baseline logs must remain bit-exact and unaltered.
   - Cryptographic SHA256 hashes of all 13 frozen files confirmed 100% identical.
2. **Holdout Fence Enforcement & Audit Distinction [`VERIFIED (Executed Unit Tests & Inspection)`]:**
   - *Audit Distinction:* **A SHA256 match establishes artifact integrity, not by itself that protected data were never accessed during execution.**
   - Holdout integrity is supported by actual code-level execution fences:
     - `check_seed_safety()` in `experiments/run_phase_3a_evolution.py` and `env/cuda_evaluator.py` actively raises `ValueError: CRITICAL SAFETY VIOLATION` if seeds in `2000–2099` are passed.
     - `test_holdout_seed_rejection()` in `tests/test_cuda_evaluator.py` actively asserts this rejection.
     - Static code and log audits confirm zero execution calls targeting seeds `2000–2099` or Objective B ranges.
3. **Legacy Benchmark Protection:**
   - Seeds `1000–1099` represent frozen legacy benchmark data from Task B. They must not be repurposed for development or training.
4. **Evaluator Backend Status:**
   - **Vectorized NumPy CPU (`VectorizedCartPoleEnv`) is the sole accepted default evaluator.**
   - The CUDA backend (`env/cuda_evaluator.py`) is recorded as **`EXPERIMENTAL / NOT ACCEPTED`** due to failure of the mandatory kernel fusion gate (~15,500 kernel launches per rollout).
5. **Stretch Features Classified as Non-Blockers:**
   - Online parameter adaptation, spiking neural networks (LIF), INT8 quantization, single-kernel CUDA fusion, and physical ESP32 oscilloscope measurements are exploratory stretch goals that do **not** block release.
   - *Strict Non-Claim:* **Do not claim online adaptation as a solved capability in this release.**
6. **Git Cleanliness:**
   - No automated Git staging, committing, file deletion, or branch switching may be executed without explicit operator consent.

---

## 9. Master Phase-Gate Compliance Matrix

| Gate ID | Gate Name | Acceptance Standard | Current Status | Executed Evidence & Technical Rationale |
|---|---|---|---|---|
| **GATE-01A** | Physics & Parity Gate | Vectorized NumPy simulator achieves max float64 state error $< 10^{-9}$ vs serial reference; trusted controller performs task. | **`PASS`** | Verified in `tests/test_phase_3a_evolution.py` (state diff $= 0.00\text{e}+00$ across 50 seeds, 500 steps). Frozen $H=8$ achieves 100% nominal success. |
| **GATE-01B** | Works Gate (Nominal) | $\ge 90/100$ nominal successes for at least one EvoEdge controller on an authorized prospective nominal split. | **`NOT RUN`** | Historical 50/50 result on seeds 200–249 passed historical Gate 0, but prospective evaluation on a new 100-seed split has not yet been authorized. |
| **GATE-02A** | Perturbation Contract | Explicit numerical parameters, units, distributions, and delay semantics defined for all 10 diagnostic conditions. | **`PASS`** | Fully specified and verified in `config/perturbations.yaml` and Section 3.3 of this document. |
| **GATE-02B1** | Aggregate Classical Non-Inferiority | Candidate controller non-inferior to Energy-Shaping + Linear Feedback on aggregate success ($L_{\text{agg}, 95} \ge -0.05$). | **`BLOCKED`** | Energy-Shaping reference has not yet been evaluated under the 10-condition suite on permitted validation seeds 250–399. Reference score cannot be fabricated. |
| **GATE-02B2** | Condition-Wise Classical Non-Inferiority | Candidate controller non-inferior across all 10 conditions ($L_{c, \text{adj}} \ge -\delta_c$) with FWER $\le 0.05$ and preregistered floors $S_c^{\text{min}}$. | **`BLOCKED PENDING PREREGISTRATION APPROVAL`** | Condition-wise margins $\delta_c$ and absolute floors $S_c^{\text{min}}$ must be justified by engineering significance and application tolerance before results are observed. Blocked pending formal preregistration approval. |
| **GATE-02C** | Compute & Baseline Parity | Equal-interaction-budget comparison against ES-trained 64×64 MLP; $\ge 3$ seeds; from-scratch and teacher-assisted arms reported separately. | **`BLOCKED`** | Phase 3A $112.5 \times 10^6$ figure represents maximum scheduled rollout capacity ($450 \text{ evals} \times 500 \text{ episodes} \times 500 \text{ steps}$). Blocked pending approval of baseline rollout accounting protocol. |
| **GATE-03A** | Numerical Edge Parity | Compiled float32 C inference matches Python float32 within $\max \|\Delta u\| \le 1.0 \times 10^{-5}$ on versioned edge corpus. | **`NOT RUN`** | Awaiting selection of final candidate controller and assembly of corpus. |
| **GATE-03B** | Closed-Loop C Parity | C controller dynamic library achieves identical trajectory, rail collision, and success flag parity in closed loop. | **`NOT RUN`** | Awaiting candidate selection and dynamic library integration test. |
| **GATE-04** | Protocol Integrity | Checkpoints, configs, and hashes intact; seeds 2000–2099 and Objective B strictly unaccessed; NumPy CPU default. | **`PASS`** | 13/13 SHA256 hashes bit-exact; code-level seed fences actively verified; NumPy CPU confirmed as default evaluator. |

---

## 10. Recommended First Approved Experiment

Following contract approval, the next authorized experiment requiring explicit operator sign-off is:

### Experiment Title: Stage 1 Descriptive Baseline Evaluation of Classical Energy-Shaping on Validation Seeds (`250–399`)
- **Objective:** Evaluate `HybridController` across the authoritative 10 diagnostic conditions on Phase 3A validation partition seeds `250–399` (150 seeds per condition = 1,500 episodes) to establish the descriptive reference score alongside Arm B's historical validation performance. This provides sample variance estimates to inform prospective power and sample-size calculations for confirmatory testing.
- **Protocol Classification:** **Stage 1 Descriptive Baseline Comparison.** This carries **zero confirmatory status** and strictly does not touch the locked confirmation partition.
- **Execution Invariant:** **Do not launch this experiment during the documentation task.** Execution halts upon contract delivery.
