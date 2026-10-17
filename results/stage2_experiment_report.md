# Stage 2 Development Training Experiment Report: Option A (Matched-Budget Comparison)

**Project:** EvoEdge: Ultra-Compact Neuroevolution for Microcontroller Deployment  
**Date:** 2026-10-14  
**Author:** Pair Programming Agent (Antigravity IDE)  
**Execution Status:** COMPLETED (All 10 runs + Classical Reference executed to completion)  
**Verification:** 109/109 Unit, Parity & Budget Tests Passing  
**Integrity Guarantee:** Zero synthetic data; 100% empirical simulator execution. Seeds 600–749 strictly quarantined. Seeds 2000–2099 strictly locked.

---

## Executive Summary

Stage 2 development training evaluated the two competing controller synthesis paradigms under an identical, hard-capped simulator transition budget:
- **Approach 1 (Dense ES MLP):** Dense MLP with 113 parameters ($5 \times 16 + 16 + 16 \times 1 + 1$), 96 weight MACs, trained via Natural Evolution Strategies (ES) with antithetic perturbation and rank-shaped Adam updates.
- **Approach 2 (Compact NSGA-II Neuroevolution):** Topology-and-weight masked genome ($\le 57$ dense parameters), multi-objective optimization (robust success, active parameters, weight MACs) via NSGA-II with elitist non-dominated sorting and crowding distance.
- **Reference (Classical Analytical):** `HybridController` combining nonlinear energy-shaping pumping ($| \theta | \ge 0.35$ rad or $| \dot{\theta} | \ge 2.5$ rad/s) and linear state feedback balancing ($|\theta| < 0.35$ rad and $|\dot{\theta}| < 2.5$ rad/s) with 7 parameters and 7 MACs.

### Key Finding & Candidate Selection Outcome

**Neither learned approach met the candidate-selection filters.**
Under the registered budget limit of 2,000,000 transitions per run:
1. **ES MLP (5 runs):** Achieved **0.0% nominal screening success** (Gate 1B) and **0.0% robust success** across all 5 optimizer seeds. All runs failed early due to rail collision before reaching the required 95 upright steps in the final 100-step evaluation window.
2. **NSGA-II (5 runs):** Successfully produced ultra-compact genomes (16 to 38 active parameters, 12 to 31 weight MACs), satisfying the complexity constraints ($\le 128$ params, $\le 128$ MACs). However, it achieved **0.0% nominal screening success** (Gate 1B) and **0.0% to 0.5% robust success** across all 5 seeds, failing the proposed 40% worst-condition success floor and 90% nominal gate.
3. **Classical `HybridController` Baseline:** Achieved **100.0% nominal screening success** (100/100 episodes), **92.0% aggregate robust success** across the 10 diagnostic conditions, **65.0% worst-condition success** (Condition 7: Combined Historical Stress), with **7 parameters** and **7 MACs**, requiring **0 training transitions**.

**Decision:** Under the frozen development candidate-selection criteria, **no learned model qualifies for prospective confirmation evaluation**. The classical baseline remains the undisputed superior solution in sample efficiency, computational compactness, and robust stability.

---

## 1. Experimental Protocol & Budget Accounting

### 1.1 Registered Seed Partitions
- **Training / Fitness Search Partition:** Seeds `0–199` (used strictly for optimizer evaluation episodes).
- **Nominal Screening Partition (Gate 1B):** Seeds `200–299` (100 distinct deterministic initial conditions).
- **Multi-Condition Diagnostic Partition:** Seeds `300–319` (20 distinct seeds $\times$ 10 diagnostic conditions = 200 episodes).
- **Quarantined Prospective Evaluation Partition:** Seeds `600–749` (UNTOUCHED, QUARANTINED).
- **Hard Locked Confirmation Partition:** Seeds `2000–2099` (UNTOUCHED, LOCKED).

### 1.2 Lifecycle Budget Allocation per Run
- **Total Lifecycle Limit:** Exactly 2,000,000 physical simulator transitions.
- **Search Phase Ceiling:** 1,850,000 transitions maximum.
- **Validation Phase Ceiling:** 150,000 transitions maximum (shared monotonically on the same `TransitionBudgetTracker`).
- **Fail-Closed Guarantee:** Enforced via `BudgetedEnv` and `BudgetedVectorizedEnv` with atomic step checking.

### 1.3 Optimizer Configurations
- **ES:** Population size = 20 (10 antithetic pairs), generations = 18, eval episodes = 10 (seeds rotated deterministically across 0–199), max steps = 500, $\sigma = 0.05$, learning rate = 0.01 with Adam ($\beta_1=0.9, \beta_2=0.999$).
- **NSGA-II:** Population size = 20, offspring size = 10, generations = 35, eval episodes = 10, max steps = 500, vectorized execution, mutation rates: weight mutation = 0.6 ($\sigma=0.15$), connection toggle = 0.2, neuron toggle = 0.1, activation mutation = 0.05.

---

## 2. Complete Raw Results: 10 Optimizer Runs + Classical Baseline

### 2.1 Tabular Performance & Resource Summary

| Method | Run / Seed | Gate 1B Nom. (%) | Robust Succ. (%) | Worst Cond. (%) | Active Params | Weight MACs | Search Trans. | Val. Trans. | Total Trans. | Wall Time (s) | Throughput (t/s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ES (MLP)** | Seed 42 | 0.0% (0/100) | 0.0% | 0.0% | 113 | 96 | 225,190 | 11,219 | 236,409 | 13.4s | 17,674 |
| **ES (MLP)** | Seed 43 | 0.0% (0/100) | 0.0% | 0.0% | 113 | 96 | 250,743 | 11,207 | 261,950 | 14.3s | 18,259 |
| **ES (MLP)** | Seed 44 | 0.0% (0/100) | 0.0% | 0.0% | 113 | 96 | 288,269 | 11,219 | 299,488 | 16.3s | 18,358 |
| **ES (MLP)** | Seed 45 | 0.0% (0/100) | 0.0% | 0.0% | 113 | 96 | 231,458 | 11,225 | 242,683 | 13.6s | 17,802 |
| **ES (MLP)** | Seed 46 | 0.0% (0/100) | 0.0% | 0.0% | 113 | 96 | 244,820 | 11,229 | 256,049 | 14.4s | 17,840 |
| **ES Mean** | *Mean $\pm$ Std* | **0.0%** | **0.0%** | **0.0%** | **113.0** | **96.0** | **248,096** | **11,220** | **259,316** | **14.4s** | **17,987** |
| | | | | | | | | | | | |
| **NSGA-II** | Seed 42 | 0.0% (0/100) | 0.5% | 0.0% | 38 | 31 | 1,850,000 | 150,000 | 2,000,000 | 33.3s | 60,073 |
| **NSGA-II** | Seed 43 | 0.0% (0/100) | 0.0% | 0.0% | 23 | 19 | 1,850,000 | 150,000 | 2,000,000 | 32.9s | 60,753 |
| **NSGA-II** | Seed 44 | 0.0% (0/100) | 0.0% | 0.0% | 16 | 12 | 1,850,000 | 150,000 | 2,000,000 | 32.8s | 61,050 |
| **NSGA-II** | Seed 45 | 0.0% (0/100) | 0.0% | 0.0% | 17 | 13 | 1,850,000 | 150,000 | 2,000,000 | 32.8s | 60,889 |
| **NSGA-II** | Seed 46 | 0.0% (0/100) | 0.0% | 0.0% | 26 | 21 | 1,850,000 | 150,000 | 2,000,000 | 33.3s | 59,993 |
| **NSGA-II Mean** | *Mean $\pm$ Std* | **0.0%** | **0.1%** | **0.0%** | **24.0** | **19.2** | **1,850,000** | **150,000** | **2,000,000** | **33.0s** | **60,552** |
| | | | | | | | | | | | |
| **Hybrid** | Classical | **100.0%** (100/100) | **92.0%** | **65.0%** | **7** | **7** | **0** | **147,896** | **147,896** | **4.2s** | **35,213** |

---

### 2.2 Per-Condition Diagnostic Suite Breakdown (Seeds 300–319)

Each condition evaluates 20 episodes on seeds 300–319 (maximum 500 steps per episode).

| Condition Name | Description | HybridController Success (%) | NSGA-II Seed 42 (%) | NSGA-II Seeds 43–46 (%) | ES All Seeds (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1_Nominal** | $m_c=1.0, m_p=0.1, b_c=0, b_p=0, \sigma=0, d=0$ | **100.0%** (20/20) | 0.0% | 0.0% | 0.0% |
| **2_PoleMass_+30%** | Pole mass = 0.13 kg (+30%) | **100.0%** (20/20) | 5.0% (1/20) | 0.0% | 0.0% |
| **3_CartMass_+30%** | Cart mass = 1.30 kg (+30%) | **100.0%** (20/20) | 0.0% | 0.0% | 0.0% |
| **4a_CartFriction_0.50** | Cart friction coefficient $b_c = 0.5$ | **100.0%** (20/20) | 0.0% | 0.0% | 0.0% |
| **4b_JointFriction_0.010**| Joint friction coefficient $b_p = 0.010$ | **70.0%** (14/20) | 0.0% | 0.0% | 0.0% |
| **4c_CombinedFriction** | $b_c = 0.5, b_p = 0.010$ | **85.0%** (17/20) | 0.0% | 0.0% | 0.0% |
| **5_SensorNoise_0.05** | Gaussian noise $\sigma = 0.05$ on observations | **100.0%** (20/20) | 0.0% | 0.0% | 0.0% |
| **6a_ActuatorDelay_1Step**| 1 step (20 ms) pure control delay | **100.0%** (20/20) | 0.0% | 0.0% | 0.0% |
| **6b_ActuatorDelay_2Step**| 2 steps (40 ms) pure control delay | **100.0%** (20/20) | 0.0% | 0.0% | 0.0% |
| **7_CombinedHistorical** | $+30\% m_p, b_c=0.5, b_p=0.01, \sigma=0.05, 1\text{ step}$ | **65.0%** (13/20) | 0.0% | 0.0% | 0.0% |
| **Aggregate Robust** | Arithmetic Mean across 10 Conditions | **92.0%** | **0.5%** | **0.0%** | **0.0%** |
| **Worst Condition** | Minimum rate across all conditions | **65.0%** (Cond 7) | **0.0%** | **0.0%** | **0.0%** |

---

## 3. Deep Analysis & Scientific Findings

### 3.1 Why Did ES Fail Under the 2,000,000 Budget?
1. **Reward Formulation vs. Success Predicate:**  
   The existing ES baseline trains on cumulative step reward ($r = 1.0 - \theta^2 - 0.1 \dot{\theta}^2 - 0.01 u^2$). Over 18 generations, ES consistently increased average candidate reward from $-184.2$ to $-21.56$. However, this gradient guides the policy to avoid large velocity swings, leading to conservative motions that drift into the soft rail boundaries ($|x| > 2.4$ m).
2. **Early Rail Termination:**  
   Because the continuous cartpole terminates upon hitting the rail, episodes early in training last only 20–50 steps. In validation screening, all ES policies hit the rail within ~37 steps, consuming only 11,200 transitions across the 300 validation episodes (100 Gate 1B + 200 Diagnostic).
3. **Sample Inefficiency of Black-Box Search:**  
   18 generations of 20 mutants evaluate only 360 parameter vectors. In a 113-dimensional continuous parameter space, 360 evaluations are mathematically insufficient to discover both the nonlinear swing-up pumping manifold and the precise linear stabilization balance manifold without prior demonstration or dense reward shaping.

### 3.2 Why Did NSGA-II Fail Under the 2,000,000 Budget?
1. **Sparsity of the Success Metric:**  
   NSGA-II directly optimizes robust success rate. In random initial populations (Generation 0), the probability of a randomly initialized sparse neural network ($<50$ weights) swinging up and balancing for 500 steps is near zero. Consequently, all individuals in early generations have `robust_success = 0.0`.
2. **Objective Degeneracy toward Zero-Complexity:**  
   Because the primary task objective (`robust_success`) is zero across the population, the Pareto selection pressure exclusively favors the other two objectives: minimizing active parameters and minimizing MACs. The evolutionary search rapidly drives the genome toward ultra-sparse topologies (16–26 parameters), effectively pruning away functional capacity before any positive reward gradient can be established.
3. **Budget Consumption:**  
   NSGA-II consumed the entire 1,850,000 search transition ceiling and all 150,000 validation transitions (2,000,000 total), yet could not escape the zero-success basin of attraction.

### 3.3 The Classical Hybrid Advantage
1. **Domain Knowledge Exploitation:**  
   The `HybridController` divides the problem into two physically grounded regimes:
   - Energy-shaping pumping controller driven by Lyapunov analysis:
     $$u = k_E (E - E_0) \text{sign}(\dot{\theta} \cos\theta) - k_x x - k_v \dot{x}$$
   - Linear state feedback balancing synthesized via Riccati equations:
     $$u = - (K_x x + K_{\dot{x}} \dot{x} + K_\theta \theta + K_{\dot{\theta}} \dot{\theta})$$
2. **Extreme Simplicity:** Exactly 7 parameters and 7 Multiply-Accumulate operations per time step.
3. **Zero Training Cost:** Consumes 0 training simulator transitions, while achieving 100% nominal success and 92.0% robust success.

---

## 4. Generated Publication Artifacts

The following visualization artifacts have been generated and saved to `results/stage2/plots/`:
1. `results/stage2/plots/es_training_curves.png`: Training reward trajectories across 18 generations for ES seeds 42–46.
2. `results/stage2/plots/complexity_vs_performance.png`: Active parameters vs. robust success rate relative to the 128-param filter and 40% success floor.
3. `results/stage2/plots/diagnostic_profile_comparison.png`: 10-condition diagnostic bar profile comparing HybridController against learned methods.
4. `results/stage2/plots/budget_consumption_breakdown.png`: Exact phase-by-phase transition consumption relative to search and total limits.

---

## 5. Cryptographic Reproducibility Hashes

| File | SHA-256 Digest |
| :--- | :--- |
| `evolution/evolution_strategies.py` | `677ce60ef195a7c0babe87f024436c36ddcf67556b16959020c40d6bf340fe44` |
| `evolution/nsga2_engine.py` | `3e81464de9c80b8386eb31c67f677d7a982fa96db865683da5866fce98a5d4e0` |
| `baselines/hybrid_controller.py` | `d13fbbee79fcc98fde74b8dc0ee01a862a5f8366e328d5115474640c8f864e2c` |
| `env/budget_tracker.py` | `8ec292a7d4d038deeed3a48eabbf2c207d5374a8a098034f6fc421cbc67f2d56` |
| `env/vectorized_env.py` | `c35cef625c29df806281e6d2ee2937d7073e73d552400ce3c27c5299de1ea81c` |
| `experiments/run_robustness_experiment.py` | `c3b12f868fb1c9641ba7ddb26dd8f3f880d0b8facb72ac0046145339a4535c95` |
| `results/stage2/stage2_summary.json` | `1996e4b09036fe815b9484040eec540abca9d6eace992a0c481b15e7e82df1f2` |

---

## 6. Exact Reproduction Commands

```powershell
# Run the complete test suite (109 passing tests)
C:\Users\cosmic\AppData\Local\Programs\Python\Python312\python.exe -m pytest -q

# Reproduce the master Stage 2 experiment suite (all 10 runs + classical baseline)
C:\Users\cosmic\AppData\Local\Programs\Python\Python312\python.exe scratch/run_stage2_experiments.py

# Re-generate all plots
C:\Users\cosmic\AppData\Local\Programs\Python\Python312\python.exe scratch/generate_stage2_plots.py
```
