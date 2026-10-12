# EvoEdge — Phase 2B Evidence and Methodology Audit Report

**Date:** 2026-10-12  
**Baseline Git Commit:** `2da1aef`  
**Configuration File:** `config/perturbations.yaml` (v2.4)  
**Configuration SHA-256:** `984053987fa92679264faeb417435f3df3b050cfefc88e7aebec4ba1481b3765`  
**Audit Scope:** Verification of raw episode data, distribution provenance, previous-action mechanism, production tests, and statistical interpretation.

---

## 1. Executive Summary & Audit Gate Statuses

An independent audit of Phase 2B evidence and methodology was conducted across all 6,500 validation episodes (`results/phase_2b_episode_validation.csv`), model training provenance, previous-action feedback paths, test assertions, and statistical reporting.

### Audit Gate Evaluation

| Audit Item | Description | Status | Evidence / Verification |
|---|---|---|---|
| **Task 1: Episode Data Reconciliation** | Reconcile every cell between raw episode CSV, summary CSV, and report text. | **PASS** | 0 discrepancies found between raw episodes and `phase_2b_aggregated_summary.csv`. Narrative claims in report reconciled with exact data. |
| **Task 2: Training Distribution Provenance** | Audit sampler bounds vs YAML protocol; verify trace to models and deterministic reproducibility. | **PASS (with Disclosed Bound Variance)** | Initial exploratory training bounds were slightly tighter than v2.4 protocol maxima; labeled as exploratory reconstructed distributions. Sampler updated to use YAML as source of truth; automated test added. |
| **Task 3: Previous-Action Hypothesis** | Audit inference path, queue reset, model weights, and delay observability. | **PASS** | $a_{t-1}$ verified to feed plant-applied action, initialized to 0.0, zero cross-episode leakage. Offline weight audit confirms active usage (output sensitivity range 0.06–0.87). Limitations of partial observability under 2-step delay disclosed. |
| **Task 4: Automated Test Strengthening** | Production helper for success predicate, edge cases, YAML sampler test. | **PASS** | Canonical `evaluate_success_predicate()` implemented in `run_robustness_experiment.py`. 7/7 tests pass in `tests/test_robustness_experiment.py`, including exact 95/94 cutoff, strict inequality, rail priority, and short episode failure. |
| **Task 5: Statistical Interpretation** | Run-level reporting, variance disclosure, exploratory labeling. | **PASS** | Generated `results/phase_2b_run_level_summary.csv` with mean, standard deviation, min, and max across the 3 independent runs. Disclosed that $N=3$ runs is exploratory and that pooled success rates obscure wide run-to-run variation. |

---

## 2. Task 1: Episode-Level Data Reconciliation

### 2.1 Recomputation vs. `phase_2b_aggregated_summary.csv`
Every episode in `results/phase_2b_episode_validation.csv` (6,500 rows) was independently grouped by controller and condition. For all 130 controller-condition combinations:
- Total episodes ($N=50$), success count, rail-hit count, and other-failure count match `phase_2b_aggregated_summary.csv` with **zero discrepancies**.
- Recomputed Wilson 95% confidence intervals match within rounding precision ($\pm 0.1\%$).

### 2.2 Reconciled Narrative Corrections in `results/phase_2b_report.md`
Prior narrative claims in `results/phase_2b_report.md` implied that Arm D was uniquely capable of solving the combined conditions. The primary episode evidence reveals:
1. **Condition 4c (Combined Friction):**
   - **Arm A (nominal, 5-in):** Achieved **48/150 (32.0%)** pooled success, driven entirely by **seed 43 which achieved 48/50 (96.0%)**.
   - **Arm D (robust, 6-in):** Achieved **29/150 (19.3%)** pooled success, driven entirely by **seed 43 which achieved 29/50 (58.0%)**.
   - **Arms B & C:** Achieved 0/150 (0.0%) and 2/150 (1.3%).
2. **Condition 7 (Combined Historical Stress):**
   - **Arm A (nominal, 5-in):** Achieved **13/150 (8.7%)** pooled success (seed 43 achieved 13/50, 26.0%).
   - **Arm B (robust, 5-in):** Achieved **1/150 (0.7%)** (seed 42 achieved 1/50).
   - **Arm C (nominal, 6-in):** Achieved **2/150 (1.3%)** (seed 44 achieved 2/50).
   - **Arm D (robust, 6-in):** Achieved **33/150 (22.0%)** (seed 43 achieved 33/50, 66.0%).

**Correction Applied:** Arm D is **not uniquely capable** of solving these stress conditions. Arm A seed 43 also solved Combined Friction and Combined Stress without the 6th input. Performance on these extreme friction conditions is characterized by high optimizer sensitivity: certain training trajectories discover aggressive pumping limit cycles while others stall or slam into the rails.

---

## 3. Task 2: Training Distribution Provenance Audit

### 3.1 Provenance of Saved Model Checkpoints
The 12 saved PyTorch models (`results/models/arm_a_seed42.pt` through `arm_d_seed44.pt`) were produced during the Phase 2B experiment run using `sample_training_condition()`.

| Parameter | YAML Protocol v2.4 Declared Max | Exploratory Sampler Upper Bound Used | Status |
|---|---|---|---|
| **Cart Mass ($m_c$)** | $[0.85, 1.30]\text{ kg}$ | $[0.85, 1.25]\text{ kg}$ | Sub-range of declared protocol |
| **Pole Mass ($m_p$)** | $[0.08, 0.130]\text{ kg}$ | $[0.08, 0.125]\text{ kg}$ | Sub-range of declared protocol |
| **Cart Friction ($b_c$)** | $[0.00, 0.50]\text{ N}\cdot\text{s/m}$ | $[0.00, 0.35]\text{ N}\cdot\text{s/m}$ | Sub-range of declared protocol |
| **Joint Friction ($b_p$)** | $[0.00, 0.010]\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | $[0.00, 0.007]\text{ N}\cdot\text{m}\cdot\text{s/rad}$ | Sub-range of declared protocol |
| **Observation Noise ($\sigma$)** | $[0.00, 0.050]$ | $[0.00, 0.035]$ | Sub-range of declared protocol |
| **Actuator Delay ($d$)** | $\{0, 1, 2\}$ control steps | $\{0, 1, 2\}$ control steps | **Exact match** |

### 3.2 Audit Classification
- **Exploratory Status:** Because the initial exploratory run sampled from slightly narrower upper bounds ($b_c \le 0.35$ vs $0.50$, $b_p \le 0.007$ vs $0.010$, $\sigma \le 0.035$ vs $0.050$), these models are classified as **reconstructed exploratory models**, not strictly trained under the expanded v2.4 maximum boundaries.
- **Reproducibility:** Given the optimizer seeds (`42, 43, 44`) and the exact RNG initialization (`np.random.default_rng(optimizer_seed)`), every training rollout condition can be deterministically reconstructed.
- **Production Guard Implemented:** `sample_training_condition()` in `experiments/run_robustness_experiment.py` has been refactored to take `suite: Optional[PerturbationSuite]` and query bounds directly from the loaded YAML configuration. An automated test (`test_training_distribution_sampling_matches_yaml`) enforces that the sampler dynamically matches `config/perturbations.yaml`.

---

## 4. Task 3: Previous-Action Hypothesis & Mechanistic Analysis

### 4.1 Queue & Input Semantics Verification
- **Plant Feedback:** The step return `info["applied_action"]` is verified to feed $a_{t-1}$.
- **Initialization:** At $t=0$, `prev_action_applied = 0.0`. Under delay $d=1$ or $d=2$, initial applied actions popped from the environment queue are $0.0$.
- **Inter-Episode Isolation:** `env.reset()` completely purges the deque of pending actions. Zero state or action leakage occurs between episodes.

### 4.2 Offline Model Weight & Sensitivity Audit
Weights in layer 1 (`fc1.weight[:, 5]`) connecting the 6th input ($a_{t-1}$) were examined for all saved 6-input models:

| Model Checkpoint | Input 5 Mean \|Weight\| | Input 5 Max \|Weight\| | Output Response Range (Holding State Fixed, Varying $a_{t-1} \in [-1, 1]$) |
|---|---|---|---|
| `arm_c_seed42.pt` | 0.1366 | 0.3843 | 0.8652 |
| `arm_c_seed43.pt` | 0.2825 | 0.5710 | 0.4367 |
| `arm_c_seed44.pt` | 0.3249 | 0.9568 | 0.7096 |
| `arm_d_seed42.pt` | 0.1266 | 0.1742 | 0.4768 |
| `arm_d_seed43.pt` | 0.1207 | 0.2541 | 0.0599 |
| `arm_d_seed44.pt` | 0.0373 | 0.1446 | 0.3363 |

**Findings:**
1. All 6-input models actively utilize the previous-action input, producing non-zero output shifts across varying $a_{t-1}$.
2. In Arm D seed 44 (which achieved 50/50 at $d=1$ and 42/50 at $d=2$), varying $a_{t-1}$ across $[-1, 1]$ modulates the output action by up to 0.336 (over 16% of total action span $[-1, 1]$).

### 4.3 Causality vs. Correlation & Limitations of Partial Observability
- **Delay Observability:** Under a 2-step transport delay ($d=2$), the plant is driven by $a_{t-2}$. The queue holds $[a_{t-1}, a_t]$. Providing only $a_{t-1}$ (the action applied at $t-1$) provides partial information about the immediate past plant command, but does not fully reveal the unexecuted action pipeline ($a_{t-2}$ and $a_{t-1}$).
- **Teacher Labeling Effect:** The expert teacher computes action $a^* = \pi^*(s)$ based solely on the physical 5-state observation $s$. The teacher target does not explicitly condition on $a_{t-1}$. Therefore, during DAgger imitation, the network learns a conditional policy $\pi(s, a_{t-1})$ where the 6th feature can act as a regularizer or implicit state-derivative corrector along the student's rollout distribution.
- **Causal Disclaimer:** The observed improvement in Arm D at $d=2$ (49.3% vs. Arm B's 6.0%) is an **empirical association under robust training**, not formal mathematical proof of optimal lead compensation.

---

## 5. Task 4: Automated Test Strengthening

The test suite in [`tests/test_robustness_experiment.py`](file:///d:/neat/tests/test_robustness_experiment.py) was expanded and unified with the production evaluation code.

### Canonical Production Helper
`evaluate_success_predicate()` was extracted in `experiments/run_robustness_experiment.py`:
```python
def evaluate_success_predicate(
    cos_thetas: List[float],
    rail_hit: bool,
    max_steps: int = 500,
    required_upright_in_window: int = 95,
    upright_threshold: float = 0.9,
    window_size: int = 100,
) -> Tuple[bool, bool, bool, int]:
```

### Verified Test Cases (7/7 Unit Tests Passing)
1. **`test_canonical_success_predicate`**:
   - Exactly 95 steps with $\cos\theta > 0.9$ in final 100 steps $\rightarrow$ `SUCCESS` (`True, False, False`).
   - Exactly 94 steps with $\cos\theta > 0.9$ in final 100 steps $\rightarrow$ `OTHER_FAILURE` (`False, False, True`).
   - Upright value strictly equal to 0.9000 $\rightarrow$ does NOT count as $> 0.9$.
   - Any rail collision $\rightarrow$ `RAIL_HIT` (`False, True, False`) regardless of upright trajectory.
   - Episode shorter than 500 steps (e.g. 499) $\rightarrow$ `OTHER_FAILURE`.
   - Partition check: `is_success + is_rail_hit + is_other_failure == 1` always holds.
2. **`test_previous_action_queue_semantics`**: Verified step-by-step queue pipeline under 2-step delay.
3. **`test_episode_reset_clears_queue_and_state`**: Verified zero inter-episode action queue leakage.
4. **`test_actual_inference_path_receives_previous_applied_action`**: Verified production inference feeds applied plant action.
5. **`test_architecture_parameter_and_mac_accounting`**: Verified 57 params/48 MACs (5-in) vs 65 params/56 MACs (6-in).
6. **`test_training_distribution_sampling_matches_yaml`**: 500 samples verified to lie strictly within YAML ranges.
7. **`test_seed_partition_guard`**: Verified runtime exception if seeds `2000..2099` are accessed for training or validation.

---

## 6. Task 5: Corrected Statistical Interpretation

### 6.1 Run-Level Performance Summary
The table below (exported to [`results/phase_2b_run_level_summary.csv`](file:///d:/neat/results/phase_2b_run_level_summary.csv)) shows the 3 individual optimizer runs (seeds 42, 43, 44), run-level mean, standard deviation, min, and max for each arm and condition ($N=50$ episodes per run, 150 pooled):

| Arm | Condition | Seed 42 Rate | Seed 43 Rate | Seed 44 Rate | Run Mean | Run Std | Run Min | Run Max | Pooled Success |
|---|---|---|---|---|---|---|---|---|---|
| **Arm A** | 1_Nominal | 0.0% | 30.0% | 96.0% | 42.0% | 49.1% | 0.0% | 96.0% | 42.0% (63/150) |
| **Arm A** | 2_PoleMass_+30% | 0.0% | 40.0% | 58.0% | 32.7% | 29.7% | 0.0% | 58.0% | 32.7% (49/150) |
| **Arm A** | 3_CartMass_+30% | 0.0% | 100.0% | 34.0% | 44.7% | 50.8% | 0.0% | 100.0% | 44.7% (67/150) |
| **Arm A** | 4a_CartFriction_0.50 | 0.0% | 32.0% | 28.0% | 20.0% | 17.4% | 0.0% | 32.0% | 20.0% (30/150) |
| **Arm A** | 4b_JointFriction_0.010 | 0.0% | 80.0% | 82.0% | 54.0% | 46.8% | 0.0% | 82.0% | 54.0% (81/150) |
| **Arm A** | 4c_CombinedFriction | 0.0% | 96.0% | 0.0% | 32.0% | 55.4% | 0.0% | 96.0% | 32.0% (48/150) |
| **Arm A** | 5_SensorNoise_0.05 | 0.0% | 84.0% | 78.0% | 54.0% | 46.9% | 0.0% | 84.0% | 54.0% (81/150) |
| **Arm A** | 6a_ActuatorDelay_1Step | 2.0% | 92.0% | 100.0% | 64.7% | 54.4% | 2.0% | 100.0% | 64.7% (97/150) |
| **Arm A** | 6b_ActuatorDelay_2Step | 0.0% | 100.0% | 90.0% | 63.3% | 55.1% | 0.0% | 100.0% | 63.3% (95/150) |
| **Arm A** | 7_CombinedHistoricalStress | 0.0% | 26.0% | 0.0% | 8.7% | 15.0% | 0.0% | 26.0% | 8.7% (13/150) |
| **Arm B** | 1_Nominal | 96.0% | 100.0% | 42.0% | 79.3% | 32.4% | 42.0% | 100.0% | 79.3% (119/150) |
| **Arm B** | 2_PoleMass_+30% | 100.0% | 100.0% | 38.0% | 79.3% | 35.8% | 38.0% | 100.0% | 79.3% (119/150) |
| **Arm B** | 3_CartMass_+30% | 98.0% | 100.0% | 48.0% | 82.0% | 29.5% | 48.0% | 100.0% | 82.0% (123/150) |
| **Arm B** | 4a_CartFriction_0.50 | 76.0% | 86.0% | 0.0% | 54.0% | 47.0% | 0.0% | 86.0% | 54.0% (81/150) |
| **Arm B** | 4b_JointFriction_0.010 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% (0/150) |
| **Arm B** | 4c_CombinedFriction | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% (0/150) |
| **Arm B** | 5_SensorNoise_0.05 | 94.0% | 100.0% | 70.0% | 88.0% | 15.9% | 70.0% | 100.0% | 88.0% (132/150) |
| **Arm B** | 6a_ActuatorDelay_1Step | 90.0% | 42.0% | 64.0% | 65.3% | 24.0% | 42.0% | 90.0% | 65.3% (98/150) |
| **Arm B** | 6b_ActuatorDelay_2Step | 12.0% | 0.0% | 6.0% | 6.0% | 6.0% | 0.0% | 12.0% | 6.0% (9/150) |
| **Arm B** | 7_CombinedHistoricalStress | 2.0% | 0.0% | 0.0% | 0.7% | 1.2% | 0.0% | 2.0% | 0.7% (1/150) |
| **Arm C** | 1_Nominal | 0.0% | 82.0% | 78.0% | 53.3% | 46.3% | 0.0% | 82.0% | 53.3% (80/150) |
| **Arm C** | 2_PoleMass_+30% | 0.0% | 92.0% | 80.0% | 57.3% | 50.0% | 0.0% | 92.0% | 57.3% (86/150) |
| **Arm C** | 3_CartMass_+30% | 2.0% | 70.0% | 76.0% | 49.3% | 41.1% | 2.0% | 76.0% | 49.3% (74/150) |
| **Arm C** | 4a_CartFriction_0.50 | 0.0% | 12.0% | 60.0% | 24.0% | 31.7% | 0.0% | 60.0% | 24.0% (36/150) |
| **Arm C** | 4b_JointFriction_0.010 | 0.0% | 2.0% | 12.0% | 4.7% | 6.4% | 0.0% | 12.0% | 4.7% (7/150) |
| **Arm C** | 4c_CombinedFriction | 0.0% | 0.0% | 4.0% | 1.3% | 2.3% | 0.0% | 4.0% | 1.3% (2/150) |
| **Arm C** | 5_SensorNoise_0.05 | 0.0% | 62.0% | 74.0% | 45.3% | 39.7% | 0.0% | 74.0% | 45.3% (68/150) |
| **Arm C** | 6a_ActuatorDelay_1Step | 4.0% | 62.0% | 82.0% | 49.3% | 40.5% | 4.0% | 82.0% | 49.3% (74/150) |
| **Arm C** | 6b_ActuatorDelay_2Step | 2.0% | 62.0% | 8.0% | 24.0% | 33.1% | 2.0% | 62.0% | 24.0% (36/150) |
| **Arm C** | 7_CombinedHistoricalStress | 0.0% | 0.0% | 4.0% | 1.3% | 2.3% | 0.0% | 4.0% | 1.3% (2/150) |
| **Arm D** | 1_Nominal | 22.0% | 94.0% | 100.0% | 72.0% | 43.4% | 22.0% | 100.0% | 72.0% (108/150) |
| **Arm D** | 2_PoleMass_+30% | 50.0% | 94.0% | 72.0% | 72.0% | 22.0% | 50.0% | 94.0% | 72.0% (108/150) |
| **Arm D** | 3_CartMass_+30% | 92.0% | 16.0% | 20.0% | 42.7% | 42.8% | 16.0% | 92.0% | 42.7% (64/150) |
| **Arm D** | 4a_CartFriction_0.50 | 94.0% | 82.0% | 32.0% | 69.3% | 32.9% | 32.0% | 94.0% | 69.3% (104/150) |
| **Arm D** | 4b_JointFriction_0.010 | 2.0% | 0.0% | 0.0% | 0.7% | 1.2% | 0.0% | 2.0% | 0.7% (1/150) |
| **Arm D** | 4c_CombinedFriction | 0.0% | 58.0% | 0.0% | 19.3% | 33.5% | 0.0% | 58.0% | 19.3% (29/150) |
| **Arm D** | 5_SensorNoise_0.05 | 20.0% | 74.0% | 98.0% | 64.0% | 40.0% | 20.0% | 98.0% | 64.0% (96/150) |
| **Arm D** | 6a_ActuatorDelay_1Step | 20.0% | 12.0% | 100.0% | 44.0% | 48.7% | 12.0% | 100.0% | 44.0% (66/150) |
| **Arm D** | 6b_ActuatorDelay_2Step | 64.0% | 0.0% | 84.0% | 49.3% | 43.9% | 0.0% | 84.0% | 49.3% (74/150) |
| **Arm D** | 7_CombinedHistoricalStress | 0.0% | 66.0% | 0.0% | 22.0% | 38.1% | 0.0% | 66.0% | 22.0% (33/150) |

### 6.2 Disclosed Statistical Limitations
1. **Sample Size:** With only $N=3$ training seeds per arm, between-model variance dominates the comparison. Standard deviations range from $0.0\%$ to $55.4\%$.
2. **Exploratory Status:** Comparisons must be interpreted as **exploratory observational patterns**, not confirmed population-level effects.
3. **Wilson Confidence Intervals:** The Wilson intervals reported in the aggregate summary quantify episode sampling uncertainty for each specific trained artifact; they do **not** bound the variance of the underlying training process across random initializations.
