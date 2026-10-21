# CartPole Swing-Up & Stabilization Experiments

**Benchmark Target:** CartPole Inverted Pendulum on a Track
**Environment Implementations:** `env/cartpole_swingup.py`, `env/vectorized_env.py`
**Evaluation Partitions:**
- Development Partition: Seeds `200..209` (10 seeds × 10 conditions = 100 episodes/controller)
- Historical Phase 3E Holdout: Seeds `2000..2099` (100 seeds × 10 conditions = 1,000 episodes)
- Prospective Partition: Seeds `3000..3099` (**BLOCKED**, unconsumed)

---

## 1. Physical Task Definition & Success Predicate

### 1.1 Physical Parameters
The system consists of an unactuated rigid pole attached to a motorized cart traversing a horizontal rail:
- Cart Mass $M_c = 1.0\text{ kg}$ (nominal), Pole Mass $m_p = 0.1\text{ kg}$ (nominal), Pole Half-Length $l = 0.5\text{ m}$.
- Rail displacement limit: $|x| < 2.4\text{ m}$.
- Gravity: $g = 9.8\text{ m/s}^2$.
- Simulation step: $\Delta t = 0.02\text{ s}$ ($50\text{ Hz}$).
- Actuator force ceiling: Continuous force magnitude $F_{\text{mag}} = 10.0\text{ N}$ (canonical continuous environment), with up to $\pm 20.0\text{ N}$ dynamic recovery headroom under combined stress perturbations.
- Initial State: Downward hanging pole $\theta_0 = \pi\text{ rad}$, zero cart velocity $\dot{x}_0 = 0$, zero angular velocity $\dot{\theta}_0 = 0$.

### 1.2 Binary Success Predicate
An episode succeeds if and only if:
1. The cart remains strictly within rail boundaries ($|x_t| < 2.4\text{ m}$) for all 500 steps ($10.0\text{ s}$).
2. The pole is balanced upright for at least 95 of the final 100 timesteps ($t \in [401, 500]$):
   $$\sum_{t=401}^{500} \mathbb{I}\left( \cos\theta_t > 0.90 \;\land\; |\dot{\theta}_t| < 2.0\text{ rad/s} \right) \ge 95$$

---

## 2. Experimental Controllers Under Comparison

| Controller | Class / Implementation | Active Parameters | Weight MACs | Memory Footprint | Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **EvoEdge Champion** | `MaskedGenome` (`stage3d_seed46`) | **69** | **60** | 0 B Heap / 64 B Stack | Frozen Research Candidate |
| **OpenAI-ES Baseline v4 (Seed 43)** | `MLP64x64Controller` (`v4_seed43`) | 4,737 | 4,608 | 0 B Heap / ~1 KB Stack | Best Baseline Seed |
| **OpenAI-ES Baseline v4 (Seed 44)** | `MLP64x64Controller` (`v4_seed44`) | 4,737 | 4,608 | 0 B Heap / ~1 KB Stack | Median Baseline Seed |
| **OpenAI-ES Baseline v4 (Seed 42)** | `MLP64x64Controller` (`v4_seed42`) | 4,737 | 4,608 | 0 B Heap / ~1 KB Stack | Minimum Baseline Seed |
| **Classical Reference** | `HybridController` | **7** | **7** | 0 B Heap / 32 B Stack | Analytical Benchmark |

---

## 3. Verified Development Evaluation Results (Seeds 200..209)

Across 10 registered diagnostic perturbation conditions (10 evaluation seeds $\times$ 10 conditions = 100 episodes per controller):

### 3.1 Aggregate Performance Metrics

| Controller | Successes / Total | Success Rate | 95% Wilson Score CI | Worst-Condition Success | Rail Crashes | Dwell Shortfalls |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EvoEdge Champion (Seed 46)** | **99 / 100** | **99.0%** | **[94.6%, 99.8%]** | **90.0%** | **0** | **1** |
| **OpenAI-ES Baseline (Seed 43 - Best)** | 98 / 100 | **98.0%** | [93.0%, 99.4%] | 80.0% | 0 | 2 |
| **OpenAI-ES Baseline (Seed 44 - Mid)** | 95 / 100 | **95.0%** | [88.8%, 97.8%] | 70.0% | 0 | 5 |
| **OpenAI-ES Baseline (Seed 42 - Min)** | 91 / 100 | **91.0%** | [83.8%, 95.2%] | 50.0% | 0 | 9 |
| **Classical Reference (`HybridController`)** | 84 / 100 | **84.0%** | [75.6%, 89.9%] | 20.0% | 4 | 12 |

### 3.2 Per-Condition Success Breakdown (10 Episodes per Cell)

| Perturbation Condition | EvoEdge (Seed 46) | v4 Baseline (Seed 42) | v4 Baseline (Seed 43) | v4 Baseline (Seed 44) | Classical Hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1: Nominal** ($M_c=1.0, m_p=0.1$) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| **2: Pole Mass +30%** ($m_p=0.13$) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| **3: Cart Mass +30%** ($M_c=1.30$) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| **4a: Cart Friction 0.50** ($b_c=0.50$) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| **4b: Joint Friction 0.010** ($b_p=0.010$) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 8/10 (80%) |
| **4c: Combined Friction** ($b_c=0.50, b_p=0.010$) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| **5: Sensor Noise 0.05** ($\sigma=0.05$) | 9/10 (90%) | 7/10 (70%) | 10/10 (100%) | 7/10 (70%) | 2/10 (20%) |
| **6a: Actuator Delay 1-Step** ($\tau=20\text{ ms}$) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| **6b: Actuator Delay 2-Step** ($\tau=40\text{ ms}$) | 10/10 (100%) | 9/10 (90%) | 10/10 (100%) | 10/10 (100%) | 6/10 (60%) |
| **7: Combined Stress** (All Perturbations) | 10/10 (100%) | 5/10 (50%) | 8/10 (80%) | 8/10 (80%) | 8/10 (80%) |
| **Worst-Condition Success Rate** | **90.0%** | **50.0%** | **80.0%** | **70.0%** | **20.0%** |

### 3.3 Paired Statistical Hypothesis Tests
Pairwise differences $\Delta = p_{\text{EvoEdge}} - p_{\text{Baseline}}$ evaluated via clustered bootstrap resampling ($B=1,000$ clusters over 10 evaluation seeds):

| Comparison | Point Diff ($\Delta$) | 95% Clustered Bootstrap CI | Paired Record (Wins / Ties / Losses) | Statistical Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **EvoEdge vs. Classical Hybrid** | **+15.0%** | **[+11.0%, +19.0%]** | 16 / 83 / 1 | EvoEdge statistically superior ($p < 0.001$) |
| **EvoEdge vs. Baseline Seed 42** | **+8.0%** | **[+3.0%, +12.0%]** | 9 / 90 / 1 | EvoEdge statistically superior ($p < 0.01$) |
| **EvoEdge vs. Baseline Seed 43** | **+1.0%** | **[-2.0%, +4.0%]** | 2 / 97 / 1 | Statistically equivalent (Non-inferior) |
| **EvoEdge vs. Baseline Seed 44** | **+4.0%** | **[-1.0%, +9.0%]** | 5 / 94 / 1 | Statistically non-inferior |

---

## 4. Failure Mode Analysis

Across all 500 executed development episodes:
1. **Rail Collisions ($|x| \ge 2.4\text{ m}$):**
   - EvoEdge: **0 episodes** (0.0%).
   - v4 Learned Baseline (all seeds): **0 episodes** (0.0%).
   - Classical Hybrid Reference: **4 episodes** (4.0%) — all occurred under Condition 6b (Actuator Delay 2-Step) due to phase lag during position recentering.
2. **Terminal Dwell Shortfalls (Survived 500 steps, but upright for $<95$ steps):**
   - EvoEdge: **1 episode** (Condition 5 Sensor Noise, Seed 205: held upright for 93 steps, narrowly missing the 95-step threshold).
   - Baseline Seed 42: **9 episodes** (5 in Combined Stress, 3 in Sensor Noise, 1 in 2-step Delay).
   - Baseline Seed 43: **2 episodes** (both in Combined Stress).
   - Baseline Seed 44: **5 episodes** (3 in Sensor Noise, 2 in Combined Stress).
   - Classical Hybrid: **12 episodes** (8 in Sensor Noise, 4 in 2-step Delay).

---

## 5. Historical Sprint 3C Defect Resolution

Prior reports in Sprint 3C claimed that the full-scale 64×64 MLP baseline collapsed (scoring 0%, 33%, and 1%). An independent forensic audit proved that this was caused by four implementation defects, which were corrected in v4:

1. **Teacher Gain Factor Omission:** The teacher model (`results/mlp64_baseline.pt`) applied $\tanh(2.5 \cdot \mathbf{W}\mathbf{x} + \mathbf{b})$. When exported to NumPy, the factor of 2.5 was omitted, causing a $(2.5)^3 \approx 15.6\times$ slope attenuation. *Correction:* Absorbed $2.5\times$ into weights and biases, verifying trajectory error $< 7.31 \times 10^{-5}$.
2. **Single-Condition Overfitting:** ES sampled only one condition per generation. *Correction:* Enforced a 5-condition representative schedule evaluated in lockstep across all candidates.
3. **Reward Loopholes:** Late-episode balance was not differentiated from swinging. *Correction:* Added terminal dwell reward (+4.0 for balance; -2.0 for swinging).
4. **Transition Starvation:** ES ran only 41 updates before budget exhaustion. *Correction:* Checked transition cap before every physical step, ensuring exactly 1,850,000 transitions.

---

## 6. CartPole Task A: Operating Regime Variation Audit

In addition to Task B (Swing-Up from $\theta_0 = \pi$), Task A (Inverted Balancing from small initial tilt $|\theta_0| \le 0.2\text{ rad}$) was evaluated across seeds `210..219`:
- **Result:** EvoEdge achieved **100/100 (100.0%)** zero-shot success, matching Classical LQR (100%) and MLP64 (100%).
- **Angle Basin Sweep ($N=20$/cell):** Linear LQR collapses completely ($0/20$) at $\theta_0 \ge 0.60\text{ rad}$ ($34.4^\circ$). EvoEdge exhibits directional recovery at large angles (recovering 10/20 at 0.40, 0.60, 0.80 rad, but failing from negative tilt at 0.50 and 0.70 rad due to directional weight bias).
- **Audit Classification:** Task A is formally classified as an *operating regime variation on the identical CartPole plant*, not as a distinct physical control benchmark.

---

## 7. Open Governance Gate: Prospective Evaluation

- **Status:** **BLOCKED (UNAUTHORIZED)**.
- **Partition:** Seeds `3000..3099` remain unconsumed and strictly preserved.
- **Prerequisite:** Prospective evaluation requires independent verification of a human cryptographic signature on the authorization manifest. Because no human signature was provided, the prospective experiment was deliberately withheld to preserve scientific integrity.
