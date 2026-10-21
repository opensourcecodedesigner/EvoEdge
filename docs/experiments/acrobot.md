# Acrobot-v1 Distinct Benchmark Experiments

**Target Environment:** Gymnasium `Acrobot-v1` (Underactuated Double Pendulum)  
**Workspace:** `work/phase3e_followup_6h/sprint3/research_upgrade/second_task/`  
**Governing Protocol:** Preregistered Protocol v1.1 ([`PROTOCOL.md`](../../work/phase3e_followup_6h/sprint3/research_upgrade/second_task/PROTOCOL.md))  
**Evaluation Partition:** Seeds `500..549` (50 seeds × 5 conditions = 250 episodes per controller)  
**Reserved Partition:** Seeds `1500..1599` (**STRICTLY PRESERVED / UNTOUCHED**)

---

## 1. Environment & Dynamic Distinctness

### 1.1 Why Acrobot is a Distinct Control Benchmark
Unlike CartPole (a single inverted pendulum on an actuated linear cart), the Acrobot is an **underactuated two-link planar robotic arm** operating in a vertical plane under gravity:
- **Underactuation:** Torque is applied **only at the second joint (the elbow)**; the shoulder joint is completely unactuated.
- **Chaotic Dynamics:** The two-link equations of motion exhibit severe nonlinear coupling, non-minimum phase behavior, and chaotic phase portraits.
- **Control Strategy:** Pumping kinetic energy through elbow oscillations to swing the unactuated shoulder upwards past the horizontal target line.

### 1.2 State Space & Normalization ($D=6$)
The observation vector is continuous and 6-dimensional:
$$\mathbf{x} = \left[ \cos\theta_1,\, \sin\theta_1,\, \cos\theta_2,\, \sin\theta_2,\, \text{clip}\left(\frac{\dot\theta_1}{4\pi}, -1, 1\right),\, \text{clip}\left(\frac{\dot\theta_2}{9\pi}, -1, 1\right) \right]$$
- Joint angle $\theta_1 = 0$ corresponds to link 1 hanging straight **downwards** in the stable resting position.
- Joint angle $\theta_2 = 0$ is the relative angle between link 2 and link 1.

### 1.3 Action Space (Discrete 3-Way Torque)
Continuous network output $y \in [-1.0, 1.0]$ is thresholded into discrete torque $u \in \{-1.0, 0.0, +1.0\}\text{ N}\cdot\text{m}$:
$$u = \begin{cases} -1.0\text{ N}\cdot\text{m} & \text{if } y < -0.33 \\ 0.0\text{ N}\cdot\text{m} & \text{if } |y| \le 0.33 \\ +1.0\text{ N}\cdot\text{m} & \text{if } y > 0.33 \end{cases}$$

### 1.4 Binary Success Predicate & 5 Diagnostic Conditions
An episode succeeds if and only if the tip of the second link crosses above the target horizontal line within 500 steps ($10.0\text{ s}$):
$$-\cos\theta_1 - \cos(\theta_1 + \theta_2) > 1.0 \quad \text{within } 500\text{ timesteps}$$

**Diagnostic Robustness Conditions:**
1. **1_Nominal:** Standard Gymnasium physical constants ($l_1=1.0, l_2=1.0, m_1=1.0, m_2=1.0$).
2. **2_LinkMass_+20%:** Link 2 mass increased by 20% ($m_2 = 1.20\text{ kg}$).
3. **3_LinkLength_+20%:** Link 2 length increased by 20% ($l_2 = 1.20\text{ m}$).
4. **4_SensorNoise_0.05:** Gaussian noise $\mathcal{N}(0, 0.05^2)$ added to normalized angular velocities.
5. **5_CombinedStress:** Link mass $+20\%$, Link length $+20\%$, and Sensor noise $\sigma = 0.05$.

---

## 2. Controller Architectures

1. **EvoEdge Compact Controller (`AcrobotCompactController`):**
   - 6 inputs, 1 hidden layer with 8 neurons (tanh activations), 1 output neuron (tanh activation).
   - Complexity: **65 parameters**, **56 weight MACs**.
2. **Learned MLP Baseline (`AcrobotMLP64Controller`):**
   - 6 inputs, 2 hidden layers with 64 neurons each (tanh activations), 1 output neuron (tanh activation).
   - Complexity: **4,673 parameters**, **4,544 weight MACs**.
3. **Calibrated Classical Benchmark (`ClassicalAcrobotSpongControllerV2`):**
   - Analytical energy-shaping controller based on Spong (1995).
   - Complexity: **4 parameters**, **4 MACs**.

---

## 3. Historical v1 Over-Budget vs. v2 Corrected-Budget Reconciliation

In the initial execution (v1), training runs exceeded the declared 150,000 transition ceiling. In v2, a fail-closed `StepBudgetTracker` enforced the hard cap before every step and halted cleanly before unviable partial batches:

| Controller / Seed | Version | Search Transitions | Budget Delta | Success Count ($N=250$) | Success Rate | Mean Steps | Checkpoint Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Spong Energy-Shaping** | v1 (Defective) | 0 | Analytical | 0 / 250 | 0.0% | 500.00 | Defective physics (inversion) |
| **Spong Energy-Shaping** | **v2 (Corrected)** | **0** | **Analytical** | **250 / 250** | **100.0%** | **110.98** | Calibrated reference |
| **Compact Seed 42** | v1 (Over-budget) | 150,500 | +500 | 0 / 250 | 0.0% | 500.00 | Archived in `checkpoints_v1_overbudget/` |
| **Compact Seed 42** | **v2 (Corrected)** | **150,000** | **Exact Hard Cap** | **0 / 250** | **0.0%** | **500.00** | Active Checkpoint v2 |
| **Compact Seed 43** | v1 (Over-budget) | 150,500 | +500 | 0 / 250 | 0.0% | 500.00 | Archived in `checkpoints_v1_overbudget/` |
| **Compact Seed 43** | **v2 (Corrected)** | **150,000** | **Exact Hard Cap** | **0 / 250** | **0.0%** | **500.00** | Active Checkpoint v2 |
| **Compact Seed 44** | v1 (Over-budget) | 150,011 | +11 | 250 / 250 | 100.0% | 155.35 | Archived in `checkpoints_v1_overbudget/` |
| **Compact Seed 44** | **v2 (Corrected)** | **150,000** | **Exact Hard Cap** | **250 / 250** | **100.0%** | **155.78** | Active Checkpoint v2 |
| **MLP64 Seed 42** | v1 (Over-budget) | 150,500 | +500 | 0 / 250 | 0.0% | 500.00 | Archived in `checkpoints_v1_overbudget/` |
| **MLP64 Seed 42** | **v2 (Corrected)** | **150,000** | **Exact Hard Cap** | **0 / 250** | **0.0%** | **500.00** | Active Checkpoint v2 |
| **MLP64 Seed 43** | v1 (Over-budget) | 150,181 | +181 | 220 / 250 | 88.0% | 201.43 | Archived in `checkpoints_v1_overbudget/` |
| **MLP64 Seed 43** | **v2 (Corrected)** | **150,000** | **Exact Hard Cap** | **220 / 250** | **88.0%** | **268.18** | Active Checkpoint v2 |
| **MLP64 Seed 44** | v1 (Over-budget) | 150,497 | +497 | 181 / 250 | 72.4% | 328.12 | Archived in `checkpoints_v1_overbudget/` |
| **MLP64 Seed 44** | **v2 (Corrected)** | **149,985** | **Clean Cutoff (-15)** | **132 / 250** | **52.8%** | **360.52** | Active Checkpoint v2 |

---

## 4. Full Seed Distribution & Statistical Reporting

To eliminate cherry-picking, the full seed distribution across all three optimizer seeds is reported:

### 4.1 Architecture-Level Aggregate Performance ($N=750$ Episodes per Architecture)

| Architecture | Seed Yield (Converged / Total) | Per-Seed Rates (Seed 42 / 43 / 44) | Mean Success Rate | Median Success Rate | Aggregate Success ($N=750$) | Best Checkpoint (Secondary) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Spong V1 (Defective)** | N/A | N/A | 0.0% | 0.0% | 0 / 250 (**0.0%**) | 0.0% |
| **Classical Spong V2 (Corrected)** | N/A | N/A | 100.0% | 100.0% | **250 / 250 (100.0%)** | 100.0% |
| **EvoEdge Compact ($H=8$, 65 params)** | **1 of 3 (33.3%)** | 0.0% / 0.0% / 100.0% | **33.3%** | **0.0%** | **250 / 750 (33.3%)** | **100.0%** (Seed 44)* |
| **Learned MLP 64×64 (4,673 params)** | **2 of 3 (66.7%)** | 0.0% / 88.0% / 52.8% | **46.9%** | **52.8%** | **352 / 750 (46.9%)** | **88.0%** (Seed 43)* |

*\*Note: Best-checkpoint performance is reported strictly as a secondary descriptive metric and is selection-sensitive because checkpoint selection was performed on the development evaluation set.*

### 4.2 Condition-Level Breakdown for Converged Models ($N=50$ Episodes/Condition)

| Controller | 1: Nominal | 2: LinkMass +20% | 3: LinkLength +20% | 4: SensorNoise 0.05 | 5: CombinedStress |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Spong V2 (Corrected)** | 50/50 (**100.0%**, 103.2 s) | 50/50 (**100.0%**, 141.3 s) | 50/50 (**100.0%**, 94.2 s) | 50/50 (**100.0%**, 100.1 s) | 50/50 (**100.0%**, 116.2 s) |
| **EvoEdge Compact (Seed 44)** | 50/50 (**100.0%**, 146.0 s) | 50/50 (**100.0%**, 168.2 s) | 50/50 (**100.0%**, 139.0 s) | 50/50 (**100.0%**, 150.3 s) | 50/50 (**100.0%**, 175.4 s) |
| **MLP 64×64 (Seed 43)** | 43/50 (**86.0%**, 247.1 s) | 34/50 (**68.0%**, 329.3 s) | 50/50 (**100.0%**, 224.2 s) | 47/50 (**94.0%**, 253.3 s) | 46/50 (**92.0%**, 286.8 s) |
| **MLP 64×64 (Seed 44)** | 31/50 (**62.0%**, 326.6 s) | 17/50 (**34.0%**, 412.2 s) | 24/50 (**48.0%**, 355.6 s) | 40/50 (**80.0%**, 287.2 s) | 20/50 (**40.0%**, 421.1 s) |

---

## 5. Classical Spong Baseline Audit

The historical 0/250 failure of the classical Spong controller was audited in `acrobot_controllers.py`. Three genuine mathematical defects were identified in `ClassicalAcrobotSpongControllerV1_Defective`:
1. **Coordinate Convention Inversion:** In Gymnasium `Acrobot-v1`, $\theta_1 = 0$ corresponds to link 1 hanging straight *downwards*. The v1 implementation assumed $\theta_1=0$ was upright, inverting the potential energy reference.
2. **Total Energy Inversion:** Total mechanical energy was calculated as $E = T - V$ instead of $E = T + V$.
3. **Pumping Sign Reversal:** Torque was applied in the same direction as relative joint velocity, braking the arm instead of pumping kinetic energy.

When corrected in `ClassicalAcrobotSpongControllerV2_Corrected`, the analytical controller achieved **250 / 250 (100.0%) [95% CI: 98.5%–100.0%]** with a mean time-to-goal of **110.98 steps** across all 5 conditions.

---

## 6. Critical Scientific Conclusions & Disclosures

1. **Classical Reliability vs. Neural Search Variance:**  
   The corrected classical controller achieved **100.0% (250/250)** across the entire development suite, while the compact neural controller achieved 100% on **only one of three training seeds** (Seed 44: 100%; Seeds 42 and 43: 0%). Analytical physics-based control provides deterministic reliability that tabula rasa ES search cannot guarantee across all random seeds under tight transition budgets.
2. **Distinguishing Within-Run from Across-Run Uncertainty:**  
   For a single model, bootstrap confidence intervals reflect sampling variability across evaluation seeds (e.g., Compact Seed 44: [98.5%, 100.0%]). However, optimization variance across independent training seeds is substantial (Compact: 0%, 0%, 100%; MLP: 0%, 88%, 52.8%).
3. **No Claim of Architectural Superiority:**  
   We explicitly disavow claims of architectural superiority over the MLP baseline based on selecting the best seed. Across all three seeds, the MLP baseline demonstrated a higher convergence yield (2/3 vs 1/3) and higher aggregate success (46.9% vs 33.3%). The true advantage of the compact architecture is its **81.1× compute efficiency** (56 vs 4,544 MACs) and representational sufficiency once the swing-up basin is reached.
4. **Task Provenance:**  
   Acrobot neural controllers were trained tabula rasa on Acrobot dynamics. They demonstrate that the compact topology and evolutionary pipeline can learn a second distinct control task; they do **not** represent zero-shot transfer of the frozen CartPole weights.
