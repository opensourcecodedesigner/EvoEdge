### Title: EvoEdge: A Verifiable, Self-Auditing Neuroevolution Framework for Resource-Constrained Edge Controllers

**Author**: PhD Researcher, EECS / Robotics

**Core Thesis**: Evolutionary search can discover ultra-compact neural controllers for non-linear dynamic systems that match or outperform classical baselines on microcontrollers, provided that learning is constrained by an immutable interaction budget, audited through pre-registered non-inferiority statistics, and verified against a cryptographically sealed confirmation vault without data leakage.

### Research Motivation and Problem Formulation

Deep reinforcement learning and neuroevolution benchmarks suffer from two persistent credibility crises:

1. **The Statistical Mirage**: As highlighted by [Agarwal et al. (2021)](https://arxiv.org/abs/2108.13264) and [Henderson et al. (2018)](https://arxiv.org/abs/1709.06560), point-estimate metrics (mean return, best-seed cherry-picking) mask high seed variance, hyperparameter overfitting, and tail failure modes.
2. **The Edge Disconnect**: Standard policies evaluated in floating-point simulators fail when ported to microcontrollers due to memory fragmentation, integer quantization shifts, and unmodeled cycle latency ([Lin et al., 2020](https://arxiv.org/abs/2007.10319)).

Current robot learning benchmarks either assume unbounded simulator compute or treat post-hoc deployment profiling as an afterthought. **EvoEdge** resolves this by embedding the auditor directly into the gym lifecycle. Rather than expanding prematurely to high-DOF quadrupeds, EvoEdge anchors its proof-of-rigor on underactuated non-linear dynamics (**CartPole swing-up and balance**) deployed to an **ESP32-class MCU (Xtensa LX7 / RISC-V, 512 KB SRAM)** before scaling outwards.

### Formalized System Architecture and Fixed Protocols

```
                          EvoEdge Architecture
+-----------------------------------------------------------------------+
|  Training & Validation Lifecycle (Open Access)                        |
|                                                                       |
|  [ES Population] ---> [Budget Wrapper] ---> [CartPole Gym Context]    |
|                             |                      |                  |
|                    (Physics Step Hook)      (Frozen Dev Seeds)        |
|                             |                      |                  |
|                             v                      v                  |
|                      Ledger Registry        Pareto Selector           |
+-----------------------------|----------------------|------------------+
                              |                      |
                              v                      v  Candidate Checkpoint
+----------------------------------------------------+------------------+
|  Tamper-Evident Confirmation Vault (Write-Once / Sealed)              |
|                                                                       |
|  [Hardware/Disk Lock] ---> [Manifest HMAC] ---> [One-Shot Evaluation] |
|                                                        |              |
|                                                        v              |
|                                              [Statistical Auditor]    |
|                                              - Stratified Cluster BCa |
|                                              - Formal TOST Bounds     |
|                                              - Tail Risk (CVaR_0.05)  |
|                                              - Hardware Map Profiler  |
|                                                        |              |
|                                                        v              |
|                                              [Cryptographic Audit BOM]|
+-----------------------------------------------------------------------+

```

#### 1. Hard Interaction-Budget and Physics Hook Protocol

* **Physics-level accounting**: The budget wrapper hooks directly into the low-level simulation integrator ($dt = 0.02\text{ s}$). If an action repeat $k=4$ or frame-skip is applied, the transition counter increments by $k=4$, tracking total simulator state transitions rather than raw agent `.step()` calls.
* **Equalized ES vs. RL budget ceiling**: The total transition limit is locked at $B_{\text{trans}} = 2.0 \times 10^6$ transitions. For an ES population of $P$ individuals evaluated over $H$ horizon steps per generation $g$:

$$\sum_{g=1}^{G} \sum_{i=1}^{P} \sum_{t=1}^{H_{g,i}} k_{\text{substeps}} \le B_{\text{trans}}$$



Once $B_{\text{trans}}$ is reached, the environment throws an uncatchable `BudgetTerminated` exception.
* **Separation of training vs. edge compute**:
* *Training compute*: Reported in cumulative floating-point operations ($\text{FLOPs}_{\text{train}}$), wall-clock time, and energy ($J$).
* *Deployment cost*: Measured via ELF symbol analysis and cycle-accurate emulation on target hardware (ESP-IDF / QEMU / Renode), extracting static Flash size (BSS/text segment), peak stack and heap allocation in internal SRAM, Multiply-Accumulates per control step ($\text{MACs/step}$), and real control-loop latency ($\mu\text{s}$).



#### 2. Persistent, Tamper-Evident Confirmation Vault

* **Persistent state ledger**: Eliminates in-memory boolean flags. Vault state is persisted to an append-only JSON-lines ledger protected by a local file lock and HMAC signature:

$$\text{HMAC-SHA256}(\text{candidate\_id} \,\Vert{}\, \text{run\_timestamp} \,\Vert{}\, \text{manifest\_digest}, K_{\text{vault}})$$



If the ledger records that `candidate_id` or the current protocol version has accessed confirmation partition $\mathcal{D}_{\text{confirm}}$, subsequent evaluation requests are blocked at the filesystem level.
* **Provenance and partition verification**: Rather than introducing arbitrary seed numbers that risk collisions with existing runs, EvoEdge cross-references a manifest of all historic development seeds. The confirmation partition draws strictly from a cryptographically sealed hash-chain seed generator:

$$\text{Seed}_i = \text{KDF}(\text{MasterSecret}, \text{"confirm\_cartpole"}, i) \quad \forall i \in [1, N_{\text{confirm}}]$$



A pre-execution validator checks that $\mathcal{D}_{\text{confirm}} \cap \mathcal{D}_{\text{historic}} = \emptyset$.
* **Candidate freeze barrier**: The confirmation test harness refuses execution until the candidate policy weights, network topology, and quantization lookup table are serialized and committed with a Git commit hash and file digest.

#### 3. Corrected Statistical Auditing Engine

* **Formal Two One-Sided Tests (TOST)**:
Instead of only evaluating a one-sided bootstrap lower bound, EvoEdge implements formal TOST for non-inferiority against the classical baseline (e.g., LQR or energy-shaping controller) and the reference uncompressed MLP:

$$H_0^-: \mu_{\text{cand}} - \mu_{\text{ref}} \le -\Delta \quad \text{vs.} \quad H_1^-: \mu_{\text{cand}} - \mu_{\text{ref}} > -\Delta$$



Using Welch's $t$-statistic adjusted for unequal variances across seeds:

$$t_L = \frac{(\bar{X}_{\text{cand}} - \bar{X}_{\text{ref}}) - (-\Delta)}{\sqrt{\frac{s_{\text{cand}}^2}{n_{\text{cand}}} + \frac{s_{\text{ref}}^2}{n_{\text{ref}}}}}$$



Non-inferiority is certified at significance $\alpha = 0.05$ if $p_L < 0.05$. Margin $\Delta$ is pre-registered at $\Delta = 0.05 \cdot \vert{}\bar{X}_{\text{ref, nominal}}\vert{}$.
* **Stratified cluster bootstrap for IQM**:
Following the [Agarwal et al. (2021)](https://arxiv.org/abs/2108.13264) specification, bootstrapping resamples at the *seed cluster level* across experimental conditions rather than pooling individual episode rollouts:

$$\text{IQM} = \frac{1}{2M} \sum_{i=\lfloor M/4 \rfloor + 1}^{\lfloor 3M/4 \rfloor} X_{(i)}$$



Confidence intervals (95%) are constructed using bias-corrected and accelerated (BCa) bootstrap percentiles over 5,000 resamples.
* **Tail risk and safety floors**:
* *Conditional Value-at-Risk*: $\text{CVaR}_{0.05}(R) = \mathbb{E}[R \mid R \le F_R^{-1}(0.05)] \ge R_{\text{stable}}$, guaranteeing the worst 5% of runs do not enter unrecoverable divergence.
* *Success floor*: Invariant requirement that swing-up and continuous balance within $\theta \le \pm 12^\circ$ for $t \ge 3.0\text{ s}$ holds across $\ge 90\%$ of confirmation seeds.



#### 4. Hardware-Aware Deployment Profiler

* **Binary map analyzer**: EvoEdge hooks into the GCC/Clang linker script (`.map` file inspection) for ESP32 builds to report exact BSS, Data, and Text segment sizes down to the byte.
* **Quantization parity test**: The auditor evaluates both the floating-point policy $\pi_{\text{FP32}}$ and the quantized integer policy $\pi_{\text{INT8}}$ on the confirmation split, computing policy divergence:

$$\mathcal{D}_{\text{action}} = \frac{1}{T} \sum_{t=1}^{T} \Vert{}\pi_{\text{FP32}}(s_t) - \pi_{\text{INT8}}(s_t)\Vert{}_2$$



If action discrepancy exceeds the non-inferiority threshold, the deployment certificate is rejected.

### Three-Milestone Execution Roadmap

| Milestone | Core Deliverable | Acceptance & Gate Criteria | Primary Artifacts |
| --- | --- | --- | --- |
| **1. Works (Physics & Swing-Up Baseline)** | Reliable continuous CartPole swing-up and balance environment under hard integrator step accounting. | Energy-shaping + LQR controller achieves 100% balance over 100 nominal seeds; physics step hook accurately counts all integrator substeps. | `cartpole_contextual_env.py`, `physics_interceptor.py`, nominal benchmark logs. |
| **2. Wins or Fairly Loses (ES vs. Classical)** | Preregistered ES optimization of compact neural network under $2.0 \times 10^6$ transition ceiling. | ES candidate evaluated against LQR and reference MLP under identical transition budgets; TOST non-inferiority evaluated on validation partition. | Frozen training logs, Pareto curve (IQM Return vs. Parameter Count), validation TOST report. |
| **3. Ships (The Reusable Audit Engine)** | Standalone, reproducible auditing engine with cryptographic vault, statistical profiler, and CLI. | One-time confirmation evaluation executed on sealed partition; zero seed leakage verified; tamper-evident deployment certificate generated. | `evoedge-cli`, `confirmation_vault.py`, `manifest.hmac`, signed audit certificate JSON/PDF. |

### Concrete Research Contributions

* **A Verifiable Auditing Protocol for Robot Learning**: Solves the reproducibility crisis in small-scale robot learning by creating an open-source, write-once audit harness that prevents data leakage and p-hacking in sim-to-real transfer studies ([Henderson et al., 2018](https://arxiv.org/abs/1709.06560); [Cobbe et al., 2019](https://arxiv.org/abs/1912.01588)).
* **Pareto-Optimal Edge Controllers**: Demonstrates that neuroevolution can discover sub-500 parameter quantized policies that match classical control performance on underactuated systems while meeting sub-32KB RAM microcontroller budgets ([Gaier & Ha, 2019](https://arxiv.org/abs/1906.04358); [Lin et al., 2020](https://arxiv.org/abs/2007.10319)).
* **Deployment-Integrated Benchmarking**: Bridges the gap between simulator gym environments and microcontrollers by unifying statistical robustness proofs with physical hardware footprint verification.

### Immediate Action Plan for Stage 2

To proceed with Stage 2 without breaking existing partitions or introducing invalid statistical tests:

1. Lock the transition budget at $2.0 \times 10^6$ integrator steps.
2. Confirm the historic seed allocation table to ensure complete isolation of the development and future confirmation partitions.
3. Formalize the non-inferiority margin at $\Delta = 0.05 \cdot \bar{R}_{\text{LQR}}$.
4. Freeze the ES-MLP candidate architecture before generating the confirmation manifest.