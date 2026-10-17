# EvoEdge — Stage 2 Protocol Reconciliation and Authorization Report

**Document Revision:** 1.1 (Final Audit Corrections Before Sign-Off)  
**Phase:** Protocol Reconciliation and Stage 2 Authorization Review  
**Date:** October 13, 2026  
**Status:** **READY FOR SIGN-OFF**  
**Authority:** Senior GPU Systems Engineer, Control Systems Researcher & Reproducibility Auditor  

---

## 1. Executive Summary & Review Scope

This report reconciles the latest EvoEdge research proposal (`docs/proposals/stage2_spec.md`) against:
1. Frozen acceptance contract `docs/success_criteria.md` v2.3.
2. Historical Phase 3A experimental records and code (`results/phase_3a/`, `experiments/run_phase_3a_evolution.py`).
3. Stage 1 classical baseline empirical report (`results/descriptive_baseline_phase3a/stage_1_baseline_report.md`).
4. Repository seed allocation ledgers and code-level holdout assertions.

Per the task mandate, this is an **inspection, audit, and documentation task only**. No models were trained, no confirmation seeds were generated or accessed, no frozen YAML/INI configs were altered, and no Git commits or deletions were made.

The updated acceptance specification has been written to:
`docs/success_criteria_v2.4_PROPOSED.md` (leaving `docs/success_criteria.md` v2.3 frozen and untouched).

---

## 2. Seed Exposure Audit & Confirmation Eligibility

An exhaustive audit of repository configurations, historical logs, raw CSVs, and code manifests establishes the following empirical seed ledger:

| Seed Range | Quantity | Historical / Active Purpose | Eligibility for Future Confirmation | Audited Sources & Evidence |
|---|---|---|---|---|
| **`0–199`** | 200 | Training & Fitness Rollouts | **BURNED (Ineligible)** | `config/perturbations.yaml`, `config/phase_3a_config.yaml`, `experiments/run_phase_3a_evolution.py` |
| **`200–249`** | 50 | Historical Gate 0 Nominal Validation | **BURNED (Ineligible)** | `results/phase_3a_report.md`, `results/smoke_test_validation_summary.json` |
| **`250–399`** | 150 | Phase 3A Validation & Stage 1 Baseline | **BURNED (Ineligible)** | `results/phase_3a/phase_3a_validation_summary.json`, `results/descriptive_baseline_phase3a/` |
| **`400–599`** | 200 | Unused buffer (isolated references to 400 and 500 in scripts) | **BUFFER (Avoid)** | `scratch/check_seeds_400_999.py`, `evolution/evolve_tinyml.py` |
| **`600–749`** | 150 | **Proposed Confirmation Partition** | **PROVISIONAL (Pending Ratification)** | Zero literal matches in active code, configs, logs, and CSVs. *Limitation:* Does not prove unlogged runs were never executed out-of-band. |
| **`1000–1099`** | 100 | Historical Task B Benchmark | **BURNED / LEGACY (Ineligible)** | `config/perturbations.yaml`, historical Task B reports |
| **`2000–2099`** | 100 | Protected Final Holdout Partition | **PROTECTED (Untouchable)** | Code-level assertion in `utils/split_integrity.py`, `test_cuda_evaluator.py` |

### Methodological Distinction: Literal Reference Scan vs. Actual Exposure Proof
- **Sources Examined:** All active `.py`, `.yaml`, `.md`, `.json`, `.csv`, `.ini` files in `config/`, `experiments/`, `evolution/`, `baselines/`, `env/`, `results/`, `tests/`, and `scratch/`.
- **Sources That Cannot Be Examined:** Ephemeral terminal executions, interactive REPL sessions, uncommitted Git branches/stashes (none currently stashed), and private external development notes.
- **Scientific Disposition:** A zero-match scan establishes absence of recorded references in the repository, but cannot guarantee that seeds were never tested out-of-band. Therefore, seeds `600–749` are classified as **`PROVISIONAL`** pending explicit operator ratification.
- The proposal's suggestion to draw seeds from an arbitrary `0000–7499` range via KDF is **REJECTED** because it would collide with burned development seeds `0–399`.
- The proposed HMAC vault, KDF seed generator, and standalone audit platform are classified as **`PROPOSED ARCHITECTURE`**, not existing verified capabilities.
- For the immediate single-machine research workflow, an unexposed prospective seed block (`600–749`, 150 seeds) is proposed, protected by a deterministic SHA-256 manifest and an append-only filesystem lock file.

---

## 3. Simulator-Transition Budget Reconciliation

### Exact Two-Million-Transition Ceiling Accounting
- **Ceiling Scope:** The budget ceiling applies **per optimizer run (per optimizer seed)**.
- **Operations That Consume Budget:**
  - Every single physics integration step ($\Delta t = 0.02\text{ s}$) executed during training rollouts, fitness evaluations, population screening, candidate validation checks, and early-stopping rollouts increments the budget counter.
  - For an ES population of $P$ individuals evaluated over $H$ horizon steps per generation $g$, and an action repeat / sub-stepping factor $k_{\text{substeps}}$:
    $$\sum_{\text{evaluations}} \sum_{t=1}^{H} k_{\text{substeps}} \le B_{\text{trans}} = 2,000,000$$
- **Operations That Do NOT Consume Budget:**
  - Static parameter counting, tensor weight initialization, gradient calculation / optimizer state updates, C code generation, and wall-clock time do not consume physics transitions.
- **Counting Semantics:**
  - **Early Termination:** If an episode terminates early due to rail boundary violation ($|x| \ge 2.4\text{ m}$), the environment halts immediately, and only the actual executed integration steps prior to termination are billed to the counter.
  - **Vectorized Batch Rollouts:** In batch simulators that step all array slots without masking, every simulated slot is counted as an actual physics call unless early termination is masked out.
  - **Exceptions and Retries:** If an exception occurs, all transitions executed prior to the exception remain billed. Retries do **not** reset the counter.
- **Fail-Closed Enforcement:**
  - The environment step hook checks the atomic transition counter on every step. When the counter reaches $B_{\text{trans}}$, it raises a catchable `BudgetExhaustedError`.
  - The training orchestrator catches `BudgetExhaustedError`, immediately halts search, and serializes the current best candidate using development data only.
- **Multi-Seed Protocol & Aggregate Budget:**
  - Each method is evaluated across $S_{\text{opt}} = 5$ independent optimizer seeds (Seeds 42, 43, 44, 45, 46).
  - Total permitted training budget across all 5 runs: $5 \times 2,000,000 = \mathbf{10,000,000\text{ actual transitions}}$ per method.
- **Comparable Candidate Selection Opportunities:**
  - Both algorithms evaluate candidates exclusively on the development partition (training seeds `0–199`, validation seeds `250–399`). Candidate selection is finalized before crossing the freeze barrier.

### Historical Phase 3A Reality vs. New Matched Budget
- **Historical Phase 3A Reality:**
  - Scheduled evaluations: 30 individuals $\times$ 15 generations $= 450$ evaluations per seed.
  - Evaluation suite: 10 conditions $\times$ 50 training seeds $= 500$ episodes per evaluation.
  - Scheduled horizon: 500 steps fixed loop.
  - Scheduled rollout capacity: $450 \times 500 \times 500 = \mathbf{112,500,000\text{ rollout slots}}$ ($1.125 \times 10^8$).
- In Phase 3A vectorized batch rollouts, tensor arrays were stepped for all 500 steps regardless of early cart collisions to maintain uniform tensor shapes.
- Therefore, $112.5 \times 10^6$ represents **scheduled loop capacity and physical integration cycles**, but does not log non-terminated effective steps.
- Floating-point compute difference: dense 64×64 MLP requires 4,480 MACs/step ($5.04 \times 10^{11}$ MACs total), whereas compact $H=8$ requires 48 MACs/step ($5.40 \times 10^9$ MACs total). The MLP performs **$93.3\times$ more compute** for the same transition count.

### Decision Options Presented for Operator Sign-Off
1. **Option A (Recommended — Fresh Matched-Budget Comparison):**
   - Both Evolution Strategies (ES) and compact neuroevolution receive the exact same registered interaction budget: $B_{\text{trans}} = 2,000,000$ actual transitions per optimizer run.
   - Label: `EQUAL-INTERACTION-BUDGET`.
2. **Option B (Descriptive Unequal-Budget Comparison):**
   - Train the ES MLP under $2.0 \times 10^6$ transitions and compare directly against historical Phase 3A neuroevolution ($112.5 \times 10^6$ scheduled iterations).
   - Label: Explicitly reported as **`UNEQUAL-BUDGET DESCRIPTIVE EVIDENCE`**.

---

## 4. Development-Confirmation Separation & Freeze Barrier

### Partition Roles
- **Development Partition (Open Access):**
  - Seeds `0–199`: Training rollouts and fitness evaluations.
  - Seeds `200–399`: Gate 1B nominal screening, validation tuning, hyperparameter selection, and candidate Pareto front extraction.
  - Invariant: All model tuning, selection, and debugging occur here.
- **Freeze Barrier:**
  - Candidate controller weights, architecture, and topology serialized.
  - Execution scripts, evaluation protocol, and configs frozen.
  - SHA-256 digests committed to audit ledger.
  - Invariant: ZERO changes permitted after crossing barrier.
- **Confirmation Partition (Locked Single-Pass Access):**
  - Provisional Seeds `600–749` (150 seeds $\times$ 10 conditions = 1,500 runs).
  - Evaluated ONCE for the locked candidate, ES, and `HybridController`.
  - Invariant: NEVER used for tuning, selection, screening, or retries.

---

## 5. Official Success Definition & Registered Statistical Protocol

- **Official Task Success Predicate (FROZEN):**
  - Episode horizon: Exactly 500 steps ($\Delta t = 0.02\text{ s}$).
  - Rail boundary violation: $|x| < 2.4\text{ m}$ strictly enforced at every step.
  - Upright balance: $\cos(\theta) > 0.9$ strictly for at least 95 of the final 100 steps ($t \in [400, 499]$).
- **Secondary Telemetry Metric:**
  - Sustained balance within $|\theta| \le 12^\circ$ for $\ge 3.0$ consecutive seconds (150 steps).
  - Status: Documented strictly as a secondary diagnostic measure. It does **not** replace the frozen official predicate.

### Corrected Statistical Protocol
- **Primary Estimands:**
  - Official success rate per method, aggregated with equal weight across the 10 diagnostic conditions:
    $$S_{\text{agg}} = \frac{1}{10} \sum_{c=1}^{10} S_c$$
  - Condition-level numerators and denominators ($k_c / n_c$).
  - Paired candidate-minus-reference aggregate difference: $\bar{D}_{\text{agg}} = S_{\text{agg}}(\text{cand}) - S_{\text{agg}}(\text{ref})$.
- **Uncertainty Quantification:**
  - **Individual Condition Rates:** Evaluated via **95% Wilson score confidence intervals** on binomial episode counts ($k_c / n_c$).
  - **Aggregate Success Rate & Differences:** Because $S_{\text{agg}}$ is an equally weighted average across 10 correlated physical conditions sharing the same underlying seed dynamic perturbations, **it cannot be treated as an ordinary binomial proportion for Wilson-interval purposes**.
  - All aggregate uncertainty intervals and candidate-minus-reference differences are evaluated exclusively via a **paired seed-cluster bootstrap** ($B = 10,000$ resamples over evaluation seeds, fixed PRNG seed 42) that preserves all 10 condition outcomes within each sampled seed.
- **Non-Inferiority Testing Rule:**
  - Framework: One-sided non-inferiority test (distinguished from two-one-sided-tests [TOST]).
  - Margin: $\delta_{\text{agg}} = -0.05$ ($-5.0$ percentage points).
  - Rule: Pass if lower one-sided 95% bootstrap bound $L_{\text{agg}, 95} \ge -0.05$.
  - Engineering basis: $\delta_{\text{agg}}$ is an a priori engineering tolerance, **not** derived post hoc from observed reference variance.
  - Multiplicity / Condition-level: All 10 condition differences and uncertainty intervals must be reported individually. Aggregate non-inferiority cannot mask single-condition collapse.
- **Secondary Metrics (IQM, CVaR):** Retained strictly as secondary diagnostic summaries; they do not replace the primary binary predicate.

---

## 6. Edge Deployment Target Disambiguation

- **Selected MCU Target:** Original **ESP32** (Espressif Systems).
  - Architecture: Dual-core 32-bit Xtensa LX6 @ 240 MHz.
  - FPU: Single-precision IEEE 754 hardware floating-point unit.
  - SRAM: 520 KB total (~320 KB usable contiguous DRAM).
  - Storage: 448 KB ROM, external SPI flash.
- **Distinction from Other Variants:**
  - ESP32-S3: Xtensa LX7 with vector instructions (not equivalent).
  - ESP32-C3 / C6: RISC-V 32-bit single-core (ESP32-C3 lacks hardware FPU; requires software float emulation).
- **Evidence Hierarchy:**
  - Current reporting: Analytical complexity (57 params, 48 MACs for compact neuroevolution; 113 params, 96 MACs for compact MLP).
  - Emulation & physical hardware testing: Deferred until algorithm comparison is complete.

---

## 7. Master Protocol Reconciliation Decision Table

| Decision Item | Stage 2 Proposal Specification | Historical EvoEdge Reality | Reconciled Protocol Recommendation | Status |
|---|---|---|---|---|
| **Lifecycle Budget** | $2.0 \times 10^6$ actual simulator transitions | $112.5 \times 10^6$ scheduled loop iterations ($450 \times 500 \times 500$) | Offer Option A (2.0M matched ceiling) vs Option B (descriptive comparison vs historical 112.5M). | **`AWAITING OPERATOR SIGN-OFF`** |
| **Official Success Predicate** | $\le \pm 12^\circ$ for $\ge 3.0$ s across $\ge 90\%$ seeds | 500 steps, $|x| < 2.4$, $\cos\theta > 0.9$ for $\ge 95/100$ final steps | Retain frozen 500-step predicate as official. Document $\pm 12^\circ$ strictly as secondary diagnostic. | **`FROZEN (Contract Preserved)`** |
| **Statistical Test** | Formal TOST equivalence ($\Delta = 0.05 \cdot \bar{R}_{\text{LQR}}$) | One-sided non-inferiority ($L_{\text{agg}, 95} \ge -0.05$) | One-sided non-inferiority via paired seed-cluster bootstrap ($B=10,000$). LQR rejected as reference. | **`PROPOSED SPECIFICATION`** |
| **Confirmation Partition** | KDF generator seeds `0000–7499` split | Seeds `0–399` burned, `1000–1099` burned, `2000–2099` protected | Authorize unexposed seeds `600–749` (150 seeds). Reject arbitrary unverified ranges. Mark provisional. | **`PROVISIONAL SPECIFICATION`** |
| **Classical Reference** | Energy-Shaping or LQR | `HybridController` (Energy-Shaping + Linear Feedback) | `HybridController` primary; LQR diagnostic only. | **`FROZEN (Stage 1 Measured)`** |
| **Edge Hardware Target** | ESP32-class (Xtensa LX7 / RISC-V, 512 KB SRAM) | ESP32 Xtensa LX6 @ 240 MHz, single-precision FPU | Target ESP32 Xtensa LX6. Separate estimates, emulation, and physical bench testing. | **`PROPOSED SPECIFICATION`** |
| **Confirmation Infrastructure** | HMAC vault, KDF seed generator, audit CLI | Minimal version-controlled manifest & execution guard | Defer complex vault; implement simple file lock and SHA-256 manifest. | **`PROPOSED (Minimized Infrastructure)`** |

---

## 8. Artifact Ledger, Hashes & Executed Verification Tests

### Critical Artifact Hashes (SHA-256)
- `docs/proposals/stage2_spec.md`: `2083bdc7a9cc187f48a9301803d5d53824560e94b8db1047d5de3442d40a5d3d` (Audit source)
- `docs/success_criteria.md` (v2.3, frozen): `4bdfca68ffdfc7217c18c1e4db95c6f6630f9a26317b629b0be9992497672ec1` (Preserved intact)
- `docs/success_criteria_v2.4_PROPOSED.md`: Newly created proposed contract specification.
- `results/stage2_reconciliation_report.md`: Newly created comprehensive reconciliation report.
- `config/phase_3a_config.yaml`: `0d8a14b7f6184da54b2a703efa3e761c273cad10e945fb0cf75d85f0924507a8`
- `config/perturbations.yaml`: `5e0d0e4b4d0c8780145b17ddec0b39f17ba7ca9364f8705163ecf1a20f25689a`
- `results/champion_neural_h8.pkl`: `d16035a57cfcdd6c094fc6670454092265c15fb71e18529ebc9bac755499d1b7`
- `results/phase_3a/models/arm_b_warmstart_seed43.json`: `384ec8dd6b27980a01f0094b82d5a5f15e8d2bf97ef186235e649bfe68a9d571`
- `results/mlp64_baseline.pt`: `2cd704f0064171190daac270de317a37fcdf8a55cb884da6edb6dc744f409842`
- `results/fixed_mlp/best_weights.npz`: `deba6cd6110e72ac3b930f84e29e16e088753fd370a8c6b38abe87ebe587690d`
- `results/descriptive_baseline_phase3a/raw_episodes.csv`: `2b39a9e476ec60c64da633b2c5eb019d1b405f466785490b175c943272986b36`

### Executed Test Suites & Results
1. `tests/test_phase_3a_evolution.py`: 8 tests executed, 8 passed (`max |x_vec - x_ref| = 0.00e+00`).
2. `tests/test_split_integrity.py`: 6 tests executed, 6 passed (Disjointness and OOD checks verified).
3. `tests/test_cuda_evaluator.py`: 13 tests executed, 13 passed (Holdout seed rejection actively asserted).

---

## 9. Final Decision & Stop Condition

### Current Disposition: **`READY FOR SIGN-OFF`**
The protocol reconciliation is complete. The decision table, seed ledger, and proposed contract `docs/success_criteria_v2.4_PROPOSED.md` are fully drafted, verified, and ready for operator review.

### Recommended Next Concrete Action
Obtain operator sign-off on:
1. Budget selection: **Option A** (matched $2.0 \times 10^6$ transition ceiling) vs. **Option B** (descriptive unequal-budget comparison).
2. Ratification of provisional confirmation partition: Seeds `600–749`.
3. Formal ratification of `docs/success_criteria_v2.4_PROPOSED.md`.

Once signed off, prepare the frozen configuration and launch the authorized Stage 2 matched-budget training run. Per instructions, execution halts here.
