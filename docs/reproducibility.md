# EvoEdge Reproducibility & Replication Guide

**Document Classification:** Technical Protocol and Verification Standard
**Governing Gate Matrix:** [Verified Results & Gate Matrix](results-and-limitations.md#4-preserved-governance-gates--boundaries)
**Repository:** [`opensourcecodedesigner/EvoEdge`](https://github.com/opensourcecodedesigner/EvoEdge)

---

## 1. System Requirements & Environment Setup

All experiments and evaluations are designed to run in standard POSIX or Windows environments with minimal external dependencies.

### 1.1 Prerequisites
- **Python:** 3.10+ (tested on Python 3.12.10 x86_64)
- **C Compiler:** Any standard ANSI C99 compiler (tested with GCC 16.1.0 on MSYS2 / UCRT64, Clang 17, and MSVC 19.x)
- **Host Workstation:** Minimum 4 CPU cores, 8 GB RAM (all evaluations run in single-thread or batch-vectorized CPU mode; no GPU required)

### 1.2 Python Dependencies
Install the required packages in your active environment:
```bash
pip install -r requirements.txt
```
Key packages include:
- `numpy >= 1.24.0`
- `pytest >= 7.4.0`
- `gym >= 0.26.0` (or `gymnasium >= 0.28.0`)

---

## 2. Automated Test Suites

The repository contains automated unit, integration, and parity regression tests.

### 2.1 Seed Partition & Split Integrity Test
Verify that training, development, and protected confirmatory evaluation partitions remain mathematically and programmatically isolated:
```bash
pytest tests/test_split_integrity.py -v
```
*Expected output: 6 passed.*

### 2.2 Baseline Implementation & Budget Verification
Verify the corrected OpenAI-ES baseline training script, teacher weight transfer scaling, reward formulation, and fail-closed transition accounting:
```bash
pytest work/phase3e_followup_6h/sprint3/project_closeout/v4_baseline/test_v4.py -v
```
*Expected output: 3 passed.*

### 2.3 Embedded C Numerical Parity & SIL Tests
Verify double-precision bit parity and single-precision bounded tolerance between Python reference models and ANSI C99 exports:
```bash
pytest tests/test_hybrid_c_parity.py -v
pytest tests/test_embedded_sil.py -v
pytest tests/test_budget_tracker.py -v
```
*Expected output: All tests passed.*

---

## 3. Development Evaluation Workflows

To reproduce the exact published numbers from raw checkpoints on development evaluation seeds without modifying or rerunning training:

### 3.1 CartPole Task B Swing-Up Evaluation (Seeds 200..209, 10 Conditions, 100 Episodes/Controller)
Evaluates the compact EvoEdge candidate (Seed 46), the 3 OpenAI-ES MLP baselines (Seeds 42, 43, 44), and the analytical classical reference:
```bash
python work/phase3e_followup_6h/sprint3/project_closeout/v4_baseline/evaluate_v4_development.py
```
**Expected Results:**
- EvoEdge Candidate (Seed 46, 69 params, 60 MACs): **99/100 (99.0%)** [95% CI: 94.6%–99.8%], 0 rail crashes, 1 dwell failure. Worst-condition success: 90.0%.
- MLP Baseline Seed 43 (Best): **98/100 (98.0%)** [95% CI: 93.0%–99.4%], worst-condition: 80.0%.
- MLP Baseline Seed 44 (Mid): **95/100 (95.0%)** [95% CI: 88.8%–97.8%], worst-condition: 70.0%.
- MLP Baseline Seed 42 (Min): **91/100 (91.0%)** [95% CI: 83.8%–95.2%], worst-condition: 50.0%.
- Classical Reference (`HybridController`, 7 params): **84/100 (84.0%)** [95% CI: 75.6%–89.9%], 4 rail crashes, 12 dwell failures. Worst-condition: 20.0%.

### 3.2 Acrobot-v1 Second Task Evaluation (Seeds 500..549, 5 Conditions, 250 Episodes/Controller)
Evaluates the second-task Acrobot models trained under the corrected budget ceiling ($\le 150,000$ transitions):
```bash
python work/phase3e_followup_6h/sprint3/research_upgrade/second_task/evaluate_acrobot_v2.py
```
**Expected Results:**
- Classical Spong V2 Reference: **250/250 (100.0%)** across all 5 conditions.
- Classical Spong V1 (Defective): **0/250 (0.0%)**.
- EvoEdge Compact (Seed 44): **250/250 (100.0%)** [Seeds 42 & 43: 0.0%]. Aggregate yield: 1 of 3 (33.3%).
- Learned MLP Baseline (Seed 43): **220/250 (88.0%)** [Seed 44: 52.8%, Seed 42: 0.0%]. Aggregate yield: 2 of 3 (66.7%).

### 3.3 Embedded C Compilation & Native Runner Execution
Compile the portable ANSI C99 controller and platform adapter with strict warnings and execute the standalone native C test suite:
```bash
# Compile native test runner
gcc -std=c99 -Wall -Wextra -Wpedantic -O3 \
    results/deployment/hybrid_controller_runner.c \
    results/deployment/hybrid_controller.c \
    results/deployment/platform_adapter.c \
    -o results/deployment/hybrid_controller_runner.exe

# Execute native runner
./results/deployment/hybrid_controller_runner.exe
```
*Expected Output: All assertions pass; host timing benchmark completes ~1,000,000 steps with $p_{50} \approx 12.10\text{ ns}$.*

---

## 4. Cryptographic Artifact Ledger

To verify that model weights, training records, and evaluation outputs have not been altered, verify SHA-256 hashes against the official ledger:

| Artifact Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- |
| `work/phase3e_followup_6h/sprint3/project_closeout/final_package/models/stage3d_nsga2_robust_seed46.json` | `0C9A57692131972D52F75BD633A64BAE2A34A7B4EC2FB94C56041065C58A2B0E` | EvoEdge CartPole candidate (69 params, 60 MACs) |
| `work/phase3e_followup_6h/sprint3/project_closeout/v4_baseline/models/mlp64_baseline.pt` | `2CD704F0064171190DAAC270DE317A37FCDF8A55CB884DA6EDB6DC744F409842` | OpenAI-ES CartPole baseline checkpoint |
| `work/phase3e_followup_6h/sprint3/research_upgrade/second_task/models/acrobot_compact_seed44.json` | `DB282929979BB141C06A5FDFECEBF1B237CF399434D45115EB0B8904857E9387` | EvoEdge Acrobot candidate (65 params, 56 MACs) |
| `work/phase3e_followup_6h/sprint3/research_upgrade/second_task/models/acrobot_mlp64_seed43.json` | `7F3A66F14178AA466C17F2587DF9523CFD867E19E2CA4ECCC83D732510A3A83B` | Learned MLP Acrobot checkpoint |
| `work/phase3e_followup_6h/sprint3/research_upgrade/second_task/results/acrobot_episodes_v2.csv` | `423DBAD6A86D497313DCFF2942990B6FDBD27A694480BA98E3410956AA0A2638` | Raw episode records for Acrobot v2 (1,750 episodes) |
| `work/phase3e_followup_6h/sprint3/research_upgrade/second_task/results/acrobot_comparison_v2.json` | `1708DFC2181940B36C673F367F906C7EF55FFFD1F0690CA77FD641A7804488E9` | Reconciled summary statistics for Acrobot v2 |
| `work/phase3e_followup_6h/sprint3/project_closeout/final_package/final_gate_matrix.json` | `E57049ED159F9F8B59BC0418EB528E43B330F77265CE9D683A35467475F256E1` | Frozen 14-gate audit and status matrix |

To verify on Windows PowerShell:
```powershell
Get-FileHash <path_to_file> -Algorithm SHA256
```
To verify on Linux / macOS:
```bash
sha256sum <path_to_file>
```

---

## 5. Seed Partition Governance & Access Rules

Strict cryptographic seed governance prevents overfitting to evaluation data and preserves the statistical validity of future confirmatory trials:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                     SEED PARTITION GOVERNANCE                          │
├───────────────────────────────┬────────────────────────────────────────┤
│ UNRESTRICTED DEVELOPMENT      │ STRICTLY QUARANTINED & PROTECTED       │
│                               │                                        │
│ • CartPole Task B Dev:        │ • CartPole Stage 2 Confirmatory:       │
│   Seeds 200..209              │   Seeds 600..749 (FROZEN & QUARANTINED)│
│ • CartPole Task A Sweep:      │ • CartPole Stage 3 Confirmatory:       │
│   Seeds 210..219              │   Seeds 2000..2099 (FROZEN & ISOLATED) │
│ • Acrobot Development:        │ • CartPole Prospective Holdout:        │
│   Seeds 500..549              │   Seeds 3000..3099 (BLOCKED: GATE 11)  │
│ • CartPole Training:          │ • Acrobot Reserved Holdout:            │
│   Seeds 42, 43, 44            │   Seeds 1500..1599 (FROZEN & UNREAD)   │
└───────────────────────────────┴────────────────────────────────────────┘
```

> [!CAUTION]
> **Prohibited Operations:**
> 1. Never evaluate models on seeds `600–749`, `2000–2099`, `3000–3099`, or `1500–1599`.
> 2. Gate 11 evaluates prospective holdout seeds `3000–3099`. This gate requires human cryptographic key authorization and MUST remain **BLOCKED** until authorized.
