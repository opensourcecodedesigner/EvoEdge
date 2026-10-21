# Contributing to EvoEdge

Thank you for your interest in contributing to the EvoEdge project. EvoEdge is an open research repository dedicated to compact, auditable, and hardware-efficient neural and classical control algorithms.

---

## 1. Development Principles & Code Quality

1. **Deterministic Reproducibility:**  
   Every experiment and evaluation must be fully seeded and deterministic. Do not introduce unseeded calls to `random` or `numpy.random`.
2. **Transition Budget Fairness:**  
   All learning algorithms under comparison must be wrapped with a fail-closed transition budget tracker. Any comparison claiming budget parity must verify and report exact search transitions consumed.
3. **ANSI C99 Standards:**  
   Embedded C deployment code must conform strictly to ANSI C99 (`-std=c99`) and compile cleanly under `-Wall -Wextra -Wpedantic` with zero warnings.
4. **Zero Dynamic Allocation in Real-Time Paths:**  
   Embedded controller libraries must never call `malloc`, `free`, `calloc`, or any dynamic allocation routines inside the control evaluation loop.

---

## 2. Seed Partition Protection Rules

To maintain the statistical integrity of confirmatory benchmarks, all contributors must observe strict seed partitioning:

- **Permitted Development Seeds:**  
  - CartPole Task B: Seeds `200..209`  
  - CartPole Task A: Seeds `210..219`  
  - Acrobot-v1: Seeds `500..549`
- **Strictly Quarantined & Forbidden Seeds:**  
  - CartPole Stage 2 Confirmatory: Seeds `600..749` (Do NOT read or evaluate)  
  - CartPole Stage 3 Confirmatory: Seeds `2000..2099` (Do NOT read or evaluate)  
  - Prospective Confirmatory Holdout: Seeds `3000..3099` (Gate 11 BLOCKED; requires formal key authorization)  
  - Acrobot Reserved Holdout: Seeds `1500..1599` (Do NOT read or evaluate)

Automated tests will reject any changes that read or execute on protected seed ranges.

---

## 3. Pull Request Workflow

1. Fork the repository and create a descriptive feature branch (e.g., `feature/stm32-bsp-adapter`).
2. Run the regression test suite:
   ```bash
   pytest tests/test_split_integrity.py
   pytest tests/test_hybrid_c_parity.py
   pytest tests/test_embedded_sil.py
   ```
3. Verify that all C artifacts compile without warnings:
   ```bash
   gcc -std=c99 -Wall -Wextra -Wpedantic -O3 \
       results/deployment/hybrid_controller_runner.c \
       results/deployment/hybrid_controller.c \
       results/deployment/platform_adapter.c \
       -o results/deployment/hybrid_controller_runner.exe
   ./results/deployment/hybrid_controller_runner.exe
   ```
4. Submit a Pull Request documenting the proposed changes, test outcomes, and any impact on computational complexity (parameters or MACs).
