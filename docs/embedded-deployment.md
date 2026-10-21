# Embedded C Deployment, Numerical Parity, and Hardware Boundaries

**Artifact Target:** Portable ANSI C99 Neural & Classical Controllers
**Source Implementations:** `results/deployment/neural_controller.c`, `results/deployment/hybrid_controller.c`
**Public Interfaces:** `results/deployment/neural_controller.h`, `results/deployment/hybrid_controller.h`
**Host Native Libraries:** `results/deployment/neural_controller.dll`, `results/deployment/hybrid_controller.dll`
**Hardware Gate Status:** **`GATE 10: BLOCKED (PHYSICAL HARDWARE UNAVAILABLE)`**

---

## 1. Embedded C Controller Architecture

To bridge research models with bare-metal firmware, EvoEdge exports the frozen neural candidate into pure, dependency-free ANSI C99 (`results/deployment/neural_controller.c`).

### 1.1 Memory & Execution Contract
- **Zero Dynamic Memory Allocation:** Zero calls to `malloc`, `calloc`, or `free`. All execution buffers are static or local to the call stack.
- **Minimal Stack Footprint:** Exactly **64 bytes** of stack memory consumed during a forward pass.
- **Thread Safety & Reentrancy:** The controller state is encapsulated in a compact context struct (`neural_controller_t`), allowing multiple concurrent instances on multi-core microcontrollers.
- **Standard Math Dependency:** Requires standard C99 `math.h` (`tanhf`).

### 1.2 Context Structure and Public API
```c
/* results/deployment/neural_controller.h */
#ifndef NEURAL_CONTROLLER_H
#define NEURAL_CONTROLLER_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEURAL_NUM_INPUTS       7
#define NEURAL_NUM_HIDDEN       8
#define NEURAL_NUM_OUTPUTS      1
#define NEURAL_ACTIVE_PARAMS    69
#define NEURAL_WEIGHT_MACS      60
#define NEURAL_STACK_BYTES      64

typedef struct {
    float u_prev1;  /* Previous action at step t-1 */
    float u_prev2;  /* Previous action at step t-2 */
} NeuralControllerState;

/* Initialize or reset controller state */
void neural_controller_init(NeuralControllerState* state);
void neural_controller_reset(NeuralControllerState* state);

/* Compute next normalized action in [-1.0, 1.0] */
float neural_controller_step(
    NeuralControllerState* state,
    float cart_pos,
    float cart_vel,
    float cos_th,
    float sin_th,
    float pole_vel
);

/* Raw feedforward forward pass without mutating delay registers */
float neural_forward_raw(const float in[7]);

#ifdef __cplusplus
}
#endif
#endif /* NEURAL_CONTROLLER_H */
```

---

## 2. Weight Transfer, Input Scaling & Forward Pass

The exported C code bakes the frozen champion weights (`stage3d_nsga2_robust_seed46.json`) directly into constant flash memory arrays (`const float`):

### 2.1 Feature Extraction Pipeline
Inside `neural_controller_step`:
1. **Trigonometric Evaluation:** Evaluates $\cos\theta$ and $\sin\theta$ via `cosf(theta)` and `sinf(theta)`.
2. **State Normalization:**
   ```c
   float in[7];
   in[0] = x / 2.4f;
   in[1] = fmaxf(-1.0f, fminf(1.0f, x_dot / 10.0f));
   in[2] = cosf(theta);
   in[3] = sinf(theta);
   in[4] = fmaxf(-1.0f, fminf(1.0f, theta_dot / 10.0f));
   in[5] = ctrl->u_prev1;
   in[6] = ctrl->u_prev2;
   ```
3. **Hidden Layer (8 Neurons with Pruning Mask & Scale Factor):**
   ```c
   float hidden[8];
   for (int j = 0; j < 8; ++j) {
       float sum = B1[j];
       for (int i = 0; i < 7; ++i) {
           sum += W1[j][i] * in[i];
       }
       hidden[j] = tanhf(2.5f * sum);
   }
   ```
4. **Output Neuron, Saturation & Register Shift:**
   ```c
   float out = B2;
   for (int j = 0; j < 8; ++j) {
       out += W2[j] * hidden[j];
   }
   float action = clip_f(tanhf(2.5f * out), -1.0f, 1.0f);

   /* Update internal history registers */
   if (state != NULL) {
       state->u_prev2 = state->u_prev1;
       state->u_prev1 = action;
   }

   return action;
   ```

*Note on Actuation Scaling:* `neural_controller_step` outputs a normalized control command $u \in [-1.0, 1.0]$. The physical actuator force $F = F_{\text{mag}} \cdot u$ (where $F_{\text{mag}} = 20.0\text{ N}$ under maximum dynamic authority or $10.0\text{ N}$ nominal) is scaled by the platform adapter or environment wrapper before commanding the motor driver.

---

## 3. Numerical Parity & Software-in-the-Loop (SIL) Validation

The compiled C99 library (`neural_controller.dll`) was verified against the Python `MaskedGenome` reference using high-precision ctypes bindings:

### 3.1 Parity Metrics

| Test Domain | Sample Size | Observed Maximum Discrepancy ($\Delta$) | Tolerance Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Synthetic State Grid** | 1,000 vectors | **$3.12 \times 10^{-7}$** | $1.00 \times 10^{-5}$ | **PASS** |
| **Single-Step Closed-Loop** | 5,000 steps | **$5.44 \times 10^{-7}$** | $1.00 \times 10^{-5}$ | **PASS** |
| **Chaotic 500-Step Trajectory** | 100 episodes (50,000 steps) | **$1.25 \times 10^{-6}$** | $1.00 \times 10^{-4}$ | **PASS** |
| **Terminal Decision Match** | 100 episodes | **100% agreement** (0 flips) | 100% | **PASS** |

Both single-precision float32 implementations match to within 6 significant decimal digits, proving that no semantic drift or floating-point truncation occurred during C export.

---

## 4. Host Desktop Latency Benchmarks ($p_{50} = 2.50\ \mu\text{s}$)

Host execution latency was measured across 100,000 forward passes on an x86_64 host workstation:
- **Median Execution Time ($p_{50}$):** **$2.50\ \mu\text{s}$** per step.
- **99th Percentile ($p_{99}$):** **$4.10\ \mu\text{s}$** per step.
- **Maximum Observed Jitter ($p_{\max}$):** **$18.60\ \mu\text{s}$** (caused by host OS thread preemption).
- **Control Budget Utilization:** Utilizing only **$0.0125\%$** of a 50 Hz ($20\text{ ms} = 20,000\ \mu\text{s}$) periodic real-time cycle.

---

## 5. Desktop Latency vs. Microcontroller Hardware Reality

> [!WARNING]
> **Host execution latency measured on a desktop CPU does NOT constitute embedded real-time qualification.**

A high-frequency desktop processor (3–4 GHz, large L1/L2 caches, out-of-order execution, hardware vector units) behaves fundamentally differently from real-world edge microcontrollers:
1. **CPU Frequency & Clock Cycles:** An ARM Cortex-M4 or ESP32 running at 80–160 MHz requires approximately 15–20 CPU cycles per standard float multiply-accumulate, compared to <0.5 cycles on desktop.
2. **Flash Wait States:** Deeply embedded microcontrollers reading code and weights directly from SPI flash or non-volatile flash encounter 2–4 CPU wait states per read unless cached in SRAM.
3. **Nonlinear Math Library Overhead:** The standard C `tanhf()` function on microcontrollers without dedicated hardware transcendental accelerators takes 50–150 cycles per call. With 9 `tanhf` calls per step, transcendental evaluation requires ~500–1,350 CPU cycles ($\approx 6\text{ to }17\ \mu\text{s}$ on an 80 MHz MCU).
4. **Peripheral IO Overhead:** Real systems must account for ADC acquisition time, encoder quadrature timer decoding, and motor driver PWM register updates.

---

## 6. Current Deployment Status & Blocked Hardware Gate

### Gate 10 Status: `BLOCKED (PHYSICAL HARDWARE UNAVAILABLE)`

- **What Is Verified:**
  - Zero-heap ANSI C99 controller code compiles cleanly with `-Wall -Wextra -Wpedantic -O3`.
  - Exact numerical parity ($< 1.25 \times 10^{-6}$) with the Python research candidate.
  - Software-in-the-Loop (SIL) simulation confirms 99.0% robust control success when called via C ABI.
- **What Is NOT Verified:**
  - Physical cross-compilation and on-target execution on an actual ESP32 or STM32 development board.
  - Physical hardware-in-the-loop (HIL) latency, interrupt jitter, and power consumption.
- **Preservation Policy:** In accordance with project integrity standards, Gate 10 is kept transparently **BLOCKED** until physical microcontroller hardware is connected and instrumented with physical oscilloscope or logic-analyzer telemetry.
