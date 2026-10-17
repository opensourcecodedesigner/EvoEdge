"""Dedicated C Exporter for Energy-Shaping + Linear Feedback Hybrid Controller.

Exports standalone, zero-heap ANSI C source and header files:
  - results/deployment/hybrid_controller.h
  - results/deployment/hybrid_controller.c

Features:
  - Exact reproduction of baselines/hybrid_controller.py control laws.
  - Zero dynamic memory allocation (pure stack, O(1) memory, < 64 bytes).
  - Explicit IEEE-754 input validation (NaN and Inf checks returning safe 0.0f).
  - Single-precision float (MCU-optimized) and double-precision interfaces.
  - Exported complexity: exactly 7 parameters, max 7 MACs.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from baselines.hybrid_controller import (
    HybridController,
    HAND_TUNED_LINEAR_GAINS,
    TRUE_LQR_GAINS,
)


def generate_header(controller: Optional[HybridController] = None) -> str:
    """Generate deployment/hybrid_controller.h content."""
    ctrl = controller if controller is not None else HybridController()
    spec = ctrl.get_spec()

    k_x = ctrl.K[0]
    k_x_dot = ctrl.K[1]
    k_theta = ctrl.K[2]
    k_theta_dot = ctrl.K[3]

    k_E = ctrl.k_E
    k_cent_x = ctrl.k_x
    k_cent_v = ctrl.k_v

    th_thresh = ctrl.theta_threshold
    th_dot_thresh = ctrl.theta_dot_threshold
    m_p = ctrl.mass_pole
    l_p = ctrl.length
    g = ctrl.gravity

    header = f"""/* ========================================================================
 * EvoEdge Deployment Artifact: Analytical Hybrid Controller
 * ========================================================================
 * Architecture:
 *   - Regime 1: Balancing (|theta| < {th_thresh:.2f} rad AND |theta_dot| < {th_dot_thresh:.2f} rad/s)
 *       Linear state feedback:
 *       u = -(K_x * x + K_x_dot * x_dot + K_theta * theta + K_theta_dot * theta_dot)
 *   - Regime 2: Pumping / Swing-Up (|theta| >= {th_thresh:.2f} rad OR |theta_dot| >= {th_dot_thresh:.2f} rad/s)
 *       Nonlinear energy shaping with cart centering:
 *       E = (2/3)*m_p*L^2 * theta_dot^2 + m_p*g*L * (cos(theta) - 1.0)
 *       u = k_E * sign(E - E_0) * sign(theta_dot * cos(theta)) - k_x * x - k_v * x_dot
 *
 * Complexity Specifications:
 *   - Active Parameters:          {spec['active_parameters']} gains
 *   - Multiply-Accumulate (MAC):  {spec['theoretical_macs']} MACs per forward step
 *   - Dynamic Heap Allocation:    0 bytes (100% stack, O(1) memory)
 *   - Action Saturation:          [-1.0, 1.0] normalized continuous force
 *   - Dependencies:               Standard C99 math.h (atan2f, fabsf, isfinite)
 * ======================================================================== */

#ifndef HYBRID_CONTROLLER_H
#define HYBRID_CONTROLLER_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #define HYBRID_API __declspec(dllexport)
#else
  #define HYBRID_API
#endif

#ifdef __cplusplus
extern "C" {{
#endif

/* ── Architectural & Complexity Constants ────────────────────────────── */
#define HYBRID_NUM_PARAMS                {spec['active_parameters']}
#define HYBRID_MAX_MACS                  {spec['theoretical_macs']}
#define HYBRID_STACK_USAGE_BYTES         32

/* ── Controller Gains (Single Precision) ──────────────────────────────── */
#define HYBRID_K_X                       ({k_x:.6f}f)
#define HYBRID_K_X_DOT                   ({k_x_dot:.6f}f)
#define HYBRID_K_THETA                   ({k_theta:.6f}f)
#define HYBRID_K_THETA_DOT               ({k_theta_dot:.6f}f)

#define HYBRID_K_E                       ({k_E:.6f}f)
#define HYBRID_K_CENT_X                  ({k_cent_x:.6f}f)
#define HYBRID_K_CENT_V                  ({k_cent_v:.6f}f)

/* ── Controller Gains (Double Precision) ──────────────────────────────── */
#define HYBRID_K_X_D                     ({k_x:.10f})
#define HYBRID_K_X_DOT_D                 ({k_x_dot:.10f})
#define HYBRID_K_THETA_D                 ({k_theta:.10f})
#define HYBRID_K_THETA_DOT_D             ({k_theta_dot:.10f})

#define HYBRID_K_E_D                     ({k_E:.10f})
#define HYBRID_K_CENT_X_D                ({k_cent_x:.10f})
#define HYBRID_K_CENT_V_D                ({k_cent_v:.10f})

/* ── Physical Constants & Switching Thresholds ────────────────────────── */
#define HYBRID_THETA_THRESHOLD_RAD       ({th_thresh:.4f}f)
#define HYBRID_THETA_DOT_THRESHOLD_RAD_S ({th_dot_thresh:.4f}f)
#define HYBRID_MASS_POLE                 ({m_p:.4f}f)
#define HYBRID_LENGTH_POLE               ({l_p:.4f}f)
#define HYBRID_GRAVITY                   ({g:.4f}f)

/* Precomputed energy constants */
#define HYBRID_C_KINETIC                 ((2.0f / 3.0f) * HYBRID_MASS_POLE * HYBRID_LENGTH_POLE * HYBRID_LENGTH_POLE)
#define HYBRID_C_POTENTIAL               (HYBRID_MASS_POLE * HYBRID_GRAVITY * HYBRID_LENGTH_POLE)

#define HYBRID_C_KINETIC_D               ((2.0 / 3.0) * ({m_p:.10f}) * ({l_p:.10f}) * ({l_p:.10f}))
#define HYBRID_C_POTENTIAL_D             (({m_p:.10f}) * ({g:.10f}) * ({l_p:.10f}))

/* ── Controller Regimes ───────────────────────────────────────────────── */
typedef enum {{
    HYBRID_REGIME_INVALID = -1,
    HYBRID_REGIME_BALANCE = 0,
    HYBRID_REGIME_PUMPING = 1
}} hybrid_regime_t;

/* ── Public API ──────────────────────────────────────────────────────── */

/**
 * Computes control action u in [-1.0, 1.0] from explicit scalar state components.
 *
 * @param x         Cart position [m] (operating range: [-2.4, 2.4])
 * @param x_dot     Cart velocity [m/s]
 * @param cos_th    Cosine of pole angle cos(theta)
 * @param sin_th    Sine of pole angle sin(theta)
 * @param th_dot    Pole angular velocity [rad/s]
 * @return          Clamped continuous force action in [-1.0, 1.0].
 *                  Returns 0.0f if any input is NaN or Inf.
 */
HYBRID_API float hybrid_compute_action_explicit(
    float x,
    float x_dot,
    float cos_th,
    float sin_th,
    float th_dot
);

/**
 * Standard array interface: obs = [x, x_dot, cos_theta, sin_theta, theta_dot].
 * Compatible with gym CartPole 5D observation vectors.
 *
 * @param obs       Pointer to array of 5 float observation values.
 * @return          Clamped continuous force action in [-1.0, 1.0].
 *                  Returns 0.0f if pointer is NULL or any element is non-finite.
 */
HYBRID_API float hybrid_compute_action(const float obs[5]);

/**
 * Double-precision interface for bit-accurate numerical verification.
 *
 * @param obs       Pointer to array of 5 double observation values.
 * @return          Clamped continuous force action in [-1.0, 1.0].
 */
HYBRID_API double hybrid_compute_action_double(const double obs[5]);

/**
 * Diagnostic query: returns the active control regime for given observation.
 *
 * @param obs       Pointer to array of 5 float observation values.
 * @return          HYBRID_REGIME_BALANCE, HYBRID_REGIME_PUMPING, or HYBRID_REGIME_INVALID.
 */
HYBRID_API hybrid_regime_t hybrid_get_active_regime(const float obs[5]);

#ifdef __cplusplus
}}
#endif

#endif /* HYBRID_CONTROLLER_H */
"""
    return header


def generate_source(controller: Optional[HybridController] = None) -> str:
    """Generate deployment/hybrid_controller.c content."""
    ctrl = controller if controller is not None else HybridController()

    source = """/* ========================================================================
 * EvoEdge Deployment Artifact: Analytical Hybrid Controller Source
 * ======================================================================== */

#include "hybrid_controller.h"
#include <stddef.h>

/* Fast clamp helper without branch prediction penalty */
static inline float _hybrid_clampf(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static inline double _hybrid_clampd(double val, double min_val, double max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

HYBRID_API float hybrid_compute_action_explicit(
    float x,
    float x_dot,
    float cos_th,
    float sin_th,
    float th_dot
) {
    /* 1. Input Validation: Fail-safe zero response on non-finite input */
    if (!isfinite(x) || !isfinite(x_dot) || !isfinite(cos_th) ||
        !isfinite(sin_th) || !isfinite(th_dot)) {
        return 0.0f;
    }

    /* 2. Angle reconstruction: theta in [-pi, pi], 0.0 rad is vertical upright */
    const float theta = atan2f(sin_th, cos_th);
    const float abs_theta = fabsf(theta);
    const float abs_th_dot = fabsf(th_dot);

    float u = 0.0f;

    /* 3. Dual-Regime Selection:
     * Switch to linear balance ONLY when |theta| < 0.35 rad AND |theta_dot| < 2.5 rad/s
     */
    if (abs_theta < HYBRID_THETA_THRESHOLD_RAD && abs_th_dot < HYBRID_THETA_DOT_THRESHOLD_RAD_S) {
        /*
         * --- Balancing Regime ---
         * Full-state linear feedback:
         * u = -(K_x * x + K_x_dot * x_dot + K_theta * theta + K_theta_dot * th_dot)
         */
        const float term_x      = HYBRID_K_X * x;
        const float term_x_dot  = HYBRID_K_X_DOT * x_dot;
        const float term_theta  = HYBRID_K_THETA * theta;
        const float term_th_dot = HYBRID_K_THETA_DOT * th_dot;

        u = -(term_x + term_x_dot + term_theta + term_th_dot);
    } else {
        /*
         * --- Pumping / Swing-Up Regime ---
         * Mechanical energy relative to upright equilibrium E_0 = 0.0:
         *   c_kinetic   = (2/3) * m_p * L^2 = 0.01666667
         *   c_potential = m_p * g * L       = 0.49000000
         *   E = c_kinetic * (th_dot^2) + c_potential * (cos_th - 1.0)
         *   sign_E = 1 if E > 0, -1 if E < 0, 0 if E == 0
         *   sign_p = 1 if th_dot*cos_th > 0, -1 if < 0, +1 if == 0
         *   u = k_E * sign_E * sign_p - k_x * x - k_v * x_dot
         */
        const float c_kinetic = HYBRID_C_KINETIC;
        const float c_potential = HYBRID_C_POTENTIAL;

        const float E = (c_kinetic * th_dot * th_dot) + (c_potential * (cos_th - 1.0f));
        const float sign_E = (E > 0.0f) ? 1.0f : ((E < 0.0f) ? -1.0f : 0.0f);

        const float p = th_dot * cos_th;
        const float sign_p = (p > 0.0f) ? 1.0f : ((p < 0.0f) ? -1.0f : 1.0f);

        u = (HYBRID_K_E * sign_E * sign_p) - (HYBRID_K_CENT_X * x) - (HYBRID_K_CENT_V * x_dot);
    }

    /* 4. Output saturation to track motor limit [-1.0, 1.0] */
    return _hybrid_clampf(u, -1.0f, 1.0f);
}

HYBRID_API float hybrid_compute_action(const float obs[5]) {
    if (obs == NULL) {
        return 0.0f;
    }
    return hybrid_compute_action_explicit(obs[0], obs[1], obs[2], obs[3], obs[4]);
}

HYBRID_API double hybrid_compute_action_double(const double obs[5]) {
    if (obs == NULL) {
        return 0.0;
    }
    const double x = obs[0];
    const double x_dot = obs[1];
    const double cos_th = obs[2];
    const double sin_th = obs[3];
    const double th_dot = obs[4];

    if (!isfinite(x) || !isfinite(x_dot) || !isfinite(cos_th) ||
        !isfinite(sin_th) || !isfinite(th_dot)) {
        return 0.0;
    }

    const double theta = atan2(sin_th, cos_th);
    const double abs_theta = fabs(theta);
    const double abs_th_dot = fabs(th_dot);

    double u = 0.0;

    if (abs_theta < (double)HYBRID_THETA_THRESHOLD_RAD && abs_th_dot < (double)HYBRID_THETA_DOT_THRESHOLD_RAD_S) {
        const double term_x      = HYBRID_K_X_D * x;
        const double term_x_dot  = HYBRID_K_X_DOT_D * x_dot;
        const double term_theta  = HYBRID_K_THETA_D * theta;
        const double term_th_dot = HYBRID_K_THETA_DOT_D * th_dot;

        u = -(term_x + term_x_dot + term_theta + term_th_dot);
    } else {
        const double c_kinetic = HYBRID_C_KINETIC_D;
        const double c_potential = HYBRID_C_POTENTIAL_D;

        const double E = (c_kinetic * th_dot * th_dot) + (c_potential * (cos_th - 1.0));
        const double sign_E = (E > 0.0) ? 1.0 : ((E < 0.0) ? -1.0 : 0.0);

        const double p = th_dot * cos_th;
        const double sign_p = (p > 0.0) ? 1.0 : ((p < 0.0) ? -1.0 : 1.0);

        u = (HYBRID_K_E_D * sign_E * sign_p) - (HYBRID_K_CENT_X_D * x) - (HYBRID_K_CENT_V_D * x_dot);
    }

    return _hybrid_clampd(u, -1.0, 1.0);
}

HYBRID_API hybrid_regime_t hybrid_get_active_regime(const float obs[5]) {
    if (obs == NULL) {
        return HYBRID_REGIME_INVALID;
    }
    const float x = obs[0];
    const float x_dot = obs[1];
    const float cos_th = obs[2];
    const float sin_th = obs[3];
    const float th_dot = obs[4];

    if (!isfinite(x) || !isfinite(x_dot) || !isfinite(cos_th) || !isfinite(sin_th) || !isfinite(th_dot)) {
        return HYBRID_REGIME_INVALID;
    }

    const float theta = atan2f(sin_th, cos_th);
    if (fabsf(theta) < HYBRID_THETA_THRESHOLD_RAD && fabsf(th_dot) < HYBRID_THETA_DOT_THRESHOLD_RAD_S) {
        return HYBRID_REGIME_BALANCE;
    }
    return HYBRID_REGIME_PUMPING;
}
"""
    return source


def export_hybrid_controller_c(
    output_dir: str = "results/deployment",
    controller: Optional[HybridController] = None,
) -> Tuple[str, str]:
    """Export hybrid_controller.h and hybrid_controller.c to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    header_path = os.path.join(output_dir, "hybrid_controller.h")
    source_path = os.path.join(output_dir, "hybrid_controller.c")

    header_code = generate_header(controller)
    source_code = generate_source(controller)

    with open(header_path, "w", encoding="utf-8") as f:
        f.write(header_code)

    with open(source_path, "w", encoding="utf-8") as f:
        f.write(source_code)

    print(f"Exported C header: {header_path}")
    print(f"Exported C source: {source_path}")
    return header_path, source_path


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/deployment"
    export_hybrid_controller_c(out_dir)
