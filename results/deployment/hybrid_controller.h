/* ========================================================================
 * EvoEdge Deployment Artifact: Analytical Hybrid Controller
 * ========================================================================
 * Architecture:
 *   - Regime 1: Balancing (|theta| < 0.35 rad AND |theta_dot| < 2.50 rad/s)
 *       Linear state feedback:
 *       u = -(K_x * x + K_x_dot * x_dot + K_theta * theta + K_theta_dot * theta_dot)
 *   - Regime 2: Pumping / Swing-Up (|theta| >= 0.35 rad OR |theta_dot| >= 2.50 rad/s)
 *       Nonlinear energy shaping with cart centering:
 *       E = (2/3)*m_p*L^2 * theta_dot^2 + m_p*g*L * (cos(theta) - 1.0)
 *       u = k_E * sign(E - E_0) * sign(theta_dot * cos(theta)) - k_x * x - k_v * x_dot
 *
 * Complexity Specifications:
 *   - Active Parameters:          7 gains
 *   - Multiply-Accumulate (MAC):  7 MACs per forward step
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
extern "C" {
#endif

/* ── Architectural & Complexity Constants ────────────────────────────── */
#define HYBRID_NUM_PARAMS                7
#define HYBRID_MAX_MACS                  7
#define HYBRID_STACK_USAGE_BYTES         32

/* ── Controller Gains (Single Precision) ──────────────────────────────── */
#define HYBRID_K_X                       (-0.800000f)
#define HYBRID_K_X_DOT                   (-1.200000f)
#define HYBRID_K_THETA                   (-35.000000f)
#define HYBRID_K_THETA_DOT               (-6.500000f)

#define HYBRID_K_E                       (0.600000f)
#define HYBRID_K_CENT_X                  (0.500000f)
#define HYBRID_K_CENT_V                  (0.250000f)

/* ── Controller Gains (Double Precision) ──────────────────────────────── */
#define HYBRID_K_X_D                     (-0.8000000000)
#define HYBRID_K_X_DOT_D                 (-1.2000000000)
#define HYBRID_K_THETA_D                 (-35.0000000000)
#define HYBRID_K_THETA_DOT_D             (-6.5000000000)

#define HYBRID_K_E_D                     (0.6000000000)
#define HYBRID_K_CENT_X_D                (0.5000000000)
#define HYBRID_K_CENT_V_D                (0.2500000000)

/* ── Physical Constants & Switching Thresholds ────────────────────────── */
#define HYBRID_THETA_THRESHOLD_RAD       (0.3500f)
#define HYBRID_THETA_DOT_THRESHOLD_RAD_S (2.5000f)
#define HYBRID_MASS_POLE                 (0.1000f)
#define HYBRID_LENGTH_POLE               (0.5000f)
#define HYBRID_GRAVITY                   (9.8000f)

/* Precomputed energy constants */
#define HYBRID_C_KINETIC                 ((2.0f / 3.0f) * HYBRID_MASS_POLE * HYBRID_LENGTH_POLE * HYBRID_LENGTH_POLE)
#define HYBRID_C_POTENTIAL               (HYBRID_MASS_POLE * HYBRID_GRAVITY * HYBRID_LENGTH_POLE)

#define HYBRID_C_KINETIC_D               ((2.0 / 3.0) * (0.1000000000) * (0.5000000000) * (0.5000000000))
#define HYBRID_C_POTENTIAL_D             ((0.1000000000) * (9.8000000000) * (0.5000000000))

/* ── Controller Regimes ───────────────────────────────────────────────── */
typedef enum {
    HYBRID_REGIME_INVALID = -1,
    HYBRID_REGIME_BALANCE = 0,
    HYBRID_REGIME_PUMPING = 1
} hybrid_regime_t;

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
}
#endif

#endif /* HYBRID_CONTROLLER_H */
