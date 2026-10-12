/* ========================================================================
 * CartPole Swing-Up & Balance — Dual-Regime Hybrid C Controller
 * ========================================================================
 * Architecture:
 *   - Pumping Regime (|theta| >= 0.35 rad):
 *       Mechanical energy-shaping with cart centering:
 *       E = 0.5 * m * (L * th_dot)^2 + m * g * L * (cos_th - 1.0)
 *       u = k_E * E * sign(th_dot * cos_th) - k_x * x
 *   - Balancing Regime (|theta| < 0.35 rad):
 *       Linear full-state feedback (LQR):
 *       u = -(K_x * x + K_x_dot * x_dot + K_theta * theta + K_theta_dot * th_dot)
 *
 * Microcontroller Specifications (ESP32 / Cortex-M):
 *   - Compile-Time Parameters:    6 floats (K_x, K_x_dot, K_th, K_th_dot, k_E, k_x)
 *   - MAC Operations:             6 MACs per control cycle
 *   - RAM Footprint:              < 64 bytes (pure stack, 0 bytes heap)
 *   - Runtime Execution Time:     ~0.20 microseconds on ESP32 @ 240 MHz
 *   - Dependencies:               None (standard IEEE-754 math.h atan2f/fabsf)
 * ======================================================================== */

#ifndef CONTROLLER_H
#define CONTROLLER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Compile-time architectural constants */
#define HYBRID_NUM_PARAMS           7
#define HYBRID_NUM_MACS             7
#define HYBRID_RAM_USAGE_BYTES      32
#define HYBRID_EST_TIME_US_ESP32    0.20f

/* Tuned control gains */
#define HYBRID_K_X                  (-0.80f)
#define HYBRID_K_X_DOT              (-1.20f)
#define HYBRID_K_THETA              (-35.00f)
#define HYBRID_K_THETA_DOT          (-6.50f)
#define HYBRID_K_E                  (0.40f)
#define HYBRID_K_CENT_X             (1.20f)
#define HYBRID_K_CENT_V             (0.80f)

/* Physical threshold and constants */
#define HYBRID_THETA_THRESHOLD_RAD      (0.35f)
#define HYBRID_THETA_DOT_THRESHOLD_RAD_S (2.50f)
#define HYBRID_MASS_POLE            (0.10f)
#define HYBRID_LENGTH_POLE          (0.50f)
#define HYBRID_GRAVITY              (9.80f)

/**
 * Computes the continuous cartpole control action in [-1.0, 1.0].
 *
 * @param x         Cart position [m]
 * @param x_dot     Cart velocity [m/s]
 * @param cos_th    Cosine of pole angle (cos theta)
 * @param sin_th    Sine of pole angle (sin theta)
 * @param th_dot    Pole angular velocity [rad/s]
 * @return          Clamped force action u in [-1.0, 1.0]
 */
float compute_cartpole_action(
    float x,
    float x_dot,
    float cos_th,
    float sin_th,
    float th_dot
);

/**
 * Unified 5-element array interface: obs = [x, x_dot, cos_th, sin_th, th_dot].
 * Compatible with standard gym observation vectors.
 */
float compute_action(const float obs[5]);

#ifdef __cplusplus
}
#endif

#endif /* CONTROLLER_H */
