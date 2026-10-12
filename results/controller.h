/* ========================================================================
 * CartPole Swing-Up & Balance — Energy-Shaping + Linear Feedback C Controller
 * ========================================================================
 * Architecture:
 *   - Pumping Regime (|theta| >= 0.35 rad OR |theta_dot| >= 2.5 rad/s):
 *       Nonlinear sign-based mechanical energy-shaping with cart centering:
 *       E = (2/3) * m * (L * th_dot)^2 + m * g * L * (cos_th - 1.0)
 *       u = k_E * sign(E - E_0) * sign(th_dot * cos_th) - k_x * x - k_v * x_dot
 *   - Balancing Regime (|theta| < 0.35 rad AND |theta_dot| < 2.5 rad/s):
 *       Linear full-state feedback:
 *       u = -(K_x * x + K_x_dot * x_dot + K_theta * theta + K_theta_dot * th_dot)
 *
 * Embedded Microcontroller Specifications:
 *   - Total Parameters:           7 float gains
 *   - Multiply-Accumulate (MAC):  7 MACs per forward evaluation
 *   - RAM Footprint:              < 64 bytes (pure stack, 0 bytes heap)
 *   - Dependencies:               Standard IEEE-754 math.h (atan2f, fabsf)
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

/* Linear feedback gains: hand-tuned */
#define HYBRID_K_X                  (-0.80f)
#define HYBRID_K_X_DOT              (-1.20f)
#define HYBRID_K_THETA              (-35.00f)
#define HYBRID_K_THETA_DOT          (-6.50f)

/* True continuous LQR gains (Q=diag(1, 0.1, 10, 0.1), R=1) */
#define HYBRID_TRUE_LQR_K_X         (-1.0000f)
#define HYBRID_TRUE_LQR_K_X_DOT     (-1.2138f)
#define HYBRID_TRUE_LQR_K_THETA     (-7.8073f)
#define HYBRID_TRUE_LQR_K_THETA_DOT (-1.8677f)

/* Pumping & Centering Gains (Fix 1 verified) */
#define HYBRID_K_E                  (0.60f)
#define HYBRID_K_CENT_X             (0.50f)
#define HYBRID_K_CENT_V             (0.25f)

/* Physical threshold and constants */
#define HYBRID_THETA_THRESHOLD_RAD       (0.35f)
#define HYBRID_THETA_DOT_THRESHOLD_RAD_S (2.50f)
#define HYBRID_MASS_POLE                 (0.10f)
#define HYBRID_LENGTH_POLE               (0.50f)
#define HYBRID_GRAVITY                   (9.80f)

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
