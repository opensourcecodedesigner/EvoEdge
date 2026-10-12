/* ========================================================================
 * CartPole Swing-Up & Balance — Energy-Shaping + Linear Feedback C Controller
 * ======================================================================== */

#include "controller.h"
#include <math.h>

/* Fast clamp helper without library overhead */
static inline float clampf(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

float compute_cartpole_action(
    float x,
    float x_dot,
    float cos_th,
    float sin_th,
    float th_dot
) {
    /* 1. Extract pole angle: theta in [-pi, pi], 0 rad is vertical upright */
    const float theta = atan2f(sin_th, cos_th);
    const float abs_theta = fabsf(theta);
    const float abs_th_dot = fabsf(th_dot);

    float u = 0.0f;

    /* 2. Dual-Regime Selection:
     * Switch to linear balance ONLY when |theta| < 0.35 rad AND |theta_dot| < 2.5 rad/s
     */
    if (abs_theta < HYBRID_THETA_THRESHOLD_RAD && abs_th_dot < HYBRID_THETA_DOT_THRESHOLD_RAD_S) {
        /*
         * --- Balancing Regime (|theta| < 0.35 rad AND |th_dot| < 2.5 rad/s) ---
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
         * --- Pumping / Swing-Up Regime (|theta| >= 0.35 rad OR |th_dot| >= 2.5 rad/s) ---
         * Corrected mechanical energy-shaping:
         *   c_kinetic   = (2/3) * m * L^2 = (2/3) * 0.1 * 0.25 = 0.016667
         *   c_potential = m * g * L       = 0.1 * 9.8 * 0.5   = 0.4900
         *   E = c_kinetic * (th_dot^2) + c_potential * (cos_th - 1.0)
         *   u = k_E * sign(E - E_0) * sign(th_dot * cos_th) - k_x * x - k_v * x_dot
         */
        const float c_kinetic = 0.016667f;
        const float c_potential = 0.4900f;

        const float E = (c_kinetic * th_dot * th_dot) + (c_potential * (cos_th - 1.0f));

        const float sign_E = (E > 0.0f) ? 1.0f : ((E < 0.0f) ? -1.0f : 0.0f);
        const float p = th_dot * cos_th;
        const float sign_p = (p > 0.0f) ? 1.0f : ((p < 0.0f) ? -1.0f : 1.0f);

        u = (HYBRID_K_E * sign_E * sign_p) - (HYBRID_K_CENT_X * x) - (HYBRID_K_CENT_V * x_dot);
    }

    /* 3. Output saturation to track motor limit [-1.0, 1.0] */
    return clampf(u, -1.0f, 1.0f);
}

float compute_action(const float obs[5]) {
    return compute_cartpole_action(obs[0], obs[1], obs[2], obs[3], obs[4]);
}
