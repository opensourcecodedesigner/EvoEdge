/* ========================================================================
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
