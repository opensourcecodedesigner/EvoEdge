/* ========================================================================
 * NEAT CartPole Swing-Up — Standalone C Controller
 * ========================================================================
 * Generated from: topology.json
 *
 * Model Architecture & Theoretical Complexity:
 *   - Total Parameters:          6
 *   - Active Synapses:           5
 *   - Hidden Neurons:            0
 *   - Theoretical MAC Operations: 6
 *   - Input Dimensions:          5
 *   - Output Dimensions:         1
 *
 * Observation Vector Format (const float state[5]):
 *   state[0] : Cart Position (normalized: x / 2.4)
 *   state[1] : Cart Velocity (normalized: clip(x_dot / 10.0, -1, 1))
 *   state[2] : Pole Cosine Angle (cos(theta))
 *   state[3] : Pole Sine Angle (sin(theta))
 *   state[4] : Pole Angular Velocity (normalized: clip(theta_dot / 10.0, -1, 1))
 *
 * Return Value:
 *   float    : Continuous force action in [-1.0, 1.0]
 * ======================================================================== */

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Activation Functions (Matching NEAT-Python Specs) ─────────────── */

static inline float _neat_act_tanh(float z) {
    float scaled = 2.5f * z;
    if (scaled > 60.0f) scaled = 60.0f;
    else if (scaled < -60.0f) scaled = -60.0f;
    return tanhf(scaled);
}

static inline float _neat_act_identity(float z) {
    return z;
}

static inline float _neat_act_relu(float z) {
    return (z > 0.0f) ? z : 0.0f;
}

static inline float _neat_act_sigmoid(float z) {
    float scaled = 5.0f * z;
    if (scaled > 60.0f) scaled = 60.0f;
    else if (scaled < -60.0f) scaled = -60.0f;
    return 1.0f / (1.0f + expf(-scaled));
}

/* ── Forward Inference Function ──────────────────────────────────────── */

/**
 * Execute one forward inference step of the evolved neural controller.
 *
 * @param state  Pointer to float array of length 5 (normalized observations).
 * @return       Control signal action in range [-1.0, 1.0].
 */
static inline float run_controller(const float state[5]) {
    /* Node 0 (output, activation: tanh, bias: 0.300747) */
    float node_0_sum = 0.0f;
    node_0_sum += state[4] * (+0.3912610f);
    node_0_sum += state[3] * (-0.7716990f);
    node_0_sum += state[2] * (-0.6827520f);
    node_0_sum += state[1] * (-0.9774280f);
    node_0_sum += state[0] * (-1.9696500f);
    float node_0 = _neat_act_tanh(+0.3007470f + node_0_sum);

    return node_0;
}

/* ── Optional Helper: Normalize Raw Gym State ────────────────────────── */

/**
 * Convenience helper to normalize raw gym observations before run_controller.
 *
 * @param raw_state   Raw environment observation [x, x_dot, cos_theta, sin_theta, theta_dot]
 * @param normalized  Output array of length 5 for normalized features
 */
static inline void normalize_state(const float raw_state[5], float normalized[5]) {
    normalized[0] = raw_state[0] / 2.4f;
    float x_dot_norm = raw_state[1] / 10.0f;
    normalized[1] = (x_dot_norm > 1.0f) ? 1.0f : ((x_dot_norm < -1.0f) ? -1.0f : x_dot_norm);
    normalized[2] = raw_state[2];
    normalized[3] = raw_state[3];
    float th_dot_norm = raw_state[4] / 10.0f;
    normalized[4] = (th_dot_norm > 1.0f) ? 1.0f : ((th_dot_norm < -1.0f) ? -1.0f : th_dot_norm);
}

#ifdef __cplusplus
}
#endif

#endif /* CONTROLLER_H */
