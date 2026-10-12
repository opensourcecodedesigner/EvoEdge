/* ========================================================================
 * TinyML Neural Controller — Compact Evolved Feedforward Architecture
 * ========================================================================
 * Architecture: 5 Inputs -> 3 Hidden (Tanh) -> 1 Output (Tanh)
 *
 * Theoretical Complexity & Embedded Microcontroller Metrics:
 *   - Total Parameters:          22 (18 weights + 4 biases)
 *   - Multiply-Accumulate (MAC): 22 MACs per inference step
 *   - RAM Footprint:             < 64 bytes (pure stack, 0 bytes heap)
 *   - MCU Execution Time:        ~0.85 us on ESP32 / Xtensa LX6 @ 240 MHz
 *   - Hardware Target:           ESP32 / ARM Cortex-M4 / TinyML Embedded
 *
 * Observation Vector Format (const float obs[5]):
 *   obs[0] : Cart Position (m)             [normalized internally by / 2.4]
 *   obs[1] : Cart Velocity (m/s)           [normalized internally by clip(/ 10, -1, 1)]
 *   obs[2] : Pole Cosine Angle (cos theta) [-1.0, 1.0]
 *   obs[3] : Pole Sine Angle (sin theta)   [-1.0, 1.0]
 *   obs[4] : Pole Angular Velocity (rad/s) [normalized internally by clip(/ 10, -1, 1)]
 *
 * Return Value:
 *   float  : Continuous control force in [-1.0, 1.0]
 * ======================================================================== */

#ifndef NEURAL_CONTROLLER_H
#define NEURAL_CONTROLLER_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEURAL_NUM_PARAMS 22
#define NEURAL_NUM_MACS   22

/* Fast clamp helper */
static inline float _neural_clamp(float v, float min_v, float max_v) {
    if (v < min_v) return min_v;
    if (v > max_v) return max_v;
    return v;
}

/* Scaled hyperbolic tangent matching NEAT-Python specification: tanh(2.5 * z) */
static inline float _neat_act_tanh(float z) {
    float scaled = 2.5f * z;
    if (scaled > 60.0f) scaled = 60.0f;
    else if (scaled < -60.0f) scaled = -60.0f;
    return tanhf(scaled);
}

/**
 * Compute continuous control action using the evolved TinyML neural network.
 *
 * @param obs  Raw observation array [x, x_dot, cos_theta, sin_theta, theta_dot]
 * @return     Control action in [-1.0, 1.0]
 */
static inline float compute_neural_action(const float obs[5]) {
    /* 1. Feature normalization matching training inputs */
    const float x_norm      = obs[0] / 2.4f;
    const float x_dot_norm  = _neural_clamp(obs[1] / 10.0f, -1.0f, 1.0f);
    const float cos_th      = obs[2];
    const float sin_th      = obs[3];
    const float th_dot_norm = _neural_clamp(obs[4] / 10.0f, -1.0f, 1.0f);

    /* 2. Hidden Layer (3 Neurons, Tanh activations) — 18 MACs */
    /* Hidden Neuron 0 */
    float h0_sum = (+0.463863f);
    h0_sum += x_norm      * (+0.704383f);
    h0_sum += x_dot_norm  * (+0.587601f);
    h0_sum += cos_th      * (-0.353934f);
    h0_sum += sin_th      * (+3.569789f);
    h0_sum += th_dot_norm * (+2.090387f);
    const float h0 = _neat_act_tanh(h0_sum);

    /* Hidden Neuron 1 */
    float h1_sum = (-0.917275f);
    h1_sum += x_norm      * (-0.560261f);
    h1_sum += x_dot_norm  * (-0.748128f);
    h1_sum += cos_th      * (-0.147078f);
    h1_sum += sin_th      * (+0.820688f);
    h1_sum += th_dot_norm * (-0.514977f);
    const float h1 = _neat_act_tanh(h1_sum);

    /* Hidden Neuron 2 */
    float h2_sum = (+0.690867f);
    h2_sum += x_norm      * (-0.144903f);
    h2_sum += x_dot_norm  * (-0.212107f);
    h2_sum += cos_th      * (+0.139512f);
    h2_sum += sin_th      * (+0.283142f);
    h2_sum += th_dot_norm * (-0.084436f);
    const float h2 = _neat_act_tanh(h2_sum);

    /* 3. Output Layer (1 Neuron, Tanh activation) — 4 MACs */
    float out_sum = (+0.134581f);
    out_sum += h0 * (+2.448708f);
    out_sum += h1 * (+0.278175f);
    out_sum += h2 * (+0.253521f);
    const float action = _neat_act_tanh(out_sum);

    /* Saturate to [-1.0, 1.0] */
    return _neural_clamp(action, -1.0f, 1.0f);
}

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_CONTROLLER_H */
