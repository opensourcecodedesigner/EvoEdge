/* ========================================================================
 * TinyML Neural Controller — Compact Evolved Feedforward Architecture
 * ========================================================================
 * Architecture: 5 Inputs -> 8 Hidden (Tanh) -> 1 Output (Tanh)
 * Status: Weights evolved, topology fixed.
 *
 * Theoretical Complexity & Embedded Microcontroller Metrics:
 *   - Total Parameters:          57
 *   - Multiply-Accumulate (MAC): 57 MACs per inference step
 *   - RAM Footprint:             < 64 bytes (pure stack, 0 bytes heap)
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

#define NEURAL_NUM_PARAMS 57
#define NEURAL_NUM_MACS   57

/* Fast clamp helper */
static inline float _neural_clamp(float v, float min_v, float max_v) {
    if (v < min_v) return min_v;
    if (v > max_v) return max_v;
    return v;
}

/* Scaled hyperbolic tangent matching training: tanh(2.5 * z) */
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

    /* 2. Hidden Layer (8 Neurons, Tanh activations) */
    /* Hidden Neuron 0 */
    float h0_sum = (+0.407136f);
    h0_sum += x_norm      * (-0.598023f);
    h0_sum += x_dot_norm  * (-2.703003f);
    h0_sum += cos_th      * (+0.398716f);
    h0_sum += sin_th      * (+0.248443f);
    h0_sum += th_dot_norm * (-0.078436f);
    const float h0 = _neat_act_tanh(h0_sum);

    /* Hidden Neuron 1 */
    float h1_sum = (+0.238957f);
    h1_sum += x_norm      * (-1.101050f);
    h1_sum += x_dot_norm  * (+0.297838f);
    h1_sum += cos_th      * (-0.428523f);
    h1_sum += sin_th      * (+0.000792f);
    h1_sum += th_dot_norm * (+2.071201f);
    const float h1 = _neat_act_tanh(h1_sum);

    /* Hidden Neuron 2 */
    float h2_sum = (+1.049509f);
    h2_sum += x_norm      * (+0.153657f);
    h2_sum += x_dot_norm  * (-3.796205f);
    h2_sum += cos_th      * (-2.703894f);
    h2_sum += sin_th      * (-0.086874f);
    h2_sum += th_dot_norm * (-1.559311f);
    const float h2 = _neat_act_tanh(h2_sum);

    /* Hidden Neuron 3 */
    float h3_sum = (+0.738620f);
    h3_sum += x_norm      * (-0.355425f);
    h3_sum += x_dot_norm  * (-0.347291f);
    h3_sum += cos_th      * (+0.176512f);
    h3_sum += sin_th      * (-1.587864f);
    h3_sum += th_dot_norm * (-2.175164f);
    const float h3 = _neat_act_tanh(h3_sum);

    /* Hidden Neuron 4 */
    float h4_sum = (-1.104928f);
    h4_sum += x_norm      * (+0.748385f);
    h4_sum += x_dot_norm  * (+2.572279f);
    h4_sum += cos_th      * (+1.019709f);
    h4_sum += sin_th      * (+10.906854f);
    h4_sum += th_dot_norm * (+16.209131f);
    const float h4 = _neat_act_tanh(h4_sum);

    /* Hidden Neuron 5 */
    float h5_sum = (-0.400138f);
    h5_sum += x_norm      * (+1.188766f);
    h5_sum += x_dot_norm  * (-1.605913f);
    h5_sum += cos_th      * (+0.626634f);
    h5_sum += sin_th      * (-0.600005f);
    h5_sum += th_dot_norm * (-4.131932f);
    const float h5 = _neat_act_tanh(h5_sum);

    /* Hidden Neuron 6 */
    float h6_sum = (-0.016123f);
    h6_sum += x_norm      * (-0.412387f);
    h6_sum += x_dot_norm  * (+4.933028f);
    h6_sum += cos_th      * (+0.026898f);
    h6_sum += sin_th      * (+1.118386f);
    h6_sum += th_dot_norm * (+5.730555f);
    const float h6 = _neat_act_tanh(h6_sum);

    /* Hidden Neuron 7 */
    float h7_sum = (+0.753834f);
    h7_sum += x_norm      * (-0.509976f);
    h7_sum += x_dot_norm  * (+0.747530f);
    h7_sum += cos_th      * (-0.540305f);
    h7_sum += sin_th      * (-2.233742f);
    h7_sum += th_dot_norm * (-4.170272f);
    const float h7 = _neat_act_tanh(h7_sum);

    /* 3. Output Layer (1 Neuron, Tanh activation) */
    float out_sum = (-0.604798f);
    out_sum += h0 * (+0.472039f);
    out_sum += h1 * (+0.843841f);
    out_sum += h2 * (+0.538032f);
    out_sum += h3 * (+1.157928f);
    out_sum += h4 * (+0.263045f);
    out_sum += h5 * (+0.730589f);
    out_sum += h6 * (+0.399687f);
    out_sum += h7 * (-0.831083f);
    const float action = _neat_act_tanh(out_sum);

    /* Saturate to [-1.0, 1.0] */
    return _neural_clamp(action, -1.0f, 1.0f);
}

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_CONTROLLER_H */
