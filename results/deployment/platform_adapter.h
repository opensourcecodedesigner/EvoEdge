/* ========================================================================
 * EvoEdge Deployment Artifact: Embedded Platform Adapter Interface
 * ========================================================================
 * Architecture:
 *   Sensor Acquisition -> Observation Conversion -> Controller Step -> Actuator Conversion
 *
 * Design Principles:
 *   - Strictly modular: Sensor conversion, controller execution, and actuator
 *     conversion are separately testable functions.
 *   - Zero dynamic memory: Uses only fixed-size stack structures (< 64 bytes).
 *   - Platform-agnostic: Pure ANSI C99, zero OS or vendor SDK dependencies.
 *   - Configurable update rate: Default 50 Hz (20.0 ms nominal loop time).
 * ======================================================================== */

#ifndef PLATFORM_ADAPTER_H
#define PLATFORM_ADAPTER_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "hybrid_controller.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Physical Sensor Data Structure (Input) ───────────────────────────── */
typedef struct {
    float cart_pos_m;        /**< Cart position x along track [m] (-2.4 to +2.4 m) */
    float cart_vel_mps;      /**< Cart linear velocity x_dot [m/s] */
    float pole_angle_rad;    /**< Pole angle theta [rad] (0 = upright vertical, +-pi = downward hanging) */
    float pole_ang_vel_rads; /**< Pole angular velocity theta_dot [rad/s] */
    uint32_t timestamp_us;   /**< Monotonic timestamp of acquisition [microseconds] */
    bool sensor_fault;       /**< Hardware fault flag from ADC / encoder peripheral */
} platform_sensor_data_t;

/* ── Observation Vector Structure (5D Gym Compatible) ─────────────────── */
typedef struct {
    float values[5];         /**< [0]: x, [1]: x_dot, [2]: cos(theta), [3]: sin(theta), [4]: theta_dot */
    bool is_valid;           /**< True if all 5 values are finite and valid */
} platform_observation_t;

/* ── Actuator Command Structure (Output) ──────────────────────────────── */
typedef struct {
    float normalized_force;  /**< Clamped continuous action u in [-1.0, 1.0] */
    float pwm_duty_pct;      /**< Motor PWM duty cycle in [0.0%, 100.0%] */
    int8_t motor_direction;  /**< +1 = Right / Push, -1 = Left / Pull, 0 = Coast / Disable */
    float target_voltage_v;  /**< Target motor armature voltage [V] */
    bool fail_safe_active;   /**< True if emergency shutdown / neutral action commanded */
} platform_actuator_cmd_t;

/* ── Platform Configuration Structure ────────────────────────────────── */
typedef struct {
    uint32_t sample_rate_hz; /**< Nominal control loop frequency [Hz] (Default: 50 Hz) */
    float dt_sec;            /**< Timestep period [s] (1.0f / sample_rate_hz = 0.02 s) */
    float max_motor_voltage; /**< Motor nominal supply rail voltage [V] (e.g. 12.0f or 24.0f) */
    float track_limit_m;     /**< Physical cart track travel boundary [m] (e.g. 2.4f) */
} platform_config_t;

/* ── Public Adapter Pipeline API ─────────────────────────────────────── */

/**
 * Returns default recommended platform configuration (50 Hz, 12.0V rail, 2.4m track).
 */
HYBRID_API platform_config_t platform_get_default_config(void);

/**
 * Stage 1: Converts raw physical sensor measurements into 5D normalized observation.
 *
 * @param sensors    Pointer to raw physical sensor measurements.
 * @param obs_out    Pointer to output observation struct.
 * @return           True on successful conversion, false on sensor fault or non-finite value.
 */
HYBRID_API bool platform_convert_sensors_to_observation(
    const platform_sensor_data_t* sensors,
    platform_observation_t* obs_out
);

/**
 * Stage 2: Executes pure hybrid controller step on 5D observation vector.
 *
 * @param obs        Pointer to valid 5D observation.
 * @return           Clamped action force in [-1.0, 1.0]. Returns 0.0f on invalid observation.
 */
HYBRID_API float platform_run_controller_step(
    const platform_observation_t* obs
);

/**
 * Stage 3: Converts normalized force action into physical actuator commands (PWM & Direction).
 *
 * @param action     Normalized force action in [-1.0, 1.0].
 * @param config     Platform electrical and mechanical configuration.
 * @param cmd_out    Pointer to output actuator command struct.
 * @return           True on successful conversion, false on invalid command.
 */
HYBRID_API bool platform_convert_action_to_actuator(
    float action,
    const platform_config_t* config,
    platform_actuator_cmd_t* cmd_out
);

/**
 * End-to-End Pipeline Step:
 * Sensor Acquisition -> Observation Conversion -> Controller Step -> Actuator Conversion.
 *
 * @param sensors    Input raw physical sensor data.
 * @param config     Platform configuration parameters.
 * @param cmd_out    Output actuator command.
 * @return           True if full pipeline executed normally; false if fail-safe activated.
 */
HYBRID_API bool platform_execute_pipeline_step(
    const platform_sensor_data_t* sensors,
    const platform_config_t* config,
    platform_actuator_cmd_t* cmd_out
);

#ifdef __cplusplus
}
#endif

#endif /* PLATFORM_ADAPTER_H */
