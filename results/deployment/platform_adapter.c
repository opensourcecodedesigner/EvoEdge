/* ========================================================================
 * EvoEdge Deployment Artifact: Embedded Platform Adapter Source
 * ======================================================================== */

#include "platform_adapter.h"

HYBRID_API platform_config_t platform_get_default_config(void) {
    platform_config_t cfg;
    cfg.sample_rate_hz   = 50U;     /* 50 Hz control loop frequency */
    cfg.dt_sec           = 0.02f;   /* 20.0 ms period */
    cfg.max_motor_voltage= 12.0f;   /* 12V DC H-bridge supply */
    cfg.track_limit_m    = 2.4f;    /* +-2.4m physical track limit */
    return cfg;
}

HYBRID_API bool platform_convert_sensors_to_observation(
    const platform_sensor_data_t* sensors,
    platform_observation_t* obs_out
) {
    if (sensors == NULL || obs_out == NULL) {
        return false;
    }

    /* Check hardware fault flag */
    if (sensors->sensor_fault) {
        obs_out->is_valid = false;
        for (int i = 0; i < 5; i++) {
            obs_out->values[i] = 0.0f;
        }
        return false;
    }

    /* Validate numerical finiteness of all sensor channels */
    if (!isfinite(sensors->cart_pos_m) ||
        !isfinite(sensors->cart_vel_mps) ||
        !isfinite(sensors->pole_angle_rad) ||
        !isfinite(sensors->pole_ang_vel_rads)) {
        obs_out->is_valid = false;
        for (int i = 0; i < 5; i++) {
            obs_out->values[i] = 0.0f;
        }
        return false;
    }

    /* Convert angle theta into cos(theta) and sin(theta) */
    const float theta = sensors->pole_angle_rad;
    obs_out->values[0] = sensors->cart_pos_m;
    obs_out->values[1] = sensors->cart_vel_mps;
    obs_out->values[2] = cosf(theta);
    obs_out->values[3] = sinf(theta);
    obs_out->values[4] = sensors->pole_ang_vel_rads;
    obs_out->is_valid  = true;

    return true;
}

HYBRID_API float platform_run_controller_step(
    const platform_observation_t* obs
) {
    if (obs == NULL || !obs->is_valid) {
        return 0.0f;
    }

    return hybrid_compute_action(obs->values);
}

HYBRID_API bool platform_convert_action_to_actuator(
    float action,
    const platform_config_t* config,
    platform_actuator_cmd_t* cmd_out
) {
    if (cmd_out == NULL) {
        return false;
    }

    const float max_voltage = (config != NULL && config->max_motor_voltage > 0.0f)
                              ? config->max_motor_voltage
                              : 12.0f;

    /* Fail-safe clamp check */
    if (!isfinite(action)) {
        cmd_out->normalized_force = 0.0f;
        cmd_out->pwm_duty_pct     = 0.0f;
        cmd_out->motor_direction  = 0;
        cmd_out->target_voltage_v = 0.0f;
        cmd_out->fail_safe_active = true;
        return false;
    }

    /* Clamp continuous force to normalized action limits [-1.0, 1.0] */
    float clamped_u = action;
    if (clamped_u > 1.0f)  clamped_u = 1.0f;
    if (clamped_u < -1.0f) clamped_u = -1.0f;

    cmd_out->normalized_force = clamped_u;
    cmd_out->fail_safe_active = false;

    /* Compute directional H-bridge actuation */
    const float abs_u = fabsf(clamped_u);
    cmd_out->pwm_duty_pct = abs_u * 100.0f;
    cmd_out->target_voltage_v = clamped_u * max_voltage;

    /* Direction deadband near zero (0.1% duty) */
    if (abs_u < 0.001f) {
        cmd_out->motor_direction = 0; /* Coast / Off */
    } else if (clamped_u > 0.0f) {
        cmd_out->motor_direction = 1; /* Forward / Push Right */
    } else {
        cmd_out->motor_direction = -1; /* Reverse / Push Left */
    }

    return true;
}

HYBRID_API bool platform_execute_pipeline_step(
    const platform_sensor_data_t* sensors,
    const platform_config_t* config,
    platform_actuator_cmd_t* cmd_out
) {
    if (cmd_out == NULL) {
        return false;
    }

    platform_observation_t obs;

    /* Stage 1: Convert raw sensor readings to 5D observation */
    if (!platform_convert_sensors_to_observation(sensors, &obs)) {
        /* Sensor fault or non-finite reading -> trigger immediate fail-safe */
        cmd_out->normalized_force = 0.0f;
        cmd_out->pwm_duty_pct     = 0.0f;
        cmd_out->motor_direction  = 0;
        cmd_out->target_voltage_v = 0.0f;
        cmd_out->fail_safe_active = true;
        return false;
    }

    /* Stage 2: Pure controller inference */
    const float action = platform_run_controller_step(&obs);

    /* Stage 3: Convert action to actuator PWM & direction command */
    return platform_convert_action_to_actuator(action, config, cmd_out);
}
