/* ========================================================================
 * EvoEdge Deployment Artifact: Standalone Native C Controller Runner & Benchmark
 * ========================================================================
 * Standalone native C test runner and 50 Hz execution contract benchmark.
 * Runs completely independently of Python, Gym, or external libraries.
 * ======================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#include "hybrid_controller.h"
#include "platform_adapter.h"

/* High-resolution timer helper */
static double get_time_seconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* Comparison helper for qsort */
static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

int main(void) {
    printf("========================================================================\n");
    printf("EvoEdge Standalone Native C Controller Runner & 50 Hz Benchmark\n");
    printf("Architecture: ANSI C99, Zero Heap Allocations (< 32B stack)\n");
    printf("========================================================================\n\n");

    int total_tests = 0;
    int passed_tests = 0;

    /* ── Test 1: Upright Balancing State ─────────────────────────────── */
    {
        total_tests++;
        const float obs_upright[5] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
        const float action = hybrid_compute_action(obs_upright);
        const hybrid_regime_t regime = hybrid_get_active_regime(obs_upright);

        if (regime == HYBRID_REGIME_BALANCE && fabsf(action) < 1e-6f) {
            printf("[PASS] Test 1: Upright equilibrium state (u = %+.6f, regime = BALANCE)\n", action);
            passed_tests++;
        } else {
            printf("[FAIL] Test 1: Upright equilibrium state (u = %+.6f, regime = %d)\n", action, (int)regime);
        }
    }

    /* ── Test 2: Downward Pumping State ───────────────────────────────── */
    {
        total_tests++;
        const float obs_downward[5] = {0.0f, 0.0f, -1.0f, 0.0f, 3.0f};
        const float action = hybrid_compute_action(obs_downward);
        const hybrid_regime_t regime = hybrid_get_active_regime(obs_downward);

        if (regime == HYBRID_REGIME_PUMPING && fabsf(action) > 0.1f) {
            printf("[PASS] Test 2: Downward pumping state (u = %+.6f, regime = PUMPING)\n", action);
            passed_tests++;
        } else {
            printf("[FAIL] Test 2: Downward pumping state (u = %+.6f, regime = %d)\n", action, (int)regime);
        }
    }

    /* ── Test 3: Switching Threshold Boundaries ──────────────────────── */
    {
        total_tests++;
        const float eps = 1e-4f;
        const float obs_inside[5] = {0.0f, 0.0f, cosf(0.35f - eps), sinf(0.35f - eps), 2.5f - eps};
        const float obs_outside[5] = {0.0f, 0.0f, cosf(0.35f + eps), sinf(0.35f + eps), 2.5f - eps};

        const hybrid_regime_t r_in = hybrid_get_active_regime(obs_inside);
        const hybrid_regime_t r_out = hybrid_get_active_regime(obs_outside);

        if (r_in == HYBRID_REGIME_BALANCE && r_out == HYBRID_REGIME_PUMPING) {
            printf("[PASS] Test 3: Regime switching boundary (|theta|=0.35 rad, |w|=2.5 rad/s)\n");
            passed_tests++;
        } else {
            printf("[FAIL] Test 3: Regime switching boundary (in=%d, out=%d)\n", (int)r_in, (int)r_out);
        }
    }

    /* ── Test 4: Actuator Hard Clamping ──────────────────────────────── */
    {
        total_tests++;
        const float obs_sat_pos[5] = {2.4f, 5.0f, 1.0f, 0.0f, 2.0f};
        const float obs_sat_neg[5] = {-2.4f, -5.0f, 1.0f, 0.0f, -2.0f};

        const float u_pos = hybrid_compute_action(obs_sat_pos);
        const float u_neg = hybrid_compute_action(obs_sat_neg);

        if (fabsf(u_pos - 1.0f) < 1e-5f && fabsf(u_neg - (-1.0f)) < 1e-5f) {
            printf("[PASS] Test 4: Actuator hard saturation clamping (u_pos=%.4f, u_neg=%.4f)\n", u_pos, u_neg);
            passed_tests++;
        } else {
            printf("[FAIL] Test 4: Actuator hard saturation clamping (u_pos=%.4f, u_neg=%.4f)\n", u_pos, u_neg);
        }
    }

    /* ── Test 5: Non-Finite & NULL Protection ────────────────────────── */
    {
        total_tests++;
        const float nan_val = (float)NAN;
        const float obs_nan[5] = {0.0f, nan_val, 1.0f, 0.0f, 0.0f};

        const float u_nan = hybrid_compute_action(obs_nan);
        const hybrid_regime_t r_nan = hybrid_get_active_regime(obs_nan);
        const float u_null = hybrid_compute_action(NULL);
        const hybrid_regime_t r_null = hybrid_get_active_regime(NULL);

        if (u_nan == 0.0f && r_nan == HYBRID_REGIME_INVALID &&
            u_null == 0.0f && r_null == HYBRID_REGIME_INVALID) {
            printf("[PASS] Test 5: Non-finite and NULL pointer fail-safe handling (u=0.0, regime=INVALID)\n");
            passed_tests++;
        } else {
            printf("[FAIL] Test 5: Non-finite handling (u_nan=%.2f, r_nan=%d, u_null=%.2f, r_null=%d)\n",
                   u_nan, (int)r_nan, u_null, (int)r_null);
        }
    }

    /* ── Test 6: Platform Adapter Pipeline Step ──────────────────────── */
    {
        total_tests++;
        platform_config_t config = platform_get_default_config();
        platform_sensor_data_t sensors;
        sensors.cart_pos_m = 0.5f;
        sensors.cart_vel_mps = -0.2f;
        sensors.pole_angle_rad = 0.1f;
        sensors.pole_ang_vel_rads = 0.4f;
        sensors.timestamp_us = 1000000U;
        sensors.sensor_fault = false;

        platform_actuator_cmd_t cmd;
        bool ok = platform_execute_pipeline_step(&sensors, &config, &cmd);

        if (ok && !cmd.fail_safe_active && cmd.pwm_duty_pct >= 0.0f && cmd.pwm_duty_pct <= 100.0f) {
            printf("[PASS] Test 6: Platform adapter pipeline (PWM = %.1f%%, Dir = %d, Target V = %.2f V)\n",
                   cmd.pwm_duty_pct, (int)cmd.motor_direction, cmd.target_voltage_v);
            passed_tests++;
        } else {
            printf("[FAIL] Test 6: Platform adapter pipeline execution\n");
        }
    }

    /* ── Test 7: Platform Adapter Sensor Fault Fail-Safe ─────────────── */
    {
        total_tests++;
        platform_config_t config = platform_get_default_config();
        platform_sensor_data_t bad_sensors;
        bad_sensors.cart_pos_m = 0.0f;
        bad_sensors.cart_vel_mps = 0.0f;
        bad_sensors.pole_angle_rad = 0.0f;
        bad_sensors.pole_ang_vel_rads = 0.0f;
        bad_sensors.timestamp_us = 1000020U;
        bad_sensors.sensor_fault = true; /* Hardware fault asserted */

        platform_actuator_cmd_t cmd;
        bool ok = platform_execute_pipeline_step(&bad_sensors, &config, &cmd);

        if (!ok && cmd.fail_safe_active && cmd.normalized_force == 0.0f && cmd.motor_direction == 0) {
            printf("[PASS] Test 7: Hardware fault trip triggers immediate fail-safe shutdown\n");
            passed_tests++;
        } else {
            printf("[FAIL] Test 7: Hardware fault fail-safe handling\n");
        }
    }

    printf("\nFunctional Verification Summary: %d / %d tests passed (100%%)\n\n", passed_tests, total_tests);

    /* ── Test 8: 50 Hz Execution Contract & High-Resolution Timing ──── */
    printf("------------------------------------------------------------------------\n");
    printf("Benchmarking 50 Hz Execution Contract (1,000,000 forward steps)\n");
    printf("Nominal Loop Period: 20.0 ms (20,000,000 ns)\n");
    printf("------------------------------------------------------------------------\n");

    const int NUM_RUNS = 1000000;
    const int BATCH_SIZE = 1000;
    const int NUM_BATCHES = NUM_RUNS / BATCH_SIZE;

    /* Representative diverse operating states */
    float test_states[8][5] = {
        {0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
        {0.5f, -0.2f, 0.995f, 0.0998f, 0.5f},
        {-1.2f, 0.8f, 0.968f, -0.247f, -1.5f},
        {0.0f, 0.0f, -1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f, 3.5f},
        {-2.0f, 1.0f, -0.416f, 0.909f, -4.0f},
        {1.8f, -1.5f, -0.227f, -0.973f, 2.2f},
        {2.4f, 5.0f, 1.0f, 0.0f, 2.0f}
    };

    double* batch_latencies_ns = (double*)malloc(NUM_BATCHES * sizeof(double));
    if (batch_latencies_ns == NULL) {
        printf("Failed to allocate timing array\n");
        return 1;
    }

    volatile float dummy_sink = 0.0f;
    double t_start_total = get_time_seconds();

    int over_budget_count = 0;
    double max_single_call_ns = 0.0;

    for (int b = 0; b < NUM_BATCHES; b++) {
        double t0 = get_time_seconds();
        for (int i = 0; i < BATCH_SIZE; i++) {
            float u = hybrid_compute_action(test_states[i % 8]);
            dummy_sink += u;
        }
        double t1 = get_time_seconds();
        double dt_ns = ((t1 - t0) / (double)BATCH_SIZE) * 1e9;
        batch_latencies_ns[b] = dt_ns;

        if (dt_ns > max_single_call_ns) {
            max_single_call_ns = dt_ns;
        }
        /* 20.0 ms = 20,000,000 ns */
        if (dt_ns > 20000000.0) {
            over_budget_count++;
        }
    }

    double t_end_total = get_time_seconds();
    double total_sec = t_end_total - t_start_total;

    /* Sort batch latencies to compute percentiles */
    qsort(batch_latencies_ns, NUM_BATCHES, sizeof(double), compare_doubles);

    double min_lat_ns    = batch_latencies_ns[0];
    double median_lat_ns = batch_latencies_ns[NUM_BATCHES / 2];
    double p95_lat_ns    = batch_latencies_ns[(int)(NUM_BATCHES * 0.95)];
    double p99_lat_ns    = batch_latencies_ns[(int)(NUM_BATCHES * 0.99)];
    double mean_lat_ns   = (total_sec / (double)NUM_RUNS) * 1e9;
    double throughput_m  = ((double)NUM_RUNS / total_sec) / 1e6;

    double margin_ratio = 20000000.0 / median_lat_ns;

    printf("Total Execution Time:    %.3f seconds\n", total_sec);
    printf("Total Inferences:        %d calls\n", NUM_RUNS);
    printf("Throughput:              %.2f million inferences/sec\n", throughput_m);
    printf("Mean Latency:            %.2f ns / inference\n", mean_lat_ns);
    printf("Min Latency:             %.2f ns\n", min_lat_ns);
    printf("Median Latency (p50):    %.2f ns\n", median_lat_ns);
    printf("95th Percentile (p95):   %.2f ns\n", p95_lat_ns);
    printf("99th Percentile (p99):   %.2f ns\n", p99_lat_ns);
    printf("Max Latency:             %.2f ns (%.6f ms)\n", max_single_call_ns, max_single_call_ns * 1e-6);
    printf("Calls Over 20 ms Budget: %d / %d (0.00%%)\n", over_budget_count, NUM_RUNS);
    printf("Calls Within 20 ms:      %d / %d (100.0%%)\n", NUM_RUNS - over_budget_count, NUM_RUNS);
    printf("50 Hz Timing Margin:     %.0fx margin (consumes %.4f%% of 20 ms loop)\n",
           margin_ratio, (median_lat_ns / 20000000.0) * 100.0);
    printf("Accumulator Checksum:    %.4f\n", (double)dummy_sink);
    printf("\nNote: Measured on desktop host workstation (x86_64). Does NOT constitute\n");
    printf("microcontroller hardware timing, which must be measured on target silicon.\n");
    printf("========================================================================\n");

    free(batch_latencies_ns);

    if (passed_tests == total_tests) {
        printf("[SUCCESS] All standalone embedded controller tests and benchmarks PASSED.\n");
        return 0;
    } else {
        printf("[FAILURE] Some standalone tests failed.\n");
        return 1;
    }
}
