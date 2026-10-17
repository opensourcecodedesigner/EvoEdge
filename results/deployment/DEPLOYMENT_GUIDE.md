# EvoEdge — Embedded Controller Deployment & Hardware Integration Guide

This guide describes how to integrate the verified `HybridController` C component into an embedded microcontroller firmware project (e.g. STM32, ESP32, TI C2000, NXP S32K).

---

## 1. Architectural Architecture & Data Flow

The embedded deployment follows a decoupled, four-stage pipeline:

```
[Physical Sensors]
       │
       ▼
1. platform_convert_sensors_to_observation(&sensors, &obs)
       │  (Converts encoder ticks -> rad/m; validates finiteness)
       ▼
2. platform_run_controller_step(&obs)
       │  (Pure analytical dual-regime C controller: 7 params, <= 7 MACs)
       ▼
3. platform_convert_action_to_actuator(action, &config, &cmd)
       │  (Clamps force, calculates PWM duty % and H-bridge direction)
       ▼
[Hardware Actuator (PWM / Timer Registers)]
```

---

## 2. File Organization

To integrate into a firmware project, add the following two source files and headers to the build system (Makefile, CMake, ESP-IDF `CMakeLists.txt`, or Keil/IAR project):

```
firmware/
├── include/
│   ├── hybrid_controller.h    <-- Pure analytical controller interface
│   └── platform_adapter.h     <-- Sensor & actuator conversion interface
└── src/
    ├── hybrid_controller.c    <-- Zero-allocation controller core (< 32B stack)
    └── platform_adapter.c     <-- Platform-independent conversion functions
```

---

## 3. Minimal Firmware Integration Example

```c
#include "platform_adapter.h"

/* Platform configuration for 50 Hz loop */
static platform_config_t g_platform_cfg;

void control_loop_init(void) {
    g_platform_cfg = platform_get_default_config();
    /* Configure hardware timer interrupt for 50 Hz (20 ms period) */
    hardware_timer_init(50 /* Hz */);
}

/* 50 Hz Timer Interrupt Service Routine (ISR) */
void TIMx_IRQHandler(void) {
    platform_sensor_data_t sensors;
    platform_actuator_cmd_t cmd;

    /* 1. Acquire raw sensors from hardware quadrature encoders */
    sensors.cart_pos_m        = bsp_read_cart_encoder_meters();
    sensors.cart_vel_mps      = bsp_read_cart_velocity_mps();
    sensors.pole_angle_rad    = bsp_read_pole_encoder_radians();
    sensors.pole_ang_vel_rads = bsp_read_pole_velocity_rads();
    sensors.timestamp_us      = bsp_get_microseconds();
    sensors.sensor_fault      = bsp_check_sensor_fault();

    /* 2. Execute pipeline step */
    bool ok = platform_execute_pipeline_step(&sensors, &g_platform_cfg, &cmd);

    /* 3. Apply to physical actuator PWM registers */
    if (ok && !cmd.fail_safe_active) {
        bsp_set_motor_direction(cmd.motor_direction);
        bsp_set_motor_pwm_duty(cmd.pwm_duty_pct);
    } else {
        /* Fail-safe shutdown on error or non-finite reading */
        bsp_set_motor_direction(0);
        bsp_set_motor_pwm_duty(0.0f);
    }
}
```

---

## 4. Hardware Adaptation Checklist

When moving from desktop SIL validation to physical hardware, complete the following verification steps:

### 4.1 Sensor Scaling & Calibration
- [ ] **Pole Encoder Counts to Radians:** Ensure zero position corresponds to **perfect vertical upright** ($\theta = 0\text{ rad}$), downward hanging corresponds to $\theta = \pm \pi\text{ rad}$, and clockwise rotation follows standard right-hand rule.
- [ ] **Cart Track Encoder Scaling:** Calibrate optical encoder pulses per meter ($k_{\text{pulse/m}}$) such that track center is $x = 0.0\text{ m}$ and limits are $x = \pm 2.4\text{ m}$.
- [ ] **Velocity Differentiation / Filtering:** Implement low-pass or Kalman filtering on raw encoder tick differences to prevent numerical derivative quantization noise in $\dot{x}$ and $\dot{\theta}$.

### 4.2 Actuator & Electrical Interfacing
- [ ] **H-Bridge PWM Frequency:** Operate PWM timer at $\ge 20\text{ kHz}$ to avoid audible motor coil whining.
- [ ] **Motor Direction Logic:** Verify direction pin polarity ($+1 \implies$ push cart right, $-1 \implies$ push cart left).
- [ ] **Deadband Compensation:** If the motor has static friction ($V_{\text{deadband}} \approx 0.5\text{ V}$), adjust `cmd_out->pwm_duty_pct` in the board-level driver to overcome stiction without altering core controller gains.

### 4.3 Control Loop Timing & Execution Contract
- [ ] **Sample Rate:** Maintain strict $50\text{ Hz} \pm 0.5\text{ Hz}$ ($\Delta t = 20.0\text{ ms}$) loop timing using a hardware timer interrupt.
- [ ] **FPU Support:** Ensure compiler target flags enable hardware single-precision FPU (e.g., `-mfloat-abi=hard -mfpu=fpv4-sp-d16` on ARM Cortex-M4F).
- [ ] **Worst-Case Execution Time (WCET):** Measure actual on-chip execution cycles using DWT cycle counter (`DWT->CYCCNT`). On a 168 MHz Cortex-M4F, expected execution is $< 200\text{ cycles}$ ($< 1.2\,\mu\text{s}$), leaving $> 99.9\%$ CPU idle time.

### 4.4 Fail-Safe & Fault Handling
- [ ] **Watchdog Timer (WDT):** Enable independent hardware watchdog ($50\text{ ms}$ timeout).
- [ ] **Limit Switches:** Connect physical mechanical track end-stops to high-priority hardware interrupts to disable H-bridge gate drives instantly upon track overrun ($|x| > 2.4\text{ m}$).
- [ ] **Zero-Force Neutral Action:** Verify that upon sensor disconnection or non-finite floating-point error, `cmd.fail_safe_active` de-energizes the motor.
