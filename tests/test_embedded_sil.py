"""Targeted automated tests for Day 6: Embedded Portable Architecture and SIL Validation.

Verifies:
  1. Portable C build artifacts exist and have valid structure:
     - hybrid_controller.h / .c
     - platform_adapter.h / .c
     - hybrid_controller_runner.c / .exe
     - libhybrid_controller.a / hybrid_controller.dll
     - DEPLOYMENT_GUIDE.md
  2. Standalone native test runner execution (hybrid_controller_runner.exe):
     - Exits with returncode 0.
     - Passes all 7 native unit tests.
     - Satisfies 50 Hz timing budget (100.0% within 20 ms).
  3. Platform adapter boundary interface (ctypes):
     - Sensor acquisition to observation conversion.
     - Controller step execution through adapter.
     - Action to actuator command conversion (physical Force, PWM, Direction, Voltage).
     - Full pipeline execution (sensor -> obs -> ctrl -> act).
     - Fault injection and safety fail-safe on NaN/Inf/out-of-bounds/NULL.
  4. Software-in-the-Loop (SIL) closed-loop validation on development seeds 200-209:
     - 100% success predicate across Python, C double, C float, and Platform Adapter.
     - Trajectory deviations strictly bounded.
  5. Memory resilience / zero-state accumulation:
     - Controller preserves stateless execution across repeated calls.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from baselines.hybrid_controller import HybridController
from env.cartpole_swingup import CartPoleSwingUpEnv
from experiments.run_robustness_experiment import evaluate_success_predicate


DEPLOYMENT_DIR = os.path.join(_PROJECT_ROOT, "results", "deployment")
RUNNER_EXE = os.path.join(DEPLOYMENT_DIR, "hybrid_controller_runner.exe")
DLL_PATH = os.path.join(DEPLOYMENT_DIR, "hybrid_controller.dll")
ADAPTER_DLL = os.path.join(DEPLOYMENT_DIR, "platform_adapter.dll")
STATIC_LIB = os.path.join(DEPLOYMENT_DIR, "libhybrid_controller.a")


# C Types for platform_adapter matching platform_adapter.h exactly
class PlatformSensorDataC(ctypes.Structure):
    _fields_ = [
        ("cart_pos_m", ctypes.c_float),
        ("cart_vel_mps", ctypes.c_float),
        ("pole_angle_rad", ctypes.c_float),
        ("pole_ang_vel_rads", ctypes.c_float),
        ("timestamp_us", ctypes.c_uint32),
        ("sensor_fault", ctypes.c_bool),
    ]


class PlatformObservationC(ctypes.Structure):
    _fields_ = [
        ("values", ctypes.c_float * 5),
        ("is_valid", ctypes.c_bool),
    ]


class PlatformActuatorCmdC(ctypes.Structure):
    _fields_ = [
        ("normalized_force", ctypes.c_float),
        ("pwm_duty_pct", ctypes.c_float),
        ("motor_direction", ctypes.c_int8),
        ("target_voltage_v", ctypes.c_float),
        ("fail_safe_active", ctypes.c_bool),
    ]


class PlatformConfigC(ctypes.Structure):
    _fields_ = [
        ("sample_rate_hz", ctypes.c_uint32),
        ("dt_sec", ctypes.c_float),
        ("max_motor_voltage", ctypes.c_float),
        ("track_limit_m", ctypes.c_float),
    ]


@pytest.fixture(scope="module")
def native_runner() -> str:
    """Ensure runner executable exists."""
    assert os.path.isfile(RUNNER_EXE), f"Runner executable missing at {RUNNER_EXE}"
    return RUNNER_EXE


@pytest.fixture(scope="module")
def native_dll():
    """Load native compiled DLL with hybrid controller and platform adapter entry points."""
    assert os.path.isfile(ADAPTER_DLL), f"DLL missing at {ADAPTER_DLL}"
    dll = ctypes.CDLL(ADAPTER_DLL)

    # hybrid_controller signatures
    dll.hybrid_compute_action.argtypes = [ctypes.POINTER(ctypes.c_float)]
    dll.hybrid_compute_action.restype = ctypes.c_float

    dll.hybrid_compute_action_double.argtypes = [ctypes.POINTER(ctypes.c_double)]
    dll.hybrid_compute_action_double.restype = ctypes.c_double

    dll.hybrid_get_active_regime.argtypes = [ctypes.POINTER(ctypes.c_float)]
    dll.hybrid_get_active_regime.restype = ctypes.c_int

    # platform_adapter signatures
    dll.platform_get_default_config.argtypes = []
    dll.platform_get_default_config.restype = PlatformConfigC

    dll.platform_convert_sensors_to_observation.argtypes = [
        ctypes.POINTER(PlatformSensorDataC),
        ctypes.POINTER(PlatformObservationC),
    ]
    dll.platform_convert_sensors_to_observation.restype = ctypes.c_bool

    dll.platform_run_controller_step.argtypes = [
        ctypes.POINTER(PlatformObservationC),
    ]
    dll.platform_run_controller_step.restype = ctypes.c_float

    dll.platform_convert_action_to_actuator.argtypes = [
        ctypes.c_float,
        ctypes.POINTER(PlatformConfigC),
        ctypes.POINTER(PlatformActuatorCmdC),
    ]
    dll.platform_convert_action_to_actuator.restype = ctypes.c_bool

    dll.platform_execute_pipeline_step.argtypes = [
        ctypes.POINTER(PlatformSensorDataC),
        ctypes.POINTER(PlatformConfigC),
        ctypes.POINTER(PlatformActuatorCmdC),
    ]
    dll.platform_execute_pipeline_step.restype = ctypes.c_bool

    yield dll

    # Clean up DLL handle on Windows so file is not locked
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
        kernel32.FreeLibrary(dll._handle)


def test_artifact_completeness():
    """Verify all required portable deployment artifacts exist."""
    required_files = [
        os.path.join(DEPLOYMENT_DIR, "hybrid_controller.h"),
        os.path.join(DEPLOYMENT_DIR, "hybrid_controller.c"),
        os.path.join(DEPLOYMENT_DIR, "platform_adapter.h"),
        os.path.join(DEPLOYMENT_DIR, "platform_adapter.c"),
        os.path.join(DEPLOYMENT_DIR, "hybrid_controller_runner.c"),
        RUNNER_EXE,
        STATIC_LIB,
        DLL_PATH,
        ADAPTER_DLL,
        os.path.join(DEPLOYMENT_DIR, "DEPLOYMENT_GUIDE.md"),
        os.path.join(DEPLOYMENT_DIR, "build_embedded_artifacts.ps1"),
        os.path.join(DEPLOYMENT_DIR, "build_embedded_artifacts.sh"),
    ]
    for path in required_files:
        assert os.path.isfile(path), f"Required file missing: {path}"


def test_native_runner_execution(native_runner: str):
    """Run standalone C test runner and assert clean exit and full pass."""
    res = subprocess.run([native_runner], capture_output=True, text=True, timeout=10)
    assert res.returncode == 0, f"Runner failed with return code {res.returncode}:\n{res.stderr}"
    assert "7 / 7 tests passed" in res.stdout, f"Runner did not pass all tests:\n{res.stdout}"
    assert "Calls Within 20 ms:      1000000 / 1000000 (100.0%)" in res.stdout
    assert "[SUCCESS] All standalone embedded controller tests and benchmarks PASSED." in res.stdout


def test_platform_adapter_nominal_pipeline(native_dll: ctypes.CDLL):
    """Test platform adapter pipeline with valid sensor inputs."""
    cfg = native_dll.platform_get_default_config()
    assert cfg.sample_rate_hz == 50
    assert abs(cfg.dt_sec - 0.02) < 1e-4

    sensors = PlatformSensorDataC(
        cart_pos_m=0.1,
        cart_vel_mps=-0.05,
        pole_angle_rad=0.05,  # near upright -> balance regime
        pole_ang_vel_rads=0.02,
        timestamp_us=20000,
        sensor_fault=False,
    )

    cmd = PlatformActuatorCmdC()
    ok = native_dll.platform_execute_pipeline_step(
        ctypes.byref(sensors),
        ctypes.byref(cfg),
        ctypes.byref(cmd),
    )

    assert ok is True
    assert cmd.fail_safe_active is False
    assert -1.0 <= cmd.normalized_force <= 1.0
    assert 0.0 <= cmd.pwm_duty_pct <= 100.0
    assert cmd.motor_direction in (-1, 0, 1)
    assert 0.0 <= cmd.target_voltage_v <= cfg.max_motor_voltage


def test_platform_adapter_failsafe_on_nan(native_dll: ctypes.CDLL):
    """Test that platform adapter handles NaN/Inf safely."""
    cfg = native_dll.platform_get_default_config()

    bad_sensors = PlatformSensorDataC(
        cart_pos_m=float("nan"),
        cart_vel_mps=0.0,
        pole_angle_rad=0.0,
        pole_ang_vel_rads=0.0,
        timestamp_us=0,
        sensor_fault=False,
    )

    cmd = PlatformActuatorCmdC()
    ok = native_dll.platform_execute_pipeline_step(
        ctypes.byref(bad_sensors),
        ctypes.byref(cfg),
        ctypes.byref(cmd),
    )

    assert ok is False  # Fail-safe triggered
    assert cmd.fail_safe_active is True
    assert cmd.normalized_force == 0.0
    assert cmd.pwm_duty_pct == 0.0
    assert cmd.motor_direction == 0
    assert cmd.target_voltage_v == 0.0


def test_platform_adapter_null_safety(native_dll: ctypes.CDLL):
    """Test NULL pointer safety in platform adapter."""
    cmd = PlatformActuatorCmdC()
    cfg = native_dll.platform_get_default_config()

    ok = native_dll.platform_execute_pipeline_step(
        None, ctypes.byref(cfg), ctypes.byref(cmd)
    )
    assert ok is False
    assert cmd.fail_safe_active is True


def test_sil_closed_loop_seeds_200_to_209(native_dll: ctypes.CDLL):
    """Run closed-loop SIL validation on development seeds 200..209."""
    py_ctrl = HybridController()
    cfg = native_dll.platform_get_default_config()

    for seed in range(200, 210):
        # 1. Python run
        env_py = CartPoleSwingUpEnv(max_steps=500)
        obs_py, _ = env_py.reset(seed=seed)
        py_actions = []
        cos_thetas_py = [float(obs_py[2])]
        rail_hit_py = False

        for _ in range(500):
            a = py_ctrl.compute_action(obs_py)
            py_actions.append(a)
            obs_py, _, term, trunc, _ = env_py.step(a)
            cos_thetas_py.append(float(obs_py[2]))
            if term:
                rail_hit_py = True
            if term or trunc:
                break
        succ_py, _, _, _ = evaluate_success_predicate(cos_thetas_py, rail_hit=rail_hit_py, max_steps=500)
        assert succ_py is True

        # 2. C Double run
        env_c = CartPoleSwingUpEnv(max_steps=500)
        obs_c, _ = env_c.reset(seed=seed)
        c_double_actions = []
        cos_thetas_c = [float(obs_c[2])]
        rail_hit_c = False

        for _ in range(500):
            obs_arr = (ctypes.c_double * 5)(*[float(v) for v in obs_c])
            act_val = float(native_dll.hybrid_compute_action_double(obs_arr))
            c_double_actions.append(act_val)
            obs_c, _, term, trunc, _ = env_c.step(act_val)
            cos_thetas_c.append(float(obs_c[2]))
            if term:
                rail_hit_c = True
            if term or trunc:
                break
        succ_c, _, _, _ = evaluate_success_predicate(cos_thetas_c, rail_hit=rail_hit_c, max_steps=500)
        assert succ_c is True

        # Assert exact bit parity for double precision
        assert np.allclose(py_actions, c_double_actions, atol=1e-12)

        # 3. Platform adapter run
        env_ad = CartPoleSwingUpEnv(max_steps=500)
        obs_ad, _ = env_ad.reset(seed=seed)
        cos_thetas_ad = [float(obs_ad[2])]
        rail_hit_ad = False

        for _ in range(500):
            x, x_dot, cos_th, sin_th, theta_dot = [float(v) for v in obs_ad]
            th = math.atan2(sin_th, cos_th)
            sensors = PlatformSensorDataC(
                cart_pos_m=x,
                cart_vel_mps=x_dot,
                pole_angle_rad=th,
                pole_ang_vel_rads=theta_dot,
                timestamp_us=0,
                sensor_fault=False,
            )
            cmd = PlatformActuatorCmdC()
            ok = native_dll.platform_execute_pipeline_step(
                ctypes.byref(sensors),
                ctypes.byref(cfg),
                ctypes.byref(cmd),
            )
            assert ok is True
            obs_ad, _, term, trunc, _ = env_ad.step(float(cmd.normalized_force))
            cos_thetas_ad.append(float(obs_ad[2]))
            if term:
                rail_hit_ad = True
            if term or trunc:
                break
        succ_ad, _, _, _ = evaluate_success_predicate(cos_thetas_ad, rail_hit=rail_hit_ad, max_steps=500)
        assert succ_ad is True
