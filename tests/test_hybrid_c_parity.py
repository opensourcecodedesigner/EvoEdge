"""Targeted automated tests for HybridController C export and numerical parity.

Verifies:
  1. C export files exist and define the expected architectural constants.
  2. C shared library compilation succeeds using GCC.
  3. C ctypes interface loads all public entry points.
  4. Numerical parity between compiled C and Python reference implementation:
     - Exact double-precision parity (tol <= 1e-12).
     - Single-precision float parity on continuous states (tol <= 1e-5).
  5. Critical boundary behavior:
     - Regime switching thresholds (|theta| = 0.35 rad, |theta_dot| = 2.5 rad/s).
     - Saturation clamping to [-1.0, 1.0].
     - Sign-convention handling at zero crossing.
  6. Robustness against non-finite inputs and NULL pointers:
     - Fail-safe zero response on NaN or Inf.
     - NULL pointer detection and invalid regime classification.
  7. Closed-loop simulation parity on approved development seeds (seeds 200..209).
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
from utils.export_hybrid_c import export_hybrid_controller_c


DEPLOYMENT_DIR = os.path.join(_PROJECT_ROOT, "results", "deployment")
HEADER_PATH = os.path.join(DEPLOYMENT_DIR, "hybrid_controller.h")
SOURCE_PATH = os.path.join(DEPLOYMENT_DIR, "hybrid_controller.c")
DLL_PATH = os.path.join(DEPLOYMENT_DIR, "hybrid_controller.dll")


@pytest.fixture(scope="module")
def compiled_dll() -> str:
    """Ensure C files are exported and compiled to DLL before running tests."""
    export_hybrid_controller_c(DEPLOYMENT_DIR)

    # Compile using gcc
    compile_cmd = [
        "gcc",
        "-O3",
        "-shared",
        "-fPIC",
        "-o",
        DLL_PATH,
        SOURCE_PATH,
        f"-I{DEPLOYMENT_DIR}",
        "-lm",
    ]
    res = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert res.returncode == 0, f"Compilation failed: {res.stderr}"
    assert os.path.exists(DLL_PATH), "DLL artifact was not created"
    return DLL_PATH


@pytest.fixture(scope="module")
def c_controller(compiled_dll):
    """Load the compiled DLL and set function prototypes."""
    dll = ctypes.CDLL(compiled_dll)

    # float hybrid_compute_action(const float obs[5]);
    dll.hybrid_compute_action.argtypes = [ctypes.POINTER(ctypes.c_float)]
    dll.hybrid_compute_action.restype = ctypes.c_float

    # double hybrid_compute_action_double(const double obs[5]);
    dll.hybrid_compute_action_double.argtypes = [ctypes.POINTER(ctypes.c_double)]
    dll.hybrid_compute_action_double.restype = ctypes.c_double

    # float hybrid_compute_action_explicit(float, float, float, float, float);
    dll.hybrid_compute_action_explicit.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    dll.hybrid_compute_action_explicit.restype = ctypes.c_float

    # hybrid_regime_t hybrid_get_active_regime(const float obs[5]);
    dll.hybrid_get_active_regime.argtypes = [ctypes.POINTER(ctypes.c_float)]
    dll.hybrid_get_active_regime.restype = ctypes.c_int

    return dll


# ── 1. Export & Compilation Tests ─────────────────────────────────────────────


def test_export_artifacts_exist():
    """Verify that hybrid_controller.h and hybrid_controller.c are generated."""
    header_path, source_path = export_hybrid_controller_c(DEPLOYMENT_DIR)
    assert os.path.exists(header_path)
    assert os.path.exists(source_path)

    with open(header_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "HYBRID_NUM_PARAMS                7" in content
    assert "HYBRID_MAX_MACS                  7" in content
    assert "HYBRID_THETA_THRESHOLD_RAD" in content
    assert "HYBRID_THETA_DOT_THRESHOLD_RAD_S" in content


def test_compilation_and_symbols(c_controller):
    """Verify that the shared library compiles cleanly and exports all required functions."""
    assert hasattr(c_controller, "hybrid_compute_action")
    assert hasattr(c_controller, "hybrid_compute_action_double")
    assert hasattr(c_controller, "hybrid_compute_action_explicit")
    assert hasattr(c_controller, "hybrid_get_active_regime")


# ── 2. Double-Precision Parity Tests (Bit-Accurate Reference) ────────────────


def test_double_precision_numerical_parity(c_controller):
    """Verify double-precision C action matches Python reference within 1e-12."""
    py_ctrl = HybridController()

    test_states = [
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.5, -0.2, math.cos(0.1), math.sin(0.1), 0.5],
        [-1.2, 0.8, math.cos(-0.25), math.sin(-0.25), -1.5],
        [0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 3.5],
        [-2.0, 1.0, math.cos(2.0), math.sin(2.0), -4.0],
        [1.8, -1.5, math.cos(-1.8), math.sin(-1.8), 2.2],
    ]

    for state in test_states:
        py_u = py_ctrl.compute_action(state)
        c_arr = (ctypes.c_double * 5)(*[float(v) for v in state])
        c_u = float(c_controller.hybrid_compute_action_double(c_arr))

        assert abs(py_u - c_u) <= 1e-12, f"Discrepancy at state {state}: Py={py_u}, C={c_u}"


# ── 3. Single-Precision Float Parity on Continuous States ─────────────────────


def test_single_precision_numerical_parity(c_controller):
    """Verify single-precision float action matches Python reference within 1e-5 on continuous states."""
    py_ctrl = HybridController()

    rng = np.random.default_rng(42)
    for _ in range(100):
        x = float(rng.uniform(-2.0, 2.0))
        x_dot = float(rng.uniform(-4.0, 4.0))
        th = float(rng.uniform(-math.pi, math.pi))
        th_dot = float(rng.uniform(-8.0, 8.0))
        cos_th = math.cos(th)
        sin_th = math.sin(th)
        obs = [x, x_dot, cos_th, sin_th, th_dot]

        # Skip near-zero energy knife-edge in single precision test (examined separately)
        E = (2.0 / 3.0) * 0.1 * 0.25 * (th_dot ** 2) + (0.1 * 9.8 * 0.5) * (cos_th - 1.0)
        if abs(E) < 1e-3:
            continue

        py_u = py_ctrl.compute_action(obs)
        c_arr = (ctypes.c_float * 5)(*[float(v) for v in obs])
        c_u = float(c_controller.hybrid_compute_action(c_arr))

        assert abs(py_u - c_u) <= 1e-5, f"Discrepancy at obs {obs}: Py={py_u}, C={c_u}"


# ── 4. Boundary & Regime Switching Tests ──────────────────────────────────────


def test_regime_switching_boundary(c_controller):
    """Verify switching behavior at |theta| = 0.35 rad and |theta_dot| = 2.5 rad/s."""
    eps = 1e-4

    # Just inside balance: regime should be 0 (HYBRID_REGIME_BALANCE)
    obs_inside = [0.0, 0.0, math.cos(0.35 - eps), math.sin(0.35 - eps), 2.5 - eps]
    arr_inside = (ctypes.c_float * 5)(*obs_inside)
    assert c_controller.hybrid_get_active_regime(arr_inside) == 0

    # Just outside angle threshold: regime should be 1 (HYBRID_REGIME_PUMPING)
    obs_out_th = [0.0, 0.0, math.cos(0.35 + eps), math.sin(0.35 + eps), 2.5 - eps]
    arr_out_th = (ctypes.c_float * 5)(*obs_out_th)
    assert c_controller.hybrid_get_active_regime(arr_out_th) == 1

    # Just outside angular velocity threshold: regime should be 1 (HYBRID_REGIME_PUMPING)
    obs_out_w = [0.0, 0.0, math.cos(0.35 - eps), math.sin(0.35 - eps), 2.5 + eps]
    arr_out_w = (ctypes.c_float * 5)(*obs_out_w)
    assert c_controller.hybrid_get_active_regime(arr_out_w) == 1


def test_actuator_saturation_clamping(c_controller):
    """Verify action clamping strictly to [-1.0, 1.0]."""
    extreme_positive = [2.4, 5.0, math.cos(0.2), math.sin(0.2), 2.0]
    extreme_negative = [-2.4, -5.0, math.cos(-0.2), math.sin(-0.2), -2.0]

    for state, expected_bound in [(extreme_positive, 1.0), (extreme_negative, -1.0)]:
        arr_f = (ctypes.c_float * 5)(*state)
        arr_d = (ctypes.c_double * 5)(*state)
        u_f = float(c_controller.hybrid_compute_action(arr_f))
        u_d = float(c_controller.hybrid_compute_action_double(arr_d))

        assert -1.0 <= u_f <= 1.0
        assert -1.0 <= u_d <= 1.0
        assert math.isclose(u_f, expected_bound, abs_tol=1e-5)
        assert math.isclose(u_d, expected_bound, abs_tol=1e-12)


def test_sign_conventions_at_zero(c_controller):
    """Verify sign(p) convention when p = theta_dot * cos(theta) = 0."""
    py_ctrl = HybridController()

    # When th_dot = 0, p = 0. Python evaluates sign_p = 1.0
    state_p_zero = [0.0, 0.0, -1.0, 0.0, 0.0]
    py_u = py_ctrl.compute_action(state_p_zero)

    arr_d = (ctypes.c_double * 5)(*state_p_zero)
    c_u_d = float(c_controller.hybrid_compute_action_double(arr_d))
    assert math.isclose(py_u, c_u_d, abs_tol=1e-12)


def test_adjacent_representable_float32_boundary(c_controller):
    """Verify energy boundary behavior at adjacent representable float32 values (ULP +-1).

    Documents Decision A and Decision B:
      - Double-precision reproduces Python bit-for-bit (diff == 0.0) across all boundaries.
      - Float32 matches Python at adjacent representable values within 1e-5 (diff ~ 2.4e-8).
      - Directly at w_exact, float32 resolves to the valid action magnitude 0.60 with sign reversal.
    """
    py_ctrl = HybridController()

    for cos_val in [-1.0, 0.0, 0.5]:
        th_val = math.acos(cos_val)
        sin_val = math.sin(th_val)
        w_exact = math.sqrt(60.0 * 0.49 * (1.0 - cos_val))
        w_f32 = np.float32(w_exact)
        w_prev = float(np.nextafter(w_f32, np.float32(-np.inf)))
        w_next = float(np.nextafter(w_f32, np.float32(np.inf)))

        # 1. Double precision must match Python exactly at all points
        for w in [w_prev, float(w_f32), w_next]:
            obs = [0.0, 0.0, cos_val, sin_val, w]
            py_u = py_ctrl.compute_action(obs)
            c_u_d = float(c_controller.hybrid_compute_action_double((ctypes.c_double * 5)(*obs)))
            assert abs(py_u - c_u_d) <= 1e-12, f"Double mismatch at cos={cos_val}, w={w}"

        # 2. Float32 must match Python at adjacent representable points (ULP +-1)
        for w in [w_prev, w_next]:
            obs = [0.0, 0.0, cos_val, sin_val, w]
            py_u = py_ctrl.compute_action(obs)
            c_u_f = float(c_controller.hybrid_compute_action((ctypes.c_float * 5)(*obs)))
            assert abs(py_u - c_u_f) <= 1e-5, f"Float mismatch at adjacent ULP: cos={cos_val}, w={w}"

        # 3. Float32 at w_exact must evaluate to valid pumping action magnitude
        obs_exact = [0.0, 0.0, cos_val, sin_val, float(w_f32)]
        c_u_f_exact = float(c_controller.hybrid_compute_action((ctypes.c_float * 5)(*obs_exact)))
        assert abs(abs(c_u_f_exact) - 0.60) <= 1e-5, f"Float invalid magnitude at exact: {c_u_f_exact}"


# ── 5. Non-Finite Inputs & Error Handling ─────────────────────────────────────


def test_non_finite_inputs_return_safe_neutral(c_controller):
    """Verify non-finite inputs (NaN, +Inf, -Inf) safely return 0.0 action and INVALID regime."""
    nan_val = float("nan")
    inf_val = float("inf")

    bad_states = [
        [nan_val, 0.0, 1.0, 0.0, 0.0],
        [0.0, inf_val, 1.0, 0.0, 0.0],
        [0.0, 0.0, -inf_val, 0.0, 0.0],
        [0.0, 0.0, 1.0, nan_val, 0.0],
        [0.0, 0.0, 1.0, 0.0, inf_val],
        [nan_val] * 5,
        [inf_val] * 5,
    ]

    for state in bad_states:
        arr_f = (ctypes.c_float * 5)(*state)
        arr_d = (ctypes.c_double * 5)(*state)

        u_f = float(c_controller.hybrid_compute_action(arr_f))
        u_d = float(c_controller.hybrid_compute_action_double(arr_d))
        regime = c_controller.hybrid_get_active_regime(arr_f)

        assert u_f == 0.0, f"Expected 0.0 for non-finite input, got {u_f}"
        assert u_d == 0.0, f"Expected 0.0 for non-finite input, got {u_d}"
        assert regime == -1, f"Expected HYBRID_REGIME_INVALID (-1), got {regime}"


def test_null_pointer_safety(c_controller):
    """Verify passing a NULL pointer returns 0.0 action and INVALID regime without crashing."""
    u_f = float(c_controller.hybrid_compute_action(None))
    u_d = float(c_controller.hybrid_compute_action_double(None))
    regime = c_controller.hybrid_get_active_regime(None)

    assert u_f == 0.0
    assert u_d == 0.0
    assert regime == -1


# ── 6. Closed-Loop Simulation Parity Tests ────────────────────────────────────


def test_closed_loop_simulation_parity(c_controller):
    """Verify 100% success and identical trajectory behavior on development seeds 200..209."""
    py_ctrl = HybridController()

    for seed in range(200, 210):
        env_py = CartPoleSwingUpEnv(max_steps=500)
        env_c = CartPoleSwingUpEnv(max_steps=500)

        obs_py, _ = env_py.reset(seed=seed)
        obs_c, _ = env_c.reset(seed=seed)

        cos_thetas_py = [float(obs_py[2])]
        cos_thetas_c = [float(obs_c[2])]
        rail_hit_py = False
        rail_hit_c = False

        step = 0
        while step < 500:
            step += 1
            u_py = py_ctrl.compute_action(obs_py)
            c_arr = (ctypes.c_double * 5)(*[float(v) for v in obs_c])
            u_c = float(c_controller.hybrid_compute_action_double(c_arr))

            assert abs(u_py - u_c) <= 1e-12, f"Action divergence at step {step}"

            obs_py, _, term_py, trunc_py, _ = env_py.step(u_py)
            obs_c, _, term_c, trunc_c, _ = env_c.step(u_c)

            cos_thetas_py.append(float(obs_py[2]))
            cos_thetas_c.append(float(obs_c[2]))

            if term_py:
                rail_hit_py = True
            if term_c:
                rail_hit_c = True

            diff = np.max(np.abs(np.array(obs_py) - np.array(obs_c)))
            assert diff <= 1e-12, f"State divergence at step {step}: diff={diff}"

            if (term_py or trunc_py) and (term_c or trunc_c):
                break

        succ_py, _, _, _ = evaluate_success_predicate(cos_thetas_py, rail_hit=rail_hit_py, max_steps=500)
        succ_c, _, _, _ = evaluate_success_predicate(cos_thetas_c, rail_hit=rail_hit_c, max_steps=500)

        assert succ_py is True, f"Python failed on seed {seed}"
        assert succ_c is True, f"C failed on seed {seed}"
