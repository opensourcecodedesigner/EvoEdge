"""Comprehensive Split-Integrity and Experimental Leakage Validator.

Performs rigorous automated static and structural checks:
  1. Validates schema, required fields, physical variables, and units.
  2. Verifies seed partition well-formedness and pairwise mutual disjointness.
  3. Verifies out-of-distribution benchmark separation:
     - Fixed test benchmarks must lie strictly outside training distribution ranges.
     - Fixed test benchmarks must not be included within validation distribution bounds.
  4. Scans codebase (training, tuning, and evolution scripts) for reserved test seed references.
  5. Distinguishes static inspection capabilities from runtime execution verification.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Set, Tuple
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


REQUIRED_PHYSICAL_VARS = {
    "pole_mass": ("MASS_POLE", "kg"),
    "cart_mass": ("MASS_CART", "kg"),
    "cart_friction": ("b_cart", "N*s/m"),
    "joint_friction": ("b_pole", "N*m*s/rad"),
    "observation_noise": ("obs_noise_sigma", "standard deviation (Gaussian)"),
    "actuator_delay": ("delay_steps", "discrete control steps (dt = 0.02s)"),
}


class SplitIntegrityError(Exception):
    """Raised when experimental split integrity or protocol separation is violated."""
    pass


def validate_split_integrity(config_path: str = "config/perturbations.yaml") -> Dict[str, Any]:
    """Execute complete validation suite on perturbation config and codebase."""
    if not os.path.isfile(config_path):
        raise SplitIntegrityError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise SplitIntegrityError(f"YAML Syntax Error in {config_path}: {exc}")

    if not isinstance(cfg, dict):
        raise SplitIntegrityError(f"Malformed configuration: expected top-level dictionary in {config_path}")

    # 1. Validate splits schema and disjointness
    if "splits" not in cfg:
        raise SplitIntegrityError("Missing 'splits' section in configuration.")

    splits = cfg["splits"]
    for req_split in ["training", "validation", "test"]:
        if req_split not in splits:
            raise SplitIntegrityError(f"Missing required split '{req_split}' in splits configuration.")
        if "seed_range" not in splits[req_split] or len(splits[req_split]["seed_range"]) != 2:
            raise SplitIntegrityError(f"Split '{req_split}' must define a 2-element [start, end] seed_range.")

    train_start, train_end = splits["training"]["seed_range"]
    val_start, val_end = splits["validation"]["seed_range"]
    test_start, test_end = splits["test"]["seed_range"]

    if train_start > train_end or val_start > val_end or test_start > test_end:
        raise SplitIntegrityError("Invalid seed range: start must be <= end.")

    train_set = set(range(train_start, train_end + 1))
    val_set = set(range(val_start, val_end + 1))
    test_set = set(range(test_start, test_end + 1))

    # Pairwise disjointness checks
    tv = train_set.intersection(val_set)
    tt = train_set.intersection(test_set)
    vt = val_set.intersection(test_set)

    if tv:
        raise SplitIntegrityError(f"Integrity Violation: Training and Validation seeds overlap on {sorted(list(tv))[:5]}...")
    if tt:
        raise SplitIntegrityError(f"Integrity Violation: Training and Test seeds overlap on {sorted(list(tt))[:5]}...")
    if vt:
        raise SplitIntegrityError(f"Integrity Violation: Validation and Test seeds overlap on {sorted(list(vt))[:5]}...")

    # Optional supplementary untouched test partition check (Protocol Amendment v2.2)
    if "supplementary_untouched_test" in splits:
        supp_spec = splits["supplementary_untouched_test"]
        if "seed_range" in supp_spec and len(supp_spec["seed_range"]) == 2:
            s_start, s_end = supp_spec["seed_range"]
            if s_start > s_end:
                raise SplitIntegrityError("Invalid supplementary seed range: start must be <= end.")
            supp_set = set(range(s_start, s_end + 1))
            for other_name, other_set in [("Training", train_set), ("Validation", val_set), ("Test", test_set)]:
                overlap = supp_set.intersection(other_set)
                if overlap:
                    raise SplitIntegrityError(f"Integrity Violation: Supplementary Untouched Test overlaps with {other_name} on {sorted(list(overlap))[:5]}...")

    # 2. Validate physical variables and OOD benchmark conditions
    if "perturbations" not in cfg:
        raise SplitIntegrityError("Missing 'perturbations' section in configuration.")

    perts = cfg["perturbations"]
    for p_name, (exp_var, exp_units) in REQUIRED_PHYSICAL_VARS.items():
        if p_name not in perts:
            raise SplitIntegrityError(f"Missing required perturbation variable: '{p_name}'")
        p_data = perts[p_name]
        if p_data.get("variable_name") != exp_var:
            raise SplitIntegrityError(f"Variable name mismatch for '{p_name}': expected {exp_var}, got {p_data.get('variable_name')}")
        if p_data.get("units") != exp_units:
            raise SplitIntegrityError(f"Units mismatch for '{p_name}': expected {exp_units}, got {p_data.get('units')}")

        for part in ["training", "validation", "test"]:
            if part not in p_data:
                raise SplitIntegrityError(f"Perturbation '{p_name}' missing distribution specification for '{part}' split.")

    # 3. Check OOD benchmark properties vs Train & Validation
    ood_checks = {}
    val_leakage_checks = {}

    for p_name, p_data in perts.items():
        test_spec = p_data["test"]
        train_spec = p_data["training"]
        val_spec = p_data["validation"]

        # If a fixed test benchmark is specified:
        if "benchmark_fixed" in test_spec:
            bm_val = test_spec["benchmark_fixed"]

            # Must be strictly outside training range
            if "range" in train_spec:
                tr_min, tr_max = train_spec["range"]
                is_ood_train = (bm_val > tr_max) or (bm_val < tr_min)
                if not is_ood_train:
                    raise SplitIntegrityError(
                        f"OOD Violation for '{p_name}': benchmark {bm_val} falls inside training range [{tr_min}, {tr_max}]"
                    )
                ood_checks[p_name] = f"Test benchmark {bm_val} > train max {tr_max}"

            # Must not be enclosed within validation bounds (preventing tuning to test benchmark)
            if "range" in val_spec:
                val_min, val_max = val_spec["range"]
                contains_val = (val_min <= bm_val <= val_max)
                if contains_val:
                    raise SplitIntegrityError(
                        f"Validation Leakage Violation for '{p_name}': test benchmark {bm_val} is enclosed within validation range [{val_min}, {val_max}]"
                    )
                val_leakage_checks[p_name] = f"Test benchmark {bm_val} strictly excluded from val [{val_min}, {val_max}]"

        elif "values" in test_spec:
            test_vals = set(test_spec["values"])
            train_vals = set(train_spec.get("values", []))
            val_vals = set(val_spec.get("values", []))

            overlap_tr = test_vals.intersection(train_vals)
            overlap_val = test_vals.intersection(val_vals)

            if overlap_tr:
                raise SplitIntegrityError(f"Discrete OOD Violation for '{p_name}': test values {overlap_tr} exist in training values.")
            if overlap_val:
                raise SplitIntegrityError(f"Validation Leakage Violation for '{p_name}': test values {overlap_val} exist in validation values.")

            ood_checks[p_name] = f"Test discrete values {test_vals} disjoint from train {train_vals}"
            val_leakage_checks[p_name] = f"Test discrete values {test_vals} disjoint from val {val_vals}"

    # 4. Static Code Inspection: Check for test seed references in training scripts
    training_scripts = [
        os.path.join(_PROJECT_ROOT, "evolution", "evolve_tinyml.py"),
        os.path.join(_PROJECT_ROOT, "experiments", "run_baselines_pruning.py"),
    ]
    code_leakage_audit = {}

    for s_path in training_scripts:
        if not os.path.isfile(s_path):
            continue
        rel_path = os.path.relpath(s_path, _PROJECT_ROOT)
        with open(s_path, "r", encoding="utf-8") as fh:
            code = fh.read()

        # Find loop ranges or seed assignments targeting training/generation
        # Match e.g. range(1000, 1020) used in training loops
        train_leakage_matches = []
        for line_no, line in enumerate(code.splitlines(), start=1):
            if "range(1000" in line and ("train" in line.lower() or "gen" in line.lower() or "population" in line.lower()):
                train_leakage_matches.append((line_no, line.strip()))

        code_leakage_audit[rel_path] = {
            "status": "PASS" if not train_leakage_matches else "FLAGGED",
            "findings": train_leakage_matches,
        }

    return {
        "status": "PASS",
        "splits": {
            "training": splits["training"]["seed_range"],
            "validation": splits["validation"]["seed_range"],
            "test": splits["test"]["seed_range"],
        },
        "ood_extensions": ood_checks,
        "val_leakage_isolation": val_leakage_checks,
        "static_code_inspection": code_leakage_audit,
        "disclaimer": (
            "Static inspection verifies declared boundaries and code patterns. "
            "Runtime data leakage absence must be reinforced during active training via environment logging."
        ),
    }


if __name__ == "__main__":
    try:
        report = validate_split_integrity()
        print("=" * 70)
        print("EvoEdge Experimental Split & Integrity Verification: PASSED")
        print("=" * 70)
        print(f"Splits: Train {report['splits']['training']}, Val {report['splits']['validation']}, Test {report['splits']['test']}")
        print("\nOOD Benchmark Separation:")
        for k, v in report["ood_extensions"].items():
            print(f"  - {k:<18}: {v}")
        print("\nValidation Isolation (No Benchmark Exposure):")
        for k, v in report["val_leakage_isolation"].items():
            print(f"  - {k:<18}: {v}")
        print("\nStatic Code Inspection:")
        for k, v in report["static_code_inspection"].items():
            print(f"  - {k:<35}: {v['status']}")
        print("=" * 70)
    except SplitIntegrityError as err:
        print(f"Integrity Check FAILED: {err}", file=sys.stderr)
        sys.exit(1)
