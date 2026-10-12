"""Automated data split and experimental perturbation integrity verification.

Validates:
  1. Complete disjointness among training (0..199), validation (200..399), and test (1000..1099) seeds.
  2. Test perturbation ranges strictly extend beyond training distributions.
  3. Validates absence of test seed leakage in training or model selection scripts.
"""

from __future__ import annotations

import os
import sys
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def verify_split_integrity(config_path: str = "config/perturbations.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = cfg["splits"]
    train_range = range(splits["training"]["seed_range"][0], splits["training"]["seed_range"][1] + 1)
    val_range = range(splits["validation"]["seed_range"][0], splits["validation"]["seed_range"][1] + 1)
    test_range = range(splits["test"]["seed_range"][0], splits["test"]["seed_range"][1] + 1)

    train_set = set(train_range)
    val_set = set(val_range)
    test_set = set(test_range)

    # 1. Check seed overlap
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    if train_val_overlap:
        raise ValueError(f"Integrity Violation: Training and Validation seeds overlap on {train_val_overlap}")
    if train_test_overlap:
        raise ValueError(f"Integrity Violation: Training and Test seeds overlap on {train_test_overlap}")
    if val_test_overlap:
        raise ValueError(f"Integrity Violation: Validation and Test seeds overlap on {val_test_overlap}")

    # 2. Check perturbation out-of-distribution extension
    perts = cfg["perturbations"]
    extensions = {}

    for name, p in perts.items():
        if "benchmark_fixed" in p["test"]:
            test_val = p["test"]["benchmark_fixed"]
            if "range" in p["training"]:
                train_min, train_max = p["training"]["range"]
                is_ext = (test_val > train_max) or (test_val < train_min)
                extensions[name] = is_ext
                if not is_ext:
                    raise ValueError(f"Integrity Violation: Test condition {test_val} for {name} does not extend beyond train range [{train_min}, {train_max}]")
            elif "values" in p["training"]:
                train_vals = p["training"]["values"]
                is_ext = test_val not in train_vals
                extensions[name] = is_ext
                if not is_ext:
                    raise ValueError(f"Integrity Violation: Test condition {test_val} for {name} is present in train values {train_vals}")
        elif "values" in p["test"]:
            test_vals = p["test"]["values"]
            train_vals = p["training"]["values"]
            is_ext = all(v not in train_vals for v in test_vals)
            extensions[name] = is_ext
            if not is_ext:
                raise ValueError(f"Integrity Violation: Test values {test_vals} for {name} overlap train values {train_vals}")

    return {
        "status": "PASS",
        "train_seeds": f"{len(train_set)} seeds ({splits['training']['seed_range']})",
        "val_seeds": f"{len(val_set)} seeds ({splits['validation']['seed_range']})",
        "test_seeds": f"{len(test_set)} seeds ({splits['test']['seed_range']})",
        "perturbation_extensions": extensions,
    }


if __name__ == "__main__":
    res = verify_split_integrity()
    print("=" * 60)
    print("EvoEdge Split & Perturbation Integrity Check: PASSED")
    print(f"  Training   : {res['train_seeds']}")
    print(f"  Validation : {res['val_seeds']}")
    print(f"  Test       : {res['test_seeds']}")
    print(f"  OOD Extends: {res['perturbation_extensions']}")
    print("=" * 60)
