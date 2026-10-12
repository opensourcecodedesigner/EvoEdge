"""Unit tests for split_integrity validator.

Tests:
  1. Valid configuration returns PASS.
  2. Overlapping seeds fail.
  3. Validation range containing test benchmark fails.
  4. Test benchmark falling inside training range fails.
  5. Malformed YAML syntax fails safely.
  6. Missing required physical variable fails.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.split_integrity import validate_split_integrity, SplitIntegrityError


class TestSplitIntegrityValidator(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(_PROJECT_ROOT, "config", "perturbations.yaml"), "r", encoding="utf-8") as f:
            self.valid_cfg = yaml.safe_load(f)

    def _write_temp_cfg(self, cfg_data: dict) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(cfg_data, f)
        return path

    def test_valid_configuration(self):
        """Current config/perturbations.yaml must pass all checks."""
        res = validate_split_integrity()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["splits"]["training"], [0, 199])
        self.assertEqual(res["splits"]["validation"], [200, 399])
        self.assertEqual(res["splits"]["test"], [1000, 1099])

    def test_overlapping_seeds_detected(self):
        """Overlapping train and validation seeds must raise SplitIntegrityError."""
        cfg = yaml.safe_load(yaml.dump(self.valid_cfg))
        cfg["splits"]["validation"]["seed_range"] = [190, 399]  # overlaps 190..199
        path = self._write_temp_cfg(cfg)
        try:
            with self.assertRaises(SplitIntegrityError) as ctx:
                validate_split_integrity(path)
            self.assertIn("Training and Validation seeds overlap", str(ctx.exception))
        finally:
            os.remove(path)

    def test_validation_leakage_detected(self):
        """Validation range enclosing test benchmark must raise SplitIntegrityError."""
        cfg = yaml.safe_load(yaml.dump(self.valid_cfg))
        # Enclose 0.13 kg test benchmark in validation
        cfg["perturbations"]["pole_mass"]["validation"]["range"] = [0.07, 0.13]
        path = self._write_temp_cfg(cfg)
        try:
            with self.assertRaises(SplitIntegrityError) as ctx:
                validate_split_integrity(path)
            self.assertIn("Validation Leakage Violation", str(ctx.exception))
        finally:
            os.remove(path)

    def test_ood_violation_detected(self):
        """Test benchmark inside training range must raise SplitIntegrityError."""
        cfg = yaml.safe_load(yaml.dump(self.valid_cfg))
        # Set training cart mass range to include benchmark 1.30 kg
        cfg["perturbations"]["cart_mass"]["training"]["range"] = [0.85, 1.35]
        path = self._write_temp_cfg(cfg)
        try:
            with self.assertRaises(SplitIntegrityError) as ctx:
                validate_split_integrity(path)
            self.assertIn("OOD Violation", str(ctx.exception))
        finally:
            os.remove(path)

    def test_malformed_yaml(self):
        """Syntax errors in YAML must fail safely with explanation."""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("splits: [unclosed list\n")
        try:
            with self.assertRaises(SplitIntegrityError) as ctx:
                validate_split_integrity(path)
            self.assertIn("YAML Syntax Error", str(ctx.exception))
        finally:
            os.remove(path)

    def test_missing_physical_variable(self):
        """Missing physical variable must raise SplitIntegrityError."""
        cfg = yaml.safe_load(yaml.dump(self.valid_cfg))
        del cfg["perturbations"]["actuator_delay"]
        path = self._write_temp_cfg(cfg)
        try:
            with self.assertRaises(SplitIntegrityError) as ctx:
                validate_split_integrity(path)
            self.assertIn("Missing required perturbation variable: 'actuator_delay'", str(ctx.exception))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
