"""Unit tests for operation accounting and parameter verification."""

import os
import pickle
import unittest
import torch
import numpy as np

from metrics.ops_counter import (
    compute_feedforward_ops,
    count_dense_layer_ops,
    audit_model_accounting,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestOpsAccounting(unittest.TestCase):
    """Verify operation accounting across standard topologies and saved models."""

    def test_h8_theoretical_accounting(self):
        """Standard 5->8->1 network must have 48 weight MACs, 57 params, 9 bias adds, 9 acts."""
        ops = compute_feedforward_ops([5, 8, 1])
        self.assertEqual(ops["weight_macs"], 48)
        self.assertEqual(ops["weights"], 48)
        self.assertEqual(ops["biases"], 9)
        self.assertEqual(ops["total_parameters"], 57)
        self.assertEqual(ops["bias_additions"], 9)
        self.assertEqual(ops["activation_evals"], 9)
        # FLOPs = 2 * 48 + 9 + 9 = 114
        self.assertEqual(ops["total_flops"], 114)

    def test_mlp64_theoretical_accounting(self):
        """Standard 5->64->64->1 network must have 4,480 weight MACs, 4,609 params."""
        ops = compute_feedforward_ops([5, 64, 64, 1])
        # 5*64 + 64*64 + 64*1 = 320 + 4096 + 64 = 4480
        self.assertEqual(ops["weight_macs"], 4480)
        self.assertEqual(ops["weights"], 4480)
        # 64 + 64 + 1 = 129
        self.assertEqual(ops["biases"], 129)
        self.assertEqual(ops["total_parameters"], 4609)
        self.assertEqual(ops["bias_additions"], 129)
        self.assertEqual(ops["activation_evals"], 129)
        # FLOPs = 2 * 4480 + 129 + 129 = 9218
        self.assertEqual(ops["total_flops"], 9218)

    def test_h3_and_h6_accounting(self):
        """Verify H=3 (22 params, 18 MACs) and H=6 (43 params, 36 MACs)."""
        ops_h3 = compute_feedforward_ops([5, 3, 1])
        self.assertEqual(ops_h3["weight_macs"], 18)
        self.assertEqual(ops_h3["total_parameters"], 22)

        ops_h6 = compute_feedforward_ops([5, 6, 1])
        self.assertEqual(ops_h6["weight_macs"], 36)
        self.assertEqual(ops_h6["total_parameters"], 43)

    def test_champion_h8_artifact_audit(self):
        """Audit the actual saved champion_neural_h8.pkl file."""
        pkl_path = os.path.join(_PROJECT_ROOT, "results", "champion_neural_h8.pkl")
        with open(pkl_path, "rb") as f:
            artifact = pickle.load(f)
        
        ops = audit_model_accounting(artifact)
        self.assertEqual(ops["weight_macs"], 48)
        self.assertEqual(ops["total_parameters"], 57)
        self.assertEqual(ops["layer_dims"], [5, 8, 1])

    def test_mlp64_baseline_artifact_audit(self):
        """Audit the actual saved mlp64_baseline.pt PyTorch file."""
        pt_path = os.path.join(_PROJECT_ROOT, "results", "mlp64_baseline.pt")
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
        
        ops = audit_model_accounting(state_dict)
        self.assertEqual(ops["weight_macs"], 4480)
        self.assertEqual(ops["total_parameters"], 4609)
        self.assertEqual(ops["layer_dims"], [5, 64, 64, 1])


if __name__ == "__main__":
    unittest.main()
