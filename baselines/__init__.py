"""Non-evolutionary baseline controllers for CartPole benchmarking.

Provides:
    - LQRController: Analytical Linear Quadratic Regulator.
    - FixedMLPController: Dense 16-hidden unit feedforward neural controller (~113 params).
"""

from baselines.lqr import LQRController, get_cartpole_linearized_system
from baselines.mlp import FixedMLPController, train_mlp_es
from baselines.hybrid_controller import HybridController

__all__ = [
    "LQRController",
    "get_cartpole_linearized_system",
    "FixedMLPController",
    "train_mlp_es",
    "HybridController",
]
