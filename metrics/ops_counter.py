"""Mathematical Operation and Parameter Accounting for Neural Controllers.

Provides rigorous, unambiguous accounting of parameters, standard Multiply-Accumulate
(weight MAC) operations, bias additions, activation function evaluations, and total FLOPs
for dense feedforward controllers (e.g. TinyML 5->H->1 and Deep MLP 5->64->64->1).

Definitions:
  - Weight MACs (Multiplications): Sum_{l} (d_{in}^{(l)} * d_{out}^{(l)})
      Each weight MAC corresponds to one multiply-accumulate operation (w * x).
      In conventional hardware and deep learning reporting (e.g., standard dense layers),
      a 5->8->1 network has 5*8 + 8*1 = 48 weight MACs.
      A 5->64->64->1 network has 5*64 + 64*64 + 64*1 = 320 + 4096 + 64 = 4,480 weight MACs.

  - Parameters: Total trainable weights and biases:
      Params = Sum_{l} (d_{in}^{(l)} * d_{out}^{(l)} + d_{out}^{(l)})
      5->8->1 has 48 weights + 9 biases = 57 parameters.
      5->64->64->1 has 4,480 weights + 129 biases = 4,609 parameters.

  - Bias Additions: Sum_{l} d_{out}^{(l)}
      5->8->1 has 8 + 1 = 9 bias additions.
      5->64->64->1 has 64 + 64 + 1 = 129 bias additions.

  - Activation Evaluations: Sum_{l} d_{out}^{(l)}
      5->8->1 has 8 hidden activations (tanh) + 1 output activation (tanh/linear) = 9.
      5->64->64->1 has 64 + 64 + 1 = 129 activation evaluations.

  - Total Floating Point Operations (FLOPs / Arithmetic Ops):
      FLOPs = 2 * Weight MACs (1 mul + 1 add per MAC) + Bias Additions + Activation Ops
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np


def count_dense_layer_ops(
    in_dim: int,
    out_dim: int,
    has_bias: bool = True,
    activation: str = "tanh",
) -> Dict[str, int]:
    """Compute exact operation counts for a single dense linear layer."""
    weight_macs = in_dim * out_dim
    weights = weight_macs
    biases = out_dim if has_bias else 0
    activations = out_dim if activation != "none" else 0
    
    # 1 MAC = 1 multiplication + 1 addition (2 FLOPs)
    total_flops = (2 * weight_macs) + biases + activations
    total_params = weights + biases

    return {
        "weight_macs": weight_macs,
        "weights": weights,
        "biases": biases,
        "parameters": total_params,
        "bias_additions": biases,
        "activation_evals": activations,
        "total_flops": total_flops,
    }


def compute_feedforward_ops(
    layer_dims: Sequence[int],
    has_bias: bool = True,
    hidden_activation: str = "tanh",
    output_activation: str = "tanh",
) -> Dict[str, Any]:
    """Compute exact layer-by-layer and aggregate operation counts for an MLP."""
    if len(layer_dims) < 2:
        raise ValueError(f"layer_dims must contain at least input and output dimensions, got {layer_dims}")

    total_weight_macs = 0
    total_weights = 0
    total_biases = 0
    total_bias_adds = 0
    total_act_evals = 0
    total_flops = 0
    layers_breakdown: List[Dict[str, Any]] = []

    for i in range(len(layer_dims) - 1):
        in_d = layer_dims[i]
        out_d = layer_dims[i + 1]
        is_output = (i == len(layer_dims) - 2)
        act = output_activation if is_output else hidden_activation

        layer_stats = count_dense_layer_ops(in_d, out_d, has_bias=has_bias, activation=act)
        layer_stats["layer_index"] = i
        layer_stats["input_dim"] = in_d
        layer_stats["output_dim"] = out_d
        layer_stats["activation"] = act
        layers_breakdown.append(layer_stats)

        total_weight_macs += layer_stats["weight_macs"]
        total_weights += layer_stats["weights"]
        total_biases += layer_stats["biases"]
        total_bias_adds += layer_stats["bias_additions"]
        total_act_evals += layer_stats["activation_evals"]
        total_flops += layer_stats["total_flops"]

    return {
        "layer_dims": list(layer_dims),
        "weight_macs": total_weight_macs,
        "weights": total_weights,
        "biases": total_biases,
        "total_parameters": total_weights + total_biases,
        "bias_additions": total_bias_adds,
        "activation_evals": total_act_evals,
        "total_flops": total_flops,
        "layers": layers_breakdown,
    }


def audit_model_accounting(model_artifact: Any) -> Dict[str, Any]:
    """Inspect an actual model artifact (numpy dict, torch Module, or state_dict) and audit ops."""
    # Check if PyTorch state dict or Module
    if hasattr(model_artifact, "state_dict"):
        state_dict = model_artifact.state_dict()
    elif isinstance(model_artifact, dict) and any("weight" in k for k in model_artifact.keys()):
        state_dict = model_artifact
    elif isinstance(model_artifact, dict) and "w1" in model_artifact:
        # Custom numpy format for TinyML champion
        w1 = model_artifact["w1"]
        w2 = model_artifact["w2"]
        b1 = model_artifact["b1"]
        b2 = model_artifact["b2"]
        layer_dims = [w1.shape[1], w1.shape[0], w2.shape[0]]
        return compute_feedforward_ops(layer_dims, has_bias=True)
    else:
        raise TypeError(f"Unrecognized model artifact type: {type(model_artifact)}")

    # Extract dimensions from sequential weight matrices
    weight_keys = [k for k in state_dict.keys() if "weight" in k]
    weight_keys.sort()
    layer_dims = []
    for i, k in enumerate(weight_keys):
        w = state_dict[k]
        if i == 0:
            layer_dims.append(w.shape[1])
        layer_dims.append(w.shape[0])

    return compute_feedforward_ops(layer_dims, has_bias=True)
