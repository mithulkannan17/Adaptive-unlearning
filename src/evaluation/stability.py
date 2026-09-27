from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def compute_stability_metrics(
    model: nn.Module,
    reference_model: nn.Module,
) -> Dict[str, float]:
    """
    Compute parameter-space stability and weight drift diagnostics.
    """
    dist_sq = 0.0
    model_norm_sq = 0.0
    ref_norm_sq = 0.0

    with torch.no_grad():
        for p_curr, p_ref in zip(model.parameters(), reference_model.parameters()):
            diff = p_curr.detach() - p_ref.detach()
            dist_sq += diff.pow(2).sum().item()
            model_norm_sq += p_curr.detach().pow(2).sum().item()
            ref_norm_sq += p_ref.detach().pow(2).sum().item()

    param_distance = float(np.sqrt(dist_sq))
    current_norm = float(np.sqrt(model_norm_sq))
    reference_norm = float(np.sqrt(ref_norm_sq))
    relative_dist = param_distance / max(reference_norm, 1e-12)

    return {
        "parameter_distance": param_distance,
        "parameter_norm": current_norm,
        "reference_parameter_norm": reference_norm,
        "relative_parameter_distance": relative_dist,
    }
