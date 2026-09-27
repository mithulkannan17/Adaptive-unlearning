from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class HealthState(str, Enum):
    GREEN = "GREEN"    # Healthy, low subspace overlap (SOM < 0.4)
    AMBER = "AMBER"    # Warning, moderate overlap (0.4 <= SOM < 0.8)
    RED = "RED"        # Critical, severe plasticity collapse risk (SOM >= 0.8)


@dataclass
class SOMMeasurement:
    som_score: float
    health_state: HealthState
    rank: int
    gradient_norm: float
    basis_norm: float


class SubspaceTracker:
    """
    Subspace Orthogonality Metric (SOM) Tracker.

    Tracks the cumulative parameter subspace occupied by sequential unlearning
    directions using an exponentially decaying low-rank orthonormal basis.
    Measures the alignment of incoming unlearning gradients with this historical
    subspace to proactively diagnose parameter fatigue and plasticity collapse.
    """

    def __init__(
        self,
        decay_factor: float = 0.9,
        max_rank: int = 5,
        green_threshold: float = 0.4,
        red_threshold: float = 0.8,
        device: torch.device | str = "cuda",
    ):
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError("decay_factor must be in (0, 1].")
        if max_rank <= 0:
            raise ValueError("max_rank must be positive.")
        if not 0.0 <= green_threshold < red_threshold <= 1.0:
            raise ValueError("Thresholds must satisfy 0 <= green < red <= 1.")

        self.decay_factor = decay_factor
        self.max_rank = max_rank
        self.green_threshold = green_threshold
        self.red_threshold = red_threshold
        self.device = torch.device(device) if isinstance(device, str) else device

        # Basis matrix of shape [D, rank]
        self.basis: Optional[torch.Tensor] = None

    @torch.no_grad()
    def compute_som(self, gradient: torch.Tensor) -> float:
        """
        Compute SOM score: projection length of normalized gradient onto historical basis.
        Returns value in [0.0, 1.0].
        """
        if self.basis is None:
            return 0.0

        grad = gradient.detach().flatten().to(self.basis.device)
        norm = torch.norm(grad)
        if norm < 1e-8:
            return 0.0

        unit_grad = grad / norm
        # Project onto basis: v = basis^T * unit_grad
        projection = torch.matmul(self.basis.T, unit_grad)
        score = torch.norm(projection).item()
        return min(max(float(score), 0.0), 1.0)

    def classify_state(self, som_score: float) -> HealthState:
        """
        Map a scalar SOM score to the condition HealthState.
        """
        if som_score < self.green_threshold:
            return HealthState.GREEN
        elif som_score < self.red_threshold:
            return HealthState.AMBER
        else:
            return HealthState.RED

    def update(self, gradient: torch.Tensor) -> SOMMeasurement:
        """
        Measure SOM for the incoming gradient, then update the historical basis.
        """
        grad = gradient.detach().flatten().to(self.device)
        grad_norm = float(torch.norm(grad).item())

        if grad_norm < 1e-8:
            som_score = 0.0
            state = HealthState.GREEN
            return SOMMeasurement(
                som_score=0.0,
                health_state=state,
                rank=self.basis.size(1) if self.basis is not None else 0,
                gradient_norm=0.0,
                basis_norm=float(torch.norm(self.basis).item()) if self.basis is not None else 0.0,
            )

        unit_grad = (grad / grad_norm).unsqueeze(1)  # [D, 1]

        if self.basis is None:
            self.basis = unit_grad.clone()
            som_score = 0.0
            state = HealthState.GREEN
        else:
            som_score = self.compute_som(grad)
            state = self.classify_state(som_score)

            # Decay previous directions and append new direction
            decayed_basis = self.basis * self.decay_factor
            combined = torch.cat([decayed_basis, unit_grad], dim=1)  # [D, k+1]

            # Re-orthonormalize using economic QR
            q, _ = torch.linalg.qr(combined, mode="reduced")

            # Retain at most max_rank orthogonal directions
            if q.size(1) > self.max_rank:
                q = q[:, : self.max_rank]

            self.basis = q

        basis_norm = float(torch.norm(self.basis).item()) if self.basis is not None else 0.0
        current_rank = self.basis.size(1) if self.basis is not None else 0

        return SOMMeasurement(
            som_score=som_score,
            health_state=state,
            rank=current_rank,
            gradient_norm=grad_norm,
            basis_norm=basis_norm,
        )

    def reset(self) -> None:
        """Reset historical basis."""
        self.basis = None


def compute_request_gradient(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Compute mean cross-entropy gradient over the forget set batch-by-batch
    to avoid GPU OOM on large deletion requests.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    accumulated_grad: Optional[torch.Tensor] = None
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.size(0)

        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        batch_grads = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                batch_grads.append(param.grad.detach().flatten())

        if batch_grads:
            flat_batch = torch.cat(batch_grads)
            if accumulated_grad is None:
                accumulated_grad = flat_batch * batch_size
            else:
                accumulated_grad += flat_batch * batch_size

        total_samples += batch_size
        model.zero_grad(set_to_none=True)

    if total_samples == 0 or accumulated_grad is None:
        return None

    accumulated_grad /= total_samples
    return accumulated_grad
