from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.workloads.uniform_random import DeletionRequest


def compute_sample_losses(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
) -> np.ndarray:
    """
    Compute per-sample cross-entropy losses under the given model.
    Used as an efficient proxy for sample influence / difficulty.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    losses = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_loss = criterion(model(inputs), targets)
            losses.append(batch_loss.detach().cpu().numpy())

    return np.concatenate(losses)


class HighInfluenceWorkload:
    """
    Workload C: High-Influence / Adversarial Deletion Workload.

    Requests the deletion of samples with the highest empirical influence
    (e.g., loss or gradient magnitude) first. This represents an adversarial
    stress-test where the model must unlearn its most influential points.
    """

    def __init__(
        self,
        dataset_size: int,
        influence_scores: Sequence[float],
        rounds: int = 20,
        samples_per_round: int = 500,
        seed: int = 42,
    ):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if len(influence_scores) != dataset_size:
            raise ValueError(
                f"influence_scores length ({len(influence_scores)}) must match dataset_size ({dataset_size})."
            )
        if rounds <= 0:
            raise ValueError("rounds must be positive.")

        self.dataset_size = dataset_size
        self.influence_scores = np.array(influence_scores, dtype=np.float32)
        self.rounds = rounds
        self.samples_per_round = samples_per_round
        self.seed = seed

    def generate(self) -> List[DeletionRequest]:
        # Sort indices by influence descending (highest influence first)
        ranked_indices = np.argsort(-self.influence_scores).tolist()

        active_set = set(range(self.dataset_size))
        requests: List[DeletionRequest] = []
        pointer = 0

        for round_id in range(1, self.rounds + 1):
            if pointer >= len(ranked_indices):
                break

            forget_size = min(self.samples_per_round, len(ranked_indices) - pointer)
            selected = ranked_indices[pointer : pointer + forget_size]
            pointer += forget_size

            forget_indices = sorted(selected)
            active_set.difference_update(forget_indices)
            remaining_indices = sorted(active_set)

            requests.append(
                DeletionRequest(
                    round_id=round_id,
                    forget_indices=forget_indices,
                    remaining_indices=remaining_indices,
                )
            )

        return requests


def generate_workload_c(
    dataset_size: int,
    influence_scores: Sequence[float],
    rounds: int = 20,
    samples_per_round: int = 500,
    seed: int = 42,
) -> List[DeletionRequest]:
    workload = HighInfluenceWorkload(
        dataset_size=dataset_size,
        influence_scores=influence_scores,
        rounds=rounds,
        samples_per_round=samples_per_round,
        seed=seed,
    )
    return workload.generate()
