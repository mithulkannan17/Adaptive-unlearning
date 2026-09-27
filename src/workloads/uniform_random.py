from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DeletionRequest:
    round_id: int
    forget_indices: List[int]
    remaining_indices: List[int]

    @property
    def forget_size(self) -> int:
        return len(self.forget_indices)

    @property
    def remaining_size(self) -> int:
        return len(self.remaining_indices)


class UniformRandomWorkload:
    """
    Workload A:

    At every round, remove approximately 1% of the
    currently remaining training samples uniformly at random.
    """

    def __init__(
        self,
        dataset_size: int = 50_000,
        deletion_fraction: float = 0.01,
        rounds: int = 20,
        seed: int = 42,
    ):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")

        if not 0 < deletion_fraction <= 1:
            raise ValueError(
                "deletion_fraction must be in (0, 1]."
            )

        if rounds <= 0:
            raise ValueError("rounds must be positive.")

        self.dataset_size = dataset_size
        self.deletion_fraction = deletion_fraction
        self.rounds = rounds
        self.seed = seed

    def generate(self) -> List[DeletionRequest]:
        rng = np.random.default_rng(self.seed)

        remaining = np.arange(
            self.dataset_size,
            dtype=np.int64,
        )

        requests = []

        for round_id in range(1, self.rounds + 1):

            if len(remaining) == 0:
                break

            # At least one sample must be deleted.
            forget_size = max(
                1,
                int(
                    np.ceil(
                        len(remaining)
                        * self.deletion_fraction
                    )
                ),
            )

            forget_size = min(
                forget_size,
                len(remaining),
            )

            selected_positions = rng.choice(
                len(remaining),
                size=forget_size,
                replace=False,
            )

            selected_positions = np.sort(
                selected_positions
            )

            forget_indices = remaining[
                selected_positions
            ]

            mask = np.ones(
                len(remaining),
                dtype=bool,
            )

            mask[selected_positions] = False

            remaining = remaining[mask]

            requests.append(
                DeletionRequest(
                    round_id=round_id,
                    forget_indices=forget_indices.tolist(),
                    remaining_indices=remaining.tolist(),
                )
            )

        return requests


def generate_workload_a(
    dataset_size: int = 50_000,
    deletion_fraction: float = 0.01,
    rounds: int = 20,
    seed: int = 42,
) -> List[DeletionRequest]:

    workload = UniformRandomWorkload(
        dataset_size=dataset_size,
        deletion_fraction=deletion_fraction,
        rounds=rounds,
        seed=seed,
    )

    return workload.generate()