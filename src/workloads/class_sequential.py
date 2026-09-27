from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from src.workloads.uniform_random import DeletionRequest


class ClassSequentialWorkload:
    """
    Workload B: Class-Sequential Unlearning.

    At each round, deletion requests are concentrated in a specific class,
    cycling or progressing through classes. This tests how unlearning algorithms
    handle extreme class imbalance and localized feature destruction.
    """

    def __init__(
        self,
        targets: Sequence[int],
        rounds: int = 20,
        samples_per_round: Optional[int] = 500,
        delete_full_class: bool = False,
        seed: int = 42,
    ):
        if len(targets) == 0:
            raise ValueError("targets cannot be empty.")

        if rounds <= 0:
            raise ValueError("rounds must be positive.")

        self.targets = np.array(targets, dtype=np.int64)
        self.dataset_size = len(self.targets)
        self.rounds = rounds
        self.samples_per_round = samples_per_round
        self.delete_full_class = delete_full_class
        self.seed = seed

        self.unique_classes = np.unique(self.targets)
        self.num_classes = len(self.unique_classes)

    def generate(self) -> List[DeletionRequest]:
        rng = np.random.default_rng(self.seed)

        # Map each class to active indices
        class_to_indices = {
            c: np.where(self.targets == c)[0].tolist()
            for c in self.unique_classes
        }
        for c in class_to_indices:
            rng.shuffle(class_to_indices[c])

        active_set = set(range(self.dataset_size))
        requests: List[DeletionRequest] = []

        for round_id in range(1, self.rounds + 1):
            if not active_set:
                break

            target_class = int(self.unique_classes[(round_id - 1) % self.num_classes])
            available_in_class = class_to_indices[target_class]

            if self.delete_full_class:
                forget_size = len(available_in_class)
            elif self.samples_per_round is not None:
                forget_size = min(self.samples_per_round, len(available_in_class))
            else:
                forget_size = max(1, len(available_in_class) // (self.rounds // self.num_classes + 1))

            if forget_size == 0:
                # If target class exhausted, pick from any available class with remaining samples
                classes_with_samples = [c for c in self.unique_classes if len(class_to_indices[c]) > 0]
                if not classes_with_samples:
                    break
                target_class = int(classes_with_samples[0])
                available_in_class = class_to_indices[target_class]
                forget_size = min(
                    self.samples_per_round or 500,
                    len(available_in_class),
                )

            selected = available_in_class[:forget_size]
            class_to_indices[target_class] = available_in_class[forget_size:]

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


def generate_workload_b(
    targets: Sequence[int],
    rounds: int = 20,
    samples_per_round: int = 500,
    seed: int = 42,
) -> List[DeletionRequest]:
    workload = ClassSequentialWorkload(
        targets=targets,
        rounds=rounds,
        samples_per_round=samples_per_round,
        seed=seed,
    )
    return workload.generate()
