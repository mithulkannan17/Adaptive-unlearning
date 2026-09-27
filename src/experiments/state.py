from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class RoundState:
    """
    Immutable-style record describing one sequential
    deletion round.
    """

    round_id: int

    new_forget_indices: List[int]

    cumulative_forget_indices: List[int]

    retain_indices: List[int]

    metrics: Dict[str, float] = field(
        default_factory=dict
    )

    metadata: Dict[str, object] = field(
        default_factory=dict
    )

    @property
    def new_forget_size(self) -> int:
        return len(self.new_forget_indices)

    @property
    def cumulative_forget_size(self) -> int:
        return len(self.cumulative_forget_indices)

    @property
    def retain_size(self) -> int:
        return len(self.retain_indices)


class SequentialState:
    """
    Maintains the global state of a sequential
    machine-unlearning experiment.
    """

    def __init__(
        self,
        dataset_size: int,
    ):
        if dataset_size <= 0:
            raise ValueError(
                "dataset_size must be positive."
            )

        self.dataset_size = dataset_size

        self.active_indices: Set[int] = set(
            range(dataset_size)
        )

        self.cumulative_forget_indices: Set[int] = set()

        self.rounds: List[RoundState] = []

    def apply_deletion(
        self,
        round_id: int,
        forget_indices: List[int],
    ) -> RoundState:

        if not forget_indices:
            raise ValueError(
                "forget_indices cannot be empty."
            )

        forget_set = set(forget_indices)

        # Every requested index must currently be active.
        invalid_indices = (
            forget_set
            - self.active_indices
        )

        if invalid_indices:
            raise ValueError(
                "Deletion request contains indices "
                "that are no longer active: "
                f"{sorted(invalid_indices)[:10]}"
            )

        self.active_indices.difference_update(
            forget_set
        )

        self.cumulative_forget_indices.update(
            forget_set
        )

        round_state = RoundState(
            round_id=round_id,
            new_forget_indices=sorted(
                forget_set
            ),
            cumulative_forget_indices=sorted(
                self.cumulative_forget_indices
            ),
            retain_indices=sorted(
                self.active_indices
            ),
        )

        self.rounds.append(round_state)

        self._validate()

        return round_state

    def _validate(self) -> None:

        if (
            len(self.active_indices)
            + len(self.cumulative_forget_indices)
            != self.dataset_size
        ):
            raise RuntimeError(
                "Sequential state invariant violated: "
                "active + forgotten != dataset size."
            )

        if (
            self.active_indices
            & self.cumulative_forget_indices
        ):
            raise RuntimeError(
                "Sequential state invariant violated: "
                "an index exists in both active and forgotten sets."
            )

    @property
    def active_size(self) -> int:
        return len(self.active_indices)

    @property
    def forgotten_size(self) -> int:
        return len(
            self.cumulative_forget_indices
        )

    @property
    def completed_rounds(self) -> int:
        return len(self.rounds)

    def summary(self) -> Dict[str, int]:

        return {
            "dataset_size": self.dataset_size,
            "active_size": self.active_size,
            "forgotten_size": self.forgotten_size,
            "completed_rounds": self.completed_rounds,
        }