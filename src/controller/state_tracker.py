from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.controller.som import HealthState, SOMMeasurement


@dataclass
class RoundDecisionLog:
    round_id: int
    requested_samples: int
    cumulative_forgotten: int
    remaining_samples: int
    som_score: float
    health_state: str
    chosen_primitive: str
    escalation_path: List[str]
    verification_passed: bool
    verification_reason: str
    unlearning_time_seconds: float
    verification_time_seconds: float
    parameter_norm: float
    parameter_distance_baseline: float
    relative_parameter_distance: float
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


class ASUCStateTracker:
    """
    Maintains and serializes the complete state and decision trajectory
    of the Adaptive Sequential Unlearning Controller (ASUC).
    """

    def __init__(self, dataset_size: int):
        self.dataset_size = dataset_size
        self.history: List[RoundDecisionLog] = []
        self.cumulative_time = 0.0

    def record_round(self, log: RoundDecisionLog) -> None:
        self.history.append(log)
        self.cumulative_time += log.unlearning_time_seconds + log.verification_time_seconds

    @property
    def total_rounds(self) -> int:
        return len(self.history)

    @property
    def total_escalations(self) -> int:
        return sum(1 for log in self.history if len(log.escalation_path) > 1)

    def summary(self) -> Dict[str, Any]:
        primitive_counts: Dict[str, int] = {}
        state_counts: Dict[str, int] = {}
        for log in self.history:
            primitive_counts[log.chosen_primitive] = primitive_counts.get(log.chosen_primitive, 0) + 1
            state_counts[log.health_state] = state_counts.get(log.health_state, 0) + 1

        return {
            "dataset_size": self.dataset_size,
            "total_rounds_completed": self.total_rounds,
            "total_unlearning_and_verification_time": self.cumulative_time,
            "total_escalations": self.total_escalations,
            "primitive_distribution": primitive_counts,
            "health_state_distribution": state_counts,
            "history": [asdict(h) for h in self.history],
        }
