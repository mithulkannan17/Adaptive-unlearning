from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.experiments.state import (
    RoundState,
    SequentialState,
)
from src.workloads.uniform_random import (
    DeletionRequest,
)


class SequentialExperimentRunner:
    """
    Generic sequential unlearning experiment engine.

    The runner is deliberately independent of the actual
    unlearning algorithm.

    Future algorithms such as:

        Gradient Ascent
        SSD
        SalUn
        ASUC

    can all be plugged into this engine.
    """

    def __init__(
        self,
        dataset_size: int,
        workload: List[DeletionRequest],
        output_dir: str = "./results/sequential",
    ):
        self.dataset_size = dataset_size

        self.workload = workload

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.state = SequentialState(
            dataset_size=dataset_size
        )

        self.results: List[Dict] = []

    def run(
        self,
        round_callback: Optional[
            Callable[[RoundState], Dict]
        ] = None,
    ) -> List[Dict]:
        """
        Execute the sequential workload.

        round_callback receives the fully updated
        RoundState and can later perform the actual
        unlearning operation.
        """

        for request in self.workload:

            round_state = (
                self.state.apply_deletion(
                    round_id=request.round_id,
                    forget_indices=request.forget_indices,
                )
            )

            callback_result = {}

            if round_callback is not None:

                callback_result = (
                    round_callback(
                        round_state
                    )
                    or {}
                )

            result = {
                "round_id": round_state.round_id,

                "new_forget_size": (
                    round_state.new_forget_size
                ),

                "cumulative_forget_size": (
                    round_state.cumulative_forget_size
                ),

                "retain_size": (
                    round_state.retain_size
                ),

                "dataset_size": (
                    self.dataset_size
                ),

                "callback": callback_result,
            }

            self.results.append(result)

            print(
                f"Round {round_state.round_id:02d} | "
                f"New Forget: "
                f"{round_state.new_forget_size:5d} | "
                f"Cumulative Forget: "
                f"{round_state.cumulative_forget_size:5d} | "
                f"Remaining: "
                f"{round_state.retain_size:5d}"
            )

        self.save_results()

        return self.results

    def save_results(self) -> Path:

        output_path = (
            self.output_dir
            / "sequential_state.json"
        )

        payload = {
            "dataset_size": self.dataset_size,
            "summary": self.state.summary(),
            "rounds": self.results,
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=4,
            )

        return output_path

    def get_current_state(self) -> SequentialState:

        return self.state