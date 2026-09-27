from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.evaluation.evaluator import UnlearningEvaluator
from src.experiments.sequential_runner import (
    SequentialExperimentRunner,
)
from src.models.resnet import create_resnet18
from src.unlearning.gradient_ascent import (
    GradientAscentUnlearner,
)
from src.utils import load_checkpoint
from src.workloads.uniform_random import (
    generate_workload_a,
)


class SequentialGradientAscent:
    """
    Sequential Gradient Ascent experiment.

    The model is NOT reset between rounds.

        theta_0
          |
        GA(Df_1)
          |
        theta_1
          |
        GA(Df_2)
          |
        theta_2
          |
         ...
    """

    def __init__(
        self,
        device: torch.device,
        train_dataset,
        test_loader: DataLoader,
        baseline_checkpoint: str,
        output_dir: str = "./results/sequential_gradient_ascent",
        unlearning_epochs: int = 5,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        num_workers: int = 2,
    ):

        self.device = device
        self.train_dataset = train_dataset
        self.test_loader = test_loader

        self.baseline_checkpoint = (
            baseline_checkpoint
        )

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.unlearner = GradientAscentUnlearner(
            device=device,
            epochs=unlearning_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.evaluator = UnlearningEvaluator(
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.model = self._load_baseline()

        self.results = []

    def _load_baseline(self):

        model = create_resnet18(
            num_classes=10
        )

        load_checkpoint(
            path=self.baseline_checkpoint,
            model=model,
            device=self.device,
        )

        return model.to(self.device)

    def run(
        self,
        workload,
    ) -> Dict:

        print("=" * 70)
        print("SEQUENTIAL GRADIENT ASCENT")
        print("=" * 70)

        print(
            f"Rounds: {len(workload)}"
        )

        print(
            f"Initial model: "
            f"{self.baseline_checkpoint}"
        )

        runner = SequentialExperimentRunner(
            dataset_size=len(
                self.train_dataset
            ),
            workload=workload,
            output_dir=str(
                self.output_dir
            ),
        )

        def process_round(round_state):

            return self._process_round(
                round_state
            )

        runner.run(
            round_callback=process_round
        )

        summary = {
            "method": "gradient_ascent",
            "num_rounds": len(
                self.results
            ),
            "results": self.results,
        }

        summary_path = (
            self.output_dir
            / "summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
            )

        print()
        print("=" * 70)
        print("SEQUENTIAL GRADIENT ASCENT COMPLETE")
        print("=" * 70)

        print(
            f"Rounds completed: "
            f"{len(self.results)}"
        )

        print(
            f"Results: {summary_path}"
        )

        return summary

    def _process_round(
        self,
        round_state,
    ) -> Dict:

        round_id = round_state.round_id

        print()
        print("-" * 70)
        print(
            f"ROUND {round_id}"
        )
        print("-" * 70)

        print(
            f"New forget set: "
            f"{round_state.new_forget_size}"
        )

        print(
            f"Cumulative forgotten: "
            f"{round_state.cumulative_forget_size}"
        )

        print(
            f"Remaining samples: "
            f"{round_state.retain_size}"
        )

        # -----------------------------------------------------
        # IMPORTANT:
        # Use ONLY the NEW deletion request for this round.
        #
        # Previous deletion requests have already been applied
        # to the model in previous rounds.
        # -----------------------------------------------------

        forget_indices = (
            round_state.new_forget_indices
        )

        retain_indices = (
            round_state.retain_indices
        )

        # -----------------------------------------------------
        # Snapshot before unlearning
        # -----------------------------------------------------

        before_model = create_resnet18(
            num_classes=10
        ).to(self.device)

        before_model.load_state_dict(
            self.model.state_dict()
        )

        # -----------------------------------------------------
        # Apply Gradient Ascent
        # -----------------------------------------------------

        start = time.perf_counter()

        result = self.unlearner.unlearn(
            model=self.model,
            train_dataset=self.train_dataset,
            forget_indices=forget_indices,
        )

        self.model = result.model

        unlearning_time = (
            time.perf_counter()
            - start
        )

        # -----------------------------------------------------
        # Evaluate CURRENT new forget set
        # -----------------------------------------------------

        metrics_new = (
            self.evaluator.evaluate(
                model=self.model,
                train_dataset=self.train_dataset,
                test_loader=self.test_loader,
                forget_indices=forget_indices,
                retain_indices=retain_indices,
                reference_model=before_model,
            )
        )

        # -----------------------------------------------------
        # Evaluate CUMULATIVE forget set
        # -----------------------------------------------------

        metrics_cumulative = (
            self.evaluator.evaluate(
                model=self.model,
                train_dataset=self.train_dataset,
                test_loader=self.test_loader,
                forget_indices=(
                    round_state
                    .cumulative_forget_indices
                ),
                retain_indices=retain_indices,
                reference_model=before_model,
            )
        )

        round_result = {
            "round_id": round_id,

            "new_forget_size": (
                round_state.new_forget_size
            ),

            "cumulative_forget_size": (
                round_state.cumulative_forget_size
            ),

            "retain_size": (
                round_state.retain_size
            ),

            "unlearning_time_seconds": (
                unlearning_time
            ),

            "new_forget": metrics_new,

            "cumulative_forget": (
                metrics_cumulative
            ),
        }

        self.results.append(
            round_result
        )

        round_path = (
            self.output_dir
            / f"round_{round_id:02d}.json"
        )

        with open(
            round_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                round_result,
                file,
                indent=4,
            )

        print()
        print(
            f"New forget accuracy: "
            f"{metrics_new['forget_accuracy']:.2f}%"
        )

        print(
            f"Cumulative forget accuracy: "
            f"{metrics_cumulative['forget_accuracy']:.2f}%"
        )

        print(
            f"Retain accuracy: "
            f"{metrics_new['retain_accuracy']:.2f}%"
        )

        print(
            f"Test accuracy: "
            f"{metrics_new['test_accuracy']:.2f}%"
        )

        print(
            f"Parameter distance: "
            f"{metrics_new['parameter_distance']:.4f}"
        )

        print(
            f"Unlearning time: "
            f"{unlearning_time:.2f}s"
        )

        return {
            "round_id": round_id,
            "test_accuracy": (
                metrics_new["test_accuracy"]
            ),
        }


def run_sequential_gradient_ascent(
    device: torch.device,
    train_dataset,
    test_loader,
    baseline_checkpoint: str,
    rounds: int = 3,
    seed: int = 42,
):

    workload = generate_workload_a(
        dataset_size=len(
            train_dataset
        ),
        deletion_fraction=0.01,
        rounds=rounds,
        seed=seed,
    )

    experiment = SequentialGradientAscent(
        device=device,
        train_dataset=train_dataset,
        test_loader=test_loader,
        baseline_checkpoint=baseline_checkpoint,
    )

    return experiment.run(
        workload
    )