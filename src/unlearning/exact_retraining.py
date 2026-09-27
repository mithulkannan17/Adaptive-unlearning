from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


@dataclass
class ExactRetrainingResult:
    model: nn.Module
    training_time_seconds: float
    epochs: int
    final_train_loss: float
    final_train_accuracy: float


class ExactRetrainer:
    """
    Exact retraining counterfactual for machine unlearning.

    Given a deletion request D_f, the model is trained from
    scratch using only:

        D_retain = D_train \\ D_f

    This provides the reference model that an approximate
    unlearning method should ideally approach.
    """

    def __init__(
        self,
        device: torch.device,
        epochs: int = 30,
        learning_rate: float = 0.1,
        batch_size: int = 128,
        num_workers: int = 2,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
    ):
        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.criterion = nn.CrossEntropyLoss()

    def retrain(
        self,
        model: nn.Module,
        train_dataset,
        forget_indices: Sequence[int],
    ) -> ExactRetrainingResult:

        forget_set = set(
            int(index)
            for index in forget_indices
        )

        if not forget_set:
            raise ValueError(
                "forget_indices cannot be empty."
            )

        dataset_size = len(train_dataset)

        invalid_indices = [
            index
            for index in forget_set
            if index < 0 or index >= dataset_size
        ]

        if invalid_indices:
            raise ValueError(
                "Invalid dataset indices: "
                f"{invalid_indices[:10]}"
            )

        retain_indices = [
            index
            for index in range(dataset_size)
            if index not in forget_set
        ]

        retain_dataset = Subset(
            train_dataset,
            retain_indices,
        )

        train_loader = DataLoader(
            retain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Exact retraining starts from a NEW model.
        #
        # We must NOT start from the already-trained baseline.
        # -----------------------------------------------------

        retrained_model = copy.deepcopy(
            model
        )

        # Remove learned parameters by reinitializing
        # every layer that has a reset_parameters method.
        self._reset_model(
            retrained_model
        )

        retrained_model = (
            retrained_model.to(self.device)
        )

        optimizer = torch.optim.SGD(
            retrained_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.epochs * 0.5),
                int(self.epochs * 0.75),
            ],
            gamma=0.1,
        )

        start_time = time.perf_counter()

        final_loss = 0.0
        final_accuracy = 0.0

        print()
        print("=" * 70)
        print("EXACT RETRAINING")
        print("=" * 70)

        print(
            f"Original training samples: "
            f"{dataset_size}"
        )

        print(
            f"Forgotten samples: "
            f"{len(forget_set)}"
        )

        print(
            f"Retained training samples: "
            f"{len(retain_indices)}"
        )

        print(
            f"Epochs: {self.epochs}"
        )

        print("=" * 70)

        for epoch in range(1, self.epochs + 1):

            retrained_model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            progress = tqdm(
                train_loader,
                desc=f"Retrain {epoch}/{self.epochs}",
                leave=False,
            )

            for images, targets in progress:

                images = images.to(
                    self.device,
                    non_blocking=True,
                )

                targets = targets.to(
                    self.device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                outputs = retrained_model(
                    images
                )

                loss = self.criterion(
                    outputs,
                    targets,
                )

                loss.backward()

                optimizer.step()

                batch_size = images.size(0)

                running_loss += (
                    loss.item()
                    * batch_size
                )

                predictions = outputs.argmax(
                    dim=1
                )

                correct += (
                    predictions == targets
                ).sum().item()

                total += batch_size

                progress.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

            scheduler.step()

            final_loss = (
                running_loss / total
            )

            final_accuracy = (
                100.0 * correct / total
            )

            print(
                f"Epoch {epoch:02d}/{self.epochs} | "
                f"Loss: {final_loss:.4f} | "
                f"Retain Train Acc: "
                f"{final_accuracy:.2f}%"
            )

        training_time = (
            time.perf_counter()
            - start_time
        )

        return ExactRetrainingResult(
            model=retrained_model,
            training_time_seconds=training_time,
            epochs=self.epochs,
            final_train_loss=final_loss,
            final_train_accuracy=final_accuracy,
        )

    @staticmethod
    def _reset_model(
        model: nn.Module,
    ) -> None:

        def reset(module):

            if hasattr(
                module,
                "reset_parameters",
            ):
                module.reset_parameters()

        model.apply(reset)