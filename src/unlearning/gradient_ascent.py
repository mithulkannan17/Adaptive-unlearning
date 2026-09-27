from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


@dataclass
class GradientAscentResult:
    model: nn.Module
    initial_forget_loss: float
    final_forget_loss: float
    initial_forget_accuracy: float
    final_forget_accuracy: float
    initial_parameter_norm: float
    final_parameter_norm: float
    training_time_seconds: float
    epochs: int


class GradientAscentUnlearner:
    """
    Gradient Ascent machine-unlearning baseline.

    The model is updated using the forget set only.

    Standard training minimizes:

        L(theta)

    Gradient ascent instead maximizes the forget-set loss:

        theta <- theta + eta * grad_theta L_f(theta)

    The intention is to make the model perform poorly on
    the samples that should be forgotten.
    """

    def __init__(
        self,
        device: torch.device,
        epochs: int = 5,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        num_workers: int = 2,
    ):
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.criterion = nn.CrossEntropyLoss()

    def unlearn(
        self,
        model: nn.Module,
        train_dataset,
        forget_indices: Sequence[int],
    ) -> GradientAscentResult:
        """
        Apply gradient ascent to the supplied model.

        Parameters
        ----------
        model:
            Current model from which unlearning begins.

        train_dataset:
            Full CIFAR-10 training dataset.

        forget_indices:
            Original dataset indices that must be forgotten.

        Returns
        -------
        GradientAscentResult
            Updated model and diagnostic metrics.
        """

        forget_indices = list(
            map(int, forget_indices)
        )

        if not forget_indices:
            raise ValueError(
                "forget_indices cannot be empty."
            )

        model = model.to(self.device)

        forget_dataset = Subset(
            train_dataset,
            forget_indices,
        )

        forget_loader = DataLoader(
            forget_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

        initial_metrics = self._evaluate_forget_set(
            model,
            forget_loader,
        )

        initial_parameter_norm = (
            self._parameter_norm(model)
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
        )

        start_time = time.perf_counter()

        for epoch in range(1, self.epochs + 1):

            model.train()

            progress = tqdm(
                forget_loader,
                desc=(
                    f"Gradient Ascent "
                    f"{epoch}/{self.epochs}"
                ),
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

                outputs = model(images)

                loss = self.criterion(
                    outputs,
                    targets,
                )

                # IMPORTANT:
                #
                # Normal training:
                #
                # optimizer.step()
                #
                # minimizes the loss.
                #
                # We want to maximize the forget loss,
                # therefore explicitly reverse the gradients.
                loss.backward()

                with torch.no_grad():
                    for parameter in model.parameters():

                        if parameter.grad is not None:
                            parameter.grad.mul_(-1.0)

                optimizer.step()

                progress.set_postfix(
                    forget_loss=f"{loss.item():.4f}"
                )

        training_time = (
            time.perf_counter()
            - start_time
        )

        final_metrics = self._evaluate_forget_set(
            model,
            forget_loader,
        )

        final_parameter_norm = (
            self._parameter_norm(model)
        )

        return GradientAscentResult(
            model=model,
            initial_forget_loss=(
                initial_metrics["loss"]
            ),
            final_forget_loss=(
                final_metrics["loss"]
            ),
            initial_forget_accuracy=(
                initial_metrics["accuracy"]
            ),
            final_forget_accuracy=(
                final_metrics["accuracy"]
            ),
            initial_parameter_norm=(
                initial_parameter_norm
            ),
            final_parameter_norm=(
                final_parameter_norm
            ),
            training_time_seconds=training_time,
            epochs=self.epochs,
        )

    @torch.no_grad()
    def _evaluate_forget_set(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
    ) -> Dict[str, float]:

        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets in forget_loader:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = self.criterion(
                outputs,
                targets,
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

        return {
            "loss": total_loss / total,
            "accuracy": (
                100.0 * correct / total
            ),
        }

    @staticmethod
    def _parameter_norm(
        model: nn.Module,
    ) -> float:

        total_squared_norm = 0.0

        with torch.no_grad():

            for parameter in model.parameters():

                total_squared_norm += (
                    parameter.detach()
                    .float()
                    .pow(2)
                    .sum()
                    .item()
                )

        return total_squared_norm ** 0.5