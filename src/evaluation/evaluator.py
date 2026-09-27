from __future__ import annotations

import time
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class UnlearningEvaluator:
    """
    Common evaluation interface for all unlearning methods.

    Measures:
        - overall test performance
        - forget-set performance
        - retain-set performance
        - parameter distance
        - parameter norm
        - evaluation time
    """

    def __init__(
        self,
        device: torch.device,
        batch_size: int = 128,
        num_workers: int = 2,
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(
        self,
        model: nn.Module,
        train_dataset,
        test_loader: DataLoader,
        forget_indices: Sequence[int],
        retain_indices: Sequence[int],
        reference_model: nn.Module | None = None,
    ) -> Dict[str, float]:

        start = time.perf_counter()

        forget_loader = self._make_loader(
            train_dataset,
            forget_indices,
            shuffle=False,
        )

        retain_loader = self._make_loader(
            train_dataset,
            retain_indices,
            shuffle=False,
        )

        forget_metrics = self._evaluate_loader(
            model,
            forget_loader,
        )

        retain_metrics = self._evaluate_loader(
            model,
            retain_loader,
        )

        test_metrics = self._evaluate_loader(
            model,
            test_loader,
        )

        results = {
            "forget_loss": forget_metrics["loss"],
            "forget_accuracy": forget_metrics["accuracy"],
            "retain_loss": retain_metrics["loss"],
            "retain_accuracy": retain_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "forget_size": float(len(forget_indices)),
            "retain_size": float(len(retain_indices)),
        }

        results["parameter_norm"] = (
            self.parameter_norm(model)
        )

        if reference_model is not None:
            results["parameter_distance"] = (
                self.parameter_distance(
                    model,
                    reference_model,
                )
            )
        else:
            results["parameter_distance"] = 0.0

        results["evaluation_time_seconds"] = (
            time.perf_counter() - start
        )

        return results

    def _make_loader(
        self,
        dataset,
        indices: Sequence[int],
        shuffle: bool,
    ) -> DataLoader:

        subset = Subset(
            dataset,
            list(indices),
        )

        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    @torch.no_grad()
    def _evaluate_loader(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Dict[str, float]:

        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets in loader:

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

            batch_size = images.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += batch_size

        if total == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
            }

        return {
            "loss": total_loss / total,
            "accuracy": (
                100.0 * correct / total
            ),
        }

    @staticmethod
    @torch.no_grad()
    def parameter_norm(
        model: nn.Module,
    ) -> float:

        squared_norm = 0.0

        for parameter in model.parameters():

            squared_norm += (
                parameter.detach()
                .float()
                .pow(2)
                .sum()
                .item()
            )

        return squared_norm ** 0.5

    @staticmethod
    @torch.no_grad()
    def parameter_distance(
        model_a: nn.Module,
        model_b: nn.Module,
    ) -> float:

        total_squared_distance = 0.0

        parameters_a = list(
            model_a.parameters()
        )

        parameters_b = list(
            model_b.parameters()
        )

        if len(parameters_a) != len(parameters_b):
            raise ValueError(
                "Models have different numbers "
                "of parameters."
            )

        for parameter_a, parameter_b in zip(
            parameters_a,
            parameters_b,
        ):

            difference = (
                parameter_a.detach().float()
                - parameter_b.detach().float()
            )

            total_squared_distance += (
                difference.pow(2)
                .sum()
                .item()
            )

        return total_squared_distance ** 0.5