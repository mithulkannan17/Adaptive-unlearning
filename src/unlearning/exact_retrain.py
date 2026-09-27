from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models.resnet import create_resnet18
from src.utils import save_checkpoint


class ExactRetrainer:
    """
    Exact counterfactual retraining engine.

    For a given forget set Df, the model is initialized from scratch
    and trained only on the retain set Dr = D \\ Df.
    """

    def __init__(
        self,
        device: torch.device,
        num_classes: int = 10,
        epochs: int = 30,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        batch_size: int = 128,
        num_workers: int = 2,
    ):
        self.device = device
        self.num_classes = num_classes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_model(self) -> nn.Module:

        model = create_resnet18(
            num_classes=self.num_classes
        )

        return model.to(self.device)

    def _create_optimizer(
        self,
        model: nn.Module,
    ) -> torch.optim.Optimizer:

        return torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def train(
        self,
        train_dataset,
        test_loader: DataLoader,
        retain_indices: Sequence[int],
        checkpoint_path: str | Path | None = None,
    ) -> Dict:

        retain_indices = list(retain_indices)

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

        model = self._create_model()

        optimizer = self._create_optimizer(model)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
        )

        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0.0
        best_epoch = 0

        history = []

        total_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            progress = tqdm(
                train_loader,
                desc=f"Exact Retrain {epoch}/{self.epochs}",
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

                loss = criterion(
                    outputs,
                    targets,
                )

                loss.backward()

                optimizer.step()

                running_loss += (
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

            scheduler.step()

            train_loss = running_loss / total
            train_accuracy = (
                100.0 * correct / total
            )

            test_loss, test_accuracy = self._evaluate(
                model,
                test_loader,
                criterion,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }
            )

            print(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"Train Acc: {train_accuracy:.2f}% | "
                f"Test Acc: {test_accuracy:.2f}%"
            )

            if test_accuracy > best_accuracy:

                best_accuracy = test_accuracy
                best_epoch = epoch

                if checkpoint_path is not None:

                    save_checkpoint(
                        path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics={
                            "test_accuracy": test_accuracy,
                            "test_loss": test_loss,
                        },
                    )

        total_time = (
            time.perf_counter()
            - total_start
        )

        return {
            "model": model,
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "total_time_seconds": total_time,
            "retain_size": len(retain_indices),
            "history": history,
        }

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
    ):

        model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in test_loader:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                targets,
            )

            running_loss += (
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

        return (
            running_loss / total,
            100.0 * correct / total,
        )