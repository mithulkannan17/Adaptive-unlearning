from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(
        self,
        train_loader,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.perf_counter()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{total_epochs}",
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

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                targets,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_time = time.perf_counter() - start_time

        average_loss = running_loss / total
        accuracy = 100.0 * correct / total

        return {
            "loss": average_loss,
            "accuracy": accuracy,
            "time_seconds": epoch_time,
        }

    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
    ) -> Dict[str, float]:

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in data_loader:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                targets,
            )

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

        average_loss = running_loss / total
        accuracy = 100.0 * correct / total

        return {
            "loss": average_loss,
            "accuracy": accuracy,
        }