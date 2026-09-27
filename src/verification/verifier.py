from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class VerificationResult:
    passed: bool
    forget_accuracy: float
    retain_accuracy: float
    forget_loss: float
    retain_loss: float
    accuracy_drop: float
    loss_increase_ratio: float
    elapsed_time_seconds: float
    reason: str


class UnlearningVerifier:
    """
    Closed-loop verification module for sequential unlearning.

    Verifies two essential properties:
    1. Forget Efficacy: Model performance on the forget set has degraded.
    2. Retain Integrity: Model performance on retain/anchor data has not collapsed.
    """

    def __init__(
        self,
        device: torch.device,
        max_acceptable_forget_accuracy: float = 50.0,
        max_allowed_retain_drop: float = 7.0,
        max_loss_increase_ratio: float = 3.0,
    ):
        self.device = device
        self.max_acceptable_forget_accuracy = max_acceptable_forget_accuracy
        self.max_allowed_retain_drop = max_allowed_retain_drop
        self.max_loss_increase_ratio = max_loss_increase_ratio
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _evaluate_loader(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

        if total == 0:
            return 0.0, 0.0
        return total_loss / total, 100.0 * correct / total

    def verify(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        forget_loader: DataLoader,
        retain_or_anchor_loader: DataLoader,
    ) -> VerificationResult:
        start_time = time.perf_counter()

        # Before unlearning metrics
        loss_before_retain, acc_before_retain = self._evaluate_loader(
            model_before, retain_or_anchor_loader
        )

        # After unlearning metrics
        loss_after_forget, acc_after_forget = self._evaluate_loader(
            model_after, forget_loader
        )
        loss_after_retain, acc_after_retain = self._evaluate_loader(
            model_after, retain_or_anchor_loader
        )

        retain_drop = acc_before_retain - acc_after_retain
        loss_ratio = (
            loss_after_retain / max(loss_before_retain, 1e-8)
            if loss_before_retain > 0
            else 1.0
        )

        # Check conditions
        passed = True
        reasons = []

        if acc_after_forget > self.max_acceptable_forget_accuracy:
            passed = False
            reasons.append(
                f"Forget accuracy {acc_after_forget:.2f}% exceeds threshold {self.max_acceptable_forget_accuracy:.2f}%"
            )

        if retain_drop > self.max_allowed_retain_drop:
            passed = False
            reasons.append(
                f"Retain accuracy drop {retain_drop:.2f}% exceeds threshold {self.max_allowed_retain_drop:.2f}%"
            )

        if loss_ratio > self.max_loss_increase_ratio:
            passed = False
            reasons.append(
                f"Retain loss ratio {loss_ratio:.2f}x exceeds allowed {self.max_loss_increase_ratio:.2f}x"
            )

        elapsed = time.perf_counter() - start_time
        summary_reason = "; ".join(reasons) if not passed else "Verification passed: forget verified and retain preserved."

        return VerificationResult(
            passed=passed,
            forget_accuracy=acc_after_forget,
            retain_accuracy=acc_after_retain,
            forget_loss=loss_after_forget,
            retain_loss=loss_after_retain,
            accuracy_drop=retain_drop,
            loss_increase_ratio=loss_ratio,
            elapsed_time_seconds=elapsed,
            reason=summary_reason,
        )
