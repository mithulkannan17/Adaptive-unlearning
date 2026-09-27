from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_utility(
    model: nn.Module,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    baseline_test_acc: Optional[float] = None,
) -> Dict[str, float]:
    """
    Evaluate retain-set and test-set generalization performance.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    def _eval_loader(loader: DataLoader) -> tuple[float, float]:
        tot_loss = 0.0
        tot_correct = 0
        tot_count = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tot_loss += loss.item()
                preds = outputs.argmax(dim=1)
                tot_correct += (preds == targets).sum().item()
                tot_count += inputs.size(0)
        if tot_count == 0:
            return 0.0, 0.0
        return tot_loss / tot_count, 100.0 * tot_correct / tot_count

    retain_loss, retain_acc = _eval_loader(retain_loader)
    test_loss, test_acc = _eval_loader(test_loader)

    results = {
        "retain_loss": retain_loss,
        "retain_accuracy": retain_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }

    if baseline_test_acc is not None:
        results["test_accuracy_drop"] = max(0.0, baseline_test_acc - test_acc)
        results["utility_retention_ratio"] = test_acc / max(baseline_test_acc, 1e-8)

    return results
