from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


def compute_sample_losses_and_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses = []
    confs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
            probs = torch.softmax(outputs, dim=1)
            batch_conf = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            losses.append(batch_loss.detach().cpu().numpy())
            confs.append(batch_conf.detach().cpu().numpy())

    return np.concatenate(losses), np.concatenate(confs)


def evaluate_forgetting_efficacy(
    model: nn.Module,
    forget_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate forgetting quality including loss, accuracy, and MIA risk score.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    forget_loss = 0.0
    forget_correct = 0
    forget_total = 0

    with torch.no_grad():
        for inputs, targets in forget_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            forget_loss += loss.item()
            preds = outputs.argmax(dim=1)
            forget_correct += (preds == targets).sum().item()
            forget_total += inputs.size(0)

    avg_loss = forget_loss / max(forget_total, 1)
    acc = 100.0 * forget_correct / max(forget_total, 1)

    # Compute lightweight Membership Inference Attack (MIA) score
    # Forget samples (label 1: member) vs Test samples (label 0: non-member)
    forget_losses, _ = compute_sample_losses_and_probs(model, forget_loader, device)
    test_losses, _ = compute_sample_losses_and_probs(model, test_loader, device)

    # Subsample test losses to match forget size for balanced comparison
    n_samples = min(len(forget_losses), len(test_losses))
    rng = np.random.default_rng(42)
    sub_test = rng.choice(test_losses, size=n_samples, replace=False)
    sub_forget = rng.choice(forget_losses, size=n_samples, replace=False)

    y_true = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])
    # Lower loss indicates higher membership probability
    y_scores = np.concatenate([-sub_forget, -sub_test])

    try:
        mia_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        mia_auc = 0.50

    return {
        "forget_loss": avg_loss,
        "forget_accuracy": acc,
        "mia_auc_score": mia_auc,
        "mia_privacy_leakage": max(0.0, (mia_auc - 0.5) * 2.0),
    }
