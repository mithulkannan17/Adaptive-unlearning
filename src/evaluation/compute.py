from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.cifar10 import get_indexed_subset
from src.evaluation.forgetting import evaluate_forgetting_efficacy
from src.evaluation.stability import compute_stability_metrics
from src.evaluation.utility import evaluate_utility


def compute_comprehensive_evaluation(
    model: nn.Module,
    baseline_model: nn.Module,
    train_dataset: Dataset,
    test_loader: DataLoader,
    forget_indices: Sequence[int],
    retain_indices: Sequence[int],
    device: torch.device,
    baseline_test_acc: Optional[float] = None,
    unlearning_time_seconds: float = 0.0,
    exact_retrain_time_seconds: Optional[float] = None,
    batch_size: int = 128,
    num_workers: int = 2,
) -> Dict[str, float]:
    """
    Run the unified evaluation suite across all 4 pillars of unlearning:
    1. Forgetting efficacy (forget loss, forget acc, MIA AUC)
    2. Utility preservation (retain acc, test acc, utility retention)
    3. Parameter stability (parameter distance, relative distance, norm)
    4. Compute efficiency (unlearning time, speedup vs exact retrain)
    """
    start_eval = time.perf_counter()

    forget_subset = get_indexed_subset(train_dataset, forget_indices)
    forget_loader = DataLoader(
        forget_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    retain_subset = get_indexed_subset(train_dataset, retain_indices)
    retain_loader = DataLoader(
        retain_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    forget_metrics = evaluate_forgetting_efficacy(model, forget_loader, test_loader, device)
    utility_metrics = evaluate_utility(model, retain_loader, test_loader, device, baseline_test_acc)
    stability_metrics = compute_stability_metrics(model, baseline_model)

    report = {
        **forget_metrics,
        **utility_metrics,
        **stability_metrics,
        "forget_size": float(len(forget_indices)),
        "retain_size": float(len(retain_indices)),
        "unlearning_time_seconds": unlearning_time_seconds,
        "evaluation_time_seconds": time.perf_counter() - start_eval,
    }

    if exact_retrain_time_seconds is not None and unlearning_time_seconds > 0:
        report["speedup_factor"] = exact_retrain_time_seconds / unlearning_time_seconds

    return report
