from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.cifar10 import get_indexed_subset, get_train_loader


@dataclass
class SalUnResult:
    model: nn.Module
    unlearning_time_seconds: float
    epochs: int
    salient_parameter_count: int
    total_parameter_count: int
    saliency_ratio: float
    initial_forget_loss: float
    final_forget_loss: float
    initial_forget_accuracy: float
    final_forget_accuracy: float


class SalUnUnlearner:
    """
    Saliency Unlearning (SalUn).

    Computes a saliency mask isolating the parameters most responsible
    for the forget set, and restricts gradient updates exclusively to those
    salient weights, preserving non-salient utility across sequential rounds.
    """

    def __init__(
        self,
        device: torch.device,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        saliency_threshold_ratio: float = 0.5,
        batch_size: int = 128,
        num_workers: int = 2,
        retain_weight: float = 1.0,
    ):
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 < saliency_threshold_ratio <= 1.0:
            raise ValueError("saliency_threshold_ratio must be in (0, 1].")

        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.saliency_threshold_ratio = saliency_threshold_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.retain_weight = retain_weight
        self.criterion = nn.CrossEntropyLoss()

    def _compute_saliency_mask(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient magnitude for each parameter on forget set,
        and threshold to top saliency_threshold_ratio fraction.
        """
        model.eval()
        model.zero_grad(set_to_none=True)

        accumulated_grads: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                accumulated_grads[name] = torch.zeros_like(param.data)

        total_samples = 0
        for inputs, targets in forget_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        accumulated_grads[name] += param.grad.abs() * batch_size

            total_samples += batch_size
            model.zero_grad(set_to_none=True)

        if total_samples > 0:
            for name in accumulated_grads:
                accumulated_grads[name] /= total_samples

        # Collect all saliency scores to find global threshold
        all_scores = torch.cat(
            [grad.flatten() for grad in accumulated_grads.values()]
        )

        k = int(self.saliency_threshold_ratio * all_scores.numel())
        k = max(1, min(k, all_scores.numel()))
        threshold = torch.kthvalue(all_scores, all_scores.numel() - k + 1).values.item()

        # Build binary masks
        masks: Dict[str, torch.Tensor] = {}
        for name, grad in accumulated_grads.items():
            masks[name] = (grad >= threshold).float().to(self.device)

        return masks

    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
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

    def unlearn(
        self,
        model: nn.Module,
        train_dataset,
        forget_indices: Sequence[int],
        retain_indices: Optional[Sequence[int]] = None,
    ) -> SalUnResult:
        start_time = time.perf_counter()

        if len(forget_indices) == 0:
            raise ValueError("forget_indices cannot be empty.")

        unlearned_model = copy.deepcopy(model).to(self.device)

        forget_subset = get_indexed_subset(train_dataset, forget_indices)
        forget_loader = DataLoader(
            forget_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        init_loss, init_acc = self._evaluate(unlearned_model, forget_loader)

        # 1. Compute Saliency Mask
        masks = self._compute_saliency_mask(unlearned_model, forget_loader)
        salient_params = sum(int(mask.sum().item()) for mask in masks.values())
        total_params = sum(mask.numel() for mask in masks.values())

        # 2. Retain loader (if available) for utility regularization
        retain_loader = None
        if retain_indices is not None and len(retain_indices) > 0:
            # Sample a manageable subset of retain samples for efficiency
            max_retain = min(len(retain_indices), len(forget_indices) * 4)
            rng = torch.Generator().manual_seed(42)
            shuffled = torch.randperm(len(retain_indices), generator=rng).tolist()
            sampled_retain_indices = [retain_indices[i] for i in shuffled[:max_retain]]
            retain_subset = get_indexed_subset(train_dataset, sampled_retain_indices)
            retain_loader = DataLoader(
                retain_subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        # 3. Masked Unlearning Optimization
        optimizer = torch.optim.SGD(
            unlearned_model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )

        # Freeze BatchNorm running statistics to prevent representation collapse
        unlearned_model.eval()
        for epoch in range(self.epochs):
            retain_iter = iter(retain_loader) if retain_loader is not None else None

            for forget_inputs, forget_targets in forget_loader:
                forget_inputs = forget_inputs.to(self.device, non_blocking=True)
                forget_targets = forget_targets.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # Random labeling / bad teaching on forget set
                num_classes = 10
                random_targets = (forget_targets + torch.randint(1, num_classes, forget_targets.shape, device=self.device)) % num_classes

                outputs_forget = unlearned_model(forget_inputs)
                loss_forget = self.criterion(outputs_forget, random_targets)

                loss = loss_forget

                if retain_iter is not None:
                    try:
                        ret_inputs, ret_targets = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        ret_inputs, ret_targets = next(retain_iter)

                    ret_inputs = ret_inputs.to(self.device, non_blocking=True)
                    ret_targets = ret_targets.to(self.device, non_blocking=True)
                    outputs_retain = unlearned_model(ret_inputs)
                    loss_retain = self.criterion(outputs_retain, ret_targets)
                    loss = loss + self.retain_weight * loss_retain

                loss.backward()

                # Apply saliency mask to gradients
                with torch.no_grad():
                    for name, param in unlearned_model.named_parameters():
                        if param.requires_grad and param.grad is not None and name in masks:
                            param.grad.mul_(masks[name])

                optimizer.step()

        final_loss, final_acc = self._evaluate(unlearned_model, forget_loader)
        unlearned_model.eval()

        elapsed = time.perf_counter() - start_time

        return SalUnResult(
            model=unlearned_model,
            unlearning_time_seconds=elapsed,
            epochs=self.epochs,
            salient_parameter_count=salient_params,
            total_parameter_count=total_params,
            saliency_ratio=salient_params / max(total_params, 1),
            initial_forget_loss=init_loss,
            final_forget_loss=final_loss,
            initial_forget_accuracy=init_acc,
            final_forget_accuracy=final_acc,
        )
