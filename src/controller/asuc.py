from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.controller.som import (
    HealthState,
    SOMMeasurement,
    SubspaceTracker,
    compute_request_gradient,
)
from src.controller.state_tracker import ASUCStateTracker, RoundDecisionLog
from src.data.cifar10 import get_indexed_subset
from src.unlearning.exact_retrain import ExactRetrainer
from src.unlearning.salun import SalUnUnlearner
from src.unlearning.ssd import SSDUnlearner
from src.verification.verifier import UnlearningVerifier, VerificationResult


@dataclass
class RequestProfile:
    cardinality: int
    cardinality_ratio: float
    class_entropy: float
    influence_norm: float
    is_class_concentrated: bool


class RequestProfiler:
    """
    Analyzes the profile of an incoming deletion request.
    Measures cardinality, class dispersion (entropy), and empirical influence.
    """

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def profile(
        self,
        forget_indices: Sequence[int],
        total_remaining: int,
        dataset: Dataset,
        gradient_norm: float = 0.0,
    ) -> RequestProfile:
        cardinality = len(forget_indices)
        ratio = cardinality / max(total_remaining, 1)

        # Class distribution and normalized entropy
        if hasattr(dataset, "targets"):
            targets = np.array(dataset.targets)[list(forget_indices)]
            class_counts = np.bincount(targets, minlength=self.num_classes)
            probs = class_counts / max(cardinality, 1)
            probs = probs[probs > 0]
            if len(probs) > 1 and self.num_classes > 1:
                entropy = -float(np.sum(probs * np.log(probs))) / np.log(self.num_classes)
            else:
                entropy = 0.0
        else:
            entropy = 1.0

        is_concentrated = entropy < 0.40

        return RequestProfile(
            cardinality=cardinality,
            cardinality_ratio=ratio,
            class_entropy=entropy,
            influence_norm=gradient_norm,
            is_class_concentrated=is_concentrated,
        )


class ASUCRouter:
    """
    Tiered unlearning router: maps (RequestProfile, HealthState) to an initial primitive.
    """

    def __init__(
        self,
        concentration_threshold: float = 0.40,
        large_request_ratio: float = 0.05,
    ):
        self.concentration_threshold = concentration_threshold
        self.large_request_ratio = large_request_ratio

    def route(
        self,
        profile: RequestProfile,
        health_state: HealthState,
    ) -> str:
        # If in RED state or class concentrated, SSD tends to collapse utility; use SalUn
        if health_state == HealthState.RED:
            return "SalUn"

        if profile.is_class_concentrated or profile.cardinality_ratio > self.large_request_ratio:
            return "SalUn"

        if health_state == HealthState.AMBER:
            return "SalUn"

        # GREEN state and balanced request -> Tier 1 (fast SSD)
        return "SSD"


class ASUCController:
    """
    Adaptive Sequential Unlearning Controller (ASUC).

    A closed-loop meta-controller that orchestrates sequential unlearning by:
    1. Profiling deletion requests.
    2. Monitoring subspace health via SubspaceTracker (SOM).
    3. Dynamically routing to the optimal unlearning primitive.
    4. Verifying unlearning & retention in a closed loop.
    5. Escalating along a tiered path (SSD -> SalUn -> Retrain) upon verification failure.
    """

    def __init__(
        self,
        device: torch.device,
        num_classes: int = 10,
        som_decay: float = 0.9,
        som_max_rank: int = 5,
        green_threshold: float = 0.4,
        red_threshold: float = 0.8,
        retrain_epochs: int = 20,
        batch_size: int = 128,
        num_workers: int = 2,
    ):
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.retrain_epochs = retrain_epochs

        self.profiler = RequestProfiler(num_classes=num_classes)
        self.som_tracker = SubspaceTracker(
            decay_factor=som_decay,
            max_rank=som_max_rank,
            green_threshold=green_threshold,
            red_threshold=red_threshold,
            device=device,
        )
        self.router = ASUCRouter()
        self.verifier = UnlearningVerifier(device=device)

        # Unlearning primitive instances
        self.ssd = SSDUnlearner(
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            dampening_constant=1.0,
            selection_weighting=10.0,
            lower_bound=0.0,
        )
        self.salun = SalUnUnlearner(
            device=device,
            epochs=1,
            learning_rate=1e-3,
            saliency_threshold_ratio=0.5,
            batch_size=batch_size,
            num_workers=num_workers,
            retain_weight=1.0,
        )
        self.retrainer = ExactRetrainer(
            device=device,
            num_classes=num_classes,
            epochs=retrain_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def _parameter_distance(self, model_a: nn.Module, model_b: nn.Module) -> float:
        dist_sq = 0.0
        with torch.no_grad():
            for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
                dist_sq += (p_a.detach() - p_b.detach()).pow(2).sum().item()
        return float(np.sqrt(dist_sq))

    def _parameter_norm(self, model: nn.Module) -> float:
        norm_sq = 0.0
        with torch.no_grad():
            for p in model.parameters():
                norm_sq += p.detach().pow(2).sum().item()
        return float(np.sqrt(norm_sq))

    def step(
        self,
        current_model: nn.Module,
        baseline_model: nn.Module,
        train_dataset: Dataset,
        forget_indices: Sequence[int],
        retain_indices: Sequence[int],
        round_id: int = 1,
        anchor_loader: Optional[DataLoader] = None,
    ) -> Tuple[nn.Module, RoundDecisionLog]:
        start_time = time.perf_counter()

        # Build forget loader
        forget_subset = get_indexed_subset(train_dataset, forget_indices)
        forget_loader = DataLoader(
            forget_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Build anchor loader for verification if not provided
        if anchor_loader is None:
            max_anchor = min(1000, len(retain_indices))
            rng = np.random.default_rng(42)
            anchor_idx = rng.choice(retain_indices, size=max_anchor, replace=False).tolist()
            anchor_subset = get_indexed_subset(train_dataset, anchor_idx)
            anchor_loader = DataLoader(
                anchor_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        # 1. Compute request gradient and measure SOM
        request_grad = compute_request_gradient(current_model, forget_loader, self.device)
        if request_grad is not None:
            som_measurement = self.som_tracker.update(request_grad)
        else:
            som_measurement = SOMMeasurement(
                som_score=0.0,
                health_state=HealthState.GREEN,
                rank=0,
                gradient_norm=0.0,
                basis_norm=0.0,
            )

        # 2. Profile request
        profile = self.profiler.profile(
            forget_indices=forget_indices,
            total_remaining=len(retain_indices),
            dataset=train_dataset,
            gradient_norm=som_measurement.gradient_norm,
        )

        # 3. Dynamic Routing
        initial_primitive = self.router.route(profile, som_measurement.health_state)
        active_primitive = initial_primitive
        escalation_path = [initial_primitive]

        # 4. Execute Unlearning & Verify with Escalation
        def execute_primitive(name: str) -> Tuple[nn.Module, float]:
            t0 = time.perf_counter()
            if name == "SSD":
                res = self.ssd.unlearn(current_model, train_dataset, forget_indices)
                return res.model, time.perf_counter() - t0
            elif name == "SalUn":
                res = self.salun.unlearn(current_model, train_dataset, forget_indices, retain_indices)
                return res.model, time.perf_counter() - t0
            elif name == "Retrain":
                # Retrain from scratch on retain_indices
                res = self.retrainer.train(train_dataset, retain_indices)
                return res.model, time.perf_counter() - t0
            else:
                raise ValueError(f"Unknown primitive: {name}")

        unlearned_model, unlearn_time = execute_primitive(active_primitive)
        verification = self.verifier.verify(current_model, unlearned_model, forget_loader, anchor_loader)

        # Controlled Escalation Loop
        if not verification.passed:
            if active_primitive == "SSD":
                active_primitive = "SalUn"
                escalation_path.append("SalUn")
                unlearned_model, t_sub = execute_primitive(active_primitive)
                unlearn_time += t_sub
                verification = self.verifier.verify(current_model, unlearned_model, forget_loader, anchor_loader)

            if not verification.passed and active_primitive == "SalUn":
                active_primitive = "Retrain"
                escalation_path.append("Retrain")
                unlearned_model, t_sub = execute_primitive(active_primitive)
                unlearn_time += t_sub
                verification = self.verifier.verify(current_model, unlearned_model, forget_loader, anchor_loader)

        # 5. Model diagnostics
        param_norm = self._parameter_norm(unlearned_model)
        param_dist = self._parameter_distance(unlearned_model, baseline_model)
        base_norm = self._parameter_norm(baseline_model)
        rel_dist = param_dist / max(base_norm, 1e-12)

        decision_log = RoundDecisionLog(
            round_id=round_id,
            requested_samples=len(forget_indices),
            cumulative_forgotten=0,  # caller updates
            remaining_samples=len(retain_indices),
            som_score=som_measurement.som_score,
            health_state=som_measurement.health_state.value,
            chosen_primitive=active_primitive,
            escalation_path=escalation_path,
            verification_passed=verification.passed,
            verification_reason=verification.reason,
            unlearning_time_seconds=unlearn_time,
            verification_time_seconds=verification.elapsed_time_seconds,
            parameter_norm=param_norm,
            parameter_distance_baseline=param_dist,
            relative_parameter_distance=rel_dist,
            extra_metrics={
                "forget_accuracy": verification.forget_accuracy,
                "retain_accuracy": verification.retain_accuracy,
                "forget_loss": verification.forget_loss,
                "retain_loss": verification.retain_loss,
                "class_entropy": profile.class_entropy,
            },
        )

        return unlearned_model, decision_log
