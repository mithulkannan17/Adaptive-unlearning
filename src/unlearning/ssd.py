
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# ============================================================
# RESULT
# ============================================================

@dataclass
class SSDResult:
    model: nn.Module
    unlearning_time: float
    forget_samples: int

    selected_parameters: int
    total_parameters: int
    selection_ratio: float

    mean_dampening_factor: float
    min_dampening_factor: float
    max_dampening_factor: float

    changed_parameters: int


# ============================================================
# SSD UNLEARNER
# ============================================================

class SSDUnlearner:
    """
    Selective Synaptic Dampening (SSD).

    SSD estimates parameter importance on:

        D  = full training dataset
        Df = forget dataset

    Parameters that are disproportionately important to the
    forget set are selectively dampened.

    Importance:

        I(theta) = mean(gradient(theta)^2)

    Parameter selection:

        I_forget(theta) >
            selection_weighting * I_full(theta)

    Dampening factor:

        factor =
            (
                dampening_constant * I_full
                /
                (I_forget + epsilon)
            ) ** exponent

    The factor is constrained to:

        lower_bound <= factor <= 1.0

    Therefore SSD never increases a parameter magnitude.
    """

    def __init__(
        self,
        device: torch.device,
        batch_size: int = 128,
        num_workers: int = 2,
        dampening_constant: float = 1.0,
        selection_weighting: float = 10.0,
        exponent: float = 1.0,
        lower_bound: float = 0.0,
        alpha: float | None = None,
        lambda_: float | None = None,
    ):
        self.device = device

        self.batch_size = batch_size
        self.num_workers = num_workers

        if alpha is not None:
            dampening_constant = alpha
        if lambda_ is not None:
            selection_weighting = lambda_

        self.dampening_constant = float(
            dampening_constant
        )

        self.selection_weighting = float(
            selection_weighting
        )

        self.exponent = float(
            exponent
        )

        self.lower_bound = float(
            lower_bound
        )

        if self.dampening_constant <= 0:
            raise ValueError(
                "dampening_constant must be > 0."
            )

        if self.selection_weighting <= 0:
            raise ValueError(
                "selection_weighting must be > 0."
            )

        if self.exponent <= 0:
            raise ValueError(
                "exponent must be > 0."
            )

        if not 0.0 <= self.lower_bound <= 1.0:
            raise ValueError(
                "lower_bound must be between 0.0 and 1.0."
            )

        # -----------------------------------------------------
        # Diagnostics
        # -----------------------------------------------------

        self.selected_parameters = 0
        self.total_parameters = 0
        self.selection_ratio = 0.0

        self.mean_dampening_factor = 1.0
        self.min_dampening_factor = 1.0
        self.max_dampening_factor = 1.0

        self.changed_parameters = 0

    # =========================================================
    # PUBLIC API
    # =========================================================

    def unlearn(
        self,
        model: nn.Module,
        train_dataset,
        forget_indices: Sequence[int],
    ) -> SSDResult:

        start_time = time.perf_counter()

        if len(forget_indices) == 0:
            raise ValueError(
                "forget_indices cannot be empty."
            )

        # -----------------------------------------------------
        # IMPORTANT: IN-PLACE STATE PROPAGATION
        #
        # The sequential unlearning engine passes the current model
        # from one round to the next. Therefore SSD MUST modify the
        # supplied model itself.
        #
        # The previous implementation used copy.deepcopy(model),
        # which meant that SSD modified a private copy and returned
        # it, while the sequential engine could continue holding the
        # original model reference. This produced the diagnostic:
        #
        #   selected_parameters > 0
        #   actual model distance = 0
        #
        # We intentionally operate on the supplied model in-place.
        # Callers that need isolation should create an independent
        # model before calling unlearn().
        # -----------------------------------------------------

        model = model.to(self.device)
        model.eval()

        forget_indices = [
            int(index)
            for index in forget_indices
        ]

        # -----------------------------------------------------
        # FULL DATASET
        # -----------------------------------------------------

        full_indices = list(
            range(
                len(train_dataset)
            )
        )

        full_loader = self._make_loader(
            train_dataset,
            full_indices,
        )

        # -----------------------------------------------------
        # FORGET DATASET
        # -----------------------------------------------------

        forget_loader = self._make_loader(
            train_dataset,
            forget_indices,
        )

        print()
        print(
            "Computing SSD importance..."
        )

        print(
            f"Full dataset: "
            f"{len(full_indices)} samples"
        )

        print(
            f"Forget dataset: "
            f"{len(forget_indices)} samples"
        )

        # -----------------------------------------------------
        # COMPUTE FULL-DATA IMPORTANCE (WITH CACHING)
        # -----------------------------------------------------

        if getattr(self, "_cached_full_importance", None) is not None:
            full_importance = self._cached_full_importance
        else:
            full_importance = (
                self._compute_importance(
                    model=model,
                    loader=full_loader,
                )
            )
            self._cached_full_importance = full_importance

        # -----------------------------------------------------
        # COMPUTE FORGET-DATA IMPORTANCE
        # -----------------------------------------------------

        forget_importance = (
            self._compute_importance(
                model=model,
                loader=forget_loader,
            )
        )

        # -----------------------------------------------------
        # APPLY SSD
        # -----------------------------------------------------

        (
            selected_parameters,
            total_parameters,
            mean_factor,
            min_factor,
            max_factor,
            changed_parameters,
        ) = self._apply_ssd(
            model=model,
            full_importance=full_importance,
            forget_importance=forget_importance,
        )

        model.eval()

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        selection_ratio = (
            selected_parameters
            /
            max(total_parameters, 1)
        )

        # -----------------------------------------------------
        # STORE DIAGNOSTICS
        # -----------------------------------------------------

        self.selected_parameters = (
            selected_parameters
        )

        self.total_parameters = (
            total_parameters
        )

        self.selection_ratio = (
            selection_ratio
        )

        self.mean_dampening_factor = (
            mean_factor
        )

        self.min_dampening_factor = (
            min_factor
        )

        self.max_dampening_factor = (
            max_factor
        )

        self.changed_parameters = (
            changed_parameters
        )

        # -----------------------------------------------------
        # STATE-PROPAGATION SAFETY CHECK
        # -----------------------------------------------------
        #
        # If SSD selected parameters but changed none of them, the
        # result is invalid for sequential unlearning.
        # -----------------------------------------------------

        if selected_parameters > 0 and changed_parameters == 0:
            raise RuntimeError(
                "SSD STATE PROPAGATION FAILURE: "
                f"{selected_parameters:,} parameters were selected "
                "but no model parameters changed."
            )

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        return SSDResult(
            model=model,
            unlearning_time=elapsed,
            forget_samples=len(
                forget_indices
            ),
            selected_parameters=(
                selected_parameters
            ),
            total_parameters=(
                total_parameters
            ),
            selection_ratio=(
                selection_ratio
            ),
            mean_dampening_factor=(
                mean_factor
            ),
            min_dampening_factor=(
                min_factor
            ),
            max_dampening_factor=(
                max_factor
            ),
            changed_parameters=(
                changed_parameters
            ),
        )

    # =========================================================
    # DATA LOADER
    # =========================================================

    def _make_loader(
        self,
        dataset,
        indices,
    ) -> DataLoader:

        subset = Subset(
            dataset,
            list(indices),
        )

        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(
                self.num_workers > 0
            ),
        )

    # =========================================================
    # IMPORTANCE ESTIMATION
    # =========================================================

    def _compute_importance(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> dict[str, torch.Tensor]:

        criterion = nn.CrossEntropyLoss()

        importance = {
            name: torch.zeros_like(
                parameter,
                device=self.device,
            )
            for name, parameter
            in model.named_parameters()
            if parameter.requires_grad
        }

        model.eval()

        num_batches = 0

        # -----------------------------------------------------
        # Fisher-style diagonal importance
        #
        # I(theta) = mean(gradient(theta)^2)
        # -----------------------------------------------------

        for images, labels in loader:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            model.zero_grad(
                set_to_none=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            for name, parameter in (
                model.named_parameters()
            ):

                if (
                    parameter.requires_grad
                    and parameter.grad is not None
                ):

                    importance[name].add_(
                        parameter.grad.detach()
                        .pow(2)
                    )

            num_batches += 1

        if num_batches == 0:
            raise RuntimeError(
                "SSD received an empty dataloader."
            )

        # -----------------------------------------------------
        # Average over minibatches
        # -----------------------------------------------------

        for name in importance:

            importance[name].div_(
                float(num_batches)
            )

        return importance

    # =========================================================
    # APPLY SSD
    # =========================================================

    def _apply_ssd(
        self,
        model: nn.Module,
        full_importance: dict[
            str,
            torch.Tensor,
        ],
        forget_importance: dict[
            str,
            torch.Tensor,
        ],
    ):
        """
        Apply SSD directly to the supplied model.

        This implementation deliberately constructs a complete updated
        parameter tensor and copies it back into the Parameter object.
        This avoids advanced-indexing assignment issues and guarantees
        that the sequential engine receives the modified model state.

        Additional diagnostics are computed from the actual tensor delta,
        not from the selection mask alone.
        """

        selected_parameters = 0
        total_parameters = 0
        changed_parameters = 0

        dampening_sum = 0.0
        dampening_count = 0

        min_factor = float("inf")
        max_factor = float("-inf")

        max_parameter_change = 0.0
        total_parameter_change = 0.0

        epsilon = 1e-12

        with torch.no_grad():

            for name, parameter in model.named_parameters():

                if name not in full_importance:
                    continue

                if name not in forget_importance:
                    continue

                full = full_importance[name]
                forget = forget_importance[name]

                total_parameters += parameter.numel()

                # -------------------------------------------------
                # SELECT PARAMETERS
                # -------------------------------------------------
                locations = (
                    forget
                    >
                    (
                        self.selection_weighting
                        * full
                    )
                )

                selected_count = int(
                    locations.sum().item()
                )

                selected_parameters += selected_count

                if selected_count == 0:
                    continue

                # -------------------------------------------------
                # COMPUTE DAMPENING FACTOR
                # -------------------------------------------------
                dampening = (
                    (
                        self.dampening_constant
                        * full
                    )
                    /
                    (
                        forget
                        + epsilon
                    )
                ).pow(
                    self.exponent
                )

                # SSD is a dampening operation only.
                dampening = torch.clamp(
                    dampening,
                    min=self.lower_bound,
                    max=1.0,
                )

                selected_dampening = (
                    dampening[locations]
                )

                dampening_sum += (
                    selected_dampening.sum().item()
                )

                dampening_count += (
                    selected_dampening.numel()
                )

                local_min = (
                    selected_dampening.min().item()
                )

                local_max = (
                    selected_dampening.max().item()
                )

                min_factor = min(
                    min_factor,
                    local_min,
                )

                max_factor = max(
                    max_factor,
                    local_max,
                )

                # -------------------------------------------------
                # CONSTRUCT THE COMPLETE UPDATED PARAMETER
                # -------------------------------------------------
                #
                # We intentionally avoid:
                #
                #     parameter[locations] *= dampening[locations]
                #
                # because advanced indexing creates a temporary tensor.
                #
                # Instead, clone the complete parameter tensor, update
                # only the selected entries, and copy the result back.
                # -------------------------------------------------

                before = parameter.detach().clone()

                updated_parameter = before.clone()

                updated_parameter[locations] = (
                    before[locations]
                    *
                    selected_dampening
                )

                parameter.copy_(
                    updated_parameter
                )

                # -------------------------------------------------
                # ACTUAL NUMERICAL CHANGE
                # -------------------------------------------------
                delta = (
                    parameter.detach()
                    -
                    before
                )

                abs_delta = delta.abs()

                changed_mask = (
                    abs_delta > 0
                )

                local_changed = int(
                    changed_mask.sum().item()
                )

                changed_parameters += (
                    local_changed
                )

                if local_changed > 0:
                    local_max_change = (
                        abs_delta.max().item()
                    )

                    local_total_change = (
                        abs_delta.sum().item()
                    )

                    max_parameter_change = max(
                        max_parameter_change,
                        local_max_change,
                    )

                    total_parameter_change += (
                        local_total_change
                    )

        # ---------------------------------------------------------
        # FINAL FACTOR STATISTICS
        # ---------------------------------------------------------
        if dampening_count > 0:
            mean_factor = (
                dampening_sum
                /
                dampening_count
            )
        else:
            mean_factor = 1.0
            min_factor = 1.0
            max_factor = 1.0

        # ---------------------------------------------------------
        # INTERNAL DIAGNOSTICS
        # ---------------------------------------------------------
        print()
        print("=" * 40)
        print("SSD INTERNAL DIAGNOSTICS")
        print("=" * 40)
        print(
            f"Selected parameters: "
            f"{selected_parameters:,}"
        )
        print(
            f"Changed parameters:  "
            f"{changed_parameters:,}"
        )
        print(
            f"Total parameters:    "
            f"{total_parameters:,}"
        )
        print(
            f"Selection ratio:     "
            f"{selected_parameters / max(total_parameters, 1):.8f}"
        )
        print(
            f"Mean factor:         "
            f"{mean_factor:.8f}"
        )
        print(
            f"Min factor:          "
            f"{min_factor:.8f}"
        )
        print(
            f"Max factor:          "
            f"{max_factor:.8f}"
        )
        print(
            f"Maximum change:      "
            f"{max_parameter_change:.10f}"
        )
        print(
            f"Total abs change:    "
            f"{total_parameter_change:.10f}"
        )
        print("=" * 40)

        return (
            selected_parameters,
            total_parameters,
            mean_factor,
            min_factor,
            max_factor,
            changed_parameters,
        )

