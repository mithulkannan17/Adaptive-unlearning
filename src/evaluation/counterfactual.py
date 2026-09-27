
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


class CounterfactualEvaluator:
    """
    Functional comparison between an approximate unlearning model
    and an exact-retraining counterfactual model.

    The evaluation intentionally focuses on model behavior rather
    than treating raw parameter distance as the primary measure.

    Metrics:
        1. Prediction agreement
        2. Mean absolute probability difference
        3. KL divergence
        4. Confidence difference
        5. Parameter distance (diagnostic only)
        6. Relative parameter distance (diagnostic only)

    These metrics are computed independently for:
        - forget set
        - retain set
        - test set
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

    @torch.no_grad()
    def compare(
        self,
        approximate_model: nn.Module,
        exact_model: nn.Module,
        train_dataset,
        test_loader: DataLoader,
        forget_indices: Sequence[int],
        retain_indices: Sequence[int],
    ) -> Dict[str, float]:

        approximate_model.eval()
        exact_model.eval()

        forget_loader = self._make_loader(
            train_dataset,
            forget_indices,
        )

        retain_loader = self._make_loader(
            train_dataset,
            retain_indices,
        )

        forget_metrics = self._compare_loader(
            approximate_model,
            exact_model,
            forget_loader,
        )

        retain_metrics = self._compare_loader(
            approximate_model,
            exact_model,
            retain_loader,
        )

        test_metrics = self._compare_loader(
            approximate_model,
            exact_model,
            test_loader,
        )

        parameter_distance = (
            self.parameter_distance(
                approximate_model,
                exact_model,
            )
        )

        exact_norm = self.parameter_norm(
            exact_model
        )

        relative_parameter_distance = (
            parameter_distance
            / max(exact_norm, 1e-12)
        )

        return {
            # -------------------------------------------------
            # Forget set
            # -------------------------------------------------

            "forget_prediction_agreement": (
                forget_metrics[
                    "prediction_agreement"
                ]
            ),

            "forget_mean_probability_difference": (
                forget_metrics[
                    "mean_probability_difference"
                ]
            ),

            "forget_kl_divergence": (
                forget_metrics[
                    "kl_divergence"
                ]
            ),

            "forget_confidence_difference": (
                forget_metrics[
                    "confidence_difference"
                ]
            ),

            # -------------------------------------------------
            # Retain set
            # -------------------------------------------------

            "retain_prediction_agreement": (
                retain_metrics[
                    "prediction_agreement"
                ]
            ),

            "retain_mean_probability_difference": (
                retain_metrics[
                    "mean_probability_difference"
                ]
            ),

            "retain_kl_divergence": (
                retain_metrics[
                    "kl_divergence"
                ]
            ),

            "retain_confidence_difference": (
                retain_metrics[
                    "confidence_difference"
                ]
            ),

            # -------------------------------------------------
            # Test set
            # -------------------------------------------------

            "test_prediction_agreement": (
                test_metrics[
                    "prediction_agreement"
                ]
            ),

            "test_mean_probability_difference": (
                test_metrics[
                    "mean_probability_difference"
                ]
            ),

            "test_kl_divergence": (
                test_metrics[
                    "kl_divergence"
                ]
            ),

            "test_confidence_difference": (
                test_metrics[
                    "confidence_difference"
                ]
            ),

            # -------------------------------------------------
            # Parameter diagnostics
            # -------------------------------------------------

            "parameter_distance": (
                parameter_distance
            ),

            "relative_parameter_distance": (
                relative_parameter_distance
            ),
        }

    def _make_loader(
        self,
        dataset,
        indices: Sequence[int],
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
            persistent_workers=self.num_workers > 0,
        )

    @staticmethod
    @torch.no_grad()
    def _compare_loader(
        approximate_model: nn.Module,
        exact_model: nn.Module,
        loader: DataLoader,
    ) -> Dict[str, float]:

        prediction_matches = 0
        total_samples = 0

        probability_difference_sum = 0.0
        kl_divergence_sum = 0.0
        confidence_difference_sum = 0.0

        for images, _ in loader:

            images = images.to(
                next(
                    approximate_model.parameters()
                ).device,
                non_blocking=True,
            )

            approximate_logits = (
                approximate_model(images)
            )

            exact_logits = (
                exact_model(images)
            )

            approximate_probabilities = (
                F.softmax(
                    approximate_logits,
                    dim=1,
                )
            )

            exact_probabilities = (
                F.softmax(
                    exact_logits,
                    dim=1,
                )
            )

            # -------------------------------------------------
            # Prediction agreement
            # -------------------------------------------------

            approximate_predictions = (
                approximate_logits.argmax(
                    dim=1
                )
            )

            exact_predictions = (
                exact_logits.argmax(
                    dim=1
                )
            )

            prediction_matches += (
                approximate_predictions
                == exact_predictions
            ).sum().item()

            # -------------------------------------------------
            # Mean absolute probability difference
            # -------------------------------------------------

            probability_difference = (
                torch.abs(
                    approximate_probabilities
                    - exact_probabilities
                )
                .mean(dim=1)
            )

            probability_difference_sum += (
                probability_difference
                .sum()
                .item()
            )

            # -------------------------------------------------
            # KL divergence
            #
            # KL(exact || approximate)
            # -------------------------------------------------

            approximate_log_probabilities = (
                F.log_softmax(
                    approximate_logits,
                    dim=1,
                )
            )

            exact_probabilities_for_kl = (
                exact_probabilities
            )

            kl_divergence = F.kl_div(
                approximate_log_probabilities,
                exact_probabilities_for_kl,
                reduction="none",
            ).sum(dim=1)

            kl_divergence_sum += (
                kl_divergence
                .sum()
                .item()
            )

            # -------------------------------------------------
            # Confidence difference
            # -------------------------------------------------

            approximate_confidence = (
                approximate_probabilities
                .max(dim=1)
                .values
            )

            exact_confidence = (
                exact_probabilities
                .max(dim=1)
                .values
            )

            confidence_difference = (
                torch.abs(
                    approximate_confidence
                    - exact_confidence
                )
            )

            confidence_difference_sum += (
                confidence_difference
                .sum()
                .item()
            )

            total_samples += images.size(0)

        if total_samples == 0:
            return {
                "prediction_agreement": 0.0,
                "mean_probability_difference": 0.0,
                "kl_divergence": 0.0,
                "confidence_difference": 0.0,
            }

        return {
            "prediction_agreement": (
                100.0
                * prediction_matches
                / total_samples
            ),

            "mean_probability_difference": (
                probability_difference_sum
                / total_samples
            ),

            "kl_divergence": (
                kl_divergence_sum
                / total_samples
            ),

            "confidence_difference": (
                confidence_difference_sum
                / total_samples
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

        parameters_a = list(
            model_a.parameters()
        )

        parameters_b = list(
            model_b.parameters()
        )

        if len(parameters_a) != len(parameters_b):
            raise ValueError(
                "Models have different "
                "parameter structures."
            )

        squared_distance = 0.0

        for parameter_a, parameter_b in zip(
            parameters_a,
            parameters_b,
        ):

            difference = (
                parameter_a.detach().float()
                - parameter_b.detach().float()
            )

            squared_distance += (
                difference.pow(2)
                .sum()
                .item()
            )

        return squared_distance ** 0.5
