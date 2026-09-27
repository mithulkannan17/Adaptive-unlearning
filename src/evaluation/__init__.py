"""
Evaluation utilities and metrics for machine unlearning experiments.
"""

from src.evaluation.compute import compute_comprehensive_evaluation
from src.evaluation.counterfactual import CounterfactualEvaluator
from src.evaluation.evaluator import UnlearningEvaluator
from src.evaluation.forgetting import (
    compute_sample_losses_and_probs,
    evaluate_forgetting_efficacy,
)
from src.evaluation.stability import compute_stability_metrics
from src.evaluation.utility import evaluate_utility

__all__ = [
    "UnlearningEvaluator",
    "CounterfactualEvaluator",
    "evaluate_forgetting_efficacy",
    "evaluate_utility",
    "compute_stability_metrics",
    "compute_comprehensive_evaluation",
    "compute_sample_losses_and_probs",
]