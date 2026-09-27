
from __future__ import annotations

import torch

from src.data.cifar10 import (
    get_cifar10_datasets,
    get_test_loader,
)
from src.evaluation.counterfactual import (
    CounterfactualEvaluator,
)
from src.models.resnet import create_resnet18
from src.unlearning.ssd import (
    SSDUnlearner,
)
from src.utils import (
    get_device,
    load_checkpoint,
)
from src.workloads.uniform_random import (
    generate_workload_a,
)


def load_model(
    checkpoint_path: str,
    device: torch.device,
):
    model = create_resnet18(
        num_classes=10
    )

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        device=device,
    )

    return model.to(device)


def main():

    device = get_device("cuda")

    print("=" * 70)
    print("SSD COUNTERFACTUAL EVALUATION")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

    train_dataset, test_dataset = (
        get_cifar10_datasets(
            data_dir="./data"
        )
    )

    test_loader = get_test_loader(
        test_dataset,
        batch_size=128,
        num_workers=2,
    )

    # ---------------------------------------------------------
    # DELETION REQUEST
    # ---------------------------------------------------------

    workload = generate_workload_a(
        dataset_size=len(train_dataset),
        deletion_fraction=0.01,
        rounds=1,
        seed=42,
    )

    forget_indices = (
        workload[0].forget_indices
    )

    forget_set = set(
        forget_indices
    )

    retain_indices = [
        index
        for index in range(
            len(train_dataset)
        )
        if index not in forget_set
    ]

    print()
    print(
        f"Forget samples: "
        f"{len(forget_indices)}"
    )

    print(
        f"Retain samples: "
        f"{len(retain_indices)}"
    )

    # ---------------------------------------------------------
    # PATHS
    # ---------------------------------------------------------

    baseline_checkpoint = (
        "./checkpoints/"
        "base_cifar10/"
        "best_model.pth"
    )

    exact_checkpoint = (
        "./checkpoints/"
        "exact_retrain_cifar10/"
        "round_01_model.pth"
    )

    # ---------------------------------------------------------
    # SSD
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING SSD")
    print("=" * 70)

    ssd_model = load_model(
        baseline_checkpoint,
        device,
    )

    ssd = SSDUnlearner(
        device=device,
        batch_size=128,
        num_workers=2,
        dampening_constant=1.0,
        selection_weighting=10.0,
        exponent=1.0,
        lower_bound=1.0,
    )

    ssd_result = ssd.unlearn(
        model=ssd_model,
        train_dataset=train_dataset,
        forget_indices=forget_indices,
    )

    ssd_model = ssd_result.model

    print(
        f"SSD unlearning time: "
        f"{ssd_result.unlearning_time:.4f}s"
    )

    # ---------------------------------------------------------
    # EXACT COUNTERFACTUAL
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING EXACT-RETRAINED COUNTERFACTUAL")
    print("=" * 70)

    exact_checkpoint_data = torch.load(
        exact_checkpoint,
        map_location=device,
        weights_only=False,
    )

    exact_model = create_resnet18(
        num_classes=10
    )

    if (
        isinstance(
            exact_checkpoint_data,
            dict,
        )
        and "model_state_dict"
        in exact_checkpoint_data
    ):

        exact_model.load_state_dict(
            exact_checkpoint_data[
                "model_state_dict"
            ]
        )

    else:

        exact_model.load_state_dict(
            exact_checkpoint_data
        )

    exact_model = exact_model.to(
        device
    )

    exact_model.eval()

    print(
        "Exact counterfactual loaded."
    )

    # ---------------------------------------------------------
    # COUNTERFACTUAL EVALUATION
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATING SSD AGAINST EXACT RETRAINING")
    print("=" * 70)

    evaluator = CounterfactualEvaluator(
        device=device,
        batch_size=128,
        num_workers=2,
    )

    metrics = evaluator.compare(
        approximate_model=ssd_model,
        exact_model=exact_model,
        train_dataset=train_dataset,
        test_loader=test_loader,
        forget_indices=forget_indices,
        retain_indices=retain_indices,
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SSD vs EXACT RETRAINING")
    print("=" * 70)

    print()
    print("FORGET SET")
    print("-" * 70)

    print(
        f"Prediction agreement: "
        f"{metrics['forget_prediction_agreement']:.2f}%"
    )

    print(
        f"Mean probability difference: "
        f"{metrics['forget_mean_probability_difference']:.6f}"
    )

    print(
        f"KL divergence: "
        f"{metrics['forget_kl_divergence']:.6f}"
    )

    print(
        f"Confidence difference: "
        f"{metrics['forget_confidence_difference']:.6f}"
    )

    print()
    print("RETAIN SET")
    print("-" * 70)

    print(
        f"Prediction agreement: "
        f"{metrics['retain_prediction_agreement']:.2f}%"
    )

    print(
        f"Mean probability difference: "
        f"{metrics['retain_mean_probability_difference']:.6f}"
    )

    print(
        f"KL divergence: "
        f"{metrics['retain_kl_divergence']:.6f}"
    )

    print(
        f"Confidence difference: "
        f"{metrics['retain_confidence_difference']:.6f}"
    )

    print()
    print("TEST SET")
    print("-" * 70)

    print(
        f"Prediction agreement: "
        f"{metrics['test_prediction_agreement']:.2f}%"
    )

    print(
        f"Mean probability difference: "
        f"{metrics['test_mean_probability_difference']:.6f}"
    )

    print(
        f"KL divergence: "
        f"{metrics['test_kl_divergence']:.6f}"
    )

    print(
        f"Confidence difference: "
        f"{metrics['test_confidence_difference']:.6f}"
    )

    print()
    print("COMPUTATIONAL COST")
    print("-" * 70)

    print(
        f"SSD unlearning time: "
        f"{ssd_result.unlearning_time:.4f}s"
    )

    print()
    print("PARAMETER DIAGNOSTICS")
    print("-" * 70)

    print(
        f"Parameter distance: "
        f"{metrics['parameter_distance']:.4f}"
    )

    print(
        f"Relative parameter distance: "
        f"{metrics['relative_parameter_distance']:.6f}"
    )

    print()
    print("=" * 70)
    print("SSD COUNTERFACTUAL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
