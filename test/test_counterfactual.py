
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
from src.unlearning.gradient_ascent import (
    GradientAscentUnlearner,
)
from src.utils import (
    get_device,
    load_checkpoint,
)
from src.workloads.uniform_random import (
    generate_workload_a,
)


def main():

    device = get_device("cuda")

    print("=" * 70)
    print("COUNTERFACTUAL COMPARISON")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
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
    # ONE DELETION REQUEST
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
    # BASELINE CHECKPOINT
    # ---------------------------------------------------------

    baseline_checkpoint = (
        "./checkpoints/"
        "base_cifar10/"
        "best_model.pth"
    )

    # ---------------------------------------------------------
    # GRADIENT ASCENT MODEL
    # ---------------------------------------------------------

    ga_model = create_resnet18(
        num_classes=10
    )

    load_checkpoint(
        path=baseline_checkpoint,
        model=ga_model,
        device=device,
    )

    ga_model = ga_model.to(device)

    print()
    print("=" * 70)
    print("RUNNING GRADIENT ASCENT")
    print("=" * 70)

    ga = GradientAscentUnlearner(
        device=device,
        epochs=5,
        learning_rate=1e-3,
        batch_size=128,
        num_workers=2,
    )

    ga_result = ga.unlearn(
        model=ga_model,
        train_dataset=train_dataset,
        forget_indices=forget_indices,
    )

    ga_model = ga_result.model

    ga_model.eval()

    print("Gradient Ascent complete.")

    # ---------------------------------------------------------
    # LOAD EXACT-RETRAINED CHECKPOINT
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    #
    # DO NOT run ExactRetrainer here.
    #
    # The exact model was already trained and saved after
    # the previous ~160 minute experiment.
    #
    # ---------------------------------------------------------

    exact_checkpoint_path = (
        "./checkpoints/"
        "exact_retrain_cifar10/"
        "round_01_model.pth"
    )

    print()
    print("=" * 70)
    print("LOADING EXACT-RETRAINED MODEL")
    print("=" * 70)

    print(
        f"Checkpoint: "
        f"{exact_checkpoint_path}"
    )

    try:

        exact_checkpoint = torch.load(
            exact_checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except FileNotFoundError:

        raise FileNotFoundError(
            "\nExact-retrained checkpoint was not found.\n\n"
            "Expected:\n"
            f"  {exact_checkpoint_path}\n\n"
            "Make sure the exact-retraining experiment "
            "successfully saved round_01_model.pth "
            "before running this comparison."
        )

    exact_model = create_resnet18(
        num_classes=10
    )

    # ---------------------------------------------------------
    # SUPPORT BOTH CHECKPOINT FORMATS
    # ---------------------------------------------------------

    if (
        isinstance(
            exact_checkpoint,
            dict,
        )
        and "model_state_dict"
        in exact_checkpoint
    ):

        exact_model.load_state_dict(
            exact_checkpoint[
                "model_state_dict"
            ]
        )

        exact_training_time = (
            exact_checkpoint.get(
                "training_time_seconds",
                None,
            )
        )

    else:

        # Fallback if the checkpoint contains
        # only the state dictionary.

        exact_model.load_state_dict(
            exact_checkpoint
        )

        exact_training_time = None

    exact_model = exact_model.to(
        device
    )

    exact_model.eval()

    print(
        "Exact-retrained model loaded."
    )

    if exact_training_time is not None:

        print(
            f"Original exact retraining time: "
            f"{exact_training_time:.2f}s"
        )

    # ---------------------------------------------------------
    # COUNTERFACTUAL COMPARISON
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARING GRADIENT ASCENT vs EXACT RETRAINING")
    print("=" * 70)

    evaluator = CounterfactualEvaluator(
        device=device,
        batch_size=128,
        num_workers=2,
    )

    metrics = evaluator.compare(
        approximate_model=ga_model,
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
    print("GRADIENT ASCENT vs EXACT RETRAINING")
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
    print(
        "Counterfactual comparison complete."
    )


if __name__ == "__main__":
    main()
