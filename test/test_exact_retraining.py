from __future__ import annotations

import torch

from src.data.cifar10 import (
    get_cifar10_datasets,
    get_test_loader,
)
from src.evaluation.evaluator import (
    UnlearningEvaluator,
)
from src.models.resnet import create_resnet18
from src.unlearning.exact_retraining import (
    ExactRetrainer,
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
    print("EXACT RETRAINING VALIDATION")
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
    # LOAD BASE MODEL ONLY AS ARCHITECTURE
    # ---------------------------------------------------------

    baseline_model = create_resnet18(
        num_classes=10
    )

    checkpoint_path = (
        "./checkpoints/"
        "base_cifar10/"
        "best_model.pth"
    )

    load_checkpoint(
        path=checkpoint_path,
        model=baseline_model,
        device=device,
    )

    baseline_model = baseline_model.to(
        device
    )

    # ---------------------------------------------------------
    # EXACT RETRAINING
    # ---------------------------------------------------------

    retrainer = ExactRetrainer(
        device=device,
        epochs=30,
        learning_rate=0.1,
        batch_size=128,
        num_workers=2,
        weight_decay=5e-4,
        momentum=0.9,
    )

    result = retrainer.retrain(
        model=baseline_model,
        train_dataset=train_dataset,
        forget_indices=forget_indices,
    )

    exact_model = result.model

    exact_checkpoint_path = (
        "./checkpoints/"
        "exact_retrain_cifar10/"
        "round_01_model.pth"
    )
    
    from pathlib import Path
    
    Path(
        "./checkpoints/"
        "exact_retrain_cifar10"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )
    
    torch.save(
        {
            "model_state_dict": (
                exact_model.state_dict()
            ),
            "forget_indices": forget_indices,
            "epochs": result.epochs,
            "training_time_seconds": (
                result.training_time_seconds
            ),
        },
        exact_checkpoint_path,
    )
    
    print(
        f"Exact checkpoint saved: "
        f"{exact_checkpoint_path}"
    )

    # ---------------------------------------------------------
    # EVALUATE EXACT MODEL
    # ---------------------------------------------------------

    evaluator = UnlearningEvaluator(
        device=device,
        batch_size=128,
        num_workers=2,
    )

    metrics = evaluator.evaluate(
        model=exact_model,
        train_dataset=train_dataset,
        test_loader=test_loader,
        forget_indices=forget_indices,
        retain_indices=retain_indices,
        reference_model=baseline_model,
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("EXACT RETRAINING RESULTS")
    print("=" * 70)

    print(
        f"Forget accuracy:   "
        f"{metrics['forget_accuracy']:.2f}%"
    )

    print(
        f"Retain accuracy:   "
        f"{metrics['retain_accuracy']:.2f}%"
    )

    print(
        f"Test accuracy:     "
        f"{metrics['test_accuracy']:.2f}%"
    )

    print(
        f"Forget loss:       "
        f"{metrics['forget_loss']:.4f}"
    )

    print(
        f"Parameter norm:    "
        f"{metrics['parameter_norm']:.4f}"
    )

    print(
        f"Distance from baseline: "
        f"{metrics['parameter_distance']:.4f}"
    )

    print(
        f"Retraining time:   "
        f"{result.training_time_seconds:.2f}s"
    )

    print()
    print(
        "Exact retraining validation complete."
    )


if __name__ == "__main__":
    main()