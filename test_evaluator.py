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
    print("EVALUATOR VALIDATION")
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
    # ONE WORKLOAD-A ROUND
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

    retain_indices = [
        index
        for index in range(len(train_dataset))
        if index not in set(forget_indices)
    ]

    print(
        f"Forget samples: {len(forget_indices)}"
    )

    print(
        f"Retain samples: {len(retain_indices)}"
    )

    # ---------------------------------------------------------
    # LOAD BASE MODEL
    # ---------------------------------------------------------

    model = create_resnet18(
        num_classes=10
    )

    checkpoint_path = (
        "./checkpoints/"
        "base_cifar10/"
        "best_model.pth"
    )

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        device=device,
    )

    model = model.to(device)

    # ---------------------------------------------------------
    # EVALUATE
    # ---------------------------------------------------------

    evaluator = UnlearningEvaluator(
        device=device,
        batch_size=128,
        num_workers=2,
    )

    results = evaluator.evaluate(
        model=model,
        train_dataset=train_dataset,
        test_loader=test_loader,
        forget_indices=forget_indices,
        retain_indices=retain_indices,
        reference_model=model,
    )

    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Forget loss:       "
        f"{results['forget_loss']:.4f}"
    )

    print(
        f"Forget accuracy:   "
        f"{results['forget_accuracy']:.2f}%"
    )

    print(
        f"Retain loss:       "
        f"{results['retain_loss']:.4f}"
    )

    print(
        f"Retain accuracy:   "
        f"{results['retain_accuracy']:.2f}%"
    )

    print(
        f"Test loss:         "
        f"{results['test_loss']:.4f}"
    )

    print(
        f"Test accuracy:     "
        f"{results['test_accuracy']:.2f}%"
    )

    print(
        f"Parameter norm:    "
        f"{results['parameter_norm']:.4f}"
    )

    print(
        f"Parameter distance:"
        f" {results['parameter_distance']:.4f}"
    )

    print(
        f"Evaluation time:   "
        f"{results['evaluation_time_seconds']:.2f}s"
    )

    print()
    print("Evaluator validation successful.")


if __name__ == "__main__":
    main()