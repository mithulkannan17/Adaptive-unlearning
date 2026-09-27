
from __future__ import annotations

import torch

from src.data.cifar10 import get_cifar10_datasets
from src.models.resnet import create_resnet18
from src.unlearning.ssd import SSDUnlearner
from src.utils import get_device, load_checkpoint
from src.workloads.uniform_random import generate_workload_a


def main():

    device = get_device("cuda")

    print("=" * 70)
    print("SSD UNLEARNING VALIDATION")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

    train_dataset, _ = get_cifar10_datasets(
        data_dir="./data"
    )

    print(
        f"Training samples: {len(train_dataset)}"
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

    forget_indices = workload[0].forget_indices

    print(
        f"Forget samples: {len(forget_indices)}"
    )

    # ---------------------------------------------------------
    # LOAD BASE MODEL
    # ---------------------------------------------------------

    checkpoint_path = (
        "./checkpoints/"
        "base_cifar10/"
        "best_model.pth"
    )

    model = create_resnet18(
        num_classes=10
    )

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        device=device,
    )

    model = model.to(device)

    # ---------------------------------------------------------
    # SSD
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING SSD")
    print("=" * 70)

    unlearner = SSDUnlearner(
        device=device,
        batch_size=128,
        num_workers=2,
        alpha=1.0,
        lambda_=1.0,
    )

    result = unlearner.unlearn(
        model=model,
        train_dataset=train_dataset,
        forget_indices=forget_indices,
    )

    ssd_model = result.model

    # ---------------------------------------------------------
    # BASIC VALIDATION
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SSD VALIDATION COMPLETE")
    print("=" * 70)

    print(
        f"Forget samples: "
        f"{result.forget_samples}"
    )

    print(
        f"Unlearning time: "
        f"{result.unlearning_time:.4f}s"
    )

    print(
        f"Model device: "
        f"{next(ssd_model.parameters()).device}"
    )

    print(
        f"Training mode: "
        f"{ssd_model.training}"
    )

    # ---------------------------------------------------------
    # PARAMETER SANITY CHECK
    # ---------------------------------------------------------

    parameter_count = 0
    nonzero_parameters = 0

    for parameter in ssd_model.parameters():

        parameter_count += parameter.numel()

        nonzero_parameters += (
            parameter.detach()
            .ne(0)
            .sum()
            .item()
        )

    zero_parameters = (
        parameter_count
        - nonzero_parameters
    )

    print()
    print(
        f"Total parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Zero parameters after SSD: "
        f"{zero_parameters:,}"
    )

    # ---------------------------------------------------------
    # PARAMETER DIFFERENCE
    # ---------------------------------------------------------

    parameter_difference = 0.0
    baseline_parameter_norm = 0.0

    for original, updated in zip(
        model.parameters(),
        ssd_model.parameters(),
    ):

        parameter_difference += (
            torch.sum(
                (original.detach() - updated.detach())
                ** 2
            )
            .item()
        )

        baseline_parameter_norm += (
            torch.sum(
                original.detach() ** 2
            )
            .item()
        )

    parameter_difference **= 0.5
    baseline_parameter_norm **= 0.5

    relative_difference = (
        parameter_difference
        /
        max(
            baseline_parameter_norm,
            1e-12,
        )
    )

    print()
    print(
        f"Parameter distance: "
        f"{parameter_difference:.6f}"
    )

    print(
        f"Relative parameter distance: "
        f"{relative_difference:.6f}"
    )

    print()
    print(
        "SSD model successfully produced."
    )


if __name__ == "__main__":
    main()
