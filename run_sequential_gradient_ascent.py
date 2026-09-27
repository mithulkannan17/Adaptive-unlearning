from __future__ import annotations

import torch

from src.data.cifar10 import (
    get_cifar10_datasets,
    get_test_loader,
)
from src.unlearning.sequential_gradient_ascent import (
    run_sequential_gradient_ascent,
)
from src.utils import get_device


def main():

    device = get_device("cuda")

    print("=" * 70)
    print("ADAPTIVE SEQUENTIAL MACHINE UNLEARNING")
    print("SEQUENTIAL GRADIENT ASCENT")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

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

    baseline_checkpoint = (
        "./checkpoints/"
        "base_cifar10/"
        "best_model.pth"
    )

    # ---------------------------------------------------------
    # IMPORTANT
    #
    # Start with 3 rounds.
    # Once validated, change rounds=20.
    # ---------------------------------------------------------

    run_sequential_gradient_ascent(
        device=device,
        train_dataset=train_dataset,
        test_loader=test_loader,
        baseline_checkpoint=baseline_checkpoint,
        rounds=3,
        seed=42,
    )


if __name__ == "__main__":
    main()