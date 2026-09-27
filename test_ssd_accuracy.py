
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.cifar10 import (
    get_cifar10_datasets,
    get_test_loader,
)
from src.models.resnet import create_resnet18
from src.unlearning.ssd import SSDUnlearner
from src.utils import (
    get_device,
    load_checkpoint,
)
from src.workloads.uniform_random import (
    generate_workload_a,
)


def evaluate_model(
    model,
    loader,
    device,
    name,
):
    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = (
        100.0 * correct / total
    )

    loss = total_loss / total

    print(
        f"{name:<25} "
        f"Loss: {loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

    return loss, accuracy


def main():

    device = get_device("cuda")

    print("=" * 70)
    print("SSD DIRECT ACCURACY SANITY CHECK")
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
    # WORKLOAD
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

    retain_loader = DataLoader(
        torch.utils.data.Subset(
            train_dataset,
            retain_indices,
        ),
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # ---------------------------------------------------------
    # BASE MODEL
    # ---------------------------------------------------------

    checkpoint_path = (
        "./checkpoints/"
        "base_cifar10/"
        "best_model.pth"
    )

    baseline = create_resnet18(
        num_classes=10
    )

    load_checkpoint(
        path=checkpoint_path,
        model=baseline,
        device=device,
    )

    baseline = baseline.to(device)

    # ---------------------------------------------------------
    # BASELINE ACCURACY
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE MODEL")
    print("=" * 70)

    evaluate_model(
        baseline,
        retain_loader,
        device,
        "Baseline Retain",
    )

    evaluate_model(
        baseline,
        test_loader,
        device,
        "Baseline Test",
    )

    # ---------------------------------------------------------
    # SSD
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING SSD")
    print("=" * 70)

    ssd = SSDUnlearner(
        device=device,
        batch_size=128,
        num_workers=2,
        dampening_constant=1.0,
        selection_weighting=10.0,
        exponent=1.0,
        lower_bound=0.0,
    )

    result = ssd.unlearn(
        model=baseline,
        train_dataset=train_dataset,
        forget_indices=forget_indices,
    )

    ssd_model = result.model

    print(
        f"SSD runtime: "
        f"{result.unlearning_time:.4f}s"
    )

    # ---------------------------------------------------------
    # SSD ACCURACY
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SSD MODEL")
    print("=" * 70)

    evaluate_model(
        ssd_model,
        retain_loader,
        device,
        "SSD Retain",
    )

    evaluate_model(
        ssd_model,
        test_loader,
        device,
        "SSD Test",
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SANITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
