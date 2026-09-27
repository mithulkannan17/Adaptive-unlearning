from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.data.cifar10 import (
    get_cifar10_datasets,
    get_test_loader,
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
    print("GRADIENT ASCENT VALIDATION")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # Load datasets
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
    # Generate exactly one Workload-A deletion
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

    print()
    print(
        f"Forget-set size: {len(forget_indices)}"
    )

    # ---------------------------------------------------------
    # Load the frozen baseline model
    # ---------------------------------------------------------

    model = create_resnet18(
        num_classes=10
    )

    checkpoint_path = (
        "./checkpoints/"
        "base_cifar10/"
        "best_model.pth"
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        device=device,
    )

    model = model.to(device)

    # ---------------------------------------------------------
    # Baseline test accuracy
    # ---------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, targets in test_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

    baseline_accuracy = (
        100.0 * correct / total
    )

    print(
        f"Baseline test accuracy: "
        f"{baseline_accuracy:.2f}%"
    )

    # ---------------------------------------------------------
    # Apply Gradient Ascent
    # ---------------------------------------------------------

    unlearner = GradientAscentUnlearner(
        device=device,
        epochs=5,
        learning_rate=1e-3,
        batch_size=128,
        num_workers=2,
    )

    result = unlearner.unlearn(
        model=model,
        train_dataset=train_dataset,
        forget_indices=forget_indices,
    )

    # ---------------------------------------------------------
    # Evaluate test accuracy after unlearning
    # ---------------------------------------------------------

    result.model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, targets in test_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            outputs = result.model(
                images
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

    final_accuracy = (
        100.0 * correct / total
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("GRADIENT ASCENT RESULTS")
    print("=" * 70)

    print(
        f"Forget samples: "
        f"{len(forget_indices)}"
    )

    print(
        f"Forget loss: "
        f"{result.initial_forget_loss:.4f}"
        f" -> "
        f"{result.final_forget_loss:.4f}"
    )

    print(
        f"Forget accuracy: "
        f"{result.initial_forget_accuracy:.2f}%"
        f" -> "
        f"{result.final_forget_accuracy:.2f}%"
    )

    print(
        f"Test accuracy: "
        f"{baseline_accuracy:.2f}%"
        f" -> "
        f"{final_accuracy:.2f}%"
    )

    print(
        f"Parameter norm: "
        f"{result.initial_parameter_norm:.4f}"
        f" -> "
        f"{result.final_parameter_norm:.4f}"
    )

    print(
        f"Unlearning time: "
        f"{result.training_time_seconds:.2f}s"
    )

    print()
    print(
        "Gradient Ascent validation complete."
    )


if __name__ == "__main__":
    main()