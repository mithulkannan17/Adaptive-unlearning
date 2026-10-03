from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.data.cifar10 import get_cifar10_datasets, get_indexed_subset, get_test_loader
from src.models.resnet import create_resnet18
from src.unlearning.salun import SalUnUnlearner
from src.utils import get_device, load_checkpoint
from src.workloads.uniform_random import generate_workload_a


def test_salun():
    device = get_device("cuda")
    print("=" * 70)
    print("TESTING SALIENCY UNLEARNING (SalUn)")
    print(f"Device: {device}")
    print("=" * 70)

    train_dataset, test_dataset = get_cifar10_datasets(data_dir="./data")
    test_loader = get_test_loader(test_dataset, batch_size=128, num_workers=2)

    workload = generate_workload_a(dataset_size=len(train_dataset), rounds=1, seed=42)
    forget_indices = workload[0].forget_indices[:200]  # quick test on 200 samples
    retain_indices = [i for i in range(len(train_dataset)) if i not in set(forget_indices)]

    print(f"Forget samples: {len(forget_indices)}")

    # Load baseline
    model = create_resnet18(num_classes=10)
    load_checkpoint("./checkpoints/base_cifar10/best_model.pth", model, device=device)
    model = model.to(device)

    salun = SalUnUnlearner(
        device=device,
        epochs=1,
        learning_rate=1e-3,
        saliency_threshold_ratio=0.3,
        batch_size=128,
        num_workers=2,
    )

    print("\nExecuting SalUn...")
    result = salun.unlearn(model, train_dataset, forget_indices, retain_indices)

    print(f"Unlearning time: {result.unlearning_time_seconds:.2f}s")
    print(f"Salient parameters: {result.salient_parameter_count:,} / {result.total_parameter_count:,} ({result.saliency_ratio * 100:.1f}%)")
    print(f"Initial forget accuracy: {result.initial_forget_accuracy:.2f}%")
    print(f"Final forget accuracy:   {result.final_forget_accuracy:.2f}%")

    # Evaluate on test set
    result.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = result.model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    test_acc = 100.0 * correct / total
    print(f"Post-SalUn test accuracy: {test_acc:.2f}%")

    assert result.final_forget_accuracy < result.initial_forget_accuracy
    assert test_acc > 85.0
    print("\nSalUn validation passed successfully.")


if __name__ == "__main__":
    test_salun()
