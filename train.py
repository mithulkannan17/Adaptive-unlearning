from __future__ import annotations

import json
from pathlib import Path

import torch

from src.config import ExperimentConfig
from src.data.cifar10 import get_cifar10_dataloaders
from src.models.resnet import create_resnet18
from src.seed import set_seed
from src.training.trainer import Trainer
from src.utils import get_device, save_checkpoint


def main() -> None:

    config = ExperimentConfig()

    config.prepare_directories()

    set_seed(config.seed)

    device = get_device(config.device)

    print("=" * 70)
    print("ADAPTIVE SEQUENTIAL MACHINE UNLEARNING")
    print("BASELINE TRAINING")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(
            f"CUDA version: {torch.version.cuda}"
        )

    print(f"Seed: {config.seed}")
    print(f"Dataset: {config.dataset}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")

    train_test = get_cifar10_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    train_loader = train_test.train
    test_loader = train_test.test

    print()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = create_resnet18(
        num_classes=config.num_classes,
    )

    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    best_accuracy = 0.0
    best_epoch = 0

    history = {
        "train": [],
        "test": [],
    }

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, config.epochs + 1):

        train_metrics = trainer.train_one_epoch(
            train_loader=train_loader,
            epoch=epoch,
            total_epochs=config.epochs,
        )

        test_metrics = trainer.evaluate(
            test_loader,
        )

        history["train"].append(
            {
                "epoch": epoch,
                **train_metrics,
            }
        )

        history["test"].append(
            {
                "epoch": epoch,
                **test_metrics,
            }
        )

        print(
            f"\nEpoch {epoch:03d}/{config.epochs:03d}"
        )

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}%"
        )

        print(
            f"Test Loss:  {test_metrics['loss']:.4f} | "
            f"Test Acc:  {test_metrics['accuracy']:.2f}%"
        )

        if test_metrics["accuracy"] > best_accuracy:

            best_accuracy = test_metrics["accuracy"]
            best_epoch = epoch

            best_checkpoint_path = (
                Path(config.checkpoint_dir)
                / "base_cifar10"
                / "best_model.pth"
            )

            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "test_accuracy": best_accuracy,
                    "test_loss": test_metrics["loss"],
                },
            )

            print(
                f"✓ New best model saved: "
                f"{best_accuracy:.2f}%"
            )

    final_checkpoint_path = (
        Path(config.checkpoint_dir)
        / "base_cifar10"
        / "final_model.pth"
    )

    save_checkpoint(
        path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=config.epochs,
        metrics={
            "test_accuracy": history["test"][-1]["accuracy"],
            "test_loss": history["test"][-1]["loss"],
        },
    )

    result_directory = (
        Path(config.result_dir)
        / "base_cifar10"
    )

    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        result_directory
        / "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "config": {
                    "seed": config.seed,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "momentum": config.momentum,
                    "weight_decay": config.weight_decay,
                },
                "best_epoch": best_epoch,
                "best_test_accuracy": best_accuracy,
                "history": history,
            },
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best test accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        f"Best checkpoint: "
        f"{Path(config.checkpoint_dir) / 'base_cifar10' / 'best_model.pth'}"
    )

    print(
        f"Training history: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()