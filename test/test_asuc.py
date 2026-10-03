from __future__ import annotations

import numpy as np
import torch

from src.controller.asuc import (
    ASUCController,
    ASUCRouter,
    RequestProfiler,
)
from src.controller.som import HealthState
from src.data.cifar10 import get_cifar10_datasets
from src.models.resnet import create_resnet18
from src.utils import get_device, load_checkpoint
from src.verification.verifier import UnlearningVerifier
from src.workloads.uniform_random import generate_workload_a


def test_asuc_pipeline():
    print("=" * 70)
    print("TESTING ASUC CONTROLLER & CLOSED-LOOP PIPELINE")
    print("=" * 70)

    device = get_device("cuda")
    train_dataset, test_dataset = get_cifar10_datasets(data_dir="./data")

    # 1. Test RequestProfiler
    profiler = RequestProfiler(num_classes=10)
    # Balanced request
    balanced_idx = list(range(100))
    p_bal = profiler.profile(balanced_idx, total_remaining=49900, dataset=train_dataset)
    print(f"Profiler (Balanced): Entropy = {p_bal.class_entropy:.4f}, Concentrated = {p_bal.is_class_concentrated}")

    # Concentrated request (all class 0)
    targets = np.array(train_dataset.targets)
    class_0_idx = np.where(targets == 0)[0][:100].tolist()
    p_conc = profiler.profile(class_0_idx, total_remaining=49900, dataset=train_dataset)
    print(f"Profiler (Class-0): Entropy = {p_conc.class_entropy:.4f}, Concentrated = {p_conc.is_class_concentrated}")
    assert p_conc.is_class_concentrated

    # 2. Test ASUCRouter
    router = ASUCRouter()
    route_green = router.route(p_bal, HealthState.GREEN)
    route_amber = router.route(p_bal, HealthState.AMBER)
    route_red = router.route(p_bal, HealthState.RED)
    route_conc = router.route(p_conc, HealthState.GREEN)

    print(f"Router Decisions: Green -> {route_green}, Amber -> {route_amber}, Red -> {route_red}, Concentrated -> {route_conc}")
    assert route_green == "SSD"
    assert route_amber == "SalUn"
    assert route_red == "SalUn"
    assert route_conc == "SalUn"

    # 3. Test Controller Step
    print("\nTesting ASUCController step execution...")
    baseline_model = create_resnet18(num_classes=10)
    load_checkpoint("./checkpoints/base_cifar10/best_model.pth", baseline_model, device=device)
    baseline_model = baseline_model.to(device)

    controller = ASUCController(
        device=device,
        num_classes=10,
        som_decay=0.9,
        som_max_rank=5,
        green_threshold=0.4,
        red_threshold=0.8,
        retrain_epochs=1,
    )

    workload = generate_workload_a(dataset_size=len(train_dataset), rounds=1, seed=42)
    forget_idx = workload[0].forget_indices[:100]
    retain_idx = [i for i in range(len(train_dataset)) if i not in set(forget_idx)]

    unlearned_model, decision = controller.step(
        current_model=baseline_model,
        baseline_model=baseline_model,
        train_dataset=train_dataset,
        forget_indices=forget_idx,
        retain_indices=retain_idx,
        round_id=1,
    )

    print(f"ASUC Step Complete:")
    print(f"  Chosen Primitive:     {decision.chosen_primitive}")
    print(f"  Health State:         {decision.health_state} (SOM: {decision.som_score:.4f})")
    print(f"  Verification Passed:  {decision.verification_passed}")
    print(f"  Verification Reason:  {decision.verification_reason}")
    print(f"  Unlearning Time:      {decision.unlearning_time_seconds:.2f}s")
    print(f"  Parameter Distance:   {decision.parameter_distance_baseline:.4f}")

    assert decision.parameter_norm > 0
    print("\nASUC Controller validation passed successfully.")


if __name__ == "__main__":
    test_asuc_pipeline()
