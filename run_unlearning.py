from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.config import ExperimentConfig
from src.controller.asuc import ASUCController
from src.data.cifar10 import get_cifar10_datasets, get_test_loader
from src.evaluation.compute import compute_comprehensive_evaluation
from src.models.resnet import create_resnet18
from src.seed import set_seed
from src.unlearning.exact_retrain import ExactRetrainer
from src.unlearning.gradient_ascent import GradientAscentUnlearner
from src.unlearning.salun import SalUnUnlearner
from src.unlearning.ssd import SSDUnlearner
from src.utils import get_device, load_checkpoint
from src.workloads.class_sequential import generate_workload_b
from src.workloads.high_influence import compute_sample_losses, generate_workload_c
from src.workloads.uniform_random import generate_workload_a


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-step machine unlearning experiment.")
    parser.add_argument("--method", type=str, default="asuc", choices=["asuc", "ssd", "salun", "ga", "gradient_ascent", "retrain"],
                        help="Unlearning method to run.")
    parser.add_argument("--workload", type=str, default="a", choices=["a", "b", "c"],
                        help="Workload type (a: Uniform, b: Class-sequential, c: High-influence).")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to forget.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/base_cifar10/best_model.pth",
                        help="Path to baseline checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./results/single_unlearning",
                        help="Directory to save unlearning results.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device("cuda")

    print("=" * 75)
    print("MACHINE UNLEARNING BENCHMARK")
    print(f"Method: {args.method.upper()} | Workload: {args.workload.upper()} | Forget Samples: {args.samples}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 75)

    train_dataset, test_dataset = get_cifar10_datasets(data_dir="./data")
    test_loader = get_test_loader(test_dataset, batch_size=128, num_workers=2)

    # 1. Load Baseline Model
    baseline_model = create_resnet18(num_classes=10)
    ckpt = load_checkpoint(args.checkpoint, baseline_model, device=device)
    baseline_model = baseline_model.to(device)
    baseline_test_acc = ckpt.get("metrics", {}).get("test_accuracy", 93.44)

    # 2. Generate Deletion Request
    if args.workload == "a":
        requests = generate_workload_a(dataset_size=len(train_dataset), rounds=1, seed=args.seed)
        forget_indices = requests[0].forget_indices[:args.samples]
    elif args.workload == "b":
        requests = generate_workload_b(targets=train_dataset.targets, rounds=1, samples_per_round=args.samples, seed=args.seed)
        forget_indices = requests[0].forget_indices
    elif args.workload == "c":
        print("Computing sample influence scores under baseline model...")
        scores = compute_sample_losses(baseline_model, train_dataset, device=device)
        requests = generate_workload_c(dataset_size=len(train_dataset), influence_scores=scores, rounds=1, samples_per_round=args.samples, seed=args.seed)
        forget_indices = requests[0].forget_indices

    forget_set = set(forget_indices)
    retain_indices = [i for i in range(len(train_dataset)) if i not in forget_set]

    print(f"Forget set: {len(forget_indices)} samples | Retain set: {len(retain_indices)} samples")

    # 3. Execute Unlearning
    unlearn_model = create_resnet18(num_classes=10)
    load_checkpoint(args.checkpoint, unlearn_model, device=device)
    unlearn_model = unlearn_model.to(device)

    start_time = time.perf_counter()
    extra_info = {}

    if args.method == "asuc":
        controller = ASUCController(device=device, num_classes=10, retrain_epochs=20)
        unlearned_model, decision = controller.step(
            current_model=unlearn_model,
            baseline_model=baseline_model,
            train_dataset=train_dataset,
            forget_indices=forget_indices,
            retain_indices=retain_indices,
            round_id=1,
        )
        unlearn_time = decision.unlearning_time_seconds
        extra_info = {
            "som_score": decision.som_score,
            "health_state": decision.health_state,
            "chosen_primitive": decision.chosen_primitive,
            "escalation_path": decision.escalation_path,
            "verification_passed": decision.verification_passed,
        }
    elif args.method == "ssd":
        ssd = SSDUnlearner(device=device, dampening_constant=1.0, selection_weighting=10.0, lower_bound=0.0)
        res = ssd.unlearn(unlearn_model, train_dataset, forget_indices)
        unlearned_model = res.model
        unlearn_time = res.unlearning_time
    elif args.method == "salun":
        salun = SalUnUnlearner(device=device, epochs=1, learning_rate=1e-3, saliency_threshold_ratio=0.5)
        res = salun.unlearn(unlearn_model, train_dataset, forget_indices, retain_indices)
        unlearned_model = res.model
        unlearn_time = res.unlearning_time_seconds
    elif args.method in ("ga", "gradient_ascent"):
        ga = GradientAscentUnlearner(device=device, epochs=5, learning_rate=1e-3)
        res = ga.unlearn(unlearn_model, train_dataset, forget_indices)
        unlearned_model = res.model
        unlearn_time = res.training_time_seconds
    elif args.method == "retrain":
        retrainer = ExactRetrainer(device=device, epochs=20)
        res = retrainer.train(train_dataset, retain_indices)
        unlearned_model = res.model
        unlearn_time = res.training_time_seconds

    # 4. Comprehensive Evaluation
    print("\nRunning comprehensive evaluation...")
    report = compute_comprehensive_evaluation(
        model=unlearned_model,
        baseline_model=baseline_model,
        train_dataset=train_dataset,
        test_loader=test_loader,
        forget_indices=forget_indices,
        retain_indices=retain_indices,
        device=device,
        baseline_test_acc=baseline_test_acc,
        unlearning_time_seconds=unlearn_time,
    )
    report["method"] = args.method
    report["workload"] = args.workload
    report.update(extra_info)

    # 5. Output Summary
    print("=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    print(f"Forget Accuracy:          {report['forget_accuracy']:.2f}% (Loss: {report['forget_loss']:.4f})")
    print(f"Retain Accuracy:          {report['retain_accuracy']:.2f}% (Loss: {report['retain_loss']:.4f})")
    print(f"Test Accuracy:            {report['test_accuracy']:.2f}% (Drop: {report.get('test_accuracy_drop', 0.0):.2f}%)")
    print(f"MIA AUC Score:            {report['mia_auc_score']:.4f} (Ideal: ~0.5000)")
    print(f"Parameter Distance:       {report['parameter_distance']:.4f}")
    print(f"Unlearning Time:          {report['unlearning_time_seconds']:.2f}s")
    if extra_info:
        print(f"SOM Score:                {extra_info.get('som_score', 0.0):.4f} ({extra_info.get('health_state')})")
        print(f"Primitive Selected:       {extra_info.get('chosen_primitive')} (Path: {' -> '.join(extra_info.get('escalation_path', []))})")
        print(f"Verification Passed:      {extra_info.get('verification_passed')}")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"{args.method}_workload_{args.workload}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"\nSaved results to: {result_file}")


if __name__ == "__main__":
    main()
