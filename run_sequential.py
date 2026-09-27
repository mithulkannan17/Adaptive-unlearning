from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from src.controller.asuc import ASUCController
from src.data.cifar10 import get_cifar10_datasets, get_test_loader
from src.evaluation.evaluator import UnlearningEvaluator
from src.experiments.state import SequentialState
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
    parser = argparse.ArgumentParser(description="Run sequential machine unlearning benchmark.")
    parser.add_argument("--method", type=str, default="asuc",
                        choices=["asuc", "ssd", "salun", "gradient_ascent", "retrain"],
                        help="Unlearning method.")
    parser.add_argument("--workload", type=str, default="a", choices=["a", "b", "c"],
                        help="Workload type (a: Uniform Random, b: Class Sequential, c: High Influence).")
    parser.add_argument("--rounds", type=int, default=20, help="Number of sequential rounds.")
    parser.add_argument("--deletion_fraction", type=float, default=0.01, help="Deletion fraction for Workload A.")
    parser.add_argument("--samples_per_round", type=int, default=500, help="Samples per round for Workload B/C.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/base_cifar10/best_model.pth",
                        help="Path to baseline checkpoint.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device("cuda")

    if args.output_dir is None:
        args.output_dir = f"./results/sequential_{args.method}_workload_{args.workload}"
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SEQUENTIAL MACHINE UNLEARNING BENCHMARK")
    print(f"Method: {args.method.upper()} | Workload: {args.workload.upper()} | Rounds: {args.rounds}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Output Directory: {output_path}")
    print("=" * 80)

    train_dataset, test_dataset = get_cifar10_datasets(data_dir="./data")
    test_loader = get_test_loader(test_dataset, batch_size=128, num_workers=2)
    dataset_size = len(train_dataset)

    # 1. Load Baseline Model
    baseline_model = create_resnet18(num_classes=10)
    load_checkpoint(args.checkpoint, baseline_model, device=device)
    baseline_model = baseline_model.to(device)
    baseline_model.eval()

    # 2. Generate Workload
    print(f"\nGenerating Workload {args.workload.upper()}...")
    if args.workload == "a":
        workload = generate_workload_a(
            dataset_size=dataset_size,
            deletion_fraction=args.deletion_fraction,
            rounds=args.rounds,
            seed=args.seed,
        )
    elif args.workload == "b":
        workload = generate_workload_b(
            targets=train_dataset.targets,
            rounds=args.rounds,
            samples_per_round=args.samples_per_round,
            seed=args.seed,
        )
    elif args.workload == "c":
        print("Computing sample influence scores under baseline model...")
        scores = compute_sample_losses(baseline_model, train_dataset, device=device)
        workload = generate_workload_c(
            dataset_size=dataset_size,
            influence_scores=scores,
            rounds=args.rounds,
            samples_per_round=args.samples_per_round,
            seed=args.seed,
        )

    # 3. Setup Method Engines
    evaluator = UnlearningEvaluator(device=device, batch_size=128, num_workers=2)
    current_model = copy.deepcopy(baseline_model).to(device)

    if args.method == "asuc":
        controller = ASUCController(device=device, num_classes=10, retrain_epochs=20)
    elif args.method == "ssd":
        ssd = SSDUnlearner(device=device, dampening_constant=1.0, selection_weighting=10.0, lower_bound=0.0)
    elif args.method == "salun":
        salun = SalUnUnlearner(device=device, epochs=1, learning_rate=1e-3, saliency_threshold_ratio=0.5)
    elif args.method == "gradient_ascent":
        ga = GradientAscentUnlearner(device=device, epochs=5, learning_rate=1e-3)
    elif args.method == "retrain":
        retrainer = ExactRetrainer(device=device, epochs=20)

    # 4. Sequential Loop
    state = SequentialState(dataset_size=dataset_size)
    round_summaries: List[Dict] = []
    total_unlearn_time = 0.0

    print("\nStarting Sequential Execution:")
    print("-" * 80)

    for request in workload:
        r_id = request.round_id
        forget_indices = request.forget_indices
        round_state = state.apply_deletion(r_id, forget_indices)
        retain_indices = round_state.retain_indices
        cum_forget_indices = round_state.cumulative_forget_indices

        round_t0 = time.perf_counter()
        meta = {}

        if args.method == "asuc":
            current_model, decision = controller.step(
                current_model=current_model,
                baseline_model=baseline_model,
                train_dataset=train_dataset,
                forget_indices=forget_indices,
                retain_indices=retain_indices,
                round_id=r_id,
            )
            round_unlearn_time = decision.unlearning_time_seconds
            meta = {
                "som_score": decision.som_score,
                "health_state": decision.health_state,
                "chosen_primitive": decision.chosen_primitive,
                "escalation_path": decision.escalation_path,
                "verification_passed": decision.verification_passed,
                "verification_time": decision.verification_time_seconds,
            }
        elif args.method == "ssd":
            res = ssd.unlearn(current_model, train_dataset, forget_indices)
            current_model = res.model
            round_unlearn_time = res.unlearning_time
        elif args.method == "salun":
            res = salun.unlearn(current_model, train_dataset, forget_indices, retain_indices)
            current_model = res.model
            round_unlearn_time = res.unlearning_time_seconds
        elif args.method == "gradient_ascent":
            res = ga.unlearn(current_model, train_dataset, forget_indices)
            current_model = res.model
            round_unlearn_time = res.training_time_seconds
        elif args.method == "retrain":
            res = retrainer.train(train_dataset, retain_indices)
            current_model = res.model
            round_unlearn_time = res.training_time_seconds

        total_unlearn_time += round_unlearn_time

        # Quick evaluation on new forget set and test set
        eval_metrics = evaluator.evaluate(
            model=current_model,
            train_dataset=train_dataset,
            test_loader=test_loader,
            forget_indices=forget_indices,
            retain_indices=retain_indices,
            reference_model=baseline_model,
        )

        round_entry = {
            "round_id": r_id,
            "new_forget_size": len(forget_indices),
            "cumulative_forget_size": len(cum_forget_indices),
            "remaining_size": len(retain_indices),
            "unlearning_time_seconds": round_unlearn_time,
            "cumulative_unlearning_time": total_unlearn_time,
            "forget_accuracy": eval_metrics["forget_accuracy"],
            "forget_loss": eval_metrics["forget_loss"],
            "retain_accuracy": eval_metrics["retain_accuracy"],
            "retain_loss": eval_metrics["retain_loss"],
            "test_accuracy": eval_metrics["test_accuracy"],
            "test_loss": eval_metrics["test_loss"],
            "parameter_norm": eval_metrics["parameter_norm"],
            "parameter_distance": eval_metrics["parameter_distance"],
            "relative_parameter_distance": eval_metrics["parameter_distance"] / max(eval_metrics["parameter_norm"], 1e-12),
            **meta,
        }
        round_summaries.append(round_entry)

        # Save incremental round file
        round_file = output_path / f"round_{r_id:02d}.json"
        with open(round_file, "w", encoding="utf-8") as f:
            json.dump(round_entry, f, indent=4)

        status_str = f" | {meta['chosen_primitive']}" if "chosen_primitive" in meta else ""
        print(
            f"Round {r_id:02d}/{args.rounds:02d} | "
            f"Forgot: {len(forget_indices):4d} | "
            f"Cumul: {len(cum_forget_indices):5d} | "
            f"Test Acc: {eval_metrics['test_accuracy']:.2f}% | "
            f"Retain Acc: {eval_metrics['retain_accuracy']:.2f}% | "
            f"Forget Acc: {eval_metrics['forget_accuracy']:.2f}% | "
            f"Time: {round_unlearn_time:.1f}s"
            f"{status_str}"
        )

    # 5. Save Final Summary
    summary = {
        "method": args.method,
        "workload": args.workload,
        "rounds_requested": args.rounds,
        "rounds_completed": len(round_summaries),
        "dataset_size": dataset_size,
        "final_cumulative_forgotten": len(cum_forget_indices),
        "final_remaining": len(retain_indices),
        "total_unlearning_time_seconds": total_unlearn_time,
        "average_round_time_seconds": total_unlearn_time / max(len(round_summaries), 1),
        "final_test_accuracy": round_summaries[-1]["test_accuracy"] if round_summaries else 0.0,
        "final_retain_accuracy": round_summaries[-1]["retain_accuracy"] if round_summaries else 0.0,
        "final_parameter_distance": round_summaries[-1]["parameter_distance"] if round_summaries else 0.0,
        "rounds": round_summaries,
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("=" * 80)
    print("SEQUENTIAL EXPERIMENT COMPLETE")
    print(f"Rounds: {len(round_summaries)} | Total Time: {total_unlearn_time:.2f}s")
    print(f"Final Test Accuracy: {summary['final_test_accuracy']:.2f}% | Final Retain: {summary['final_retain_accuracy']:.2f}%")
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
