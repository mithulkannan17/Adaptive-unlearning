from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.data.cifar10 import get_cifar10_datasets
from src.models.resnet import create_resnet18
from src.unlearning.ssd import SSDUnlearner
from src.utils import get_device, load_checkpoint
from src.workloads.uniform_random import generate_workload_a


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "./data"

CHECKPOINT_PATH = (
    "./checkpoints/"
    "base_cifar10/"
    "best_model.pth"
)

OUTPUT_DIR = (
    "./results/"
    "sequential_ssd"
)

CHECKPOINT_OUTPUT_DIR = (
    "./checkpoints/"
    "sequential_ssd"
)

BATCH_SIZE = 128
NUM_WORKERS = 2

ROUNDS = 20

# ------------------------------------------------------------
# SSD configuration
# ------------------------------------------------------------

SSD_DAMPENING_CONSTANT = 1.0
SSD_SELECTION_WEIGHTING = 10.0
SSD_EXPONENT = 1.0
SSD_LOWER_BOUND = 1.0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # We do not force deterministic CUDA algorithms here
    # because that can significantly reduce performance.


# ============================================================
# MODEL UTILITIES
# ============================================================

def parameter_distance(
    model_a,
    model_b,
) -> float:
    """
    Euclidean distance between complete parameter vectors.
    """

    distance_squared = 0.0

    with torch.no_grad():

        for parameter_a, parameter_b in zip(
            model_a.parameters(),
            model_b.parameters(),
        ):

            difference = (
                parameter_a.detach()
                -
                parameter_b.detach()
            )

            distance_squared += (
                difference.pow(2)
                .sum()
                .item()
            )

    return distance_squared ** 0.5


def parameter_norm(model) -> float:
    """
    Euclidean norm of the complete parameter vector.
    """

    norm_squared = 0.0

    with torch.no_grad():

        for parameter in model.parameters():

            norm_squared += (
                parameter.detach()
                .pow(2)
                .sum()
                .item()
            )

    return norm_squared ** 0.5


def count_parameters(model) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_changed_parameters(
    model_before,
    model_after,
    tolerance: float = 0.0,
) -> int:
    """
    Count parameters whose values changed between
    two model states.
    """

    changed = 0

    with torch.no_grad():

        for before, after in zip(
            model_before.parameters(),
            model_after.parameters(),
        ):

            difference = (
                before.detach()
                -
                after.detach()
            ).abs()

            if tolerance > 0.0:

                changed += (
                    difference > tolerance
                ).sum().item()

            else:

                changed += (
                    difference > 0.0
                ).sum().item()

    return int(changed)


def clone_model_state(model):

    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def state_dict_distance(
    state_a,
    state_b,
) -> float:
    """
    Distance between two state dictionaries.
    """

    distance_squared = 0.0

    with torch.no_grad():

        for key in state_a:

            tensor_a = state_a[key]
            tensor_b = state_b[key]

            difference = (
                tensor_a.detach()
                -
                tensor_b.detach()
            )

            distance_squared += (
                difference.pow(2)
                .sum()
                .item()
            )

    return distance_squared ** 0.5


def verify_models_are_independent(
    model_a,
    model_b,
) -> bool:
    """
    Verify that two models do not share parameter storage.
    """

    for parameter_a, parameter_b in zip(
        model_a.parameters(),
        model_b.parameters(),
    ):

        if (
            parameter_a.data_ptr()
            ==
            parameter_b.data_ptr()
        ):
            return False

    return True


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def save_sequential_checkpoint(
    model,
    path: Path,
    round_number: int,
    cumulative_forget,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "round": round_number,
            "model_state_dict": model.state_dict(),
            "cumulative_forget": sorted(
                list(cumulative_forget)
            ),
            "method": "SSD",
            "dataset": "cifar10",
            "seed": SEED,
        },
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("ADAPTIVE SEQUENTIAL MACHINE UNLEARNING")
    print("SEQUENTIAL SSD")
    print("=" * 70)

    device = get_device("cuda")

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

    print("=" * 70)

    # ========================================================
    # DATA
    # ========================================================

    train_dataset, _ = (
        get_cifar10_datasets(
            data_dir=DATA_DIR
        )
    )

    dataset_size = len(
        train_dataset
    )

    print(
        f"Training samples: "
        f"{dataset_size}"
    )

    # ========================================================
    # LOAD FROZEN BASELINE
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING FROZEN BASELINE MODEL")
    print("=" * 70)

    baseline_model = create_resnet18(
        num_classes=10
    )

    load_checkpoint(
        path=CHECKPOINT_PATH,
        model=baseline_model,
        device=device,
    )

    baseline_model = (
        baseline_model
        .to(device)
    )

    baseline_model.eval()

    for parameter in (
        baseline_model.parameters()
    ):
        parameter.requires_grad = False

    baseline_parameter_count = (
        count_parameters(
            baseline_model
        )
    )

    baseline_parameter_norm = (
        parameter_norm(
            baseline_model
        )
    )

    baseline_state = (
        clone_model_state(
            baseline_model
        )
    )

    print(
        f"Baseline parameter count: "
        f"{baseline_parameter_count:,}"
    )

    print(
        f"Baseline parameter norm: "
        f"{baseline_parameter_norm:.6f}"
    )

    # ========================================================
    # LOAD INDEPENDENT CURRENT MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING INDEPENDENT SEQUENTIAL MODEL")
    print("=" * 70)

    current_model = create_resnet18(
        num_classes=10
    )

    load_checkpoint(
        path=CHECKPOINT_PATH,
        model=current_model,
        device=device,
    )

    current_model = (
        current_model
        .to(device)
    )

    current_model.eval()

    independent = (
        verify_models_are_independent(
            baseline_model,
            current_model,
        )
    )

    if not independent:

        raise RuntimeError(
            "Baseline and current model share "
            "parameter storage."
        )

    print(
        "Model independence check: PASSED"
    )

    initial_distance = (
        parameter_distance(
            baseline_model,
            current_model,
        )
    )

    print(
        f"Initial parameter distance: "
        f"{initial_distance:.10f}"
    )

    if initial_distance > 1e-6:

        raise RuntimeError(
            "Baseline and current model "
            "do not contain identical "
            "initial parameters."
        )

    # ========================================================
    # WORKLOAD
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATING WORKLOAD A")
    print("=" * 70)

    workload = (
        generate_workload_a(
            dataset_size=dataset_size,
            deletion_fraction=0.01,
            rounds=ROUNDS,
            seed=SEED,
        )
    )

    print(
        f"Workload rounds: "
        f"{len(workload)}"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    cumulative_forget = set()

    total_unlearning_time = 0.0

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    output_path = Path(
        OUTPUT_DIR
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_output_path = Path(
        CHECKPOINT_OUTPUT_DIR
    )

    checkpoint_output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SEQUENTIAL SSD
    # ========================================================

    print()
    print("=" * 70)
    print("SEQUENTIAL SSD")
    print("=" * 70)

    print(
        f"Rounds: "
        f"{len(workload)}"
    )

    print(
        f"Initial model: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        "Baseline reference: FROZEN"
    )

    print()

    # ========================================================
    # ROUND LOOP
    # ========================================================

    for round_number, request in enumerate(
        workload,
        start=1,
    ):

        print("-" * 70)
        print(
            f"ROUND {round_number}"
        )
        print("-" * 70)

        new_forget_indices = [
            int(index)
            for index in request.forget_indices
        ]

        # ----------------------------------------------------
        # CUMULATIVE FORGET SET
        # ----------------------------------------------------

        cumulative_forget.update(
            new_forget_indices
        )

        print(
            f"New forget set: "
            f"{len(new_forget_indices)}"
        )

        print(
            f"Cumulative forgotten: "
            f"{len(cumulative_forget)}"
        )

        print(
            f"Remaining samples: "
            f"{dataset_size - len(cumulative_forget)}"
        )

        # ----------------------------------------------------
        # SAVE CURRENT STATE BEFORE SSD
        # ----------------------------------------------------

        state_before = (
            clone_model_state(
                current_model
            )
        )

        parameter_norm_before = (
            parameter_norm(
                current_model
            )
        )

        # ----------------------------------------------------
        # RUN SSD
        # ----------------------------------------------------

        print()
        print(
            "Running SSD on current model..."
        )

        unlearner = SSDUnlearner(
            device=device,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            dampening_constant=(
                SSD_DAMPENING_CONSTANT
            ),
            selection_weighting=(
                SSD_SELECTION_WEIGHTING
            ),
            exponent=SSD_EXPONENT,
            lower_bound=SSD_LOWER_BOUND,
        )

        round_start = time.perf_counter()

        result = (
            unlearner.unlearn(
                model=current_model,
                train_dataset=train_dataset,
                forget_indices=(
                    new_forget_indices
                ),
            )
        )

        round_time = (
            time.perf_counter()
            -
            round_start
        )

        # ----------------------------------------------------
        # EXPLICITLY PROPAGATE UPDATED MODEL
        # ----------------------------------------------------

        updated_model = (
            result.model
            .to(device)
        )

        updated_model.eval()

        # ----------------------------------------------------
        # VERIFY SSD ACTUALLY CHANGED THE MODEL
        # ----------------------------------------------------

        state_after = (
            clone_model_state(
                updated_model
            )
        )

        round_model_distance = (
            state_dict_distance(
                state_before,
                state_after,
            )
        )

        actual_changed_parameters = (
            count_changed_parameters(
                current_model,
                updated_model,
            )
        )

        # ----------------------------------------------------
        # HARD VALIDATION
        # ----------------------------------------------------

        if (
            result.selected_parameters > 0
            and
            round_model_distance <= 1e-12
        ):

            raise RuntimeError(
                "\n"
                "SSD STATE PROPAGATION FAILURE\n"
                "--------------------------------\n"
                f"Round: {round_number}\n"
                f"Selected parameters: "
                f"{result.selected_parameters:,}\n"
                f"Actual model distance: "
                f"{round_model_distance:.12f}\n"
                "\n"
                "SSD selected parameters but "
                "the returned model is identical "
                "to the input model.\n"
                "\n"
                "Stopping immediately rather than "
                "producing invalid sequential results."
            )

        # ----------------------------------------------------
        # THIS IS THE CRITICAL LINE
        #
        # The UPDATED model becomes the input
        # for the NEXT deletion request.
        # ----------------------------------------------------

        current_model = updated_model

        # ----------------------------------------------------
        # VERIFY CURRENT MODEL IS DIFFERENT
        # FROM THE FROZEN BASELINE
        # ----------------------------------------------------

        distance_from_baseline = (
            parameter_distance(
                baseline_model,
                current_model,
            )
        )

        current_parameter_norm = (
            parameter_norm(
                current_model
            )
        )

        relative_distance = (
            distance_from_baseline
            /
            max(
                baseline_parameter_norm,
                1e-12,
            )
        )

        total_unlearning_time += (
            round_time
        )

        # ----------------------------------------------------
        # SAVE ROUND CHECKPOINT
        # ----------------------------------------------------

        round_checkpoint = (
            checkpoint_output_path
            /
            f"round_{round_number:02d}_model.pth"
        )

        save_sequential_checkpoint(
            model=current_model,
            path=round_checkpoint,
            round_number=round_number,
            cumulative_forget=cumulative_forget,
        )

        # ----------------------------------------------------
        # ROUND RESULT
        # ----------------------------------------------------

        round_result = {

            "round": round_number,

            "new_forget_samples": (
                len(new_forget_indices)
            ),

            "cumulative_forget_samples": (
                len(cumulative_forget)
            ),

            "remaining_samples": (
                dataset_size
                -
                len(cumulative_forget)
            ),

            "unlearning_time": (
                round_time
            ),

            "cumulative_unlearning_time": (
                total_unlearning_time
            ),

            "selected_parameters": (
                result.selected_parameters
            ),

            "actual_changed_parameters": (
                actual_changed_parameters
            ),

            "total_parameters": (
                result.total_parameters
            ),

            "selection_ratio": (
                result.selection_ratio
            ),

            "parameter_norm_before": (
                parameter_norm_before
            ),

            "parameter_norm_after": (
                current_parameter_norm
            ),

            "parameter_change_this_round": (
                round_model_distance
            ),

            "baseline_parameter_norm": (
                baseline_parameter_norm
            ),

            "parameter_distance_from_baseline": (
                distance_from_baseline
            ),

            "relative_parameter_distance": (
                relative_distance
            ),

            "checkpoint": str(
                round_checkpoint
            ),
        }

        results.append(
            round_result
        )

        # ----------------------------------------------------
        # PRINT DIAGNOSTICS
        # ----------------------------------------------------

        print()

        print(
            f"Selected parameters: "
            f"{result.selected_parameters:,}"
        )

        print(
            f"Actually changed parameters: "
            f"{actual_changed_parameters:,}"
        )

        print(
            f"Selection ratio: "
            f"{result.selection_ratio:.6f}"
        )

        print(
            f"Parameter norm before: "
            f"{parameter_norm_before:.6f}"
        )

        print(
            f"Parameter norm after: "
            f"{current_parameter_norm:.6f}"
        )

        print(
            f"Parameter change this round: "
            f"{round_model_distance:.10f}"
        )

        print(
            f"Parameter distance from baseline: "
            f"{distance_from_baseline:.10f}"
        )

        print(
            f"Relative parameter distance: "
            f"{relative_distance:.10f}"
        )

        print(
            f"Unlearning time: "
            f"{round_time:.2f}s"
        )

        print(
            f"Total unlearning time: "
            f"{total_unlearning_time:.2f}s"
        )

        print()

        print(
            f"Round {round_number:02d} | "
            f"New Forget: "
            f"{len(new_forget_indices):5d} | "
            f"Cumulative Forget: "
            f"{len(cumulative_forget):5d} | "
            f"Remaining: "
            f"{dataset_size - len(cumulative_forget):5d}"
        )

    # ========================================================
    # FINAL DIAGNOSTICS
    # ========================================================

    final_parameter_distance = (
        parameter_distance(
            baseline_model,
            current_model,
        )
    )

    final_parameter_norm = (
        parameter_norm(
            current_model
        )
    )

    final_relative_distance = (
        final_parameter_distance
        /
        max(
            baseline_parameter_norm,
            1e-12,
        )
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_model_path = (
        checkpoint_output_path
        /
        "final_model.pth"
    )

    save_sequential_checkpoint(
        model=current_model,
        path=final_model_path,
        round_number=len(results),
        cumulative_forget=cumulative_forget,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "method": "SSD",

        "experiment": (
            "sequential_unlearning"
        ),

        "dataset": "cifar10",

        "seed": SEED,

        "rounds_requested": ROUNDS,

        "rounds_completed": (
            len(results)
        ),

        "initial_training_samples": (
            dataset_size
        ),

        "final_cumulative_forget": (
            len(cumulative_forget)
        ),

        "final_remaining_samples": (
            dataset_size
            -
            len(cumulative_forget)
        ),

        "baseline_parameter_count": (
            baseline_parameter_count
        ),

        "baseline_parameter_norm": (
            baseline_parameter_norm
        ),

        "final_parameter_norm": (
            final_parameter_norm
        ),

        "final_parameter_distance": (
            final_parameter_distance
        ),

        "final_relative_parameter_distance": (
            final_relative_distance
        ),

        "total_unlearning_time": (
            total_unlearning_time
        ),

        "average_round_time": (
            total_unlearning_time
            /
            max(len(results), 1)
        ),

        "ssd_configuration": {

            "dampening_constant": (
                SSD_DAMPENING_CONSTANT
            ),

            "selection_weighting": (
                SSD_SELECTION_WEIGHTING
            ),

            "exponent": (
                SSD_EXPONENT
            ),

            "lower_bound": (
                SSD_LOWER_BOUND
            ),
        },

        "model_integrity": {

            "baseline_current_independent": True,

            "baseline_frozen": True,

            "initial_parameter_distance": (
                initial_distance
            ),

            "sequential_model_propagation": True,

            "round_change_validation": True,
        },

        "results": results,
    }

    summary_path = (
        output_path
        /
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("SEQUENTIAL SSD COMPLETE")
    print("=" * 70)

    print(
        f"Rounds completed: "
        f"{len(results)}"
    )

    print(
        f"Total forgotten: "
        f"{len(cumulative_forget)}"
    )

    print(
        f"Final remaining: "
        f"{dataset_size - len(cumulative_forget)}"
    )

    print(
        f"Total SSD time: "
        f"{total_unlearning_time:.2f}s"
    )

    print(
        f"Average round time: "
        f"{total_unlearning_time / max(len(results), 1):.2f}s"
    )

    print(
        f"Final parameter norm: "
        f"{final_parameter_norm:.10f}"
    )

    print(
        f"Final parameter distance: "
        f"{final_parameter_distance:.10f}"
    )

    print(
        f"Final relative parameter distance: "
        f"{final_relative_distance:.10f}"
    )

    print(
        f"Final model: "
        f"{final_model_path}"
    )

    print(
        f"Results: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()