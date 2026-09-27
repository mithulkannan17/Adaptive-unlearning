from src.experiments.sequential_runner import (
    SequentialExperimentRunner,
)
from src.workloads.uniform_random import (
    generate_workload_a,
)


def main():

    dataset_size = 50_000

    workload = generate_workload_a(
        dataset_size=dataset_size,
        deletion_fraction=0.01,
        rounds=20,
        seed=42,
    )

    runner = SequentialExperimentRunner(
        dataset_size=dataset_size,
        workload=workload,
        output_dir="./results/sequential_test",
    )

    results = runner.run()

    state = runner.get_current_state()

    print()
    print("=" * 70)
    print("SEQUENTIAL ENGINE VALIDATION")
    print("=" * 70)

    print(
        f"Dataset size:       {state.dataset_size}"
    )

    print(
        f"Forgotten:           {state.forgotten_size}"
    )

    print(
        f"Remaining:           {state.active_size}"
    )

    print(
        f"Completed rounds:    {state.completed_rounds}"
    )

    assert (
        state.forgotten_size
        + state.active_size
        == dataset_size
    )

    assert (
        state.completed_rounds
        == 20
    )

    assert len(results) == 20

    print()
    print("Sequential engine validation successful.")


if __name__ == "__main__":
    main()