from src.workloads.uniform_random import (
    generate_workload_a,
)


def main():

    dataset_size = 50_000

    requests = generate_workload_a(
        dataset_size=dataset_size,
        deletion_fraction=0.01,
        rounds=20,
        seed=42,
    )

    print("=" * 70)
    print("WORKLOAD A VALIDATION")
    print("=" * 70)

    all_forget = set()

    previous_remaining = dataset_size

    for request in requests:

        forget_indices = set(
            request.forget_indices
        )

        # No sample may be forgotten twice.
        overlap = all_forget.intersection(
            forget_indices
        )

        assert len(overlap) == 0, (
            f"Duplicate forget indices detected "
            f"in round {request.round_id}"
        )

        all_forget.update(
            forget_indices
        )

        expected_remaining = (
            previous_remaining
            - request.forget_size
        )

        assert (
            request.remaining_size
            == expected_remaining
        )

        print(
            f"Round {request.round_id:02d} | "
            f"Forgot: {request.forget_size:5d} | "
            f"Remaining: {request.remaining_size:5d} | "
            f"Cumulative forgotten: {len(all_forget):5d}"
        )

        previous_remaining = (
            request.remaining_size
        )

    print()
    print("Validation successful.")
    print(
        f"Total forgotten: {len(all_forget)}"
    )
    print(
        f"Final remaining: {previous_remaining}"
    )


if __name__ == "__main__":
    main()