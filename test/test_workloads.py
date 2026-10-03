from __future__ import annotations

import numpy as np

from src.workloads.class_sequential import generate_workload_b
from src.workloads.high_influence import generate_workload_c
from src.workloads.uniform_random import generate_workload_a


def test_workloads():
    print("=" * 70)
    print("TESTING WORKLOAD GENERATORS (A, B, C)")
    print("=" * 70)

    dataset_size = 1000
    targets = np.random.randint(0, 10, size=dataset_size)
    influence_scores = np.random.randn(dataset_size)

    # 1. Test Workload A (Uniform Random)
    reqs_a = generate_workload_a(dataset_size=dataset_size, deletion_fraction=0.05, rounds=5, seed=42)
    print(f"Workload A: Generated {len(reqs_a)} rounds.")
    forgotten_a = set()
    for req in reqs_a:
        f_set = set(req.forget_indices)
        assert len(f_set.intersection(forgotten_a)) == 0, "Duplicate forget indices in Workload A"
        forgotten_a.update(f_set)
        assert req.remaining_size == dataset_size - len(forgotten_a)
    print("[OK] Workload A validation passed.")

    # 2. Test Workload B (Class Sequential)
    reqs_b = generate_workload_b(targets=targets, rounds=5, samples_per_round=20, seed=42)
    print(f"Workload B: Generated {len(reqs_b)} rounds.")
    forgotten_b = set()
    for r_idx, req in enumerate(reqs_b):
        f_set = set(req.forget_indices)
        assert len(f_set.intersection(forgotten_b)) == 0, "Duplicate forget indices in Workload B"
        forgotten_b.update(f_set)
        # Verify samples belong to the target class
        target_class = r_idx % 10
        classes_in_round = targets[req.forget_indices]
        assert np.all(classes_in_round == target_class), f"Class mismatch in round {r_idx}"
    print("[OK] Workload B validation passed.")

    # 3. Test Workload C (High Influence)
    reqs_c = generate_workload_c(dataset_size=dataset_size, influence_scores=influence_scores, rounds=5, samples_per_round=20, seed=42)
    print(f"Workload C: Generated {len(reqs_c)} rounds.")
    forgotten_c = set()
    prev_min_influence = float("inf")
    for req in reqs_c:
        f_set = set(req.forget_indices)
        assert len(f_set.intersection(forgotten_c)) == 0, "Duplicate forget indices in Workload C"
        forgotten_c.update(f_set)
        round_scores = influence_scores[req.forget_indices]
        assert np.max(round_scores) <= prev_min_influence + 1e-5, "Workload C must sort by highest influence first"
        prev_min_influence = np.min(round_scores)
    print("[OK] Workload C validation passed.")

    print("\nAll workload generators validated successfully!")


if __name__ == "__main__":
    test_workloads()
