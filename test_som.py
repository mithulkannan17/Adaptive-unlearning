from __future__ import annotations

import torch

from src.controller.som import HealthState, SubspaceTracker


def test_subspace_tracker():
    print("=" * 70)
    print("TESTING SUBSPACE ORTHOGONALITY METRIC (SOM)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 1000

    tracker = SubspaceTracker(
        decay_factor=0.9,
        max_rank=5,
        green_threshold=0.4,
        red_threshold=0.8,
        device=device,
    )

    # 1. First gradient: Basis is initialized, score should be 0.0 (Green)
    g1 = torch.randn(dim, device=device)
    m1 = tracker.update(g1)
    print(f"Step 1 - SOM: {m1.som_score:.4f}, State: {m1.health_state.value}, Rank: {m1.rank}")
    assert m1.health_state == HealthState.GREEN
    assert m1.rank == 1

    # 2. Orthogonal gradient: SOM score should be near 0.0 (Green)
    # Construct vector orthogonal to g1
    unit_g1 = g1 / torch.norm(g1)
    random_v = torch.randn(dim, device=device)
    g2_orth = random_v - torch.dot(random_v, unit_g1) * unit_g1
    m2 = tracker.update(g2_orth)
    print(f"Step 2 (Orthogonal) - SOM: {m2.som_score:.4f}, State: {m2.health_state.value}, Rank: {m2.rank}")
    assert m2.som_score < 0.2
    assert m2.health_state == HealthState.GREEN

    # 3. Collinear / parallel gradient: SOM score should be high (Red)
    g3_parallel = unit_g1 * 5.0
    m3 = tracker.update(g3_parallel)
    print(f"Step 3 (Collinear) - SOM: {m3.som_score:.4f}, State: {m3.health_state.value}, Rank: {m3.rank}")
    assert m3.som_score > 0.8
    assert m3.health_state == HealthState.RED

    print("\nSOM Tracker validation passed successfully.")


if __name__ == "__main__":
    test_subspace_tracker()
