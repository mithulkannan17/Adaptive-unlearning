# Project Memory & Architectural Context
## Project: ASUC-SOM (Adaptive Sequential Machine Unlearning Engine)

---

## 1. Key Architectural Decisions & Rationale

### 1.1. Why Use SOM Over Raw Parameter Distance ($\|\theta_t - \theta_0\|$)?
- **Observation**: Scalar weight distance $\|\theta_t - \theta_0\|$ increases monotonically with every deletion round regardless of model health. It cannot distinguish between safe updates along new dimensions and damaging updates that repeatedly distort the same critical weight channels.
- **Decision**: SOM maintains an orthonormal basis $Q_k$ of past gradient directions. The cosine projection $\|Q_k^T g_t\|_2 / \|g_t\|_2$ measures **geometric directional overlap**. If overlap $\ge 0.80$, it flags parameter fatigue *before* accuracy collapses.

### 1.2. Why Tiered Routing (SSD + SalUn) Instead of Pure SalUn or Pure Retrain?
- **SSD**: Computes in $< 5$ seconds ($150\times$ speedup) via direct Fisher dampening, but causes rank collapse if repeated indiscriminately on overlapping subspaces.
- **SalUn**: Slower (~$15$ seconds, $15.6\times$ speedup) but preserves retain utility through gradient regularization.
- **Decision**: Use SSD as the fast default for healthy, balanced batches, and dynamically route to SalUn only when SOM enters the Amber/Red state or when requests are class-concentrated ($H(c) < 0.40$).

### 1.3. Why Closed-Loop Verification with Controlled Escalation?
- Approximate unlearning algorithms can occasionally fail due to stochastic batch variance.
- **Decision**: Implement a checkpoint rollback gate that verifies forget accuracy ($\le 15\%$) and retain drop ($\le 3\%$) on held-out validation slices before committing model updates.

---

## 2. Experimental Findings & Known Failure Modes

### 2.1. The Static SSD Collapse Phenomenon
- **Finding**: In `sequential_ssd_workload_a`, SSD maintains high accuracy for Rounds 1–2 (99.53%, 99.43%), begins degrading at Round 3 (49.42%), and totally collapses by Round 4 (9.96% retain accuracy / random chance).
- **Root Cause**: Repeated parameter dampening without retain regularization drives critical weights to near zero, destroying network representational capacity.
- **Paper Impact**: Serves as the central baseline proof demonstrating why static unlearning cannot be used for continuous sequential compliance.

### 2.2. Class-Concentrated Deletions (Workload B)
- When 500 samples of a single class are deleted, Fisher information is concentrated on a small set of classification filters. Single-pass dampening removes those filters entirely, destroying the class boundary.
- **Mitigation**: Request profiler flags $H(c) = 0.0$ and immediately routes to Tier 2 (SalUn) with retain loss regularization.

---

## 3. Hardware & Benchmark Environment

- **Primary GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, CUDA 12.x).
- **Host CPU**: AMD Ryzen / Intel Core (Windows 11).
- **PyTorch Stack**: PyTorch 2.x, Torchvision, CUDA backend.
- **Dataset**: CIFAR-10 (50,000 train, 10,000 test, 10 classes, $32 \times 32 \times 3$).
- **Model Backbone**: ResNet-18 adapted for CIFAR-10 ($11,173,962$ parameters).

---

## 4. Key File Map & Responsibilities

| File Path | Description |
| :--- | :--- |
| `train.py` | Baseline training pipeline for CIFAR-10 ResNet-18. |
| `run_sequential.py` | Main sequential benchmark runner supporting ASUC, SSD, SalUn, GA. |
| `run_unlearning.py` | Single-shot unlearning benchmark runner. |
| `evaluate.py` | Scans `results/` and generates summary comparison tables. |
| `src/controllers/asuc_controller.py` | Request profiler, tiered router, and ASUC state machine. |
| `src/controllers/subspace_tracker.py` | SOM low-rank basis $Q_k$ and Gram-Schmidt projection engine. |
| `src/unlearning/ssd.py` | Selective Synaptic Dampening implementation. |
| `src/unlearning/salun.py` | Saliency Unlearning with retain regularization. |
| `src/unlearning/exact_retrain.py` | Clean scratch retraining engine. |
| `src/verification/verifier.py` | Closed-loop validation gate and rollback logic. |
| `visualizer/server.py` | HTTP server exposing live results APIs on port 8080. |
| `visualizer/app.js` | Frontend logic, Chart.js graphs, and 60 FPS SOM 3D Canvas. |
| `scripts/generate_paper_figures.py` | Generates 300 DPI PNG & vector PDF publication figures + LaTeX tables. |
