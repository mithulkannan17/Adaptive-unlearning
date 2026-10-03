# ASUC-SOM: Adaptive Sequential Machine Unlearning

[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An adaptive, closed-loop machine unlearning system designed for continuous data deletion requests (e.g., GDPR Article 17 / CCPA compliance). 

Most existing unlearning methods (SSD, SalUn, Gradient Ascent) were built for one-time deletions. When applied sequentially across multiple rounds, they suffer from **parameter fatigue**—rapidly destroying model accuracy. ASUC-SOM monitors parameter subspace overlap in real time, dynamically routes incoming requests to the optimal unlearning primitive, and verifies model stability before committing updates.

---

## Benchmark Results

Evaluated on **ResNet-18 on CIFAR-10** across 20 sequential deletion rounds ($500$ samples per round = $10,000$ total deleted samples):

| Method | Routing Strategy | Workload | Retain Acc (%) | Test Acc (%) | Collapse Round | Total Compute | Speedup vs Retrain |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ASUC-SOM (Ours)** | **Adaptive Meta-Controller** | **Workload A (Uniform)** | **97.19%** | **90.89%** | **None (Stable)** | ~15,655s | **1.3× – 148×** |
| **ASUC-SOM (Ours)** | **Adaptive Meta-Controller** | **Workload B (Class-Seq)** | **98.26%** | **92.33%** | **None (Stable)** | ~8,614s | **2.4×** |
| **ASUC-SOM (Ours)** | **Adaptive Meta-Controller** | **Workload C (High-Loss)** | **99.11%** | **90.75%** | **None (Stable)** | ~8,565s | **2.5×** |
| **SSD** (Foster et al., 2024) | Static Fisher Dampening | Workload A | 10.00% | 10.00% | **Round 3** | ~317s | 66.5× (Collapsed) |
| **SalUn** (Fan et al., 2024) | Static Gradient Masking | Workload A | 95.03% | 88.03% | Round 1 (Degraded) | ~1,355s | 15.6× |
| **Gradient Ascent** | Unconstrained Loss Ascent | Workload A | 99.47% | 93.19% | Round 2 (Under-erased) | ~70s | 298.6× |
| **Exact Retraining** | Scratch Retrain (Gold Standard) | Workload A | 99.10% | 93.04% | None (Stable) | ~21,100s | 1.0× (Baseline) |

### Key Findings
1. **Static unlearning fails under repetition**: Static SSD drops to random guessing ($10\%$) by Round 3.
2. **ASUC-SOM maintains utility**: Retains $>97\%$ accuracy across all 20 rounds without triggering full retrains on every step.
3. **Multi-workload stability**: Consistently stable on uniform random deletions, class-targeted deletions, and high-influence sample removals.

---

## How It Works

```
Incoming Deletion Batch D_f
           │
           ▼
┌─────────────────────────────────────┐
│ 1. Request Profiler                 │
│    - Batch size |D_f|               │
│    - Class entropy H(c)             │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ 2. Subspace Overlap Monitor (SOM)   │
│    - Maintains orthonormal basis Q  │
│    - Overlap score = ||Q^T g_t||    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ 3. Tiered Router (ASUC)             │
│    - Green (<0.40): Tier 1 (SSD)    │
│    - Amber (0.40-0.80): Tier 2 (SalUn)
│    - Red (>=0.80): Escalate         │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ 4. Closed-Loop Verifier             │
│    - Forget Acc <= 10% (Target)     │
│    - Retain Acc preservation check  │
└──────────────────┬──────────────────┘
                   │
         Passed    │    Failed
         ┌─────────┴─────────┐
         ▼                   ▼
    Commit Model      Rollback Checkpoint & Escalate
```

- **Subspace Orthogonality Metric (SOM)**: Tracks historical gradient directions using an orthonormal basis $Q_k$. When a new deletion request produces gradient $g_t$, the projection $\|Q_k^T g_t\| / \|g_t\|$ measures whether the deletion damages already-modified weights.
- **Dynamic Tiered Routing**:
  - **Tier 1 (SSD)**: Fast Fisher dampening when parameter space is healthy.
  - **Tier 2 (SalUn)**: Saliency-masked gradient unlearning with retain regularization when subspace overlap is elevated or requests are class-concentrated.
  - **Tier 3 (Retrain fallback)**: Triggered only when severe fatigue cannot be resolved by approximate unlearning.
- **Closed-Loop Verification**: Validates forget and retain accuracy on a held-out anchor set before committing the update.

---

## Project Structure

```
adaptive-unlearning/
├── configs/               # Hyperparameter and workload YAML configs
├── checkpoints/           # Pre-trained base models (ResNet-18)
├── data/                  # CIFAR-10 data loader directory
├── src/
│   ├── controllers/       # ASUC meta-controller and SOM subspace tracker
│   ├── unlearning/        # SSD, SalUn, Gradient Ascent, Exact Retrain
│   ├── verification/      # Closed-loop verification engine
│   └── evaluation/        # MIA risk evaluator and retention metrics
├── results/               # Experiment logs, histories, and summaries
├── visualizer/            # Interactive real-time dashboard (HTML/CSS/JS + server.py)
├── scripts/
│   └── generate_paper_figures.py  # Publication figures (PDF/PNG) and LaTeX tables
├── train.py               # Train baseline CIFAR-10 model
├── run_unlearning.py      # Single-shot unlearning runner
├── run_sequential.py      # Sequential benchmark runner
└── evaluate.py            # Results scanner and markdown report generator
```

---

## Getting Started

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/mithulkannan17/Adaptive-unlearning.git
cd Adaptive-unlearning

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train Base Model

Train the baseline ResNet-18 model on CIFAR-10 (achieves ~93.4% test accuracy):

```bash
python train.py --config configs/base_cifar10.yaml
```

### 3. Run Sequential Unlearning Benchmarks

Run 20 sequential unlearning rounds across different methods:

```bash
# Run ASUC-SOM (Ours) on Workload A
python run_sequential.py --method asuc --workload a --rounds 20 --output_dir ./results/sequential_asuc_workload_a

# Run Baseline: Selective Synaptic Dampening (SSD)
python run_sequential.py --method ssd --workload a --rounds 20 --output_dir ./results/sequential_ssd_workload_a

# Run Baseline: Saliency Unlearning (SalUn)
python run_sequential.py --method salun --workload a --rounds 20 --output_dir ./results/sequential_salun_workload_a

# Run Baseline: Gradient Ascent
python run_sequential.py --method gradient_ascent --workload a --rounds 20 --output_dir ./results/sequential_gradient_ascent_workload_a
```

### 4. Evaluate & Summarize Results

Scan the `results/` directory and print a formatted summary comparison table:

```bash
python evaluate.py --results_dir ./results
```

---

## Interactive Dashboard (ARIA Engine)

Launch the visualizer to inspect round-by-round accuracy trajectories, live SOM subspace 3D dynamics, and test custom request profiles:

```bash
python visualizer/server.py --port 8080
```

Open your browser at **`http://localhost:8080`**.

---

## Generating Research Paper Figures & LaTeX Tables

To generate camera-ready figures (vector PDF + 300 DPI PNG) and LaTeX table snippets for papers:

```bash
python scripts/generate_paper_figures.py --results_dir ./results --output_dir ./paper_figures --dpi 300
```

This outputs:
- `fig1_sequential_accuracy.pdf / .png`: Retain vs Forget accuracy over 20 rounds.
- `fig2_som_dynamics_drift.pdf / .png`: SOM overlap scores with safety thresholds and parameter drift.
- `fig3_speedup_pareto.pdf / .png`: Speedup vs Retain utility Pareto frontier.
- `fig4_class_erasure_radar.pdf / .png`: Class-wise retention polar plot.
- `fig5_paper_composite_2x2.pdf / .png`: 4-panel main figure for two-column paper formats.
- `table1_latex_results.tex`: Formatted LaTeX `\begin{table*}` summary.

---

## License

This project is licensed under the MIT License.
