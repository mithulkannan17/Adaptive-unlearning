# ASUC-SOM: Closed-Loop Adaptive Sequential Machine Unlearning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ASUC-SOM** is an adaptive, closed-loop meta-controller for sequential machine unlearning on deep neural networks. In continuous compliance settings (such as GDPR "Right to be Forgotten"), deletion requests arrive sequentially across multiple rounds. Repeated application of static approximate unlearning primitives (e.g., Gradient Ascent, Selective Synaptic Dampening, or Saliency Unlearning) induces **parameter fatigue** and **capacity collapse**, causing catastrophic destruction of retain utility.

ASUC-SOM addresses this by dynamically profiling incoming requests, monitoring the unlearning subspace via the **Subspace Orthogonality Metric (SOM)**, routing to the optimal primitive, and validating the unlearned model in a closed-loop verification pipeline with controlled escalation.

---

## Architecture Overview

```
                          Incoming Deletion Request (Df)
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │         Request Profiler          │
                     │  - Cardinality & ratio            │
                     │  - Class entropy (dispersion)     │
                     │  - Empirical influence norm       │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │      Subspace Tracker (SOM)       │
                     │  - Decaying low-rank basis        │
                     │  - Projection overlap score       │
                     │  - Green / Amber / Red Health     │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │           Tiered Router           │
                     │  Green + Balanced   ──► Tier 1: SSD│
                     │  Amber / Concentr.  ──► Tier 2:SalUn│
                     │  Red / Fatigue      ──► Escalate  │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │        Unlearning Primitive       │
                     │       (SSD / SalUn / Retrain)     │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │       Closed-Loop Verifier        │
                     │  - Forget efficacy validation     │
                     │  - Retain anchor preservation     │
                     └─────────┬───────────────────┬─────┘
                               │ Passed            │ Failed
                               ▼                   ▼
                     ┌──────────────────┐  ┌───────────────────────┐
                     │ Commit Model     │  │ Controlled Escalation │
                     │ Update State     │  │ SSD ──► SalUn ──► Retrain
                     └──────────────────┘  └───────────────────────┘
```

### Core Components

1. **Request Profiler (`src/controller/asuc.py`)**:
   - Calculates cardinality ($|D_f|$), cardinality ratio ($|D_f| / |D_{remain}|$), and normalized class entropy:
     $$H(c) = -\frac{1}{\log C} \sum_{c=1}^C p(c) \log p(c)$$
   - Identifies class-concentrated requests (e.g., single-class deletion) which induce localized feature collapse.

2. **Subspace Orthogonality Metric (SOM) (`src/controller/som.py`)**:
   - Tracks historical unlearning directions using an exponentially decaying orthonormal basis $Q \in \mathbb{R}^{D \times k}$:
     $$\text{SOM}(g_t, Q) = \| Q^T \hat{g}_t \|_2 \in [0, 1]$$
   - **Green State** ($\text{SOM} < 0.40$): Low overlap with prior unlearning; parameter plasticity is healthy.
   - **Amber State** ($0.40 \le \text{SOM} < 0.80$): Moderate overlap; parameter fatigue beginning to accumulate.
   - **Red State** ($\text{SOM} \ge 0.80$): Critical overlap; high risk of catastrophic capacity collapse.

3. **Tiered Dynamic Router (`src/controller/asuc.py`)**:
   - Routes healthy, balanced requests to **Tier 1 (SSD)** for near-zero compute overhead.
   - Routes fatigued or class-concentrated requests to **Tier 2 (SalUn)** with retain regularization.
   - Triggers proactive escalation when in Red condition.

4. **Closed-Loop Verifier (`src/verification/verifier.py`)**:
   - Rapidly evaluates post-unlearning forget accuracy drop and retain utility on a lightweight anchor set ($< 1$s).
   - If verification fails, orchestrates deterministic escalation: $\text{SSD} \to \text{SalUn} \to \text{Retrain}$.

---

## Repository Structure

```
adaptive-unlearning/
├── checkpoints/              # Model weights (base CIFAR-10, retrained counterfactuals)
│   ├── base_cifar10/         # Baseline ResNet-18 (93.44% test accuracy)
│   └── exact_retrain_cifar10/# Exact retraining counterfactuals
├── configs/                  # Experiment configurations
├── data/                     # CIFAR-10 dataset
├── results/                  # Benchmark JSON outputs and metrics
├── src/
│   ├── config.py             # Global ExperimentConfig dataclass
│   ├── seed.py               # Deterministic random seed utilities
│   ├── utils.py              # Checkpoint I/O and device allocation
│   ├── controller/           # ASUC-SOM Meta-Controller
│   │   ├── asuc.py           # Profiler, Router, and Controller orchestrator
│   │   ├── som.py            # Subspace Orthogonality Metric (SOM) tracker
│   │   └── state_tracker.py  # Audit logging and state serialization
│   ├── data/
│   │   └── cifar10.py        # CIFAR-10 dataloaders with index tracking
│   ├── evaluation/           # Evaluation metrics suite
│   │   ├── compute.py        # Unified 4-pillar benchmark report
│   │   ├── counterfactual.py # Counterfactual distance & KL divergence
│   │   ├── evaluator.py      # Standard UnlearningEvaluator interface
│   │   ├── forgetting.py     # Forget loss, accuracy, and MIA AUC score
│   │   ├── stability.py      # Parameter distance, relative drift, norms
│   │   └── utility.py        # Retain and test set accuracy retention
│   ├── experiments/
│   │   ├── sequential_runner.py # Generic sequential experiment runner
│   │   └── state.py          # Sequential unlearning state machine
│   ├── models/
│   │   └── resnet.py         # CIFAR-adapted ResNet-18 architecture
│   ├── training/
│   │   └── trainer.py        # Standard base model trainer
│   ├── unlearning/           # Unlearning primitives
│   │   ├── exact_retrain.py  # Exact retraining from scratch (ground truth)
│   │   ├── gradient_ascent.py# Gradient Ascent baseline
│   │   ├── salun.py          # Saliency Unlearning (SalUn)
│   │   ├── sequential_gradient_ascent.py # Sequential GA runner
│   │   └── ssd.py            # Selective Synaptic Dampening (SSD)
│   ├── verification/
│   │   └── verifier.py       # Closed-loop verifier and safety gates
│   └── workloads/            # Deletion request generators
│       ├── class_sequential.py # Workload B: Class-Sequential deletion
│       ├── high_influence.py   # Workload C: Adversarial / High-Influence
│       └── uniform_random.py   # Workload A: Uniform Random deletion
├── evaluate.py               # Master benchmark analyzer & Markdown reporter
├── run_sequential.py         # Multi-round sequential unlearning benchmark CLI
├── run_unlearning.py         # Single-step unlearning CLI
├── train.py                  # Baseline training script
└── requirements.txt          # Python dependencies
```

---

## Supported Unlearning Primitives

| Primitive | Mechanism | Speed | Retain Stability | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **SSD** | Selective Synaptic Dampening: dampens weights where $I_f(\theta) \gg I(\theta)$ | Fast (~40s) | High for uniform; fragile for concentrated | Tier 1 (Green State) |
| **SalUn** | Saliency Unlearning: updates top $p\%$ salient parameters with retain regularization | Moderate (~45s) | High across classes | Tier 2 (Amber / Class-concentrated) |
| **GA** | Gradient Ascent: directly maximizes loss on forget set | Fast (~10s) | Low; collapses rapidly under repetition | Static Baseline Comparison |
| **Retrain** | Exact Retraining from scratch on $D \setminus D_f$ | Slow (~2.5h) | Optimal (Ground Truth Counterfactual) | Final Safety Fallback |

---

## Supported Workloads

- **Workload A (Uniform Random)**:
  At each round, approximately 1% of the currently remaining training set is deleted uniformly at random across 20 rounds (total forgotten: 9,113 samples).
- **Workload B (Class-Sequential)**:
  Deletion requests cycle through specific classes, creating localized class-imbalance stress.
- **Workload C (High Influence)**:
  Samples with the highest cross-entropy loss and gradient influence on the model are unlearned first, representing an adversarial worst-case sequence.

---

## Quickstart Guide

### 1. Installation

```bash
git clone <repo-url>
cd adaptive-unlearning
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train Base Model

A pre-trained ResNet-18 checkpoint is included in `checkpoints/base_cifar10/best_model.pth` achieving **93.44% test accuracy**. To retrain from scratch:

```bash
python train.py
```

### 3. Run Single-Step Unlearning

```bash
# Run ASUC controller on Workload A (500 samples)
python run_unlearning.py --method asuc --workload a --samples 500

# Run SSD baseline
python run_unlearning.py --method ssd --workload a --samples 500

# Run SalUn baseline
python run_unlearning.py --method salun --workload b --samples 500
```

### 4. Run Sequential Unlearning Benchmark (20 Rounds)

```bash
# Execute ASUC-SOM adaptive controller across 20 rounds of Workload A
python run_sequential.py --method asuc --workload a --rounds 20

# Execute SSD baseline across 20 rounds
python run_sequential.py --method ssd --workload a --rounds 20

# Execute Gradient Ascent baseline across 20 rounds
python run_sequential.py --method gradient_ascent --workload a --rounds 20
```

### 5. Generate Benchmark Evaluation Report

```bash
python evaluate.py --results_dir ./results
```

This compiles round histories across all methods, detects rounds-to-collapse, and outputs a formatted markdown table in `results/benchmark_comparison.md`.

---

## Running Unit and Validation Tests

All individual components have self-contained validation suites:

```bash
# Test Subspace Orthogonality Metric (SOM)
python test_som.py

# Test Workload Generators (A, B, C)
python test_workloads.py

# Test Saliency Unlearning (SalUn)
python test_salun.py

# Test ASUC Controller & Closed-Loop Verification
python test_asuc.py

# Test Counterfactual Evaluator vs Retraining
python test_counterfactual.py
```

---

## Experimental Benchmark Summary

| Method | Workload | Rounds | Retain Acc (%) | Test Acc (%) | Collapse Round | Speedup vs Retrain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Retrain** | A | 20 | **99.5%** | **93.4%** | None (Ground Truth) | 1.0x (Baseline) |
| **ASUC-SOM (Ours)**| A | 20 | **> 98.0%** | **> 91.5%** | **None (Stable)** | **~150x** |
| **SSD (Static)** | A | 20 | 95.2% | 88.4% | Round 14 | ~200x |
| **SalUn (Static)**| A | 20 | 96.1% | 89.2% | Round 16 | ~180x |
| **GA (Static)** | A | 20 | < 80.0% | < 75.0% | **Round 3** | ~900x |

---

## Citation

```bibtex
@article{asuc_som_2026,
  title={ASUC-SOM: Closed-Loop Adaptive Sequential Machine Unlearning via Subspace Orthogonality},
  author={Adaptive Unlearning Research Team},
  year={2026}
}
```
