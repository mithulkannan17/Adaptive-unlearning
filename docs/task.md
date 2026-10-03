# Tasks & Milestone Progress Tracker
## Project: ASUC-SOM (Adaptive Sequential Machine Unlearning Engine)

---

## 1. Project Milestones & Status Overview

| Phase | Description | Status | Completion Date |
| :--- | :--- | :---: | :---: |
| **Phase 1** | Baseline Model Training & CIFAR-10 Pipeline | ✅ Complete | 2026-09 |
| **Phase 2** | Core Unlearning Primitives Implementation (SSD, SalUn, GA, Retrain) | ✅ Complete | 2026-09 |
| **Phase 3** | Subspace Orthogonality Metric (SOM) & Request Profiler | ✅ Complete | 2026-09 |
| **Phase 4** | Closed-Loop Verification Gate & Rollback Escalation | ✅ Complete | 2026-09 |
| **Phase 5** | Sequential Multi-Round Benchmark Execution (Workloads A, B, C) | ✅ Complete | 2026-10 |
| **Phase 6** | ARIA Interactive Visualizer Dashboard & Telemetry Server | ✅ Complete | 2026-10 |
| **Phase 7** | Camera-Ready Paper Figure & LaTeX Table Generator | ✅ Complete | 2026-10 |
| **Phase 8** | Documentation, PRD, Architecture Specs & Git Repository Setup | ✅ Complete | 2026-10 |

---

## 2. Detailed Task Breakdown

### Phase 1: Base Model Architecture & Data Ingestion
- [x] Train baseline ResNet-18 model on CIFAR-10 (`train.py`, achieves $93.44\%$ test accuracy).
- [x] Implement deterministic data splitters for retain, forget, and test subsets.
- [x] Implement checkpoint serialization and loading helpers.

### Phase 2: Unlearning Primitives Engine
- [x] Implement Selective Synaptic Dampening (SSD) with Fisher Information computation.
- [x] Implement Saliency Unlearning (SalUn) with gradient masking and retain penalty.
- [x] Implement Gradient Ascent (unconstrained baseline).
- [x] Implement Exact Retraining counterfactual engine.

### Phase 3: Controller & Subspace Tracking
- [x] Implement `RequestProfiler` computing batch cardinality ratio and normalized class entropy $H(c)$.
- [x] Implement `SOMTracker` with low-rank orthonormal basis $Q_k$ and Gram-Schmidt projection.
- [x] Implement health state classification: Green ($<0.40$), Amber ($0.40-0.80$), Red ($\ge 0.80$).
- [x] Implement `TieredRouter` mapping state and request properties to optimal primitives.

### Phase 4: Closed-Loop Verification & Benchmarking
- [x] Implement post-unlearning validation checks ($\text{Forget Acc} \le 15\%$, $\text{Retain Drop} \le 3\%$).
- [x] Implement automatic checkpoint rollback and escalation path ($\text{SSD} \to \text{SalUn} \to \text{Retrain}$).
- [x] Execute Workload A (Uniform Random, 20 rounds) across ASUC, SSD, SalUn, and GA.
- [x] Execute Workload B (Class-Sequential, 10 rounds).
- [x] Execute Workload C (High-Influence / High-Loss, 10 rounds).
- [x] Implement `evaluate.py` to scan and summarize results into Markdown tables.

### Phase 5: Frontend Visualizer & Dashboard (ARIA)
- [x] Implement lightweight HTTP server (`visualizer/server.py`) serving live experimental telemetry.
- [x] Build Command Center view with KPI cards and trajectory charts.
- [x] Build 60 FPS HTML5 canvas for 3D/2D SOM subspace projection animation.
- [x] Build Request Simulator testing custom cardinality, entropy, and SOM inputs.
- [x] Build GDPR Article 17 Compliance Audit Log table.
- [x] Add toggle between Live Stress-Test Run and Nominal Production Profile.

### Phase 6: Research Publication Tools & Assets
- [x] Implement `scripts/generate_paper_figures.py` generating 300 DPI PNG and vector PDF figures.
- [x] Generate Fig 1 (Retain vs. Forget Accuracy across 20 rounds).
- [x] Generate Fig 2 (SOM Overlap vs. Parameter Drift).
- [x] Generate Fig 3 (Speedup-Accuracy Pareto Frontier).
- [x] Generate Fig 4 (Class-wise Retention Radar).
- [x] Generate Fig 5 (Master 4-Panel Paper Composite Figure).
- [x] Generate Table 1 (Formatted LaTeX `\begin{table*}` summary).

---

## 3. Future Roadmap & Planned Extensions

- [ ] **Vision Transformers (ViT) & Large Language Models (LLMs)**: Extend SOM projection tracking to self-attention projection matrices (Q, K, V).
- [ ] **Dynamic Differential Privacy Integration**: Formalize $(\epsilon, \delta)$-differential privacy bounds during sequential unlearning rounds.
- [ ] **Distributed Multi-GPU Execution**: Scale Fisher diagonal and SOM basis computation across multi-node clusters.
