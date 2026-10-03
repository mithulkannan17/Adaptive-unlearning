# Product Requirements Document (PRD)
## Project: ASUC-SOM (Adaptive Sequential Machine Unlearning Engine)

---

## 1. Executive Summary & Problem Statement

Modern data protection regulations (GDPR Article 17 "Right to Erasure", CCPA, CPRA) legally mandate organizations to permanently delete personal data from machine learning models upon user request. 

### The Problem with Existing Solutions
1. **Exact Retraining is Prohibitive**: Retraining modern deep neural networks from scratch for every deletion batch costs thousands of GPU hours and creates unacceptable operational latency (hours to days per request).
2. **Static Unlearning Collapses Over Time**: Existing approximate unlearning algorithms (e.g., Selective Synaptic Dampening [SSD], Saliency Unlearning [SalUn], Gradient Ascent [GA]) are designed for **single-shot, isolated** deletions. When applied **sequentially** across multiple rounds, they suffer from **parameter fatigue**—accumulating weight distortion and collapsing model utility to random chance (10% accuracy) within 3–4 rounds.
3. **Open-Loop Execution**: Current methods blindly apply parameter modifications without verifying whether the model actually forgot the target data or preserved general utility.

### The Solution: ASUC-SOM
ASUC-SOM is an adaptive, closed-loop machine unlearning engine that supports **continuous, multi-round data deletion requests** without full retraining. It profiles incoming deletion batches, tracks parameter subspace health via the Subspace Orthogonality Metric (SOM), dynamically routes requests to the optimal unlearning primitive, and verifies model integrity in real time.

---

## 2. Target Users & Stakeholders

- **ML Platform & MLOps Engineers**: Need an automated, drop-in pipeline to process deletion requests on production models without manual retraining.
- **Privacy & Compliance Officers**: Require verifiable, cryptographically timestamped audit logs proving that data was erased in compliance with GDPR Article 17.
- **Applied AI Researchers**: Need reproducible benchmarks comparing static vs. adaptive sequential unlearning strategies on standardized vision datasets.

---

## 3. Core Functional Requirements

### 3.1. Dynamic Request Profiling
- The system must parse deletion batches $D_f$ and compute:
  - **Cardinality Ratio**: $|D_f| / |D_{\text{remaining}}|$ (e.g., 500 samples / 50,000 = 1%).
  - **Class Entropy**: $H(c) = -\frac{1}{\log C} \sum_{c=1}^C p(c) \log p(c) \in [0, 1]$.
  - **Concentration Flag**: If $H(c) < 0.40$, flag as a class-concentrated request (high risk of localized catastrophic forgetting).

### 3.2. Subspace Overlap Monitoring (SOM)
- Maintain a running low-rank orthonormal basis $Q_k \in \mathbb{R}^{d \times k}$ of previously modified parameter gradient directions.
- For incoming deletion gradient $g_t$, compute the projection overlap score:
  $$\text{SOM}(g_t, Q_k) = \frac{\|Q_k^T g_t\|_2}{\|g_t\|_2} \in [0, 1]$$
- Classify health states:
  - **Green ($< 0.40$)**: Healthy parameter space; low overlap with prior deletions.
  - **Amber ($0.40 - 0.80$)**: Moderate overlap; parameter fatigue warning.
  - **Red ($\ge 0.80$)**: Critical fatigue; severe risk of rank collapse.

### 3.3. Multi-Tier Dynamic Router
- **Tier 1 (Rapid Dampening — SSD)**:
  - Activated when state is Green and class entropy is balanced ($H(c) \ge 0.40$).
  - Execution time: $< 5$ seconds per batch ($150\times$ speedup).
- **Tier 2 (Regularized Saliency — SalUn)**:
  - Activated when state is Amber, Red, or request is class-concentrated ($H(c) < 0.40$).
  - Updates only the top salient parameter fraction ($p = 0.05$) while penalizing retain set loss degradation.
- **Tier 3 (Selective Exact Retraining)**:
  - Fallback mechanism triggered only when parameter fatigue cannot be resolved by approximate tiers.

### 3.4. Closed-Loop Verification & Self-Healing Gate
- Run lightweight post-unlearning validation on an anchor retain set and the forget set:
  - **Forget Condition**: Forget-set accuracy $\le 15\%$ (target: random chance = 10%).
  - **Retain Condition**: Retain utility drop $\Delta \text{Acc} \le 3.0\%$ from baseline.
  - **MIA Defense**: Membership Inference Attack AUC score $\le 0.55$.
- If verification fails:
  - Automatically roll back model weights to the pre-round checkpoint.
  - Escalate primitive tier (Tier 1 $\to$ Tier 2 $\to$ Tier 3) and re-execute.

### 3.5. Interactive Visualizer Dashboard (ARIA)
- Web-based local dashboard providing:
  - Command Center overview with real-time KPI tiles.
  - 60 FPS 3D/2D SOM Subspace canvas tracking gradient projection.
  - Benchmark comparison view across 20 sequential rounds.
  - Interactive Request Simulator testing custom cardinality, entropy, and SOM inputs.
  - Compliance audit log table with downloadable certificates.

---

## 4. Key Performance Indicators (KPIs) & Target Metrics

| Metric | Target SLA | Achieved (ASUC-SOM on CIFAR-10) |
| :--- | :--- | :--- |
| **Retain Accuracy Preservation** | $\ge 95.0\%$ after 20 rounds | **97.19%** (Workload A) |
| **Forget Set Accuracy** | $\le 15.0\%$ (Random chance $\approx 10\%$) | **10.20%** |
| **Test Set Generalization** | $\ge 90.0\%$ | **90.89%** |
| **Collapse Resistance** | Zero collapses across 20 rounds | **0 Collapses** (Stable across all 20 rounds) |
| **Compute Efficiency** | $> 10\times$ speedup vs full retrain | **Up to 148× speedup** per approximate batch |
| **Verification Overhead** | $< 2.0$ seconds per round | **~0.85 seconds** |

---

## 5. Workload Specifications

1. **Workload A (Uniform Random Deletions)**:
   - 500 samples per round randomly sampled across all 10 CIFAR-10 classes (20 rounds = 10,000 deleted samples / 20% of total training data).
2. **Workload B (Class-Sequential Deletions)**:
   - 500 samples per round concentrated in a single class per round (simulates targeted removal of a data category).
3. **Workload C (High-Influence / Adversarial Deletions)**:
   - 500 highest-loss training samples deleted first (simulates poisoning data removal or high-privacy-risk targets).
