# System Architecture & Technical Specification
## Project: ASUC-SOM (Adaptive Sequential Machine Unlearning Engine)

---

## 1. High-Level Architectural Flow

```
+-------------------------------------------------------------------------------+
|                             DATA & WORKLOAD LAYER                             |
|  - CIFAR-10 Dataset (50,000 train / 10,000 test)                              |
|  - Sequential Workload Generators: Workload A (Uniform), B (Class), C (Loss)   |
+---------------------------------------+---------------------------------------+
                                        | Deletion Request Batch D_f (t)
                                        v
+-------------------------------------------------------------------------------+
|                            CONTROLLER & ROUTING LAYER                         |
|                                                                               |
|  +----------------------------+        +-----------------------------------+  |
|  |     Request Profiler       |        |   Subspace Overlap Monitor (SOM)  |  |
|  |  - Cardinality Ratio       |        |  - Orthonormal Basis Q_k in R^d*k |  |
|  |  - Class Entropy H(c)      |        |  - Gram-Schmidt Update            |  |
|  |  - Dispersion Detection    |        |  - Overlap: ||Q^T g_t|| / ||g_t|| |  |
|  +--------------+-------------+        +-----------------+-----------------+  |
|                 |                                        |                    |
|                 +-------------------+--------------------+                    |
|                                     | Health State & Entropy                  |
|                                     v                                         |
|  +-------------------------------------------------------------------------+  |
|  |                         Tiered Decision Router                          |  |
|  |    Green + Balanced  --> Tier 1: SSD (Rapid Fisher Dampening)           |  |
|  |    Amber / Concentr. --> Tier 2: SalUn (Saliency Regularized Gradient)  |  |
|  |    Red / Fatigue     --> Tier 3: Selective Exact Retraining             |  |
|  +----------------------------------+--------------------------------------+  |
+-------------------------------------|-----------------------------------------+
                                      | Chosen Primitive + Parameters
                                      v
+-------------------------------------------------------------------------------+
|                         UNLEARNING PRIMITIVES ENGINE                          |
|  - SSD: Fisher Importance Ratio Dampening (theta_i = theta_i * gamma)         |
|  - SalUn: Top-p% Saliency Masked Optimization with Retain Penalty Loss        |
|  - Exact Retrain: Full Clean Scratch Training on D_remain                     |
+-------------------------------------+-----------------------------------------+
                                      | Unlearned Model Checkpoint theta_t*
                                      v
+-------------------------------------------------------------------------------+
|                         CLOSED-LOOP VERIFICATION GATE                         |
|  - Forget Set Evaluation: Check if Accuracy <= 15% (Target ~10%)              |
|  - Retain Set Evaluation: Check if Retain Utility Drop <= 3%                  |
|  - Decision:                                                                  |
|      * PASS   --> Commit Checkpoint, Update SOM Basis Q_k, Save Round Log     |
|      * FAIL   --> Rollback Model to theta_{t-1}, Escalate Tier (SSD->SalUn)   |
+-------------------------------------+-----------------------------------------+
                                      | Telemetry & Metrics
                                      v
+-------------------------------------------------------------------------------+
|                       VISUALIZATION & TELEMETRY LAYER                         |
|  - HTTP Visualizer Server (server.py on port 8080)                            |
|  - ARIA Web UI (index.html, style.css, app.js)                                |
|  - Live 60 FPS SOM 3D Canvas Projection                                       |
|  - Paper Figure & LaTeX Table Generator (scripts/generate_paper_figures.py)  |
+-------------------------------------------------------------------------------+
```

---

## 2. Core Subsystems

### 2.1. Request Profiler (`src/controllers/asuc_controller.py`)
Analyzes incoming deletion batch $D_f = \{(x_i, y_i)\}_{i=1}^{N_f}$:

1. **Normalized Class Entropy**:
   $$p(c) = \frac{1}{N_f} \sum_{i=1}^{N_f} \mathbb{I}(y_i = c)$$
   $$H(c) = -\frac{1}{\log C} \sum_{c=1}^C p(c) \log \left(p(c) + \epsilon\right)$$
   - When $H(c) < 0.40$, the request is heavily concentrated in few classes. Static dampening on concentrated requests destroys class-specific filters, so the router automatically routes to Tier 2 (SalUn).

2. **Cardinality Ratio**:
   $$\rho = \frac{|D_f|}{|D_{\text{remaining}}|}$$
   - If $\rho > 0.05$ (large batch $> 5\%$ of dataset), Tier 1 single-pass dampening is bypassed in favor of iterative saliency masking.

---

### 2.2. Subspace Overlap Monitor (SOM) (`src/controllers/subspace_tracker.py`)
Tracks parameter movement geometry across deletion rounds $t = 1, 2, \dots$:

1. **Gradient Computation**: Computes empirical forget-set gradient vector $g_t = \nabla_\theta \mathcal{L}(D_f; \theta_{t-1})$.
2. **Normalized Gradient**: $\hat{g}_t = \frac{g_t}{\|g_t\|_2}$.
3. **Orthonormal Subspace Basis $Q_k$**: Maintains $k$ orthonormal basis vectors $Q_k = [q_1, q_2, \dots, q_k] \in \mathbb{R}^{d \times k}$ (with $k = 16$ or $32$).
4. **Subspace Projection & Overlap Metric**:
   $$\text{SOM}(g_t, Q_k) = \| Q_k^T \hat{g}_t \|_2 = \sqrt{\sum_{j=1}^k (q_j^T \hat{g}_t)^2} \in [0, 1]$$
5. **Incremental Gram-Schmidt Basis Update**:
   $$v = \hat{g}_t - \sum_{j=1}^k (q_j^T \hat{g}_t) q_j, \quad q_{\text{new}} = \frac{v}{\|v\|_2}$$
   - When basis capacity $k$ is reached, the oldest or lowest-energy basis vector is rotated out with exponential decay weighting $\lambda = 0.95$.

---

### 2.3. Unlearning Primitives (`src/unlearning/`)

#### A. Selective Synaptic Dampening (SSD) (`src/unlearning/ssd.py`)
- Calculates empirical Fisher information diagonals for retain and forget sets:
  $$F_{\text{retain}, i} = \frac{1}{|D_r|} \sum_{x \in D_r} \left(\frac{\partial \mathcal{L}(x)}{\partial \theta_i}\right)^2, \quad F_{\text{forget}, i} = \frac{1}{|D_f|} \sum_{x \in D_f} \left(\frac{\partial \mathcal{L}(x)}{\partial \theta_i}\right)^2$$
- Relative importance score:
  $$R_i = \frac{F_{\text{forget}, i}}{F_{\text{retain}, i} + \epsilon}$$
- Selection & Dampening:
  $$\theta_i \leftarrow \theta_i \cdot \min\left(1.0, \frac{\alpha}{R_i}\right) \quad \text{for } R_i \ge \tau$$

#### B. Saliency Unlearning (SalUn) (`src/unlearning/salun.py`)
- Computes first-order gradient saliency on the forget set:
  $$S_i = \left| \frac{\partial \mathcal{L}(D_f)}{\partial \theta_i} \right|$$
- Generates binary parameter mask $M \in \{0, 1\}^d$ selecting top $p = 5\%$ salient weights.
- Optimizes masked parameters using a joint unlearning loss with retain penalty:
  $$\mathcal{L}_{\text{total}} = -\mathcal{L}_{\text{CE}}(D_f; \theta) + \beta \cdot \mathcal{L}_{\text{CE}}(D_r; \theta)$$
  $$\theta \leftarrow \theta - \eta \cdot (M \odot \nabla_\theta \mathcal{L}_{\text{total}})$$

#### C. Exact Retraining (`src/unlearning/exact_retrain.py`)
- Clean re-initialization and SGD training on the remaining dataset $D_{\text{remaining}} = D_{\text{train}} \setminus \bigcup_{\tau=1}^t D_f^{(\tau)}$.
- Provides the gold-standard counterfactual baseline.

---

### 2.4. Closed-Loop Verifier & Escalator (`src/verification/verifier.py`)
After primitive execution produces candidate model $\theta_t^*$:

1. Evaluates accuracy on validation forget subset $D_{f,\text{val}}$ and anchor retain subset $D_{r,\text{anchor}}$:
   $$\text{PASS} \iff \left(\text{Acc}(D_{f,\text{val}}) \le 0.15\right) \land \left(\text{Acc}(D_{r,\text{anchor}}) \ge \text{Acc}_{\text{baseline}} - 0.03\right)$$
   *(i.e., forget validation accuracy $\le 15.0\%$ and retain drop $\le 3.0\%$ relative to baseline).*
2. **Escalation Path**:
   - Level 0 (Initial): Chosen primitive (usually SSD).
   - Level 1 (First Escalation): Revert checkpoint $\to$ Execute SalUn with increased retain weight ($\beta = 1.5$).
   - Level 2 (Second Escalation): Revert checkpoint $\to$ Trigger localized fine-tuning on retain anchor.
   - Level 3 (Terminal Fallback): Revert checkpoint $\to$ Exact Retrain.

---

## 3. Data & Artifact Storage Schema

Each sequential run writes round diagnostics into `results/<experiment_name>/`:
- `round_01.json` through `round_20.json`:
  ```json
  {
    "round_id": 1,
    "new_forget_size": 500,
    "cumulative_forget_size": 500,
    "remaining_size": 49500,
    "unlearning_time_seconds": 4.25,
    "forget_accuracy": 10.20,
    "retain_accuracy": 99.35,
    "test_accuracy": 93.20,
    "relative_parameter_distance": 0.0142,
    "som_score": 0.125,
    "health_state": "GREEN",
    "chosen_primitive": "SSD",
    "verification_passed": true
  }
  ```
- `summary.json`: Aggregated metrics, total compute time, and final test/retain utility.
