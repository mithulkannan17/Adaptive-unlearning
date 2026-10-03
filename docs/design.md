# UI/UX Design System & Direction
## Platform: ARIA (AI Rights & Amnesia Engine) Dashboard

---

## 1. Design Philosophy

ARIA is a telemetry-first, dark-mode analytics console engineered for machine learning practitioners and compliance officers. 

### Core Design Principles
1. **High Information Density**: Present multi-round metric trajectories, subspace geometry, and audit records with zero unnecessary whitespace.
2. **Instant Visual Hierarchy**: Critical safety alerts (e.g., model collapse warnings, unlearning state transitions) must immediately command attention through high-contrast accent colors.
3. **Smooth Micro-Interactions**: Utilize subtle glassmorphic surfaces, glow accents, and responsive hover transitions without lag or bloated CSS frameworks.

---

## 2. Color Palette & Design Tokens

```css
:root {
  /* Surface & Backgrounds */
  --bg:         #060911;  /* Deep Navy-Black Base */
  --bg2:        #0b101d;  /* Secondary Container Surface */
  --bg3:        #101726;  /* Elevated Card Background */
  --card:       rgba(11, 17, 33, 0.90);
  --border:     rgba(255, 255, 255, 0.10);
  
  /* Primary Accent Tiers */
  --cyan:       #06b6d4;  /* ASUC Meta-Controller / Tier 1 (SSD) */
  --purple:     #8b5cf6;  /* Tier 2 (SalUn Saliency Unlearning) */
  --green:      #10b981;  /* Healthy Subspace (SOM < 0.40) / Verified PASS */
  --amber:      #f59e0b;  /* Caution Subspace (0.40 - 0.80) / Exact Retrain */
  --red:        #f43f5e;  /* Critical Subspace (SOM >= 0.80) / Collapse Warning */

  /* Typography */
  --text:       #f8fafc;  /* Primary Crisp White */
  --text2:      #cbd5e1;  /* Secondary Muted Slate */
  --text3:      #8192a6;  /* Subdued Metadata / Labels */
}
```

---

## 3. Typography Hierarchy

| Role | Font Family | Size | Weight | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Brand / Display** | `Outfit`, sans-serif | 1.15rem – 1.35rem | 800 | Topbar brand, Section titles, Page headings |
| **Primary UI** | `Inter`, sans-serif | 0.92rem – 0.95rem | 500 – 600 | Navigation links, Card headers, Button labels |
| **Metric KPIs** | `Outfit`, sans-serif | 1.85rem | 800 | Retain accuracy, speedup multipliers, sample counts |
| **Data / Monospace** | `JetBrains Mono`, monospace | 0.78rem – 0.88rem | 500 – 700 | Telemetry tables, SOM scores, audit identifiers |

---

## 4. Component Design Specifications

### 4.1. Navigation Sidebar
- **Width**: `260px` fixed with subtle `1px solid var(--border)` right border.
- **Brand Block**: ARIA gradient logo icon + title with subtitle.
- **Section Badges**: Uppercase muted tracking labels (`PLATFORM`, `ANALYSIS`).
- **Live Indicator**: Pulsing green dot with glow animation representing active runtime state.

### 4.2. Metric KPI Cards
- 4-column responsive grid displaying:
  1. **Model Integrity**: Retain accuracy post-unlearning.
  2. **Test Utility**: Generalization on held-out test data.
  3. **Forget Efficacy**: Residual forget-set accuracy (Target $\le 10\%$).
  4. **Compute Speedup**: Multiplier vs. exact retraining from scratch.
- Top colored accent stripe corresponding to metric type.

### 4.3. SOM Subspace Dynamics 3D/2D Canvas
- **Canvas Container**: High-DPI hardware-accelerated HTML5 canvas.
- **Render Elements**:
  - Rotating historical orthonormal basis vectors $Q = [q_1, q_2, q_3]$.
  - Dynamic request gradient vector $g_t$ (red).
  - Subspace projection shadow $Q^T g_t$ with residual dashed line.
  - Glowing concentric safety threshold rings (Green at 0.40, Amber at 0.80).
  - Real-time SOM cosine score dial.

### 4.4. Request Simulator
- 2-column interactive playground:
  - **Left (Controls)**: Sliders for Cardinality ($50 - 5,000$), Class Entropy ($0.0 - 1.0$), and SOM Overlap ($0.0 - 1.0$).
  - **Right (Decision Box)**: Real-time decision badge, routing tier, projected speedup, and GDPR compliance projection synced with the backend router endpoint.

### 4.5. Compliance Audit Trail
- Cryptographically formatted tabular log with status badges:
  - `Art. 17 ✓ Compliant` (Green)
  - `Art. 17 ~ Marginal` (Amber)
  - `Art. 17 ✗ Non-Compliant` (Red)

---

## 5. Responsive Breakpoints

- **Desktop ($> 1200\text{px}$)**: Full 4-column KPI grid, side-by-side 2-column chart grids, 260px sidebar.
- **Laptop / Tablet ($900\text{px} - 1200\text{px}$)**: 2-column KPI grid, single-column chart boxes.
- **Compact ($< 900\text{px}$)**: Icon-only collapsed sidebar (50px), single-column stacked layout.
