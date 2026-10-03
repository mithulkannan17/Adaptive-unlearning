# Engineering & Coding Rules
## Project: ASUC-SOM (Adaptive Sequential Machine Unlearning Engine)

---

## 1. Code Style & Standards

- **Language & Runtime**: Python 3.10+ (tested on Python 3.11).
- **Formatting**: PEP 8 compliance. Use 4 spaces per indentation level. Max line length: 100 characters where practical.
- **Type Annotations**: Use Python standard type hints (`typing.Optional`, `typing.Tuple`, `typing.List`, `pathlib.Path`, `torch.Tensor`, `torch.nn.Module`) across all public functions and class constructors.
- **Imports Structure**:
  1. Standard library imports (`os`, `sys`, `json`, `math`, `pathlib`, `argparse`).
  2. Third-party packages (`torch`, `torch.nn`, `numpy`, `scipy`, `pandas`, `matplotlib`).
  3. Internal local modules (`src.controllers`, `src.unlearning`, `src.verification`, `src.evaluation`).

---

## 2. PyTorch & CUDA Best Practices

1. **Explicit Device Handling**:
   - Always resolve device dynamically: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
   - Never hardcode `"cuda:0"` inside reusable library modules in `src/`.
2. **GPU Memory Management**:
   - Wrap validation, metric computation, and inference loops inside `with torch.no_grad():`.
   - Explicitly call `torch.cuda.empty_cache()` before and after heavy Fisher matrix or saliency gradient calculations.
   - Delete intermediate gradient tensors (`del grads`, `del fisher_retain`) before executing subsequent training passes.
3. **Reproducibility & Determinism**:
   - Seed all random number generators at the entry point of every script (`torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)`).
   - Use `torch.backends.cudnn.deterministic = True` for benchmark evaluation runs.

---

## 3. Unlearning Safety & Invariance Rules

1. **State Isolation**:
   - An unlearning primitive must never modify the baseline checkpoint file on disk directly. It must operate on an in-memory clone of the model weights (`copy.deepcopy(model)` or `checkpoint state_dict`).
2. **Rollback Safety**:
   - Every sequential execution loop must preserve a verified rollback state dict $\theta_{t-1}$ before initiating the current round's unlearning update.
3. **Deterministic Verification**:
   - Verification sets ($D_{f,\text{val}}$, $D_{r,\text{anchor}}$) must remain isolated from the training forget set to prevent evaluation leakage.

---

## 4. Experiment Logging & File IO Rules

1. **JSON Artifact Integrity**:
   - All diagnostic files written to `results/` must be valid JSON with explicit UTF-8 encoding.
   - Floats stored in JSON must be cast with standard Python `float(val)` to avoid `TypeError: Object of type float32 is not JSON serializable`.
2. **Path Sanitization**:
   - Use `pathlib.Path` for all filesystem operations to ensure cross-platform compatibility between Windows (`\`) and POSIX (`/`).
3. **Non-destructive Directory Creation**:
   - Always call `path.mkdir(parents=True, exist_ok=True)` before attempting to write checkpoints, logs, or figures.

---

## 5. Git & Version Control Conventions

- **Branch Strategy**: `main` contains stable, tested releases and benchmark logs.
- **Commit Format**:
  - `feat: <description>` — New feature or algorithm implementation.
  - `fix: <description>` — Bug fix or numeric correction.
  - `refactor: <description>` — Code cleanup without changing runtime behavior.
  - `docs: <description>` — Documentation or README updates.
  - `perf: <description>` — Performance or memory optimization.
- **Excluded Files**:
  - Never commit virtual environments (`venv/`), raw tarballs (`*.tar.gz`), large PyTorch checkpoints (`*.pth`), or Python cache (`__pycache__/`, `*.pyc`). Ensure `.gitignore` is maintained.
