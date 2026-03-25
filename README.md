# LogiPrune

**Smarter AI training through propositional logic and information theory.**

LogiPrune is a preprocessing library that analyzes the logical and informational structure of your dataset *before* training begins, and uses that structure to reduce the hyperparameter search space — without sacrificing accuracy.

Two complementary modules. One library. Two papers.

---

## What it does

**LogiPrune (Paper 1)** asks: *what logical relationship exists between feature A and feature B?*
It finds implications (A→B), biconditionals (A↔B), incompatibilities (A→¬B), and disjunctions (A∨B), then uses those relationships to eliminate redundant features and restrict the hyperparameter grid.

**LogiPruneEntropy (Paper 2)** asks: *how complex is that relationship?*
It computes the Shannon entropy H* of the 4-cell truth table distribution for each feature pair — a continuous measure of boundary complexity — and uses it to select the appropriate model depth and size a priori.

---

## Installation

```bash
pip install logiprune                    # core (SVC, RF, any estimator)
pip install "logiprune[xgboost]"         # with XGBoost support
```

---

## Quick start

### Paper 1 — Propositional grid pruning (SVC / RF / any estimator)

```python
from logiprune import LogiPrune
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

base_grid = {
    'svc__C':      [0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__gamma':  ['scale', 'auto', 0.01, 0.1],
}

lp = LogiPrune(base_grid=base_grid, verbose=True)
lp.fit(X_train, y_train)

X_pruned    = lp.transform(X_train)
pruned_grid = lp.pruned_grid()

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
gs   = GridSearchCV(pipe, pruned_grid, cv=5, scoring='f1')
gs.fit(X_pruned, y_train)

print(lp.report())
# → Config savings: 93.8%  |  Features eliminated: 1  |  ...
```

### Paper 2 — Entropy-based complexity selection (XGBoost)

```python
from logiprune import LogiPruneEntropy
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

xgb_grid = {
    'xgb__n_estimators':     [100, 200, 300],
    'xgb__max_depth':        [3, 5, 7],
    'xgb__learning_rate':    [0.05, 0.1, 0.3],
    'xgb__subsample':        [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0],
}

lpe = LogiPruneEntropy(base_grid=xgb_grid, verbose=True)
lpe.fit(X_train, y_train)

X_pruned    = lpe.transform(X_train)
pruned_grid = lpe.pruned_grid()

pipe = Pipeline([('xgb', XGBClassifier(eval_metric='logloss', verbosity=0))])
gs   = GridSearchCV(pipe, pruned_grid, cv=5, scoring='f1')
gs.fit(X_pruned, y_train)

print(lpe.report())
# → H_min: 0.76  |  Config savings: 77.8%  |  Complexity: low
```

### Combined pipeline (optimal)

```python
from logiprune import LogiPrune, LogiPruneEntropy

# Stage 1: propositional pruning (Paper 1)
lp = LogiPrune(base_grid=base_grid)
lp.fit(X_train, y_train)
X_p1    = lp.transform(X_train)
grid_p1 = lp.pruned_grid()

# Stage 2: entropy complexity selection (Paper 2)
lpe = LogiPruneEntropy(base_grid=grid_p1)
lpe.fit(X_p1, y_train)
X_final    = lpe.transform(X_p1)
grid_final = lpe.pruned_grid()

# Stage 3: adaptive search on the reduced space
# → plug grid_final into FLAML, Optuna, or GridSearchCV
```

---

## Empirical results

### Paper 1 — Five-method benchmark (SVC)

| Dataset | Method | Time | F1 | Config savings |
|---|---|---|---|---|
| breast_cancer | Baseline | 14.6s | 0.9861 | — |
| | FLAML | 14.6s | 0.9861 | 0% (uses full budget) |
| | **LogiPrune** | **0.7s** | **0.9726** | **93.8%** |
| digits_0v1 | Baseline | 1.5s | 1.000 | — |
| | **LogiPrune** | **1.4s** | **1.000** | **93.8%** |
| synth_lo_1k | Baseline | 32.0s | 0.9100 | — |
| | **LogiPrune** | **10.5s** | **0.9100** | **33.3%** |

On structured data (n=10,000, RF): LogiPrune is the **only method that is simultaneously faster (+29.3%) and better (F1 +0.0011) than baseline GridSearch**.

### Paper 2 — XGBoost entropy benchmark

| Dataset | H* | Grid | Savings | ΔF1 | ΔTime |
|---|---|---|---|---|---|
| breast_cancer | 0.76 | 108→24 | 77.8% | **+0.0133** | +76.9% |
| synth_hi_2k | 1.20 | 108→24 | 77.8% | +0.0005 | +74.3% |
| wine | 1.18 | 108→24 | 77.8% | 0.0000 | +74.5% |

**ΔF1 ≥ 0 in all datasets.** H*=0.76 on breast_cancer correctly identifies that shallow trees suffice — eliminating the deep configurations that hurt generalization.

---

## How it works

### Paper 1: Propositional vector

For each feature pair, LogiPrune sweeps discretization thresholds and finds the one where the logical relationship is most stable. It classifies each pair as:

- **Biconditional (A↔B):** one feature eliminated after accuracy validation
- **Implication (A→B):** linear kernel sufficient → restrict to `kernel=['linear']`
- **Incompatibility (A→¬B):** mutual exclusion structure → restrict kernels
- **Disjunction (A∨B):** compressed via t-conorms, **only when both A⊢D and B⊢D** (disjunction elimination rule ∨E)
- **Contingency:** full grid required

### Paper 2: Truth table entropy

For each feature pair at threshold T, the 4-cell weight distribution π(T) = (w₁₁, w₁₀, w₀₁, w₀₀) has Shannon entropy H(T) = −Σ wᵢⱼ · log₂(wᵢⱼ) ∈ [0, 2.0] bits. H* = min H(T) across the threshold sweep captures the best-case simplicity of the relationship.

| H* | Complexity | XGBoost restriction |
|---|---|---|
| [0.0, 0.5) | Very simple | max_depth=[2,3], n_estimators=[50,100] |
| [0.5, 1.0) | Simple | max_depth=[3,4,5], n_estimators=[100,200] |
| [1.0, 1.5) | Moderate | max_depth=[4,5,6], n_estimators=[200,300] |
| [1.5, 2.0) | Complex | Full grid |

### The feedback loop

After Paper 1 eliminates a feature B via A↔B, Paper 2 checks:
`if H*(A, D | without B) > H*(A, D | with B) + δ: reinstate B`

This detects when B acts as a "moderator" — its presence simplifies the A→D relationship even though A↔B suggested redundancy.

---

## Parameters

### LogiPrune (Paper 1)

| Parameter | Default | Description |
|---|---|---|
| `base_grid` | required | Full GridSearchCV parameter grid |
| `min_confidence` | 0.75 | Minimum confidence for structural relations |
| `acc_drop_tolerance` | 0.04 | Max accuracy drop for feature elimination |
| `theta_disj_gate` | 0.85 | Both A⊢D and B⊢D must reach this for disjunction compression |
| `theta_elevation` | 0.92 | Confidence for full pair elevation to implication |
| `discretizer_strategy` | 'percentile' | 'percentile', 'minmax', or 'zscore_clip' |
| `verbose` | False | Print progress |

### LogiPruneEntropy (Paper 2)

| Parameter | Default | Description |
|---|---|---|
| `base_grid` | required | Full hyperparameter grid |
| `acc_drop_tolerance` | 0.04 | Max accuracy drop for feature elimination |
| `feedback_delta` | 0.10 | Entropy increase that triggers feature reinstatement |
| `discretizer_strategy` | 'percentile' | Normalization strategy |
| `verbose` | False | Print progress |

---

## Recommended pipeline

```
Dataset
  → LogiPrune       (Paper 1: removes redundant features, restricts kernel/depth)
  → LogiPruneEntropy (Paper 2: restricts n_estimators, max_depth by entropy)
  → FLAML / Optuna  (searches the reduced space adaptively)
  → best model
```

LogiPrune+FLAML Pareto-dominates FLAML alone: same budget, smaller space, better or equal results.

---

## When it works best

- Medical diagnostics (blood panels, imaging features)
- Sensor fusion (IoT, process control)
- Financial features (ratios from shared base quantities)
- Image descriptors (pixel/feature correlations)

## When to expect modest gains

Purely synthetic Gaussian datasets with independent features have high entropy throughout. The propositional gate and entropy signal correctly recognize this and apply minimal restrictions, protecting accuracy at the cost of smaller savings.

---

## Citation

If you use LogiPrune in your research, please cite both papers:

```bibtex
@article{peralta2026logiprune,
  title   = {LogiPrune: Propositional Disjunction Elimination
             for Hyperparameter Search Space Pruning},
  author  = {Peralta Del Riego, V{\'i}ctor Manuel},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
}

@article{peralta2026logiprune_entropy,
  title   = {LogiPrune-Entropy: A Priori Model Complexity Selection
             via Truth Table Shannon Entropy},
  author  = {Peralta Del Riego, V{\'i}ctor Manuel},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
}
```

---

## License

MIT © Víctor Manuel Peralta Del Riego, 2026
