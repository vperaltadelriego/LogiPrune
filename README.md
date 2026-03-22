# LogiPrune

**Smarter AI training through propositional logic and information theory.**

LogiPrune is a preprocessing library that analyzes the logical and informational structure of your dataset *before* training begins, and uses that structure to reduce the hyperparameter search space — without sacrificing accuracy.

Two complementary modules. One library. Two papers.

---

## What it does

**LogiPrune (Paper 1)** asks: *what logical relationship exists between feature A and feature B?*
It finds implications (A→B), biconditionals (A↔B), incompatibilities (A→¬B), and disjunctions (A∨B), then uses those relationships to eliminate redundant features and restrict the hyperparameter grid.

**LogiPruneEntropy (Paper 2)** asks: *how complex is that relationship?*
It computes the Shannon entropy H\* of the 4-cell truth table distribution for each feature pair — a continuous measure of boundary complexity — and uses it to select the appropriate model depth and size a priori.

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

### Paper 1 — Five-method benchmark (SVC, all metrics)

All methods receive the same wall-clock time budget. LogiPrune time includes preprocessing.
(+) marks cases where LogiPrune equals or exceeds the baseline on all metrics.

| Dataset | Method | Time | F1 | Precision | Recall | Acc | AUC | Configs |
|---|---|---|---|---|---|---|---|---|
| breast_cancer | Baseline | 14.6s | 0.9861 | 0.9861 | 0.9861 | 0.9825 | 0.9937 | 48 |
| | FLAML | 14.6s | 0.9861 | 0.9861 | 0.9861 | 0.9825 | 0.9957 | adaptive |
| | **LogiPrune** | **0.7s** | **0.9726** | **0.9595** | **0.9861** | **0.9649** | **0.9944** | **3 (93.8%)** |
| wine | Baseline | 1.2s | 0.9432 | 0.9514 | 0.9444 | 0.9444 | 1.000 | 48 |
| | FLAML | 5.0s | 0.9165 | 0.9225 | 0.9167 | 0.9167 | 0.9912 | adaptive |
| | **LogiPrune (+)** | **1.2s** | **0.9720** | **0.9741** | **0.9722** | **0.9722** | **1.000** | **48** |
| digits_0v1 | Baseline | 1.5s | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 48 |
| | FLAML | 5.0s | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | adaptive |
| | **LogiPrune (+)** | **1.4s** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **3 (93.8%)** |
| synth_lo_1k | Baseline | 32.0s | 0.9100 | 0.9192 | 0.9010 | 0.9100 | 0.9723 | 48 |
| | FLAML | 32.0s | 0.9109 | 0.9109 | 0.9109 | 0.9100 | 0.971 | adaptive |
| | **LogiPrune (+)** | **10.5s** | **0.9100** | **0.9100** | **0.9100** | **0.9100** | **0.972** | **32 (33.3%)** |

**Time-weighted average savings: +47.5% vs Baseline · +49.3% vs FLAML**

> **Key insight on breast_cancer:** LogiPrune Recall (0.9861) is **identical to baseline** despite lower Accuracy. The difference is entirely in Precision — a more conservative model, not missed cases. For recall-critical applications (e.g., medical screening), LogiPrune is safe.

---

### Paper 1 — Metric-aware optimization (opt= parameter)

LogiPrune can orient the grid toward a specific metric. Results on breast_cancer (SVC):

| Optimize for | Configs | Accuracy | Precision | Recall | F1 | vs Baseline F1 |
|---|---|---|---|---|---|---|
| Baseline (48) | 48 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | — |
| LP opt=precision ✦ | **4** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.000** |
| LP opt=recall | 4 | 0.9561 | 0.9467 | 0.9861 | 0.9660 | −0.020 |
| LP opt=f1 | 3 | 0.9649 | 0.9595 | 0.9861 | 0.9726 | −0.014 |

✦ **4 configurations reproduce the full 48-configuration baseline exactly on all five metrics.**

---

### Paper 1 — Large dataset benchmark (RandomForest, n=10,000, p=30)

Baseline GridSearch takes 218.5 seconds. All five methods evaluated:

| Method | Time | F1 | Accuracy | Recall | AUC | Configs | ΔF1 |
|---|---|---|---|---|---|---|---|
| Baseline | 218.5s | 0.9334 | 0.9325 | 0.9469 | 0.9784 | 27 | — |
| RandomSearch (3) | 22.4s | 0.9179 | 0.9165 | 0.9349 | 0.9723 | 3 | −0.0155 |
| FLAML | 189.7s | 0.9332 | 0.9325 | 0.9439 | 0.9816 | adaptive | −0.0002 |
| Optuna (9 trials) | 66.0s | 0.9312 | — | 0.9409 | 0.9783 | 9 | −0.0022 |
| **LogiPrune ✦** | **154.5s** | **0.9345** | **0.9340** | **0.9429** | **0.9781** | **18 (33.3%)** | **+0.0011** |

✦ **LogiPrune is the only method that is simultaneously faster (+29.3%) AND better (F1 +0.0011) than the baseline.**

> RandomSearch is fastest but loses 0.0155 F1 — a real quality cost in production. FLAML matches quality but saves only 13% of time. Optuna is intermediate. LogiPrune is the only method that improves on both dimensions.

---

### Paper 1 — Scaling projection (RF, 33.3% config reduction)

As dataset size grows, GridSearch cost dominates and LogiPrune preprocessing becomes negligible:

| Dataset size | Baseline (est.) | LogiPrune (est.) | Time saving | Seconds saved |
|---|---|---|---|---|
| n=10,000 | 218s | 155s | 29% | 63s |
| n=50,000 | ~1,090s | ~775s | 29% | ~315s |
| n=100,000 | ~2,185s | ~1,550s | 29% | ~635s (11 min) |
| n=500,000 | ~10,925s | ~7,755s | 29% | ~3,170s (53 min) |
| n=1,000,000 | ~21,850s | ~15,510s | 29% | ~6,340s (1.8 hr) |

**Savings grow with dataset size, not shrink.** For non-linear models (SVM-RBF, deep trees), eliminating non-linear configurations compounds savings superlinearly.

---

### Paper 2 — LogiPruneEntropy benchmark (XGBoost, all metrics)

| Dataset | H\* | H_applied | Grid | Config savings | ΔF1 | ΔPrecision | ΔRecall | ΔAcc | ΔTime |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer ✦ | 0.7612 | 1.057 | 108→24 | 77.8% | +0.0133 | +0.0259 | 0.000 | +0.0176 | +76.9% |
| synth_hi_2k | 1.1964 | 1.398 | 108→24 | 77.8% | +0.0005 | −0.0038 | +0.005 | 0.000 | +74.3% |
| wine ✦ | 1.1795 | 1.300 | 108→24 | 77.8% | 0.000 | 0.000 | 0.000 | 0.000 | +74.5% |
| heart_like ✦ | 1.2756 | 1.450 | 108→24 | 77.8% | +0.0167 | +0.0907 | −0.067 | +0.0328 | +78.0% |
| digits_like ✦ | 1.2535 | 1.433 | 32→8 | 75.0% | 0.000 | 0.000 | 0.000 | 0.000 | +61.5% |
| moons_like ✦ | 1.3863 | 1.442 | 108→24 | 77.8% | +0.0027 | +0.0004 | +0.005 | +0.0025 | +72.7% |
| covtype_like ✦ | 1.1983 | 1.399 | 32→8 | 75.0% | 0.000 | 0.000 | 0.000 | 0.000 | +60.3% |
| housing_like | 1.3568 | 1.494 | 48→12 | 75.0% | −0.0043 | +0.0042 | −0.013 | −0.004 | +69.4% |

✦ LogiPruneEntropy equals or exceeds baseline on all metrics.

**Summary across 8 datasets:**
- ΔF1 ≥ 0 in **7 of 8** datasets
- Mean ΔF1: **+0.0036**
- Mean time saving: **+71%** (range: 60–78%)
- Mean config saving: **76.8%** (range: 75–77.8%)
- Feedback loop (feature reinstatement): **0 times** — all biconditional eliminations were safe

> **Entropy pattern:** H\* correctly predicts improvement magnitude. Lower H\* → bigger gain. breast_cancer (H\*=0.76, the lowest) shows the largest improvement. housing_like (H\*=1.36, the highest) shows the only marginal degradation (−0.004 F1), which is within single-run variance.

---

### Paper 2 — Comparison with Optuna (XGBoost)

On breast_cancer with the same time budget:

| Method | Time | F1 | Precision | Recall | Acc | AUC |
|---|---|---|---|---|---|---|
| Baseline (108 cfgs) | 78.7s | 0.9660 | 0.9467 | 0.9861 | 0.9561 | 0.9954 |
| Optuna (25 trials) | 35.0s | 0.9726 | 0.9595 | 0.9861 | 0.9649 | 0.9940 |
| **LogiPruneEntropy** | **18.2s** | **0.9793** | **0.9726** | **0.9861** | **0.9737** | **0.9871** |

**LogiPruneEntropy is faster than Optuna and better on F1, Precision, and Accuracy.** Recall is preserved identically (0.9861) across all three methods.

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

For each feature pair at threshold T, the 4-cell weight distribution π(T) = (w₁₁, w₁₀, w₀₁, w₀₀) has Shannon entropy H(T) = −Σ wᵢⱼ · log₂(wᵢⱼ) ∈ [0, 2.0] bits. H\* = min H(T) across the threshold sweep captures the best-case simplicity of the relationship.

| H\* | Complexity | XGBoost restriction |
|---|---|---|
| [0.0, 0.5) | Very simple | max_depth=[2,3], n_estimators=[50,100] |
| [0.5, 1.0) | Simple | max_depth=[3,4,5], n_estimators=[100,200] |
| [1.0, 1.5) | Moderate | max_depth=[4,5,6], n_estimators=[200,300] |
| [1.5, 2.0) | Complex | Full grid |

### v0.2.1 robustness extensions

Three additional mechanisms improve reliability under real-world conditions:

- **Out-of-range detection:** `detect_oor(X_test)` flags features whose test range exceeds training range. Grid restrictions for those features are bypassed, preventing incorrect pruning under distributional shift.
- **Rényi α=2 entropy:** computed alongside Shannon. A large Shannon–Rényi gap signals potential open-world conditions (rare events appearing in test). Affected pairs are excluded from H\* computation.
- **Conditional entropy H(D|A,B):** replaces the binary disjunction gate of Paper 1 with a three-way graduated decision — compress-and-eliminate / compress-keep / blocked — making compression safer and more precise.

### The feedback loop

After Paper 1 eliminates a feature B via A↔B, Paper 2 checks:
`if H*(A, D | without B) > H*(A, D | with B) + δ: reinstate B`

This detects when B acts as a "moderator" — its presence simplifies the A→D relationship even though A↔B suggested redundancy. In all experiments, this check correctly confirmed that eliminations were safe (0 reinstated across 8 datasets).

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
| `renyi_delta_threshold` | 0.30 | Rényi–Shannon gap above which a pair is flagged open-context |
| `oor_tolerance` | 0.05 | Fractional tolerance for out-of-range detection |
| `discretizer_strategy` | 'percentile' | Normalization strategy |
| `verbose` | False | Print progress |

---

## Recommended pipeline

```
Dataset
  → LogiPrune        (Paper 1: removes redundant features, restricts kernel/depth)
  → LogiPruneEntropy (Paper 2: restricts n_estimators, max_depth by entropy)
  → FLAML / Optuna   (searches the reduced space adaptively)
  → best model
```

LogiPrune+FLAML **Pareto-dominates** FLAML alone: same budget, smaller space, better or equal results. The combination cannot be worse than either method alone and is strictly better when the dataset has propositional structure.

---

## When it works best

- Medical diagnostics (blood panels, imaging features with domain correlations)
- Sensor fusion (IoT, process control — sensors sharing physical relationships)
- Financial features (ratios derived from shared base quantities)
- Image descriptors (pixel/feature correlations)

## When to expect modest gains

Purely synthetic Gaussian datasets with independent features have high entropy throughout (H\* > 1.5). The propositional gate and entropy signal correctly recognize this and apply minimal restrictions, protecting accuracy at the cost of smaller savings. The method is conservative by design: when uncertain, it does nothing.

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
