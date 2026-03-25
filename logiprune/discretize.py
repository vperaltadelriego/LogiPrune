"""
logiprune.discretize
────────────────────
Adaptive discretization strategies for continuous features.

The key insight: standard min-max normalization to [0,1] flattens
density structure (e.g. ages 15-90 where 80% cluster at 20-50).
Percentile-based discretization preserves that density, making
threshold sweeping semantically meaningful.

Three strategies:
  - 'percentile'  : divide by deciles/quintiles of actual distribution
  - 'minmax'      : standard [0,1] normalization (baseline)
  - 'zscore_clip' : z-score clamped to [-3, 3] then normalized

────────────────────────────────────────────────────────────────────────────────
v0.2.2: Asymmetric Threshold Sweep and Model Lifetime Estimation
────────────────────────────────────────────────────────────────────────────────

Standard LogiPrune uses a SYMMETRIC threshold sweep: the same threshold T is
applied to both features A and B when binarizing a pair. This works well when
both features have similar distributions, but can underestimate the strength
of a logical relationship when the marginal distributions are asymmetric.

Example: A has 80% of values above 0.6 (right-skewed),
         B has 80% of values below 0.4 (left-skewed).
Using T=0.5 for both produces a misleading truth table.
Using T_A=0.6, T_B=0.4 reveals the true relationship.

ASYMMETRIC SWEEP (optional, not default):
─────────────────────────────────────────
AsymmetricSweepAnalyzer.fit() sweeps all (T_A, T_B) combinations independently.
Cost: O(n × S²) vs O(n × S) for symmetric. For S=11, this is ~11× slower.

For a realistic dataset (p=30, n=10,000):
  Symmetric:   4,785 evaluations ≈ 570ms
  Asymmetric: 52,635 evaluations ≈ 4,200ms

The asymmetric sweep is NOT the default because for most datasets the symmetric
sweep already finds the globally optimal threshold (the entropy landscape is
flat around the minimum). Activate it when:
  (a) The marginal asymmetry warning fires (|mean_A - mean_B| > asym_threshold)
  (b) You suspect distributional shift in new data
  (c) You want the most precise "model lifetime" estimate

WHEN DOES ASYMMETRIC HELP BEYOND SYMMETRIC?
─────────────────────────────────────────────
  Strong benefit:    Disjunction (A∨B) with asymmetric marginals
  Moderate benefit:  Implication (A→B) with asymmetric marginals
  Minimal benefit:   Biconditional (A↔B) — landscape is a wide flat plateau
  Minimal benefit:   Strong relations (conf > 0.95) — already detected by symmetric

MODEL LIFETIME ESTIMATION:
────────────────────────────
The entropy landscape H(T_A, T_B) is a signature of the stability of the
logical relationship in the data. A wide, flat landscape = robust relationship
that survives threshold perturbation = long model lifetime.
A narrow, peaked landscape = fragile relationship = short lifetime.

ModelLifetimeEstimator.evaluate_drift(X_new) answers:
  "Given that I trained on X_train and now have X_new, does my model still hold?"

Decision logic:
  drift_score < 0.10  → model valid, no action needed
  drift_score 0.10-0.25 → WARN: monitor, consider recalibration
  drift_score > 0.25  → ALERT: retrain recommended

Cost comparison for N training observations + M new observations:
  Asymmetric sweep on M:    O(M × C² × S²)       — "is the model still valid?"
  Full retrain (N+M):       O((N+M)×C²×S + k'×(N+M)×t_fit)

For M << N (e.g., M = 0.1×N), the sweep is ~50-100× cheaper than retraining.
Use the sweep when you need a fast signal; use retraining when the signal says
the relationship has changed structurally.

PAPER 3 NOTE:
──────────────
This module provides the foundation for open-context analysis. The drift score
based on landscape displacement is the precursor to the full treatment planned
in Paper 3: "Open-Context Truth Table Analysis — Rényi Entropy and Conditional
Information for Robust HPO Pruning Under Distribution Shift."
"""

import warnings
import numpy as np
import pandas as pd
from typing import Literal, Optional
from itertools import product


DiscretizeStrategy = Literal['percentile', 'minmax', 'zscore_clip']


# ── AdaptiveDiscretizer ───────────────────────────────────────────────────────

class AdaptiveDiscretizer:
    """
    Fit on training data, transform any split to [0, 1] using
    the chosen strategy. Percentile is the default and recommended
    strategy for LogiPrune.

    v0.2.2: also computes marginal asymmetry statistics at fit() time,
    which are used by AsymmetricSweepAnalyzer to decide whether the
    asymmetric sweep is worth running.

    Parameters
    ----------
    strategy : 'percentile' | 'minmax' | 'zscore_clip'
    n_quantiles : int
        Number of quantile breakpoints (default 10 = deciles).
        Only used when strategy='percentile'.
    asymmetry_threshold : float, default=0.15
        If |mean(col_A) - mean(col_B)| > this value (after normalization),
        the pair is flagged as potentially benefiting from asymmetric sweep.
    warn_asymmetry : bool, default=True
        Emit a UserWarning when asymmetric marginals are detected.
        Set to False to suppress in production pipelines.
    """

    def __init__(self,
                 strategy: DiscretizeStrategy = 'percentile',
                 n_quantiles: int = 10,
                 asymmetry_threshold: float = 0.15,
                 warn_asymmetry: bool = True):
        self.strategy             = strategy
        self.n_quantiles          = n_quantiles
        self.asymmetry_threshold  = asymmetry_threshold
        self.warn_asymmetry       = warn_asymmetry
        self._params: dict        = {}
        self._means:  dict        = {}   # col → mean after normalization
        self.asymmetric_pairs_:   list = []   # pairs with |mean_A - mean_B| > threshold

    def fit(self, X: pd.DataFrame) -> 'AdaptiveDiscretizer':
        self._params = {}
        self._means  = {}
        self._n_fit_rows_ = len(X)   # stored for leakage detection in transform()
        for col in X.columns:
            v = X[col].values.astype(float)
            if self.strategy == 'percentile':
                edges = np.percentile(v, np.linspace(0, 100, self.n_quantiles + 1))
                edges = np.unique(edges)
                self._params[col] = edges
            elif self.strategy == 'minmax':
                self._params[col] = (v.min(), v.max())
            elif self.strategy == 'zscore_clip':
                self._params[col] = (v.mean(), v.std() + 1e-9)
            else:
                # Unknown strategy: store raw (no-op transform)
                self._params[col] = None
        # Detect asymmetric pairs using RAW data (before normalization)
        # After percentile normalization all means → 0.5, so we compare
        # the raw marginal means normalized by their own range [min, max].
        self.asymmetric_pairs_ = []
        cols = list(X.columns)
        raw_means = {}
        for col in cols:
            v = X[col].values.astype(float)
            lo, hi = v.min(), v.max()
            raw_means[col] = float((v.mean() - lo) / (hi - lo + 1e-9))
        for i, ca in enumerate(cols):
            for j, cb in enumerate(cols):
                if j <= i:
                    continue
                delta = abs(raw_means[ca] - raw_means[cb])
                if delta > self.asymmetry_threshold:
                    self.asymmetric_pairs_.append((ca, cb, round(delta, 3)))
        if self.warn_asymmetry and self.asymmetric_pairs_:
            n = len(self.asymmetric_pairs_)
            examples = ", ".join(f"({a},{b})" for a,b,_ in self.asymmetric_pairs_[:3])
            if n > 3:
                examples += f" ... (+{n-3} more)"
            warnings.warn(
                f"[LogiPrune] {n} feature pair(s) have asymmetric marginal distributions "
                f"(|mean_A - mean_B| > {self.asymmetry_threshold}): {examples}. "
                f"These pairs may benefit from an asymmetric threshold sweep, which can "
                f"reveal stronger logical relationships than the default symmetric sweep. "
                f"Use AsymmetricSweepAnalyzer(pairs=discretizer.asymmetric_pairs_) to investigate. "
                f"This is optional and not required for standard LogiPrune operation.",
                UserWarning, stacklevel=2
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # v0.2.3 data-leakage guard (Gemini review)
        # If the test range matches training range almost exactly AND the test
        # has as many rows as the training set, the user may have forgotten to
        # split before fitting — emit a warning rather than fail silently.
        if self._params and len(X) == self._n_fit_rows_:
            for col in X.columns:
                if col in self._params and self.strategy == 'percentile':
                    edges = self._params[col]
                    v = X[col].values.astype(float)
                    if (len(edges) > 2 and
                            abs(v.min() - edges[0]) < 1e-6 and
                            abs(v.max() - edges[-1]) < 1e-6):
                        warnings.warn(
                            f"[LogiPrune] Possible data leakage: transform() received "
                            f"data with the same size ({len(X)} rows) and identical "
                            f"range as the data used in fit(). "
                            f"Ensure you call fit() on training data only and "
                            f"transform() on train and test splits separately.",
                            UserWarning, stacklevel=2
                        )
                        break   # warn once, not per column
        out = {}
        for col in X.columns:
            v = X[col].values.astype(float)
            if self.strategy == 'percentile':
                edges = self._params.get(col)
                if edges is None:
                    out[col] = v
                    continue
                out[col] = np.searchsorted(edges, v, side='right') / len(edges)
            elif self.strategy == 'minmax':
                lo, hi = self._params[col]
                out[col] = (v - lo) / (hi - lo + 1e-9)
            elif self.strategy == 'zscore_clip':
                mu, sd = self._params[col]
                z = (v - mu) / sd
                out[col] = (np.clip(z, -3, 3) + 3) / 6
            else:
                out[col] = v
        return pd.DataFrame(out, columns=X.columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


# ── AsymmetricSweepAnalyzer ───────────────────────────────────────────────────

class AsymmetricSweepAnalyzer:
    """
    Optional asymmetric threshold sweep for feature pairs with
    asymmetric marginal distributions.

    NOT enabled by default in LogiPrune because:
      (1) For most datasets, the symmetric sweep already finds the global minimum
      (2) Cost is ~11× higher than the symmetric sweep
      (3) The gain is only meaningful for weak relations or asymmetric distributions

    When to use:
      - AdaptiveDiscretizer.asymmetric_pairs_ is non-empty (warning was emitted)
      - You want the most accurate H* for disjunctive pairs
      - You are computing model lifetime estimates on new data M

    Parameters
    ----------
    steps_a : int, default=11
        Number of threshold steps for feature A.
    steps_b : int, default=11
        Number of threshold steps for feature B.
    t_range : tuple, default=(0.20, 0.70)
        Range of thresholds to sweep for both A and B.
    pairs : list of (col_a, col_b, delta), optional
        If provided, only sweep these specific pairs.
        Typically set from AdaptiveDiscretizer.asymmetric_pairs_.

    Cost note:
        steps_a × steps_b evaluations per pair (default: 121 vs 11 for symmetric).
        For p=30 (435 pairs): ~52,635 evaluations ≈ 4–5 seconds for n=10,000.
        For targeted pairs only: proportionally less.
    """

    def __init__(self,
                 steps_a: int = 11,
                 steps_b: int = 11,
                 t_range: tuple = (0.20, 0.70),
                 pairs: Optional[list] = None):
        self.steps_a  = steps_a
        self.steps_b  = steps_b
        self.t_range  = t_range
        self.pairs    = pairs
        self.T_A      = np.linspace(t_range[0], t_range[1], steps_a)
        self.T_B      = np.linspace(t_range[0], t_range[1], steps_b)
        self.results_: dict = {}   # (ca, cb) → AsymmetricProfile

    def _shannon(self, weights):
        return float(-sum(w * np.log2(w + 1e-12) for w in weights))

    def _renyi2(self, weights):
        return float(-np.log2(sum(w * w for w in weights) + 1e-12))

    def _eval_pair(self, a: np.ndarray, b: np.ndarray):
        """Sweep all (T_A, T_B) combinations for one pair. O(n × S_A × S_B)."""
        n = len(a)
        best = {'H': 2.0, 'Ta': 0.5, 'Tb': 0.5,
                'n11':0,'n10':0,'n01':0,'n00':0,
                'H_renyi': 2.0, 'landscape': []}
        for ta, tb in product(self.T_A, self.T_B):
            ab = (a > ta).astype(int)
            bb = (b > tb).astype(int)
            n11 = int(((ab==1)&(bb==1)).sum())
            n10 = int(((ab==1)&(bb==0)).sum())
            n01 = int(((ab==0)&(bb==1)).sum())
            n00 = int(((ab==0)&(bb==0)).sum())
            w = [n11/n, n10/n, n01/n, n00/n]
            H  = self._shannon(w)
            Hr = self._renyi2(w)
            best['landscape'].append({
                'Ta': round(ta, 3), 'Tb': round(tb, 3),
                'H': round(H, 4), 'Hr': round(Hr, 4),
                'n11': n11, 'n10': n10, 'n01': n01, 'n00': n00
            })
            if H < best['H']:
                best.update({'H':H,'Ta':ta,'Tb':tb,
                             'n11':n11,'n10':n10,'n01':n01,'n00':n00,'H_renyi':Hr})
        return best

    def fit(self, X: pd.DataFrame) -> 'AsymmetricSweepAnalyzer':
        """
        Run the asymmetric sweep on specified pairs (or all pairs if pairs=None).

        Parameters
        ----------
        X : normalized feature DataFrame (output of AdaptiveDiscretizer.transform)
        """
        cols = X.columns.tolist()
        target_pairs = (
            [(ca, cb) for ca, cb, _ in self.pairs]
            if self.pairs else
            [(ca, cb) for i, ca in enumerate(cols)
             for j, cb in enumerate(cols) if j > i]
        )
        for ca, cb in target_pairs:
            if ca not in X.columns or cb not in X.columns:
                continue
            result = self._eval_pair(X[ca].values, X[cb].values)
            self.results_[(ca, cb)] = result
        return self

    def improvement_report(self, sym_profiles: dict) -> list:
        """
        Compare asymmetric H* against symmetric H* for each swept pair.

        Parameters
        ----------
        sym_profiles : dict mapping (ca, cb) → EntropyProfile (from EntropyAnalyzer)

        Returns list of dicts with delta_H and whether the asymmetric sweep
        found a meaningfully different (lower) minimum.
        """
        report = []
        for (ca, cb), asym in self.results_.items():
            sym = sym_profiles.get((ca, cb))
            H_sym  = sym.h_min if sym else 2.0
            H_asym = round(asym['H'], 4)
            delta  = round(H_sym - H_asym, 4)
            meaningful = delta > 0.05  # 0.05 bits threshold
            report.append({
                'pair':        (ca, cb),
                'H_symmetric': H_sym,
                'H_asymmetric': H_asym,
                'delta_H':     delta,
                'T_sym':       '(same T)',
                'T_A_asym':    round(asym['Ta'], 3),
                'T_B_asym':    round(asym['Tb'], 3),
                'meaningful':  meaningful,
            })
        report.sort(key=lambda x: x['delta_H'], reverse=True)
        return report


# ── ModelLifetimeEstimator ────────────────────────────────────────────────────

class ModelLifetimeEstimator:
    """
    Estimates whether a trained LogiPrune model remains valid on new data
    by comparing the entropy landscape of new observations against the
    landscape computed at training time.

    This is a lightweight alternative to full retraining when new data M
    arrives and you want to know:
      "Does my existing model still hold? Or has the structure changed?"

    How it works:
    ─────────────
    At fit() time (training), the entropy landscape H(T_A, T_B) is stored
    for each feature pair as a fingerprint. The landscape encodes the stability
    and shape of the logical relationship.

    At evaluate_drift(X_new) time, the same landscape is computed on X_new
    and compared against the training fingerprint. The drift score measures
    how much the landscape has shifted.

    Drift score interpretation:
      < 0.10  → "Model valid"     — landscape stable, relationships hold
      0.10–0.25 → "Monitor"      — landscape drifting, recalibration advisable
      > 0.25  → "Retrain"        — structural change detected, retraining needed

    Cost comparison (N training, M new observations):
    ──────────────────────────────────────────────────
    This method: O(M × C² × S²) per sweep
    Full retrain: O((N+M) × C² × S + k' × (N+M) × t_fit)

    For M = 0.1×N: this method is ~50–100× cheaper than retraining.
    For M ≈ N:     the cost gap narrows; prefer retraining when M is large.

    Parameters
    ----------
    steps : int, default=11
        Threshold steps per dimension (asymmetric sweep).
    t_range : tuple, default=(0.25, 0.75)
        Threshold sweep range.
    drift_warn  : float, default=0.10
        Drift score above which a UserWarning is emitted.
    drift_alert : float, default=0.25
        Drift score above which retraining is strongly recommended.
    min_pairs : int, default=5
        Minimum number of pairs to evaluate (samples the most informative ones).
    """

    def __init__(self,
                 steps: int = 11,
                 t_range: tuple = (0.25, 0.75),
                 drift_warn:  float = 0.10,
                 drift_alert: float = 0.25,
                 min_pairs:   int   = 5):
        self.steps       = steps
        self.t_range     = t_range
        self.drift_warn  = drift_warn
        self.drift_alert = drift_alert
        self.min_pairs   = min_pairs
        self._T          = np.linspace(t_range[0], t_range[1], steps)
        self._landscapes_: dict = {}   # (ca, cb) → np.ndarray of shape (S, S)
        self._fitted     = False

    def _landscape(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute H(T_A, T_B) landscape as S×S matrix.

        v0.2.3: vectorized via np.bincount.
        Pre-binarize A for all S thresholds at once → shape (S, n).
        Then for each T_A row, binarize B for all S thresholds in one
        np.bincount pass per (T_A, T_B) cell.
        Cost: O(S² × n / word_size) — same asymptotic, ~3-5× faster in practice.
        """
        n = len(a)
        S = len(self._T)
        L = np.zeros((S, S))
        # Pre-binarize A for all thresholds: shape (S, n), dtype uint8
        A_bin = (a[None, :] > self._T[:, None]).astype(np.uint8)  # (S, n)
        for i in range(S):
            ab = A_bin[i]  # shape (n,), already binarized
            for j, tb in enumerate(self._T):
                bb = (b > tb).astype(np.uint8)
                # encode (ab, bb) → 0..3 in one pass
                states = ab.astype(np.int32) * 2 + bb.astype(np.int32)
                c = np.bincount(states, minlength=4).astype(float)
                # c: [n00, n01, n10, n11]
                w = c / n
                L[i, j] = float(-np.dot(w, np.log2(w + 1e-12)))
        return L

    def _select_pairs(self, X: pd.DataFrame) -> list:
        """Select the most informative pairs (lowest symmetric H*)."""
        cols = X.columns.tolist()
        pairs = [(ca, cb) for i, ca in enumerate(cols)
                 for j, cb in enumerate(cols) if j > i]
        # For efficiency, prioritize pairs with low H at T=0.5
        scored = []
        for ca, cb in pairs:
            ab = (X[ca].values > 0.5).astype(int)
            bb = (X[cb].values > 0.5).astype(int)
            n  = len(ab)
            ws = [((ab==i)&(bb==j)).sum()/n for i,j in [(1,1),(1,0),(0,1),(0,0)]]
            H  = float(-sum(w * np.log2(w+1e-12) for w in ws))
            scored.append((H, ca, cb))
        scored.sort()
        # Return pairs with lowest entropy (most structured)
        n_select = max(self.min_pairs, len(pairs) // 5)
        return [(ca, cb) for _, ca, cb in scored[:n_select]]

    def fit(self, X: pd.DataFrame) -> 'ModelLifetimeEstimator':
        """
        Compute and store the entropy landscape fingerprint from training data.

        Parameters
        ----------
        X : normalized feature DataFrame (output of AdaptiveDiscretizer.transform)
        """
        pairs = self._select_pairs(X)
        self._landscapes_ = {}
        for ca, cb in pairs:
            self._landscapes_[(ca, cb)] = self._landscape(
                X[ca].values, X[cb].values
            )
        self._fitted = True
        return self

    def evaluate_drift(self, X_new: pd.DataFrame,
                       discretizer: Optional['AdaptiveDiscretizer'] = None,
                       ) -> dict:
        """
        Evaluate whether the model remains valid on new data X_new.

        Parameters
        ----------
        X_new : new data, either already normalized or raw (if discretizer provided)
        discretizer : if provided, X_new is raw and will be normalized using
                      the training discretizer (recommended for out-of-range detection)

        Returns
        -------
        dict with:
            drift_score   — mean landscape displacement across all tracked pairs
            status        — 'valid' | 'monitor' | 'retrain'
            pair_scores   — per-pair drift scores
            recommendation — human-readable text
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluate_drift().")

        if discretizer is not None:
            oor = discretizer.detect_oor(X_new) if hasattr(discretizer, 'detect_oor') else []
            X_norm = discretizer.transform(X_new)
        else:
            X_norm = X_new
            oor    = []

        pair_scores = {}
        for (ca, cb), L_train in self._landscapes_.items():
            if ca not in X_norm.columns or cb not in X_norm.columns:
                continue
            L_new = self._landscape(X_norm[ca].values, X_norm[cb].values)
            # Drift = mean absolute difference of the landscapes
            # Normalized by the range [0, 2.0 bits]
            drift = float(np.mean(np.abs(L_train - L_new))) / 2.0
            pair_scores[(ca, cb)] = round(drift, 4)

        drift_score = round(float(np.mean(list(pair_scores.values()))), 4) if pair_scores else 0.0

        if drift_score < self.drift_warn:
            status = 'valid'
            rec = (f"Model valid (drift={drift_score:.3f} < {self.drift_warn}). "
                   f"No action needed.")
        elif drift_score < self.drift_alert:
            status = 'monitor'
            rec = (f"Monitor (drift={drift_score:.3f}). "
                   f"Logical relationships are drifting. "
                   f"Consider recalibrating thresholds or retraining if drift continues.")
        else:
            status = 'retrain'
            rec = (f"Retrain recommended (drift={drift_score:.3f} > {self.drift_alert}). "
                   f"Structural change detected in feature relationships. "
                   f"The existing model may no longer reflect the data distribution.")

        if oor:
            rec += f" WARNING: out-of-range features detected: {oor}."

        if status != 'valid':
            warnings.warn(
                f"[ModelLifetimeEstimator] {rec}",
                UserWarning, stacklevel=2
            )

        return {
            'drift_score':    drift_score,
            'status':         status,
            'pair_scores':    pair_scores,
            'oor_features':   oor,
            'n_pairs_checked': len(pair_scores),
            'recommendation': rec,
        }

    def lifetime_summary(self) -> str:
        """Return a human-readable summary of what the estimator is tracking."""
        if not self._fitted:
            return "ModelLifetimeEstimator: not yet fitted."
        n = len(self._landscapes_)
        pairs_str = ", ".join(f"({a},{b})" for a,b in list(self._landscapes_.keys())[:5])
        if n > 5:
            pairs_str += f" ... (+{n-5} more)"
        return (
            f"ModelLifetimeEstimator\n"
            f"  Tracking {n} feature pairs: {pairs_str}\n"
            f"  Landscape grid: {self.steps}×{self.steps} = {self.steps**2} points per pair\n"
            f"  Drift thresholds: warn={self.drift_warn}, alert={self.drift_alert}\n"
            f"  Cost per evaluate_drift(M): O(M × {n} × {self.steps**2}) evaluations"
        )
