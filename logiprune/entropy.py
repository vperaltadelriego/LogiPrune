"""
logiprune.entropy
─────────────────
LogiPrune-Entropy: Truth Table Entropy Analysis for Model Complexity Selection.

Core concept (Paper 2):
─────────────────────────────────────────────────────────────────────────────
While LogiPrune (Paper 1) asks WHAT logical relationship exists between features,
LogiPrune-Entropy asks HOW COMPLEX that relationship is — and uses that complexity
measure to restrict the model's hyperparameter space a priori.

The Truth Table Profile of a feature pair (A, B) at threshold T is:
    π(T) = (w₁₁, w₁₀, w₀₁, w₀₀)
    where wᵢⱼ = count(a=i, b=j) / n

The Truth Table Entropy is the Shannon entropy of this 4-cell distribution:
    H_shannon(T) = -Σ wᵢⱼ · log₂(wᵢⱼ)

H ranges from 0 (all observations in one cell — perfectly predictable)
to 2.0 bits (uniform distribution — maximum complexity, no structure).

────────────────────────────────────────────────────────────────────────────────
Extension v0.2: Three improvements over the base Shannon analysis
────────────────────────────────────────────────────────────────────────────────

1. OUT-OF-RANGE DETECTION (context_open flag)
   Classical truth tables assume a closed world: all possible states were
   observed at training time. When transform() receives test data whose
   range exceeds training range by more than oor_tolerance, the method
   flags those features as open-context and relaxes their grid restrictions.
   This prevents incorrect pruning under distributional shift.

   Check: if X_test.max() > X_train.max() * (1 + oor_tolerance)
              OR X_test.min() < X_train.min() * (1 - oor_tolerance)
           → mark feature open_context, bypass its grid restriction.

2. RÉNYI α=2 ENTROPY AS OPEN-CONTEXT DETECTOR
   Shannon entropy weights all cells equally by probability p.
   Rényi entropy of order α=2 (collision entropy) weights by p²:

       H_renyi(π) = -log₂(Σ wᵢⱼ²)

   H_renyi ≤ H_shannon always. A large gap signals that rare cells are
   inflating Shannon more than expected — a sign of potential distributional
   novelty or an open-world condition. If the relative gap
   |H_shannon - H_renyi| / H_shannon exceeds renyi_delta_threshold (0.30),
   the pair is flagged and its restrictions are relaxed.
   Computational cost: O(n) per threshold — identical to Shannon.

3. CONDITIONAL ENTROPY H(D|A,B) AS CONTINUOUS ∨E GATE
   LogiPrune (Paper 1) uses a binary θ-gate for disjunction elimination:
   compress (A,B)→F only if conf(A⊢D) ≥ θ AND conf(B⊢D) ≥ θ.

   H(D|A,B) provides a continuous alternative:
       H(D|A,B) = -Σ p(a,b) · [p(d=1|a,b)log p(d=1|a,b) + p(d=0|a,b)log p(d=0|a,b)]

   Decision rule (three-way):
       H(D|A,B) < h_compress  → 'compress_and_eliminate'  (replace A,B with F)
       h_compress ≤ H < h_retain → 'compress_keep'         (add F, keep A,B)
       H(D|A,B) ≥ h_retain    → 'blocked'                 (∨E gate blocked)

   Cost: O(n) — single pass building the 8-state joint table.

────────────────────────────────────────────────────────────────────────────────
Open-context research frontier (Paper 3 — not yet implemented)
────────────────────────────────────────────────────────────────────────────────
The three extensions above handle mild distributional shift (range-based OOR
detection, Rényi divergence signal). A deeper open problem remains:

    When p(A,B,D) changes structurally between training and test — not just
    in range, but in shape — neither Shannon nor Rényi from training data
    can detect it. Robust handling requires:
      - Tsallis entropy (non-extensive, captures long-range correlations)
      - Online entropy estimation from streaming test data
      - Explicit shift detection (Maximum Mean Discrepancy, etc.)

    Planned as Paper 3: "Open-Context Truth Table Analysis — Rényi Entropy
    and Conditional Information for Robust HPO Pruning Under Distribution
    Shift." Not integrated here to avoid uncalibrated hyperparameters (q in
    Tsallis, bandwidth in MMD) in an already-working system.

XGBoost grid restriction rules (unchanged from v0.1):
─────────────────────────────────────────────────────
H* ∈ [0.0, 0.5): Very simple  → max_depth=[2,3],   n_estimators=[50,100]
H* ∈ [0.5, 1.0): Simple       → max_depth=[3,4,5], n_estimators=[100,200]
H* ∈ [1.0, 1.5): Moderate     → max_depth=[4,5,6], n_estimators=[200,300]
H* ≥ 1.5:         No struct    → full grid (no restriction)
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TruthTableProfile:
    """
    Truth table analysis for a feature pair at one threshold.

    v0.2 additions: h_renyi, renyi_delta, open_context_flag.
    """
    col_a: str; col_b: str; threshold: float
    n11: int; n10: int; n01: int; n00: int; n_total: int
    w11: float; w10: float; w01: float; w00: float
    entropy: float          # Shannon H, 0–2.0 bits
    dominant: float
    dominant_cell: str      # '11','10','01','00'
    # v0.2
    h_renyi: float = 0.0           # Rényi α=2 entropy
    renyi_delta: float = 0.0       # |H_sh - H_ry| / H_sh
    open_context_flag: bool = False # True if renyi_delta > threshold
    complexity_class: str = 'medium'

    def __repr__(self):
        oc = " [OPEN]" if self.open_context_flag else ""
        return (f"TTP({self.col_a}×{self.col_b} @{self.threshold:.2f}: "
                f"H={self.entropy:.3f} H_renyi={self.h_renyi:.3f}{oc} [{self.complexity_class}])")


@dataclass
class ConditionalEntropyResult:
    """
    H(D|A,B) — the continuous ∨E gate result (v0.2).

    gate_decision is one of:
        'compress_and_eliminate' — replace (A,B) with F, drop both
        'compress_keep'          — add F, retain A and B
        'blocked'                — do not compress
    """
    col_a: str; col_b: str; col_d: str
    h_cond: float
    gate_decision: str
    n_states: int


@dataclass
class EntropyProfile:
    """
    Per-pair summary across all threshold sweeps.

    v0.2 additions: h_renyi_min, open_context.
    """
    col_a: str; col_b: str
    h_min: float; h_mean: float; h_std: float; best_threshold: float
    profiles: list
    h_renyi_min: float = 0.0
    open_context: bool = False


# ── Entropy functions (all O(n) or O(k) for k cells) ─────────────────────────

def _shannon(weights):
    """Shannon entropy. H = -Σ w·log₂(w). Range [0, log₂(k)] bits."""
    return float(-sum(w * np.log2(w + 1e-12) for w in weights))


def _renyi_alpha2(weights):
    """
    Rényi collision entropy (α=2). H_R = -log₂(Σ w²).

    Properties:
      - H_R ≤ H_Shannon always (equality only at uniform distribution)
      - More sensitive to distribution of rare events than Shannon
      - Same scale as Shannon for 4 cells: range [0, 2.0] bits
      - Cost: O(k) where k = number of cells
    """
    return float(-np.log2(sum(w * w for w in weights) + 1e-12))


def _complexity_class(h: float) -> str:
    if h < 0.5: return 'very_low'
    if h < 1.0: return 'low'
    if h < 1.5: return 'medium'
    return 'high'


# ── Core computation ──────────────────────────────────────────────────────────

def truth_table_profile(col_a: str, col_b: str,
                         a: np.ndarray, b: np.ndarray,
                         threshold: float,
                         renyi_delta_threshold: float = 0.30,
                         ) -> TruthTableProfile:
    """
    Compute the truth table profile at one threshold.

    v0.2: adds Rényi α=2 and open-context detection.

    Parameters
    ----------
    renyi_delta_threshold : float
        If |H_shannon - H_renyi| / H_shannon > this value, flag open context.
        Default 0.30 means a 30% gap triggers the flag.
    """
    ab = (a > threshold).astype(np.uint8)
    bb = (b > threshold).astype(np.uint8)
    n  = len(a)
    # v0.2.3: bincount vectorization — encode (ab,bb) as 0..3 in one pass
    states_4 = ab.astype(np.int32) * 2 + bb.astype(np.int32)
    counts_4 = np.bincount(states_4, minlength=4)
    # states: 0=⟨0,0⟩ 1=⟨0,1⟩ 2=⟨1,0⟩ 3=⟨1,1⟩
    n00,n01,n10,n11 = int(counts_4[0]),int(counts_4[1]),int(counts_4[2]),int(counts_4[3])
    w11,w10,w01,w00 = n11/n, n10/n, n01/n, n00/n
    ws = [w11, w10, w01, w00]

    H_sh = _shannon(ws)
    H_ry = _renyi_alpha2(ws)
    # Relative Rényi-Shannon gap: rises when rare cells appear
    delta = abs(H_sh - H_ry) / (H_sh + 1e-9)

    dom = max(ws); dom_cell = ['11','10','01','00'][ws.index(dom)]

    return TruthTableProfile(
        col_a=col_a, col_b=col_b, threshold=threshold,
        n11=n11, n10=n10, n01=n01, n00=n00, n_total=n,
        w11=round(w11,4), w10=round(w10,4), w01=round(w01,4), w00=round(w00,4),
        entropy=round(H_sh,4), dominant=round(dom,4), dominant_cell=dom_cell,
        h_renyi=round(H_ry,4),
        renyi_delta=round(delta,4),
        open_context_flag=bool(delta > renyi_delta_threshold),
        complexity_class=_complexity_class(H_sh),
    )


def conditional_entropy_gate(a: np.ndarray, b: np.ndarray,
                               d: np.ndarray,
                               threshold: float,
                               h_compress: float = 0.55,
                               h_retain:   float = 0.90,
                               ) -> ConditionalEntropyResult:
    """
    Compute H(D|A,B) as a continuous gate for disjunction elimination (∨E).

    Replaces the binary θ-gate of Paper 1 with a three-way decision:

        H(D|A,B) < h_compress  → 'compress_and_eliminate'
                                  Safe to replace (A,B) with F=f(A,B)
        h_compress ≤ H < h_retain → 'compress_keep'
                                   Create F but retain A and B
        H(D|A,B) ≥ h_retain    → 'blocked'
                                  ∨E gate blocked

    The default thresholds (0.55, 0.90) align the midpoint of the three-way
    decision with the behavior of the binary gate at θ=0.85.

    Cost: O(n) — single pass building the 8-state joint table.

    Parameters
    ----------
    a, b : continuous feature arrays (will be binarized at threshold)
    d    : continuous target array (will be binarized at 0.5)
    h_compress : float
        H(D|A,B) below this → safe to compress and eliminate.
    h_retain : float
        H(D|A,B) above this → ∨E blocked.
    """
    ab = (a > threshold).astype(np.uint8)
    bb = (b > threshold).astype(np.uint8)
    db = (d > 0.5).astype(np.uint8)
    n  = len(a)

    # --- v0.2.3 vectorized via np.bincount (Gemini review, 10-50x faster) ---
    # Encode the 8-state joint (ab, bb, db) as a single integer in [0,7]
    # state = ab*4 + bb*2 + db  →  shape (n,), values in {0..7}
    states_8 = ab.astype(np.int32) * 4 + bb.astype(np.int32) * 2 + db.astype(np.int32)
    counts_8 = np.bincount(states_8, minlength=8).astype(float)   # shape (8,)

    # Aggregate: for each (ab,bb) cell, sum counts across db=0 and db=1
    # cell index c = ab*2 + bb  →  c in {0,1,2,3}
    # counts_8[c*2]   = n(ab, bb, db=0)
    # counts_8[c*2+1] = n(ab, bb, db=1)
    h_cond   = 0.0
    n_states = 0
    for c in range(4):
        n0 = counts_8[c * 2]
        n1 = counts_8[c * 2 + 1]
        n_ab = n0 + n1
        if n_ab == 0:
            continue
        n_states += 1
        p_ab = n_ab / n
        p1   = n1 / n_ab
        p0   = 1.0 - p1
        h_ab = -(p1 * np.log2(p1 + 1e-12) + p0 * np.log2(p0 + 1e-12))
        h_cond += p_ab * h_ab

    if h_cond < h_compress:
        decision = 'compress_and_eliminate'
    elif h_cond < h_retain:
        decision = 'compress_keep'
    else:
        decision = 'blocked'

    return ConditionalEntropyResult(
        col_a='A', col_b='B', col_d='__target__',
        h_cond=round(float(h_cond), 4),
        gate_decision=decision,
        n_states=n_states,
    )


# ── EntropyAnalyzer ───────────────────────────────────────────────────────────

class EntropyAnalyzer:
    """
    Computes entropy profiles for all feature pairs and translates
    them into XGBoost hyperparameter grid restrictions.

    v0.2 improvements:
    ──────────────────
    1. Out-of-range detection: detect_oor(X_test) compares test range
       against stored training range. OOR features bypass grid restrictions.

    2. Rényi α=2 alongside Shannon: pairs with a large Shannon-Rényi gap
       are flagged open_context and excluded from H* computation, making
       restrictions conservative (safer) under distributional novelty.

    3. Conditional entropy H(D|A,B) in feedback_check: the feature
       reinstatement check now uses both the Shannon delta AND the
       H(D|A,B) gate, making it more sensitive to semi-redundant features.

    Parameters
    ----------
    steps : array-like, default np.arange(0.25, 0.76, 0.05)
        Threshold sweep values.
    feedback_delta : float, default=0.10
        Shannon entropy increase that triggers feature reinstatement.
    renyi_delta_threshold : float, default=0.30
        Relative Rényi-Shannon gap above which a pair is flagged open_context.
    oor_tolerance : float, default=0.05
        Fractional tolerance for OOR detection (5% of training range).
    """

    def __init__(self,
                 steps=None,
                 feedback_delta: float = 0.10,
                 renyi_delta_threshold: float = 0.30,
                 oor_tolerance: float = 0.05):
        self.steps = (np.arange(0.25, 0.76, 0.05)
                      if steps is None else np.array(steps))
        self.feedback_delta        = feedback_delta
        self.renyi_delta_threshold = renyi_delta_threshold
        self.oor_tolerance         = oor_tolerance

        # Fitted state
        self.entropy_profiles_:        dict = {}
        self.feature_target_profiles_: dict = {}
        self.dataset_h_min_:     float = 2.0
        self.dataset_h_mean_:    float = 2.0
        self.dataset_h_renyi_min_: float = 2.0
        self.open_context_pairs_: list = []
        # v0.2 OOR
        self._train_min_: Optional[pd.Series] = None
        self._train_max_: Optional[pd.Series] = None
        self._oor_features_: list = []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sweep_pair(self, ca, cb, a, b) -> EntropyProfile:
        profiles=[]; h_vals=[]; hr_vals=[]; any_open=False
        for th in self.steps:
            ttp = truth_table_profile(ca, cb, a, b, th,
                                       self.renyi_delta_threshold)
            profiles.append(ttp)
            h_vals.append(ttp.entropy)
            hr_vals.append(ttp.h_renyi)
            if ttp.open_context_flag:
                any_open = True
        h_min = min(h_vals)
        best_th = self.steps[np.argmin(h_vals)]
        return EntropyProfile(
            col_a=ca, col_b=cb,
            h_min=round(h_min,4),
            h_mean=round(float(np.mean(h_vals)),4),
            h_std=round(float(np.std(h_vals)),4),
            best_threshold=round(float(best_th),3),
            profiles=profiles,
            h_renyi_min=round(min(hr_vals),4),
            open_context=any_open,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EntropyAnalyzer':
        """
        Fit on training data. Stores range for OOR detection.
        Computes Shannon + Rényi for all pairs and feature-target relationships.
        """
        cols = X.columns.tolist()
        ya   = y.values.astype(float)
        y_norm = (ya - ya.min()) / (ya.max() - ya.min() + 1e-9)

        # Store training range (v0.2 — used by detect_oor)
        self._train_min_ = X.min()
        self._train_max_ = X.max()

        # Feature pairs
        for i, ca in enumerate(cols):
            for j, cb in enumerate(cols):
                if j <= i: continue
                ep = self._sweep_pair(ca, cb, X[ca].values, X[cb].values)
                self.entropy_profiles_[(ca, cb)] = ep
                if ep.open_context:
                    self.open_context_pairs_.append((ca, cb))

        # Feature vs target
        for col in cols:
            ep = self._sweep_pair(col, '__target__', X[col].values, y_norm)
            self.feature_target_profiles_[col] = ep

        all_h  = [ep.h_min       for ep in self.entropy_profiles_.values()]
        all_hr = [ep.h_renyi_min for ep in self.entropy_profiles_.values()]
        self.dataset_h_min_       = round(float(np.min(all_h)),4)  if all_h else 2.0
        self.dataset_h_mean_      = round(float(np.mean(all_h)),4) if all_h else 2.0
        self.dataset_h_renyi_min_ = round(float(np.min(all_hr)),4) if all_hr else 2.0
        return self

    def detect_oor(self, X_test: pd.DataFrame) -> list:
        """
        Detect features in X_test that are out of training range.

        Compares test min/max against stored training min/max with a
        fractional buffer of oor_tolerance (default 5%). Features that
        exceed the buffer are flagged as open_context; their grid
        restrictions will be bypassed in xgb_grid_from_entropy().

        Returns list of OOR feature names. Also emits a UserWarning.
        """
        if self._train_min_ is None:
            raise RuntimeError("Call fit() before detect_oor().")
        oor = []
        for col in X_test.columns:
            if col not in self._train_min_.index:
                continue
            lo = self._train_min_[col]; hi = self._train_max_[col]
            span = abs(hi - lo) + 1e-9
            if (X_test[col].max() > hi + self.oor_tolerance * span or
                    X_test[col].min() < lo - self.oor_tolerance * span):
                oor.append(col)
        self._oor_features_ = oor
        if oor:
            warnings.warn(
                f"[LogiPruneEntropy] Out-of-range features detected: {oor}. "
                f"Grid restrictions based on these features are bypassed "
                f"(oor_tolerance={self.oor_tolerance}). "
                f"Consider retraining on combined data if shift is persistent.",
                UserWarning, stacklevel=2
            )
        return oor

    def feedback_check(self, eliminated_features: list,
                        X: pd.DataFrame, y: pd.Series) -> list:
        """
        Check which eliminated features should be reinstated.

        v0.2: checks both Shannon ΔH (as in v0.1) AND the H(D|A,B)
        conditional entropy gate. Reinstatement is triggered if either
        signal indicates the elimination was unsafe.
        """
        to_reinstate = []
        ya    = y.values.astype(float)
        y_norm = (ya - ya.min()) / (ya.max() - ya.min() + 1e-9)

        for col_rm in eliminated_features:
            partner = None
            for (ca, cb) in self.entropy_profiles_:
                if ca == col_rm or cb == col_rm:
                    partner = cb if ca == col_rm else ca
                    break
            if partner is None or partner not in X.columns:
                continue

            # Signal 1: Shannon delta (v0.1 behavior preserved)
            h_before = (self.feature_target_profiles_[partner].h_min
                        if partner in self.feature_target_profiles_ else 2.0)
            ep_after = self._sweep_pair(partner, '__target__',
                                         X[partner].values, y_norm)
            h_after  = ep_after.h_min
            sh_fired = (h_after - h_before) > self.feedback_delta

            # Signal 2: Conditional entropy H(D|A,B) gate (v0.2)
            hcond_fired = False
            if col_rm in X.columns and partner in X.columns:
                ep_key = self.entropy_profiles_.get(
                    (col_rm, partner),
                    self.entropy_profiles_.get((partner, col_rm))
                )
                th = ep_key.best_threshold if ep_key else 0.5
                ceg = conditional_entropy_gate(
                    X[col_rm].values, X[partner].values, y_norm, th
                )
                # 'compress_keep' means both features are needed → reinstate
                hcond_fired = (ceg.gate_decision == 'compress_keep')

            if sh_fired or hcond_fired:
                parts = []
                if sh_fired:
                    parts.append(f"ΔH_shannon={h_after-h_before:.3f}")
                if hcond_fired:
                    parts.append(f"H(D|A,B)='{ceg.gate_decision}'")
                to_reinstate.append({
                    'col':     col_rm, 'partner': partner,
                    'h_before': round(h_before,4), 'h_after': round(h_after,4),
                    'delta':    round(h_after-h_before,4),
                    'reason':   ' + '.join(parts),
                })
        return to_reinstate

    def xgb_grid_from_entropy(self, base_grid: dict,
                               oor_features: Optional[list] = None,
                               ) -> dict:
        """
        Translate entropy profile into XGBoost grid restrictions.

        v0.2: OOR features and Rényi-flagged open-context pairs are
        excluded from the H* computation, conservatively raising the
        effective H* and relaxing restrictions for uncertain pairs.

        Parameters
        ----------
        base_grid : dict  — full XGBoost parameter grid
        oor_features : list, optional — from detect_oor(); if provided,
            features in this list are excluded from H* computation.
        """
        g = dict(base_grid)

        # Exclude OOR and Rényi open-context pairs from H* computation
        excluded = set(self.open_context_pairs_)
        if oor_features:
            for (ca, cb) in self.entropy_profiles_:
                if ca in oor_features or cb in oor_features:
                    excluded.add((ca, cb))

        valid_h = [ep.h_min for (ca, cb), ep in self.entropy_profiles_.items()
                   if (ca, cb) not in excluded]
        h_pairs = float(np.min(valid_h)) if valid_h else self.dataset_h_min_

        # Feature-target entropy (also exclude OOR features)
        ft_vals = []
        for col, ep in self.feature_target_profiles_.items():
            if oor_features and col in oor_features:
                continue
            ft_vals.append(ep.h_min)
        ft_h = float(np.mean(ft_vals)) if ft_vals else 2.0

        # Weighted combination: 60% pair structure, 40% target predictability
        h = round(0.6 * h_pairs + 0.4 * ft_h, 4)
        self._applied_h = h

        def restrict(suffix, values):
            for k in g:
                if k.endswith(suffix):
                    merged = [v for v in values if v in g[k]]
                    if merged:
                        g[k] = merged

        if h < 0.5:
            restrict('max_depth',        [2, 3])
            restrict('n_estimators',     [50, 100])
            restrict('subsample',        [0.9, 1.0])
            restrict('colsample_bytree', [0.9, 1.0])
            restrict('learning_rate',    [0.1, 0.3])
        elif h < 1.0:
            restrict('max_depth',        [3, 4, 5])
            restrict('n_estimators',     [100, 200])
            restrict('subsample',        [0.8, 1.0])
            restrict('colsample_bytree', [0.8, 1.0])
        elif h < 1.5:
            restrict('max_depth',        [4, 5, 6])
            restrict('n_estimators',     [200, 300])
            restrict('subsample',        [0.7, 0.8, 1.0])
        # else h >= 1.5: no restriction (open or complex structure)

        return g

    def complexity_report(self) -> pd.DataFrame:
        """DataFrame of all pair profiles sorted by h_min ascending."""
        rows = []
        for (ca, cb), ep in sorted(self.entropy_profiles_.items(),
                                    key=lambda x: x[1].h_min):
            rows.append(dict(
                col_a=ca, col_b=cb,
                h_min=ep.h_min, h_renyi_min=ep.h_renyi_min,
                h_mean=ep.h_mean, h_std=ep.h_std,
                best_threshold=ep.best_threshold,
                open_context=ep.open_context,
                complexity=_complexity_class(ep.h_min),
            ))
        return pd.DataFrame(rows)

    def summary(self) -> str:
        total = len(self.entropy_profiles_)
        by_class = {}
        for ep in self.entropy_profiles_.values():
            c = _complexity_class(ep.h_min)
            by_class[c] = by_class.get(c, 0) + 1
        lines = [
            "EntropyAnalyzer Summary (v0.2)",
            f"  Total pairs analyzed:              {total}",
            f"  Dataset H_min (Shannon):           {self.dataset_h_min_}",
            f"  Dataset H_min (Rényi α=2):         {self.dataset_h_renyi_min_}",
            f"  Dataset H_mean (Shannon):          {self.dataset_h_mean_}",
            f"  Open-context pairs (Rényi flag):   {len(self.open_context_pairs_)}",
            f"  OOR features (last detect_oor()):  {len(self._oor_features_)}",
            "  Complexity distribution:",
        ]
        for c in ['very_low','low','medium','high']:
            n = by_class.get(c, 0)
            lines.append(f"    {c:<12}: {n:>4} ({100*n/max(total,1):.1f}%)")
        if hasattr(self, '_applied_h'):
            lines.append(f"  Applied H (0.6×pairs + 0.4×feat-target): {self._applied_h}")
        return "\n".join(lines)
