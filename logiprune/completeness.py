"""
logiprune.completeness
──────────────────────
Propositional Completeness Analysis for Open-Context Models.

This module addresses a question orthogonal to the HPO pruning of Papers 1 & 2:

    Given a set of known causes C₁, C₂, ..., Cₖ for outcome B,
    how complete is that causal model over the available data?
    And what can we say about causes that are NOT yet in the data?

Three components:

1. PropositionalModel
   ─────────────────
   Represents M = {C₁→B, C₂→B, ..., Cₖ→B} as a set of implications
   with empirically measured confidence, support, and violation rates.
   Stores the "empirical floor" — the baseline violation rate —
   against which new batches are compared.

2. CompletenessAnalyzer
   ────────────────────
   Computes three indices over M and new data batches:

   ICC (Índice de Completitud Causal):
       ICC = P(B=1 | some Cᵢ active) × P(some Cᵢ active | B=1)
       ICC = 1.0  → model is exhaustively complete over the dataset.
                    (A ∨ ¬A) → B holds: every B=1 has a known cause.
       ICC < 1.0  → residue exists. Possible unknown cause ¬K.

   ISR (Índice de Estructura del Residuo):
       ISR = H*(Residue(M)) / 2.0
       ISR ≈ 0   → residue has propositional structure → ¬K is discoverable.
       ISR ≈ 1   → residue is noise → no structured unknown cause.

   IVT (Índice de Violación Temporal):
       IVT(t) = violation_rate(batch_t) / baseline_violation_rate
       IVT ≈ 1   → violations stable, consistent with model noise floor.
       IVT > 2   → drift (passive or active).
       IVT accelerating across consecutive batches → adversarial signal.

3. AbductiveProposer
   ─────────────────
   When ISR is low (residue has structure), runs SWTS over Residue(M)
   to propose candidate unknown causes ¬K, ranked by:
   (a) plausibility: how strong is the proposed implication ¬K→B in the residue?
   (b) coherence:    is ¬K consistent with all existing causes in M?
   (c) novelty:      is ¬K not already implied by the existing causes?

────────────────────────────────────────────────────────────────────────────────
Formal grounding
────────────────────────────────────────────────────────────────────────────────
The CoherentistUpdater (planned) implements the AGM belief expansion operator
(Gärdenfors 1988) applied to propositional implication sets:

    M* = M ∪ {¬K→B}  is accepted iff:
      (1) ¬K is consistent with M (no contradiction)
      (2) Plausibility(¬K) > threshold (violations are structured, not noise)
      (3) ICC(M*) > ICC(M) (the residue decreases)

This is XAI at the model level (not instance level): it explains what causes
might be missing from the model, not why a specific prediction was made.

────────────────────────────────────────────────────────────────────────────────
On intentional violations (adversarial context)
────────────────────────────────────────────────────────────────────────────────
Once a model M is known, actors with incentives can exploit its causal gaps:
  - Produce B without activating any known cause (evasion)
  - Suppress B when a known cause is active (manipulation)

The IVT tracks violation rates across batches and flags:
  - Stable IVT:            model noise, no action
  - Monotone IVT growth:   passive drift (distribution shift)
  - Accelerating IVT:      adversarial signal — violations are growing
                           faster than drift alone would explain

The distinction matters because:
  - Passive drift → recalibrate thresholds, possibly retrain
  - Adversarial signal → the model's causal gaps are being exploited;
    adding ¬K to M may help, but the actor will adapt

Paper 3 (planned): full treatment of open-context completeness analysis.
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .discretize import AdaptiveDiscretizer
from .sweeper import SWTSSweeper


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Implication:
    """
    A single known implication Cᵢ → B with empirical statistics.
    """
    antecedent:      str    # feature name (or expression like 'not_C')
    consequent:      str    # always the outcome variable name
    confidence:      float  # P(B=1 | Cᵢ=1)
    support:         float  # P(Cᵢ=1)
    n_total:         int
    n_violations:    int    # cases where Cᵢ=1 and B=0
    violation_rate:  float  # n_violations / (n_total × support)
    # The "empirical floor" — baseline violation rate from training data
    # Used to detect when new batches exceed the expected noise level
    floor_rate:      float = 0.0

    def __repr__(self):
        return (f"Impl({self.antecedent}→{self.consequent}: "
                f"conf={self.confidence:.4f} supp={self.support:.4f} "
                f"viol={self.n_violations}/{int(self.n_total*self.support):.0f})")


@dataclass
class CompletenessReport:
    """
    Full completeness analysis for a model M on a dataset.
    """
    # Core indices
    ICC:   float   # Causal Completeness Index [0,1]
    ISR:   float   # Residue Structure Index   [0,1]

    # Supporting statistics
    n_total:          int
    n_B_positive:     int    # |{x: B(x)=1}|
    n_explained:      int    # |{x: B(x)=1 ∧ some Cᵢ(x)=1}|
    n_residue:        int    # n_B_positive - n_explained

    # Implication details
    implications:     list   # list of Implication objects
    causal_precision: float  # P(B=1 | some Cᵢ active)
    causal_recall:    float  # P(some Cᵢ active | B=1)

    # Residue entropy (for ISR)
    H_residue:        float  # Shannon H* of residue pairs

    def __repr__(self):
        status = "COMPLETE" if self.ICC > 0.95 else \
                 "PARTIAL"  if self.ICC > 0.70 else "INCOMPLETE"
        return (f"Completeness({status}: ICC={self.ICC:.4f} "
                f"ISR={self.ISR:.4f} residue={self.n_residue}/{self.n_B_positive})")


@dataclass
class ViolationBatch:
    """
    Violation statistics for one temporal batch.
    """
    batch_id:         int
    n_obs:            int
    violation_rates:  dict   # antecedent → rate
    IVT:              float  # ratio vs baseline
    IVT_acceleration: float  # IVT(t) - IVT(t-1), positive = accelerating
    signal:           str    # 'stable' | 'drift' | 'adversarial'


@dataclass
class AbductiveCandidate:
    """
    A proposed unknown cause ¬K suggested by the residue structure.
    """
    feature:        str
    relation:       str       # proposed implication type (A→B, ¬A→B, etc.)
    plausibility:   float     # confidence of proposed implication in residue
    coherence:      float     # consistency with existing M (1 = no conflict)
    novelty:        float     # how different from existing causes (1 = fully novel)
    score:          float     # combined rank score
    support_in_residue: float


# ── PropositionalModel ────────────────────────────────────────────────────────

class PropositionalModel:
    """
    Represents M = {C₁→B, C₂→B, ..., Cₖ→B} fitted from training data.

    The model stores each implication with its empirical statistics and
    sets the "floor" violation rate — the baseline noise level of the model
    against which new batches are compared.

    Parameters
    ----------
    outcome : str
        Name of the outcome variable B.
    threshold : float, default=0.5
        Binarization threshold for antecedents.
    min_confidence : float, default=0.85
        Minimum confidence to include an implication in M.
    negate : list of str, optional
        Features to use in negated form (¬C instead of C).
        E.g., negate=['C'] means use C=0 as the antecedent.
    """

    def __init__(self,
                 outcome: str,
                 threshold: float = 0.5,
                 min_confidence: float = 0.85,
                 negate: Optional[list] = None):
        self.outcome        = outcome
        self.threshold      = threshold
        self.min_confidence = min_confidence
        self.negate         = set(negate or [])
        self.implications_: list = []
        self._fitted        = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PropositionalModel':
        """
        Fit M from training data using an internal threshold sweep.

        For each feature, sweeps thresholds in [0.1, 0.9] to find the
        threshold that maximises confidence for C→B or ¬C→B.
        This handles skewed distributions (e.g. Beta(2,5)) where a
        fixed threshold of 0.5 would dilute confidence.
        Only implication directions with conf ≥ min_confidence are kept.
        """
        n     = len(X)
        b     = (y.values > 0.5).astype(int)
        steps = np.linspace(0.10, 0.90, 17)   # 17-point sweep
        self.implications_ = []

        for col in X.columns:
            if col == self.outcome:
                continue
            v = X[col].values.astype(float)

            best_conf = 0.0; best = None
            for t in steps:
                for negated in [False, True]:
                    if negated:
                        a = (v <= t).astype(int)
                        ante = f"\u00ac{col}"
                    else:
                        a = (v > t).astype(int)
                        ante = col
                    n_ante = int(a.sum())
                    # Skip degenerate splits
                    if n_ante < max(5, int(0.005 * n)):
                        continue
                    n_viol = int(((a == 1) & (b == 0)).sum())
                    conf   = 1.0 - n_viol / n_ante
                    supp   = n_ante / n
                    if conf > best_conf:
                        best_conf = conf
                        best = (ante, col, negated, t, conf, supp, n_ante, n_viol)

            if best and best_conf >= self.min_confidence:
                ante, col_, neg, t_, conf, supp, n_ante, n_viol = best
                self.implications_.append(Implication(
                    antecedent=ante,
                    consequent=self.outcome,
                    confidence=round(conf, 6),
                    support=round(supp, 4),
                    n_total=n,
                    n_violations=n_viol,
                    violation_rate=round(n_viol / max(n_ante,1), 8),
                    floor_rate=round(n_viol / max(n_ante,1), 8),
                ))
                # Store threshold used so active_mask can use it
                self.implications_[-1]._threshold = t_
                self.implications_[-1]._negated   = neg

        self._fitted = True
        return self

    def active_mask(self, X: pd.DataFrame) -> np.ndarray:
        """
        Boolean array: True for rows where at least one cause in M is active.
        Uses the per-implication threshold found during fit() sweep.
        """
        if not self._fitted or not self.implications_:
            return np.zeros(len(X), dtype=bool)
        mask = np.zeros(len(X), dtype=bool)
        for impl in self.implications_:
            col = impl.antecedent.lstrip('\u00ac').lstrip('¬')
            if col not in X.columns:
                continue
            v   = X[col].values.astype(float)
            t_  = getattr(impl, '_threshold', self.threshold)
            neg = getattr(impl, '_negated',   impl.antecedent.startswith('¬'))
            if neg:
                a = (v <= t_).astype(bool)
            else:
                a = (v > t_).astype(bool)
            mask |= a
        return mask

    def summary(self) -> str:
        if not self._fitted:
            return "PropositionalModel: not fitted."
        lines = [f"PropositionalModel(outcome='{self.outcome}', "
                 f"{len(self.implications_)} implications):"]
        for impl in sorted(self.implications_, key=lambda x: -x.confidence):
            lines.append(f"  {impl}")
        return "\n".join(lines)


# ── CompletenessAnalyzer ──────────────────────────────────────────────────────

class CompletenessAnalyzer:
    """
    Measures three completeness indices for a PropositionalModel M on data.

    ICC (Causal Completeness Index):
    ─────────────────────────────────
    ICC = causal_precision × causal_recall

    causal_precision = P(B=1 | some Cᵢ active)
        "When a known cause is present, how often does B occur?"
        Low → causes are noisy / unreliable.

    causal_recall = P(some Cᵢ active | B=1)
        "When B occurs, how often is there a known cause?"
        Low → many B=1 cases have no known cause → model is incomplete.

    ICC = 1.0 → (A ∨ ¬A) → B holds effectively:
                 every occurrence of B has a known cause,
                 and every known cause produces B.
    ICC < 1.0 → residue exists, possibly containing unknown cause ¬K.

    ISR (Residue Structure Index):
    ───────────────────────────────
    H* of the residue observations (those not explained by any known cause).
    ISR = H*(residue) / 2.0  (normalized to [0,1])
    ISR ≈ 0 → residue has strong propositional structure → ¬K is discoverable.
    ISR ≈ 1 → residue is uniformly distributed → pure noise, no ¬K.

    Parameters
    ----------
    threshold : float, default=0.5
        Binarization threshold for features (should match PropositionalModel).
    isr_steps : int, default=7
        Number of threshold steps for residue H* sweep (fewer for speed).
    """

    def __init__(self,
                 threshold: float = 0.5,
                 isr_steps: int = 7):
        self.threshold = threshold
        self.isr_steps = isr_steps

    def analyze(self, model: PropositionalModel,
                X: pd.DataFrame, y: pd.Series) -> CompletenessReport:
        """
        Compute ICC, ISR, and supporting statistics for M on (X, y).
        """
        n   = len(X)
        b   = (y.values > 0.5).astype(int)
        n_B = int(b.sum())

        if n_B == 0:
            warnings.warn("[CompletenessAnalyzer] No positive outcomes in data.",
                          UserWarning, stacklevel=2)

        # Active mask: rows where at least one known cause is active
        active = model.active_mask(X)

        # Causal precision: P(B=1 | some cause active)
        n_active = int(active.sum())
        if n_active > 0:
            causal_precision = float((b[active] == 1).mean())
        else:
            causal_precision = 0.0

        # Causal recall: P(some cause active | B=1)
        if n_B > 0:
            b_mask = (b == 1)
            causal_recall = float(active[b_mask].mean())
        else:
            causal_recall = 0.0

        ICC = round(causal_precision * causal_recall, 4)

        # Residue: B=1 but no known cause active
        residue_mask = (b == 1) & (~active)
        n_explained  = int(((b == 1) & active).sum()) if n_B > 0 else 0
        n_residue    = n_B - n_explained

        # ISR: entropy of residue if it exists
        H_residue = 2.0  # default: max entropy (noise)
        if n_residue >= 10 and len(model.implications_) > 0:
            X_res = X[residue_mask].reset_index(drop=True)
            H_residue = self._residue_entropy(X_res)

        ISR = round(H_residue / 2.0, 4)

        return CompletenessReport(
            ICC=ICC, ISR=ISR,
            n_total=n, n_B_positive=n_B,
            n_explained=n_explained, n_residue=n_residue,
            implications=model.implications_,
            causal_precision=round(causal_precision, 4),
            causal_recall=round(causal_recall, 4),
            H_residue=round(H_residue, 4),
        )

    def _residue_entropy(self, X_res: pd.DataFrame) -> float:
        """Compute minimum Shannon entropy across feature pairs in the residue."""
        if X_res.shape[1] < 2 or len(X_res) < 5:
            return 2.0
        T = np.linspace(0.25, 0.75, self.isr_steps)
        cols = X_res.columns.tolist()
        min_H = 2.0
        for i, ca in enumerate(cols):
            for j, cb in enumerate(cols):
                if j <= i:
                    continue
                a = X_res[ca].values
                b = X_res[cb].values
                for t in T:
                    ab = (a > t).astype(np.int32)
                    bb = (b > t).astype(np.int32)
                    states = ab * 2 + bb
                    c = np.bincount(states, minlength=4).astype(float)
                    w = c / len(a)
                    H = float(-np.dot(w, np.log2(w + 1e-12)))
                    if H < min_H:
                        min_H = H
        return min_H


# ── ViolationTracker ──────────────────────────────────────────────────────────

class ViolationTracker:
    """
    Tracks violation rates across temporal batches to detect:
      - Stable noise:        IVT ≈ 1.0 across batches
      - Passive drift:       IVT growing monotonically but slowly
      - Adversarial signal:  IVT accelerating (growing faster over time)

    The distinction between drift and adversarial is the second derivative
    of the IVT time series: drift has near-zero acceleration; adversarial
    has positive and growing acceleration.

    Parameters
    ----------
    model : PropositionalModel
        The fitted model M whose violations are tracked.
    acceleration_threshold : float, default=0.5
        If IVT acceleration exceeds this value for 2+ consecutive batches,
        flag as adversarial.
    drift_threshold : float, default=2.0
        IVT above this value (without acceleration) flags as drift.
    """

    def __init__(self,
                 model: PropositionalModel,
                 acceleration_threshold: float = 0.5,
                 drift_threshold:        float = 2.0):
        self.model                   = model
        self.acceleration_threshold  = acceleration_threshold
        self.drift_threshold         = drift_threshold
        self._history:               list = []   # list of ViolationBatch
        self._batch_counter:         int  = 0

    def update(self, X: pd.DataFrame, y: pd.Series) -> ViolationBatch:
        """
        Process a new batch and return the ViolationBatch with IVT and signal.
        """
        self._batch_counter += 1
        b = (y.values > 0.5).astype(int)
        n = len(X)

        # Compute violation rate per implication
        vrates = {}
        for impl in self.model.implications_:
            col = impl.antecedent.lstrip('¬')
            if col not in X.columns:
                continue
            v = X[col].values
            if impl.antecedent.startswith('¬'):
                a = (v <= self.model.threshold).astype(int)
            else:
                a = (v > self.model.threshold).astype(int)
            n_ante = int(a.sum())
            if n_ante == 0:
                vrates[impl.antecedent] = 0.0
                continue
            n_viol = int(((a == 1) & (b == 0)).sum())
            vrates[impl.antecedent] = round(n_viol / n_ante, 6)

        # Mean violation rate across all implications
        mean_vrate = float(np.mean(list(vrates.values()))) if vrates else 0.0

        # Mean floor rate from model
        floor_rates = [impl.floor_rate for impl in self.model.implications_
                       if impl.antecedent in vrates]
        mean_floor  = float(np.mean(floor_rates)) if floor_rates else 1e-4

        # IVT calibration: when floor_rate is very small (near-perfect model),
        # use an absolute violation rate instead of a ratio to avoid explosion.
        # A model with conf=1.0 has floor_rate=0; we treat any violation as
        # a meaningful signal by setting the effective floor to 1/n_train.
        n_train_est = self.model.implications_[0].n_total if self.model.implications_ else 1000
        floor_abs_min = 1.0 / max(n_train_est, 100)   # at least 1 violation in training
        mean_floor  = max(mean_floor, floor_abs_min)

        IVT = round(mean_vrate / mean_floor, 4)

        # IVT acceleration: difference from last batch
        if self._history:
            last_IVT = self._history[-1].IVT
            acceleration = round(IVT - last_IVT, 4)
        else:
            acceleration = 0.0

        # Signal classification
        n_accelerating = sum(
            1 for b_ in self._history[-2:]
            if b_.IVT_acceleration > self.acceleration_threshold
        )
        if acceleration > self.acceleration_threshold and n_accelerating >= 1:
            signal = 'adversarial'
        elif IVT > self.drift_threshold:
            signal = 'drift'
        else:
            signal = 'stable'

        # Emit warning for non-stable signals
        if signal != 'stable':
            msg = (
                f"[ViolationTracker] Batch {self._batch_counter}: "
                f"signal='{signal}' IVT={IVT:.3f} "
                f"(baseline={mean_floor:.6f}, current={mean_vrate:.6f})"
            )
            if signal == 'adversarial':
                msg += (
                    f" — Violation rate is ACCELERATING across "
                    f"{n_accelerating+1} consecutive batches. "
                    f"This may indicate intentional exploitation of model gaps."
                )
            warnings.warn(msg, UserWarning, stacklevel=2)

        batch = ViolationBatch(
            batch_id=self._batch_counter,
            n_obs=n,
            violation_rates=vrates,
            IVT=IVT,
            IVT_acceleration=acceleration,
            signal=signal,
        )
        self._history.append(batch)
        return batch

    def history_summary(self) -> pd.DataFrame:
        """Return IVT history as a DataFrame."""
        if not self._history:
            return pd.DataFrame()
        rows = []
        for b in self._history:
            rows.append({
                'batch': b.batch_id,
                'n_obs': b.n_obs,
                'IVT': b.IVT,
                'acceleration': b.IVT_acceleration,
                'signal': b.signal,
                'mean_vrate': round(np.mean(list(b.violation_rates.values())), 6)
                if b.violation_rates else 0.0,
            })
        return pd.DataFrame(rows)


# ── AbductiveProposer ─────────────────────────────────────────────────────────

class AbductiveProposer:
    """
    When the residue has structure (ISR < isr_threshold), proposes
    candidate unknown causes ¬K by running SWTS over Residue(M).

    A candidate ¬K is ranked by three criteria:
      plausibility: conf(¬K → B) in the residue data
      coherence:    1 - max_overlap(¬K, existing causes in M)
      novelty:      how different ¬K is from any existing antecedent

    Combined score = plausibility × coherence × novelty

    Parameters
    ----------
    isr_threshold : float, default=0.60
        Only propose candidates if ISR < this value.
        (If ISR ≥ threshold, residue is noise — no discovery warranted.)
    min_plausibility : float, default=0.75
        Minimum confidence of proposed ¬K→B in residue to be reported.
    max_candidates : int, default=5
        Maximum number of candidates to return.
    """

    def __init__(self,
                 isr_threshold:   float = 0.60,
                 min_plausibility: float = 0.75,
                 max_candidates:  int   = 5):
        self.isr_threshold    = isr_threshold
        self.min_plausibility = min_plausibility
        self.max_candidates   = max_candidates

    def propose(self,
                model:   PropositionalModel,
                report:  CompletenessReport,
                X:       pd.DataFrame,
                y:       pd.Series,
                ) -> list:
        """
        Propose candidate unknown causes from the residue.

        Returns list of AbductiveCandidate, sorted by score descending.
        Returns empty list if ISR ≥ isr_threshold or residue is too small.
        """
        if report.ISR >= self.isr_threshold:
            return []   # residue is noise, no discovery warranted

        if report.n_residue < 10:
            return []   # not enough residue observations

        # Build residue mask: B=1 AND no known cause active
        b      = (y.values > 0.5).astype(int)
        active = model.active_mask(X)
        res_mask = (b == 1) & (~active)

        X_res = X[res_mask].reset_index(drop=True)
        y_res = y[res_mask].reset_index(drop=True)

        if len(X_res) < 10:
            return []

        # Known antecedents (for coherence and novelty computation)
        known_antes = {impl.antecedent for impl in model.implications_}

        candidates = []
        n_res = len(X_res)

        for col in X_res.columns:
            v = X_res[col].values

            for negated in [False, True]:
                if negated:
                    a        = (v <= model.threshold).astype(int)
                    ante_name = f"¬{col}"
                else:
                    a        = (v > model.threshold).astype(int)
                    ante_name = col

                n_ante = int(a.sum())
                if n_ante < 5:
                    continue

                # All residue outcomes are B=1, so conf(¬K→B) = n_ante/n_ante
                # But we want: among all observations (not just residue),
                # how often does ¬K predict B=1?
                # Use the residue for discovery, full data for validation.
                plausibility = float(n_ante / n_res)  # support in residue

                if plausibility < self.min_plausibility:
                    continue

                # Coherence: does ¬K contradict any existing cause?
                # Approximation: if ¬K is the negation of an existing cause,
                # coherence is lower (they compete).
                base_col = col
                if base_col in known_antes or f"¬{base_col}" in known_antes:
                    coherence = 0.5   # related to known cause — partial overlap
                else:
                    coherence = 1.0   # fully novel feature

                # Novelty: 1 if feature not used in any form in M
                if ante_name in known_antes:
                    novelty = 0.0     # already in M — skip
                elif col in {a.lstrip('¬') for a in known_antes}:
                    novelty = 0.3     # same feature, different polarity
                else:
                    novelty = 1.0     # genuinely new feature

                if novelty == 0.0:
                    continue

                score = round(plausibility * coherence * novelty, 4)

                candidates.append(AbductiveCandidate(
                    feature=col,
                    relation=f"{ante_name} → {model.outcome}",
                    plausibility=round(plausibility, 4),
                    coherence=round(coherence, 4),
                    novelty=round(novelty, 4),
                    score=score,
                    support_in_residue=round(n_ante / n_res, 4),
                ))

        # Deduplicate by feature (keep best polarity)
        seen = {}
        for c in sorted(candidates, key=lambda x: -x.score):
            if c.feature not in seen:
                seen[c.feature] = c

        return sorted(seen.values(), key=lambda x: -x.score)[:self.max_candidates]
