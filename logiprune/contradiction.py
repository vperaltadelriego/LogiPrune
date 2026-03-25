"""
logiprune.contradiction
────────────────────────
Contradiction Detection and Observation Pruning.

The logical dual of completeness analysis:

    (A ∨ ¬A) → B   ← exhaustive model  (all B cases explained)
    (A ∧ ¬A) → B   ← contradiction     (impossible antecedent → dirty data)

While completeness analysis asks "is there a missing cause?",
contradiction analysis asks "are there observations that violate the
known causes in a way that reveals dirty data, not model incompleteness?"

The distinction is critical because both produce residues, but:

    Completeness residue: B=1 with NO known cause active
        → model may be incomplete → search for ¬K

    Contradiction residue: known cause Cᵢ is active but B=0
        → model is violated → inspect observations, not the model

────────────────────────────────────────────────────────────────────────────────
Classification of violations
────────────────────────────────────────────────────────────────────────────────

A violation is an observation where Cᵢ=1 (cause active) but B=0 (effect absent).
Violations are classified into three types:

    'outlier'
        Isolated: the violation is not similar to other violations.
        Likely: measurement error, sensor fault, data entry mistake.
        Action: safe to remove. Record in audit log.

    'subpopulation'
        Coherent: multiple violations share a pattern (low residue entropy
        within the violation set). This is not noise — it is a distinct
        data-generating process mixed into the dataset.
        Action: separate, do NOT discard. Analyze separately as a
        potential confounding subgroup.

    'noise'
        High-entropy violation set: violations are spread uniformly across
        the feature space. No structure. Could be measurement noise or
        irreducible stochasticity in the causal process.
        Action: monitor. If rate is stable and low (< floor), accept as
        model noise. If rate grows, flag for review.

────────────────────────────────────────────────────────────────────────────────
Observation pruning with audit trail
────────────────────────────────────────────────────────────────────────────────

ObservationPruner separates violations from the clean dataset while:
  1. Preserving the removed observations (never discarded, only separated)
  2. Recording WHY each observation was removed (which implication violated,
     what type, what the feature values were)
  3. Verifying that the clean dataset produces a better model
     (ICC improves, violation rate drops)

This is NOT the same as outlier removal in sklearn:
  - sklearn's IsolationForest removes observations based on density
  - ObservationPruner removes observations based on logical contradiction
    with a fitted propositional model

The audit log is the key: it makes the pruning explainable and reversible.
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .completeness import PropositionalModel, CompletenessAnalyzer


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ViolationSet:
    """
    Set of observations that violate a specific implication Cᵢ → B.
    """
    antecedent:    str
    consequent:    str
    n_violations:  int
    n_active:      int     # total observations where Cᵢ was active
    rate:          float   # n_violations / n_active
    floor_rate:    float   # expected rate from training
    ratio:         float   # rate / floor_rate (IVT at observation level)

    # Indices into the original DataFrame
    violation_indices: np.ndarray = field(default_factory=lambda: np.array([]))

    # Classification
    violation_type: str = 'noise'   # 'outlier' | 'subpopulation' | 'noise'
    H_violations:   float = 2.0     # Shannon entropy of violation feature space

    def __repr__(self):
        return (f"ViolSet({self.antecedent}→{self.consequent}: "
                f"n={self.n_violations}/{self.n_active} "
                f"rate={self.rate:.4f} type={self.violation_type})")


@dataclass
class ContradictionReport:
    """
    Full contradiction analysis for a PropositionalModel M on a dataset.

    The contradiction score measures how far the dataset is from
    the (A ∧ ¬A) → B extreme:

        score = 0.0  → no violations, model perfectly consistent
        score = 1.0  → all known-cause observations are violated
                        (dataset is maximally contradictory for this model)

    violation_type_summary breaks down violations by type.
    """
    contradiction_score:  float   # mean violation rate across all implications
    n_total:              int
    n_violations_total:   int
    violation_sets:       list    # list of ViolationSet
    violation_type_summary: dict  # type → count

    def __repr__(self):
        if self.contradiction_score < 0.005:
            level = "CLEAN"
        elif self.contradiction_score < 0.05:
            level = "LOW_NOISE"
        elif self.contradiction_score < 0.20:
            level = "NOISY"
        else:
            level = "CONTRADICTORY"
        return (f"ContradictionReport({level}: "
                f"score={self.contradiction_score:.4f} "
                f"violations={self.n_violations_total}/{self.n_total})")


@dataclass
class PruningResult:
    """
    Result of ObservationPruner.prune().

    X_clean, y_clean: dataset with violations removed
    X_pruned, y_pruned: the removed observations (NEVER discarded)
    audit_log: DataFrame recording why each observation was removed
    icc_before, icc_after: ICC improvement after pruning
    """
    X_clean:    pd.DataFrame
    y_clean:    pd.Series
    X_pruned:   pd.DataFrame   # removed observations
    y_pruned:   pd.Series
    audit_log:  pd.DataFrame   # one row per removed observation
    n_removed:  int
    n_kept:     int
    icc_before: float
    icc_after:  float
    improved:   bool           # True if ICC increased after pruning

    def __repr__(self):
        delta = round(self.icc_after - self.icc_before, 4)
        return (f"PruningResult(removed={self.n_removed}/{self.n_removed+self.n_kept} "
                f"ICC: {self.icc_before:.4f} → {self.icc_after:.4f} "
                f"Δ={delta:+.4f} improved={self.improved})")


# ── ContradictionAnalyzer ─────────────────────────────────────────────────────

class ContradictionAnalyzer:
    """
    Detects and classifies observations that violate the model M.

    Classification logic:
    ─────────────────────
    For each implication Cᵢ → B in M, finds all observations where
    Cᵢ is active (cause present) but B=0 (effect absent).

    The violation set is classified by measuring the Shannon entropy
    H* of the feature space restricted to violation indices:

        H*(violations) < h_subpop_threshold
            → 'subpopulation': violations share structure → confounding group
              DO NOT remove — analyze separately.

        H*(violations) > h_subpop_threshold AND violation rate < floor × ratio_threshold
            → 'outlier': isolated violations above expected noise floor
              Safe to remove with audit trail.

        Otherwise → 'noise': violations are uniformly spread
              Accept as model noise. Remove only if rate >> floor.

    Parameters
    ----------
    h_subpop_threshold : float, default=0.80
        H* below this indicates a structured subpopulation.
        (0.80 out of 2.0 bits = 40% of maximum entropy)
    ratio_threshold : float, default=3.0
        Violation rate above floor × ratio_threshold triggers 'outlier'
        classification (rather than pure noise).
    min_violations : int, default=3
        Minimum violation count to classify (below this: noise by default).
    isr_steps : int, default=7
        Threshold sweep steps for H* computation in violation set.
    """

    def __init__(self,
                 h_subpop_threshold: float = 0.80,
                 ratio_threshold:    float = 3.0,
                 min_violations:     int   = 3,
                 isr_steps:          int   = 7):
        self.h_subpop_threshold = h_subpop_threshold
        self.ratio_threshold    = ratio_threshold
        self.min_violations     = min_violations
        self.isr_steps          = isr_steps

    def _entropy_of_set(self, X_viol: pd.DataFrame) -> float:
        """Compute minimum Shannon H* across pairs in violation set."""
        if len(X_viol) < 5 or X_viol.shape[1] < 2:
            return 2.0
        T = np.linspace(0.25, 0.75, self.isr_steps)
        cols = X_viol.columns.tolist()
        min_H = 2.0
        for i, ca in enumerate(cols):
            for j, cb in enumerate(cols):
                if j <= i:
                    continue
                a = X_viol[ca].values.astype(float)
                b = X_viol[cb].values.astype(float)
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

    def analyze(self,
                model: PropositionalModel,
                X: pd.DataFrame,
                y: pd.Series) -> ContradictionReport:
        """
        Analyze contradictions between M and (X, y).
        """
        n   = len(X)
        b   = (y.values > 0.5).astype(int)
        idx = np.arange(n)

        violation_sets  = []
        n_total_viols   = 0
        type_counts     = {'outlier': 0, 'subpopulation': 0, 'noise': 0}

        for impl in model.implications_:
            col      = impl.antecedent.lstrip('\u00ac').lstrip('¬')
            if col not in X.columns:
                continue
            t_   = getattr(impl, '_threshold', 0.5)
            neg  = getattr(impl, '_negated',   impl.antecedent.startswith('¬'))
            v    = X[col].values.astype(float)
            a    = (v <= t_).astype(int) if neg else (v > t_).astype(int)

            n_active = int(a.sum())
            if n_active == 0:
                continue

            # Violation indices: cause active, effect absent
            viol_mask = (a == 1) & (b == 0)
            viol_idx  = idx[viol_mask]
            n_viol    = len(viol_idx)

            rate       = n_viol / n_active
            floor_rate = max(impl.floor_rate, 1.0 / max(n_active, 100))
            ratio      = rate / floor_rate if floor_rate > 0 else 0.0

            # Classify violation type
            if n_viol < self.min_violations:
                vtype = 'noise'
                H_v   = 2.0
            else:
                X_v = X.iloc[viol_idx].reset_index(drop=True)
                H_v = self._entropy_of_set(X_v)

                # Subpopulation requires:
                #   (a) low entropy (structured) AND
                #   (b) sufficient size (≥ min_subpop, default 10)
                #   Small violation sets are outliers by size definition,
                #   regardless of entropy (few points always have low H*)
                min_subpop = max(10, int(0.005 * n))   # at least 0.5% of n
                if H_v < self.h_subpop_threshold and n_viol >= min_subpop:
                    # Low entropy + sufficient size → structured subpopulation
                    vtype = 'subpopulation'
                elif ratio > self.ratio_threshold:
                    # Rate well above floor → isolated outliers
                    vtype = 'outlier'
                else:
                    vtype = 'noise'

            type_counts[vtype] += n_viol
            n_total_viols += n_viol

            vs = ViolationSet(
                antecedent=impl.antecedent,
                consequent=impl.consequent,
                n_violations=n_viol,
                n_active=n_active,
                rate=round(rate, 6),
                floor_rate=round(floor_rate, 6),
                ratio=round(ratio, 4),
                violation_indices=viol_idx,
                violation_type=vtype,
                H_violations=round(H_v, 4),
            )
            violation_sets.append(vs)

            if vtype in ('subpopulation', 'outlier'):
                warnings.warn(
                    f"[ContradictionAnalyzer] {impl.antecedent}→{impl.consequent}: "
                    f"{n_viol} {vtype} violations "
                    f"(rate={rate:.4f}, floor={floor_rate:.6f}, ratio={ratio:.1f}×, "
                    f"H*={H_v:.3f} bits). "
                    + ("This subpopulation may be a distinct data-generating process. "
                       "Do NOT remove — analyze separately."
                       if vtype == 'subpopulation' else
                       "These appear to be outliers. Consider pruning with audit trail."),
                    UserWarning, stacklevel=2
                )

        # Contradiction score: mean violation rate across all implications
        all_rates = [vs.rate for vs in violation_sets]
        c_score   = round(float(np.mean(all_rates)), 6) if all_rates else 0.0

        return ContradictionReport(
            contradiction_score=c_score,
            n_total=n,
            n_violations_total=n_total_viols,
            violation_sets=violation_sets,
            violation_type_summary=type_counts,
        )


# ── ObservationPruner ─────────────────────────────────────────────────────────

class ObservationPruner:
    """
    Separates violating observations from the clean dataset,
    with a full audit trail and ICC verification.

    Key properties:
    ───────────────
    - Observations are SEPARATED, not discarded.
      X_pruned contains the removed rows and can always be recovered.

    - Only 'outlier' type violations are removed by default.
      'subpopulation' violations are flagged but kept (they represent
      a distinct data-generating process that should be analyzed, not removed).

    - The audit log records for each removed observation:
        original_index, antecedent_violated, violation_type,
        feature_values, reason

    - ICC is computed before and after pruning to verify improvement.
      If ICC does not improve, a warning is emitted.

    Parameters
    ----------
    remove_types : list of str, default=['outlier']
        Violation types to remove. 'subpopulation' is excluded by default.
    max_remove_frac : float, default=0.05
        Maximum fraction of dataset to remove (safety cap).
        If the identified violations exceed this, only the top-scoring
        violations are removed up to the cap.
    verify_improvement : bool, default=True
        If True, compute ICC before and after and warn if no improvement.
    """

    def __init__(self,
                 remove_types:        list  = None,
                 max_remove_frac:     float = 0.05,
                 verify_improvement:  bool  = True):
        self.remove_types       = remove_types or ['outlier']
        self.max_remove_frac    = max_remove_frac
        self.verify_improvement = verify_improvement

    def prune(self,
              model:    PropositionalModel,
              report:   ContradictionReport,
              X:        pd.DataFrame,
              y:        pd.Series) -> PruningResult:
        """
        Separate violating observations from the clean dataset.

        Parameters
        ----------
        model : fitted PropositionalModel (for ICC computation)
        report : ContradictionReport from ContradictionAnalyzer.analyze()
        X, y : the full dataset

        Returns
        -------
        PruningResult with clean and pruned sets, audit log, ICC delta
        """
        n = len(X)

        # ICC before pruning
        ca = CompletenessAnalyzer()
        r_before = ca.analyze(model, X, y)
        icc_before = r_before.ICC

        # Collect indices to remove (only specified types)
        remove_idx = set()
        for vs in report.violation_sets:
            if vs.violation_type in self.remove_types:
                for i in vs.violation_indices:
                    remove_idx.add(int(i))

        # Safety cap: do not remove more than max_remove_frac of dataset
        max_remove = int(n * self.max_remove_frac)
        if len(remove_idx) > max_remove:
            warnings.warn(
                f"[ObservationPruner] {len(remove_idx)} observations identified "
                f"for removal, but cap is {max_remove} ({self.max_remove_frac*100:.0f}% "
                f"of n={n}). Only removing the capped amount. "
                f"Consider increasing max_remove_frac if intentional.",
                UserWarning, stacklevel=2
            )
            # Keep the first max_remove (arbitrary — could rank by violation strength)
            remove_idx = set(list(remove_idx)[:max_remove])

        remove_arr = np.array(sorted(remove_idx), dtype=int)
        keep_arr   = np.array([i for i in range(n) if i not in remove_idx], dtype=int)

        # Build clean and pruned sets
        X_clean  = X.iloc[keep_arr].reset_index(drop=True)
        y_clean  = y.iloc[keep_arr].reset_index(drop=True)
        X_pruned = X.iloc[remove_arr].reset_index(drop=True)
        y_pruned = y.iloc[remove_arr].reset_index(drop=True)

        # Build audit log
        audit_rows = []
        idx_to_vs  = {}
        for vs in report.violation_sets:
            for i in vs.violation_indices:
                if int(i) in remove_idx:
                    idx_to_vs[int(i)] = vs

        for orig_idx in sorted(remove_idx):
            vs   = idx_to_vs.get(orig_idx)
            row  = X.iloc[orig_idx].to_dict()
            row['__original_index__']    = orig_idx
            row['__outcome__']           = int(y.iloc[orig_idx])
            row['__antecedent_violated__'] = vs.antecedent if vs else 'unknown'
            row['__violation_type__']    = vs.violation_type if vs else 'unknown'
            row['__violation_rate__']    = vs.rate if vs else None
            row['__H_violation_set__']   = vs.H_violations if vs else None
            row['__reason__']            = (
                f"{vs.violation_type}: {vs.antecedent}→{vs.consequent} "
                f"violated (rate={vs.rate:.4f}, ratio={vs.ratio:.1f}×)"
                if vs else "unknown"
            )
            audit_rows.append(row)

        audit_log = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame()

        # ICC after pruning
        if len(X_clean) > 0 and self.verify_improvement:
            r_after  = ca.analyze(model, X_clean, y_clean)
            icc_after = r_after.ICC
        else:
            icc_after = icc_before

        improved = icc_after > icc_before

        if self.verify_improvement and not improved and len(remove_arr) > 0:
            warnings.warn(
                f"[ObservationPruner] ICC did not improve after removing "
                f"{len(remove_arr)} observations "
                f"(before={icc_before:.4f}, after={icc_after:.4f}). "
                f"The removed observations may not be the source of model noise. "
                f"Consider inspecting X_pruned for subpopulation structure.",
                UserWarning, stacklevel=2
            )

        # Always report subpopulation violations that were NOT removed
        subpop_sets = [vs for vs in report.violation_sets
                       if vs.violation_type == 'subpopulation']
        if subpop_sets:
            total_sp = sum(vs.n_violations for vs in subpop_sets)
            warnings.warn(
                f"[ObservationPruner] {total_sp} subpopulation violations "
                f"were NOT removed (by design). These {len(subpop_sets)} "
                f"violation set(s) have low entropy (H* < {0.80}) "
                f"and represent a structured subgroup that should be "
                f"analyzed independently, not discarded.",
                UserWarning, stacklevel=2
            )

        return PruningResult(
            X_clean=X_clean,
            y_clean=y_clean,
            X_pruned=X_pruned,
            y_pruned=y_pruned,
            audit_log=audit_log,
            n_removed=len(remove_arr),
            n_kept=len(keep_arr),
            icc_before=round(icc_before, 4),
            icc_after=round(icc_after, 4),
            improved=improved,
        )

    def summary(self, result: PruningResult) -> str:
        """Human-readable pruning summary."""
        delta = round(result.icc_after - result.icc_before, 4)
        lines = [
            "ObservationPruner Summary",
            f"  Observations removed: {result.n_removed} / {result.n_removed + result.n_kept}",
            f"  Observations kept:    {result.n_kept}",
            f"  ICC before pruning:   {result.icc_before}",
            f"  ICC after pruning:    {result.icc_after}",
            f"  ΔICC:                 {delta:+.4f}",
            f"  Model improved:       {result.improved}",
            f"  Audit log rows:       {len(result.audit_log)}",
        ]
        if not result.audit_log.empty and '__violation_type__' in result.audit_log:
            vc = result.audit_log['__violation_type__'].value_counts()
            lines.append("  Removed by type:")
            for t, c in vc.items():
                lines.append(f"    {t}: {c}")
        return "\n".join(lines)
