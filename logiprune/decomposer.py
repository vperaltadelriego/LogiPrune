"""
logiprune.decomposer
─────────────────────
Metric-Aware Subpopulation Decomposition.

Connects the contradiction/completeness analysis (v0.2.4) with the
metric-aware grid biasing of LogiPrune (Paper 1).

The core insight:
─────────────────
When optimizing for a specific metric (Recall, Precision, F1), the
hyperparameter grid alone cannot compensate for structural gaps in
the causal model. If a subpopulation exists where the cause of B is
different from the known causes in M, no amount of hyperparameter
tuning will capture those cases — the ceiling is causal, not parametric.

MetricAwareDecomposer makes that ceiling explicit and, when possible,
provides a path to raise it:

  1. Computes the metric ceiling implied by the causal structure:
       recall_ceiling    = causal_recall    from ICC
       precision_ceiling = causal_precision from ICC
       f1_ceiling        = harmonic mean of the two

  2. Detects whether a subpopulation exists that limits the ceiling.

  3. If a subpopulation exists AND the AbductiveProposer finds a
     candidate cause ¬K, proposes one of two strategies:
       Strategy A (ensemble): train a second model on the subpopulation
                              with the candidate feature included
       Strategy B (augment):  add the candidate feature to the main grid

  4. Estimates the expected metric gain from each strategy.

This is zero-cost when no subpopulation is detected — the analysis
runs only if ContradictionAnalyzer finds structured violations.
All computation is O(n × C²) on the residue — same as SWTS.

Integration with LogiPrune (Paper 1):
──────────────────────────────────────
The `contradiction_adjusted_confidence` method provides a drop-in
replacement for raw implication confidence in SWTS scoring:

    conf_adjusted = conf_raw × (1 - subpopulation_weight)

An implication with conf=0.92 but 15% subpopulation violations gets
conf_adjusted = 0.78, below the restriction threshold. This prevents
LogiPrune from incorrectly restricting the kernel for an implication
that is structurally unreliable for part of the population.
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .completeness import (PropositionalModel, CompletenessAnalyzer,
                            CompletenessReport, AbductiveProposer)
from .contradiction import ContradictionAnalyzer, ContradictionReport


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MetricCeiling:
    """
    The theoretical maximum for a metric given the causal structure of M.

    This ceiling cannot be exceeded by hyperparameter tuning alone.
    It can only be raised by adding new features (causes) to the model.
    """
    recall_ceiling:    float   # = causal_recall from ICC
    precision_ceiling: float   # = causal_precision from ICC
    f1_ceiling:        float   # harmonic mean of the two
    icc:               float

    # Whether the ceiling is binding for the requested metric
    is_binding:        bool    # True if ceiling < achievable_with_tuning
    limiting_factor:   str     # 'coverage' | 'precision' | 'none'

    def __repr__(self):
        return (f"MetricCeiling(Recall≤{self.recall_ceiling:.3f} "
                f"Prec≤{self.precision_ceiling:.3f} "
                f"F1≤{self.f1_ceiling:.3f} "
                f"binding={self.is_binding})")


@dataclass
class SubpopulationStrategy:
    """
    A proposed strategy for handling a detected subpopulation.
    """
    strategy:           str    # 'ensemble' | 'augment' | 'report_only'
    subpopulation_size: int
    candidate_feature:  Optional[str]
    candidate_relation: Optional[str]
    expected_delta:     float  # estimated metric improvement
    cost_overhead:      str    # 'none' | 'low' | 'medium'
    description:        str

    def __repr__(self):
        return (f"Strategy({self.strategy}: "
                f"n_subpop={self.subpopulation_size} "
                f"feature={self.candidate_feature} "
                f"Δ≈{self.expected_delta:+.3f})")


@dataclass
class DecompositionReport:
    """
    Full metric-aware decomposition report.
    """
    metric:              str    # 'recall' | 'precision' | 'f1'
    ceiling:             MetricCeiling
    has_subpopulation:   bool
    subpopulation_sizes: dict   # antecedent → n_violations (subpop type)
    strategies:          list   # list of SubpopulationStrategy
    adjusted_confidences: dict  # antecedent → adjusted_conf
    n_obs_total:         int
    n_obs_subpop:        int

    def summary(self) -> str:
        lines = [
            f"DecompositionReport (optimize_for='{self.metric}')",
            f"  Metric ceilings: Recall≤{self.ceiling.recall_ceiling:.3f}  "
            f"Prec≤{self.ceiling.precision_ceiling:.3f}  "
            f"F1≤{self.ceiling.f1_ceiling:.3f}",
            f"  Subpopulation detected: {self.has_subpopulation}  "
            f"(n={self.n_obs_subpop}/{self.n_obs_total})",
        ]
        if self.adjusted_confidences:
            lines.append("  Confidence adjustments:")
            for ante, (raw, adj) in self.adjusted_confidences.items():
                lines.append(f"    {ante}: {raw:.4f} → {adj:.4f}")
        if self.strategies:
            lines.append("  Recommended strategies:")
            for s in self.strategies:
                lines.append(f"    [{s.cost_overhead} cost] {s.description}")
        return "\n".join(lines)


# ── MetricAwareDecomposer ─────────────────────────────────────────────────────

class MetricAwareDecomposer:
    """
    Connects subpopulation analysis with metric-aware HPO decisions.

    This is the integration layer between v0.2.4 (contradiction/completeness)
    and Paper 1 (metric-aware grid biasing).

    Parameters
    ----------
    metric : str
        Target metric: 'recall', 'precision', or 'f1'.
    min_subpop_frac : float, default=0.02
        Minimum fraction of training data to qualify as a subpopulation.
        Below this, violations are treated as noise.
    isr_threshold : float, default=0.60
        ISR below this → residue has structure → subpopulation is structural.
    propose_candidates : bool, default=True
        Whether to run AbductiveProposer on the residue.
    verbose : bool, default=False
    """

    def __init__(self,
                 metric:             str   = 'recall',
                 min_subpop_frac:    float = 0.02,
                 isr_threshold:      float = 0.60,
                 propose_candidates: bool  = True,
                 verbose:            bool  = False):
        self.metric             = metric.lower()
        self.min_subpop_frac    = min_subpop_frac
        self.isr_threshold      = isr_threshold
        self.propose_candidates = propose_candidates
        self.verbose            = verbose

        assert self.metric in ('recall', 'precision', 'f1'), \
            "metric must be 'recall', 'precision', or 'f1'"

    def _log(self, msg):
        if self.verbose:
            print(f"[Decomposer] {msg}")

    def analyze(self,
                model:     PropositionalModel,
                X:         pd.DataFrame,
                y:         pd.Series,
                ) -> DecompositionReport:
        """
        Run the full metric-aware decomposition.

        Parameters
        ----------
        model : fitted PropositionalModel
        X : training features (normalized or raw)
        y : training labels

        Returns
        -------
        DecompositionReport with ceilings, strategies, adjusted confidences
        """
        n = len(X)

        # Step 1: Completeness → metric ceilings
        ca = CompletenessAnalyzer()
        comp = ca.analyze(model, X, y)
        ceiling = self._compute_ceiling(comp)
        self._log(f"Metric ceilings: {ceiling}")

        # Step 2: Contradiction → subpopulation detection
        contra = ContradictionAnalyzer(
            ratio_threshold=2.0,
        )
        c_report = contra.analyze(model, X, y)

        # Identify structural subpopulations
        subpop_sets = {
            vs.antecedent: vs
            for vs in c_report.violation_sets
            if vs.violation_type == 'subpopulation'
            and vs.n_violations >= int(n * self.min_subpop_frac)
        }

        has_subpop  = len(subpop_sets) > 0
        n_subpop    = sum(vs.n_violations for vs in subpop_sets.values())
        subpop_sizes = {k: v.n_violations for k, v in subpop_sets.items()}

        self._log(f"Subpopulations: {has_subpop}  n={n_subpop}")

        # Step 3: Adjusted confidences for LogiPrune integration
        adjusted = {}
        for impl in model.implications_:
            ante = impl.antecedent
            if ante in subpop_sets:
                vs  = subpop_sets[ante]
                # Weight = fraction of active cases that are subpopulation
                w   = vs.n_violations / max(vs.n_active, 1)
                adj = round(impl.confidence * (1.0 - w), 4)
                adjusted[ante] = (impl.confidence, adj)
                self._log(f"  {ante}: conf {impl.confidence:.4f} → {adj:.4f} "
                          f"(subpop weight {w:.3f})")
            else:
                adjusted[ante] = (impl.confidence, impl.confidence)

        # Step 4: Strategies
        strategies = []

        if not has_subpop:
            strategies.append(SubpopulationStrategy(
                strategy='report_only',
                subpopulation_size=0,
                candidate_feature=None,
                candidate_relation=None,
                expected_delta=0.0,
                cost_overhead='none',
                description=(
                    f"No subpopulation detected. "
                    f"Metric ceiling ({self.metric.upper()}) "
                    f"is achievable through hyperparameter tuning. "
                    f"Proceed with standard LogiPrune."
                )
            ))
        else:
            # Is the ceiling binding for the requested metric?
            if ceiling.is_binding:
                # Subpopulation is limiting the target metric
                # Try AbductiveProposer on the residue
                candidate = None
                if self.propose_candidates and comp.n_residue >= 10:
                    ap = AbductiveProposer(isr_threshold=0.85,
                                          min_plausibility=0.15)
                    cands = ap.propose(model, comp, X, y)
                    candidate = cands[0] if cands else None

                if candidate:
                    # Strategy A: ensemble
                    # Expected gain: n_subpop / n_total × 0.5
                    # (conservative: assuming 50% recovery with new feature)
                    delta_a = round(n_subpop / n * 0.5, 4)
                    strategies.append(SubpopulationStrategy(
                        strategy='ensemble',
                        subpopulation_size=n_subpop,
                        candidate_feature=candidate.feature,
                        candidate_relation=candidate.relation,
                        expected_delta=delta_a,
                        cost_overhead='medium',
                        description=(
                            f"Train a secondary model on the subpopulation "
                            f"(n={n_subpop}) with '{candidate.feature}' added. "
                            f"Combine with main model via soft voting. "
                            f"Expected Δ{self.metric.upper()}≈+{delta_a:.3f}. "
                            f"Requires '{candidate.feature}' to be available "
                            f"at inference time."
                        )
                    ))

                    # Strategy B: augment main model
                    delta_b = round(n_subpop / n * 0.3, 4)
                    strategies.append(SubpopulationStrategy(
                        strategy='augment',
                        subpopulation_size=n_subpop,
                        candidate_feature=candidate.feature,
                        candidate_relation=candidate.relation,
                        expected_delta=delta_b,
                        cost_overhead='low',
                        description=(
                            f"Add '{candidate.feature}' to the main feature set "
                            f"and re-run LogiPrune. "
                            f"Expected Δ{self.metric.upper()}≈+{delta_b:.3f}. "
                            f"Lower overhead than ensemble, smaller expected gain."
                        )
                    ))
                else:
                    # No candidate feature — subpopulation cause is unobservable
                    strategies.append(SubpopulationStrategy(
                        strategy='report_only',
                        subpopulation_size=n_subpop,
                        candidate_feature=None,
                        candidate_relation=None,
                        expected_delta=0.0,
                        cost_overhead='none',
                        description=(
                            f"Subpopulation of n={n_subpop} detected but no "
                            f"candidate feature found in current dataset. "
                            f"The limiting cause is not observable — "
                            f"{self.metric.upper()} ceiling of "
                            f"{getattr(ceiling, self.metric+'_ceiling', 0):.3f} "
                            f"cannot be raised with current features. "
                            f"Collect additional features or domain knowledge."
                        )
                    ))
            else:
                # Subpopulation exists but does NOT limit the target metric
                strategies.append(SubpopulationStrategy(
                    strategy='report_only',
                    subpopulation_size=n_subpop,
                    candidate_feature=None,
                    candidate_relation=None,
                    expected_delta=0.0,
                    cost_overhead='none',
                    description=(
                        f"Subpopulation of n={n_subpop} detected, "
                        f"but it does not limit {self.metric.upper()} "
                        f"(ceiling={getattr(ceiling, self.metric+'_ceiling', 0):.3f} "
                        f"is above practical threshold). "
                        f"Standard LogiPrune sufficient."
                    )
                ))

        # Emit warnings for binding ceilings
        if ceiling.is_binding and has_subpop:
            warnings.warn(
                f"[MetricAwareDecomposer] {self.metric.upper()} ceiling "
                f"({getattr(ceiling, self.metric+'_ceiling', 0):.3f}) "
                f"is limited by a structural subpopulation (n={n_subpop}). "
                f"Hyperparameter tuning cannot exceed this ceiling. "
                f"See DecompositionReport.strategies for recommendations.",
                UserWarning, stacklevel=2
            )

        return DecompositionReport(
            metric=self.metric,
            ceiling=ceiling,
            has_subpopulation=has_subpop,
            subpopulation_sizes=subpop_sizes,
            strategies=strategies,
            adjusted_confidences=adjusted,
            n_obs_total=n,
            n_obs_subpop=n_subpop,
        )

    def _compute_ceiling(self, comp: CompletenessReport) -> MetricCeiling:
        """Translate ICC components into metric ceilings."""
        rc   = comp.causal_recall      # P(some cause active | B=1)
        pc   = comp.causal_precision   # P(B=1 | some cause active)
        f1c  = (2 * rc * pc / (rc + pc + 1e-9)) if (rc + pc) > 0 else 0.0
        icc  = comp.ICC

        # Determine whether the ceiling is binding
        # (i.e., whether it's below what tuning alone could achieve)
        # Heuristic: if ICC > 0.95, ceiling is not binding in practice
        is_binding = icc < 0.95

        if self.metric == 'recall':
            limiting = 'coverage' if rc < 0.95 else 'none'
        elif self.metric == 'precision':
            limiting = 'precision' if pc < 0.95 else 'none'
        else:
            limiting = 'coverage' if f1c < 0.90 else 'none'

        return MetricCeiling(
            recall_ceiling=round(rc, 4),
            precision_ceiling=round(pc, 4),
            f1_ceiling=round(f1c, 4),
            icc=round(icc, 4),
            is_binding=is_binding,
            limiting_factor=limiting,
        )

    def contradiction_adjusted_confidence(self,
                                           report: DecompositionReport,
                                           antecedent: str) -> float:
        """
        Return the contradiction-adjusted confidence for an implication.

        Drop-in replacement for raw confidence in SWTS scoring:

            raw_conf = 0.92
            adj_conf = decomposer.contradiction_adjusted_confidence(
                           report, antecedent='¬mean_concavity')
            # → 0.78 if 15% of active cases are subpopulation violations

        Use in LogiPrune core to prevent incorrect kernel restrictions
        for implications that are structurally unreliable for a subgroup.
        """
        pair = report.adjusted_confidences.get(antecedent)
        if pair is None:
            return 1.0   # unknown antecedent: no adjustment
        _, adj = pair
        return adj
