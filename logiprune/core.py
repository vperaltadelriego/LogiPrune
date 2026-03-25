"""
logiprune.core
──────────────
The main LogiPrune estimator. Follows the scikit-learn API:
fit / transform / fit_transform, plus pruned_grid() and report().
"""

import numpy as np
import pandas as pd
import time
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from .discretize import AdaptiveDiscretizer
from .sweeper    import SWTSSweeper
from .fuzzy      import FuzzyDisjunctionAnalyzer, TCONORMS
from .relations  import (
    A_OR_B, CONTINGENCY, A_IMPLIES_B, B_IMPLIES_A,
    BICONDITIONAL, INCOMPATIBLE, classify_pair,
)

# ── Relation pruning weights ────────────────────────────────────────────────
# Higher weight = sweeper prefers this relation over lower-weight ones.
# A∨B has low weight because it requires the fuzzy layer, not direct pruning.
PRUNING_WEIGHT = {
    BICONDITIONAL: 1.00,
    A_IMPLIES_B:   0.90,
    B_IMPLIES_A:   0.90,
    INCOMPATIBLE:  0.65,
    A_OR_B:        0.20,
    CONTINGENCY:   0.00,
}

IMPL_RELS = {A_IMPLIES_B, B_IMPLIES_A, BICONDITIONAL}

# Empirical constant: seconds per row per GridSearch configuration
def _benchmark_cost_per_row(n_probe: int = 500) -> float:
    """
    v0.2.3: Dynamic micro-benchmark to replace the hardcoded 1e-5 constant.
    (Gemini review — constant was calibrated on a specific machine and
    diverges on laptops, servers, and cloud VMs.)

    Trains a tiny LogisticRegression on n_probe random rows and returns
    the measured seconds-per-row-per-config. Falls back to 1e-5 on failure.

    Cost: runs once at import time, ~5-20ms.
    """
    try:
        import time
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        X_p = rng.standard_normal((n_probe, 5))
        y_p = (X_p[:, 0] > 0).astype(int)
        t0 = time.perf_counter()
        LogisticRegression(max_iter=50, random_state=0).fit(X_p, y_p)
        elapsed = time.perf_counter() - t0
        # cost per row per config ≈ elapsed / n_probe, with a floor
        return max(elapsed / n_probe, 1e-8)
    except Exception:
        return 1e-5   # safe fallback

_COST_PER_ROW_PER_CONFIG: float = _benchmark_cost_per_row()


class LogiPrune:
    """
    Propositional structure recovery for hyperparameter search space pruning.

    Parameters
    ----------
    base_grid : dict
        The full hyperparameter grid you would pass to GridSearchCV.
    min_confidence : float, default=0.75
        Minimum confidence to consider a propositional relation structural.
    acc_drop_tolerance : float, default=0.04
        Maximum accuracy drop allowed when eliminating a feature.
        Measured by cross-validation with a LogisticRegression proxy.
    theta_disj_gate : float, default=0.85
        Both A⊢D and B⊢D must achieve this confidence for a disjunctive
        pair to be compressed (propositional disjunction gate, rule ∨E).
    theta_elevation : float, default=0.92
        If the best synthetic feature F achieves this confidence with D
        and a non-contingency relation, the pair is "elevated": fully
        replaced by F and the grid restricted to linear kernels.
    min_residual_frac : float, default=0.05
        Stop iterating when the uncovered residual is smaller than this
        fraction of the total dataset.
    max_layers : int, default=8
        Hard cap on the number of propositional layers extracted.
    coverage_threshold : float, default=0.60
        A layer must cover at least this fraction of its residual.
    discretizer_strategy : str, default='percentile'
        Normalization strategy. Options: 'percentile', 'minmax', 'zscore_clip'.
    cv : int, default=3
        Cross-validation folds for feature elimination validation.
    verbose : bool, default=False
        Print progress to stdout.

    Attributes
    ----------
    layers_ : list of dict
        Propositional layers found, each with relation, columns, confidence.
    elevated_pairs_ : list of tuple
        Feature pairs compressed to implication-grade synthetic features.
    compressed_pairs_ : list of tuple
        Feature pairs compressed (gate passed, but not elevated).
    blocked_pairs_ : list of tuple
        Feature pairs where the propositional gate blocked compression.
    eliminated_features_ : list of str
        Features eliminated via validated biconditional rule.
    synthetic_features_ : dict
        Map from synthetic feature name to (tconorm_name, col_a, col_b).
    pruned_grid_ : dict
        The pruned hyperparameter grid, ready for GridSearchCV.
    n_configs_base_ : int
        Number of configurations in the original grid.
    n_configs_pruned_ : int
        Number of configurations in the pruned grid.
    preprocessing_time_ : float
        Seconds spent in LogiPrune preprocessing.
    """

    def __init__(
        self,
        base_grid: dict,
        min_confidence: float = 0.75,
        acc_drop_tolerance: float = 0.04,
        theta_disj_gate: float = 0.85,
        theta_elevation: float = 0.92,
        min_residual_frac: float = 0.05,
        max_layers: int = 8,
        coverage_threshold: float = 0.60,
        discretizer_strategy: str = 'percentile',
        cv: int = 3,
        verbose: bool = False,
    ):
        self.base_grid            = base_grid
        self.min_confidence       = min_confidence
        self.acc_drop_tolerance   = acc_drop_tolerance
        self.theta_disj_gate      = theta_disj_gate
        self.theta_elevation      = theta_elevation
        self.min_residual_frac    = min_residual_frac
        self.max_layers           = max_layers
        self.coverage_threshold   = coverage_threshold
        self.discretizer_strategy = discretizer_strategy
        self.cv                   = cv
        self.verbose              = verbose

        # Fitted attributes (set in fit)
        self.layers_             = []
        self.elevated_pairs_     = []
        self.compressed_pairs_   = []
        self.blocked_pairs_      = []
        self.eliminated_features_ = []
        self.synthetic_features_ = {}
        self.pruned_grid_        = {}
        self.n_configs_base_     = 0
        self.n_configs_pruned_   = 0
        self.preprocessing_time_ = 0.0

        self._discretizer: Optional[AdaptiveDiscretizer] = None
        self._removed: set = set()
        self._fitted: bool = False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _count_configs(self, grid: dict) -> int:
        t = 1
        for v in grid.values():
            t *= len(v)
        return t

    def _layer_score(self, result) -> float:
        """Weighted SWTS score for layer selection."""
        th      = result.threshold
        penalty = 4.0 * th * (1.0 - th)
        weight  = PRUNING_WEIGHT.get(result.relation, 0.0)
        return result.confidence * result.support * result.stability * penalty * weight

    def _marginal_continue(self, n_res: int, n_cols: int, iteration: int) -> bool:
        if iteration == 0:
            return True
        n_steps  = 11
        t_cost   = n_res * (n_cols ** 2) * n_steps * 2e-8
        n_base   = self._count_configs(self.base_grid)
        t_saving = n_base * 0.5 * n_res * _COST_PER_ROW_PER_CONFIG
        return t_cost <= t_saving

    def _swts_conf_with_target(self, col: np.ndarray, y_norm: np.ndarray):
        """Run SWTS for a single column against the normalized target."""
        df = pd.DataFrame({'col': col, 'D': y_norm})
        sw = SWTSSweeper(min_confidence=0.0)
        sw.fit(df, col_pairs=[('col', 'D')])
        r = sw.results_.get(('col', 'D'))
        return (r.confidence, r.relation) if r else (0.0, CONTINGENCY)

    def _build_f_candidates(self, a: np.ndarray, b: np.ndarray) -> dict:
        na, nb = 1.0 - a, 1.0 - b
        return {
            'max(A,B)':    np.maximum(a, b),
            'max(¬A,B)':   np.maximum(na, b),
            'max(A,¬B)':   np.maximum(a, nb),
            'mean(A,B)':   (a + b) / 2,
            'mean(¬A,B)':  (na + b) / 2,
            'mean(A,¬B)':  (a + nb) / 2,
            'prob(A,B)':   a + b - a * b,
        }

    def _best_f_candidate(self, a, b, y_norm):
        """Find the F-candidate with highest propositional confidence with D."""
        best = {'name': None, 'conf': -1.0, 'rel': CONTINGENCY,
                'F': None, 'elevated': False}
        for fname, F in self._build_f_candidates(a, b).items():
            df = pd.DataFrame({'F': F, 'D': y_norm})
            sw = SWTSSweeper(min_confidence=0.0)
            sw.fit(df, col_pairs=[('F', 'D')])
            r = sw.results_.get(('F', 'D'))
            if r and r.confidence > best['conf']:
                best = {
                    'name': fname, 'conf': r.confidence, 'rel': r.relation,
                    'F': F,
                    'elevated': (r.relation in IMPL_RELS
                                 and r.confidence >= self.theta_elevation),
                }
        return best

    def _propositional_gate(self, a, b, y_norm, bf) -> tuple:
        """
        Propositional disjunction gate (rule ∨E).
        Returns (passes: bool, reason: str).
        """
        if bf['elevated']:
            return True, 'elevated'

        conf_a, _ = self._swts_conf_with_target(a, y_norm)
        conf_b, _ = self._swts_conf_with_target(b, y_norm)

        if conf_a >= self.theta_disj_gate and conf_b >= self.theta_disj_gate:
            return True, f'both_imply_D(A={conf_a:.3f},B={conf_b:.3f})'

        return False, f'blocked(A={conf_a:.3f},B={conf_b:.3f})'

    def _validate_elimination(self, X_norm: pd.DataFrame,
                               col_remove: str, y: pd.Series) -> bool:
        """Accept feature elimination only if accuracy drop ≤ acc_drop_tolerance."""
        cols_with    = X_norm.columns.tolist()
        cols_without = [c for c in cols_with if c != col_remove]
        if not cols_without:
            return False
        cv_obj = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        pipe   = Pipeline([('sc', StandardScaler()),
                           ('lr', LogisticRegression(max_iter=300, random_state=0))])
        a_with    = cross_val_score(pipe, X_norm[cols_with],    y, cv=cv_obj).mean()
        a_without = cross_val_score(pipe, X_norm[cols_without], y, cv=cv_obj).mean()
        return (a_with - a_without) <= self.acc_drop_tolerance

    def _grid_restriction(self, relation: str, confidence: float) -> dict:
        """Translate a propositional relation into grid constraints."""
        g = dict(self.base_grid)
        if relation == BICONDITIONAL or (relation in IMPL_RELS and confidence >= 0.95):
            g = {k: ([v[0]] if 'kernel' in k else
                     (['scale'] if 'gamma' in k else
                      [v[0], v[1]] if 'C' in k else v))
                 for k, v in g.items()}
            g['svc__kernel'] = ['linear']
            g['svc__gamma']  = ['scale']
            g['svc__C']      = [0.1, 1, 10]
        elif relation in IMPL_RELS:
            g['svc__kernel'] = ['linear', 'poly']
        return g

    def _merge_grids(self, grids: list) -> dict:
        if not grids:
            return dict(self.base_grid)
        result = dict(grids[0])
        for g in grids[1:]:
            for k, v in g.items():
                if k in result:
                    merged = [x for x in v if x in result[k]]
                    if merged:
                        result[k] = merged
        return result

    def _log(self, msg: str):
        if self.verbose:
            print(f"[LogiPrune] {msg}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LogiPrune':
        """
        Analyze X, y and build the Logic Map.

        Parameters
        ----------
        X : pd.DataFrame
            Training features (numeric, continuous).
        y : pd.Series
            Training target (classification).

        Returns
        -------
        self
        """
        t0 = time.time()
        self.n_configs_base_ = self._count_configs(self.base_grid)

        # Step 1 — Adaptive discretization
        self._discretizer = AdaptiveDiscretizer(strategy=self.discretizer_strategy)
        X_norm = self._discretizer.fit_transform(X)

        ya     = y.values.astype(float)
        y_norm = (ya - ya.min()) / (ya.max() - ya.min() + 1e-9)

        X_work      = X.copy()
        X_norm_work = X_norm.copy()
        residual    = np.arange(len(X))
        all_grids   = []
        self._removed            = set()
        self.layers_             = []
        self.elevated_pairs_     = []
        self.compressed_pairs_   = []
        self.blocked_pairs_      = []
        self.eliminated_features_ = []
        self.synthetic_features_ = {}

        mi_scores = dict(zip(X.columns,
                             mutual_info_classif(X_norm, y, random_state=0)))
        cv_obj = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        for iteration in range(self.max_layers):
            n_res = len(residual)
            if n_res / len(X) < self.min_residual_frac:
                self._log(f"Iter {iteration}: residual {n_res/len(X):.1%} < min — stopping.")
                break
            if not self._marginal_continue(n_res, X_norm_work.shape[1], iteration):
                self._log(f"Iter {iteration}: marginal criterion — SWTS cost > expected saving — stopping.")
                break

            X_res  = X_norm_work.iloc[residual].reset_index(drop=True)
            y_res  = y.iloc[residual].values.astype(float)
            yn_res = (y_res - y_res.min()) / (y_res.max() - y_res.min() + 1e-9)

            # Run SWTS
            sweeper = SWTSSweeper(min_confidence=self.min_confidence)
            sweeper.fit(X_res)

            candidates = [
                ((ca, cb), r, self._layer_score(r))
                for (ca, cb), r in sweeper.results_.items()
                if r.confidence >= self.min_confidence
            ]
            if not candidates:
                self._log(f"Iter {iteration}: no structural pairs found — stopping.")
                break
            candidates.sort(key=lambda x: x[2], reverse=True)

            # ── Process A∨B pairs ──────────────────────────────────────────
            for (ca, cb), r, _ in candidates:
                if r.relation != A_OR_B:
                    continue
                if (ca in self._removed or cb in self._removed
                        or ca not in X_res.columns
                        or cb not in X_res.columns):
                    continue

                av = X_res[ca].values
                bv = X_res[cb].values
                bf = self._best_f_candidate(av, bv, yn_res)
                passes, reason = self._propositional_gate(av, bv, yn_res, bf)

                if not passes:
                    self.blocked_pairs_.append((ca, cb, reason))
                    self._log(f"  A∨B [{ca}∨{cb}] BLOCKED: {reason}")
                    continue

                sn = f"LP_{ca[:6]}_{cb[:6]}"
                # Add synthetic to working DataFrames
                X_norm_work[sn] = 0.0
                X_norm_work.loc[X_norm_work.index[residual], sn] = bf['F']
                X_norm_work = X_norm_work.drop(columns=[ca, cb], errors='ignore')
                X_work = X_work.drop(columns=[ca, cb], errors='ignore')
                X_work[sn] = bf['F']
                self._removed.add(ca)
                self._removed.add(cb)
                self.synthetic_features_[sn] = (bf['name'], ca, cb)

                if bf['elevated']:
                    self.elevated_pairs_.append((ca, cb, bf['name'], bf['conf']))
                    all_grids.append(self._grid_restriction(bf['rel'], bf['conf']))
                    self._log(f"  A∨B [{ca}∨{cb}] ELEVATED via {bf['name']} conf={bf['conf']:.3f}")
                else:
                    self.compressed_pairs_.append((ca, cb, bf['name'], bf['conf']))
                    self._log(f"  A∨B [{ca}∨{cb}] compressed via {bf['name']} conf={bf['conf']:.3f}")

            # ── Best non-A∨B structural layer ──────────────────────────────
            best_ca = best_cb = best_r = None
            for (ca, cb), r, _ in candidates:
                if (r.relation != A_OR_B
                        and ca not in self._removed
                        and cb not in self._removed
                        and ca in X_norm_work.columns
                        and cb in X_norm_work.columns):
                    best_ca, best_cb, best_r = ca, cb, r
                    break
            if best_r is None:
                self._log(f"Iter {iteration}: no non-A∨B layer available — stopping.")
                break

            # Biconditional: validate and possibly eliminate
            if best_r.relation == BICONDITIONAL:
                mi_ca = mi_scores.get(best_ca, 0)
                mi_cb = mi_scores.get(best_cb, 0)
                col_keep   = best_ca if mi_ca >= mi_cb else best_cb
                col_remove = best_cb if col_keep == best_ca else best_ca
                if (col_remove not in self._removed
                        and mi_scores.get(col_keep, 0) >= 0.05
                        and self._validate_elimination(X_norm_work, col_remove, y)):
                    X_norm_work = X_norm_work.drop(columns=[col_remove], errors='ignore')
                    X_work      = X_work.drop(columns=[col_remove], errors='ignore')
                    self._removed.add(col_remove)
                    self.eliminated_features_.append(col_remove)
                    self._log(f"  BICONDITIONAL [{best_ca}↔{best_cb}] → eliminated [{col_remove}]")

            all_grids.append(self._grid_restriction(best_r.relation, best_r.confidence))
            self.layers_.append({
                'iteration': iteration,
                'relation':  best_r.relation,
                'col_a':     best_ca,
                'col_b':     best_cb,
                'confidence': round(best_r.confidence, 4),
                'threshold':  round(best_r.threshold, 3),
            })
            self._log(f"Iter {iteration}: layer [{best_ca} {best_r.relation} {best_cb}] conf={best_r.confidence:.3f}")

            # Update residual
            X_res_upd = X_norm_work.iloc[residual].reset_index(drop=True)
            if best_ca not in X_res_upd.columns or best_cb not in X_res_upd.columns:
                break
            th = best_r.threshold
            a  = (X_res_upd[best_ca].values > th).astype(int)
            b  = (X_res_upd[best_cb].values > th).astype(int)

            if best_r.relation == BICONDITIONAL:
                covered = a == b
            elif best_r.relation == A_IMPLIES_B:
                covered = ~((a == 1) & (b == 0))
            elif best_r.relation == B_IMPLIES_A:
                covered = ~((a == 0) & (b == 1))
            elif best_r.relation == INCOMPATIBLE:
                covered = ~((a == 1) & (b == 1))
            else:
                covered = np.ones(len(residual), dtype=bool)

            if covered.mean() < self.coverage_threshold:
                self._log(f"  Coverage {covered.mean():.1%} < {self.coverage_threshold:.1%} — stopping.")
                break
            residual = residual[~covered]

        # Finalize
        self.pruned_grid_      = self._merge_grids(all_grids) if all_grids else dict(self.base_grid)
        self.n_configs_pruned_ = self._count_configs(self.pruned_grid_)
        self._X_work           = X_work
        self._X_norm_work      = X_norm_work
        self.preprocessing_time_ = round(time.time() - t0, 3)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the learned feature transformations to new data.

        Parameters
        ----------
        X : pd.DataFrame  (must have same columns as training X)

        Returns
        -------
        pd.DataFrame  with synthetic features added and redundant features removed.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X_out = X.copy()

        # Apply synthetic features
        for sn, (fname, ca, cb) in self.synthetic_features_.items():
            if ca in X_out.columns and cb in X_out.columns:
                a  = X_out[ca].values.astype(float)
                b  = X_out[cb].values.astype(float)
                na, nb = 1.0 - a, 1.0 - b
                candidates = {
                    'max(A,B)':    np.maximum(a, b),
                    'max(¬A,B)':   np.maximum(na, b),
                    'max(A,¬B)':   np.maximum(a, nb),
                    'mean(A,B)':   (a + b) / 2,
                    'mean(¬A,B)':  (na + b) / 2,
                    'mean(A,¬B)':  (a + nb) / 2,
                    'prob(A,B)':   a + b - a * b,
                }
                X_out[sn] = candidates.get(fname, np.maximum(a, b))
                X_out = X_out.drop(columns=[ca, cb], errors='ignore')

        # Remove eliminated features
        for col in self.eliminated_features_:
            if col in X_out.columns:
                X_out = X_out.drop(columns=[col])

        return X_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def pruned_grid(self) -> dict:
        """Return the pruned hyperparameter grid for GridSearchCV."""
        if not self._fitted:
            raise RuntimeError("Call fit() before pruned_grid().")
        return self.pruned_grid_

    def savings_summary(self) -> dict:
        """Return a dict summarising the achieved pruning."""
        if not self._fitted:
            raise RuntimeError("Call fit() before savings_summary().")
        cfg_sav = (self.n_configs_base_ - self.n_configs_pruned_) / self.n_configs_base_
        return {
            'configs_base':      self.n_configs_base_,
            'configs_pruned':    self.n_configs_pruned_,
            'config_savings_pct': round(cfg_sav * 100, 1),
            'features_eliminated': len(self.eliminated_features_),
            'pairs_elevated':    len(self.elevated_pairs_),
            'pairs_compressed':  len(self.compressed_pairs_),
            'pairs_blocked':     len(self.blocked_pairs_),
            'preprocessing_time_s': self.preprocessing_time_,
            'layers_found':      len(self.layers_),
        }

    def report(self) -> str:
        """Print a human-readable summary of what LogiPrune found."""
        if not self._fitted:
            raise RuntimeError("Call fit() before report().")
        s = self.savings_summary()
        lines = [
            "\n══════════════════════════════════════════════════",
            "  LogiPrune Report",
            "══════════════════════════════════════════════════",
            f"  Preprocessing time:   {s['preprocessing_time_s']:.2f}s",
            f"  Configs (base):       {s['configs_base']}",
            f"  Configs (pruned):     {s['configs_pruned']}",
            f"  Config savings:       {s['config_savings_pct']}%",
            "",
            f"  Features eliminated:  {s['features_eliminated']}  {self.eliminated_features_}",
            f"  Pairs elevated:       {s['pairs_elevated']}",
            f"  Pairs compressed:     {s['pairs_compressed']}",
            f"  Pairs blocked (gate): {s['pairs_blocked']}",
            "",
            "  Propositional layers found:",
        ]
        for l in self.layers_:
            lines.append(
                f"    [{l['col_a']} {l['relation']} {l['col_b']}] "
                f"conf={l['confidence']}  th={l['threshold']}"
            )
        lines.append("══════════════════════════════════════════════════")
        return "\n".join(lines)
