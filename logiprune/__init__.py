"""
logiprune
─────────
Propositional Structure Recovery and Truth Table Entropy Analysis
for Hyperparameter Search Space Pruning.

Two complementary modules — one library:

  LogiPrune (Paper 1)
  ───────────────────
  Recovers directional propositional structure between feature pairs
  (implications, biconditionals, incompatibilities, disjunctions) and
  translates that structure into hyperparameter grid restrictions.
  Works with SVC, RandomForest, XGBoost, and extensible to any estimator.

  LogiPruneEntropy (Paper 2)
  ──────────────────────────
  Computes the Shannon entropy of the 4-cell truth table distribution
  for each feature pair and uses it as an a priori model complexity signal.
  Low entropy → shallow trees, few estimators.
  High entropy → full grid required.
  Includes a feedback loop that reinstates features whose elimination
  increases the entropy of the feature-target relationship.

Quick start — Paper 1 (grid pruning)
──────────────────────────────────────
    from logiprune import LogiPrune

    lp = LogiPrune(base_grid=my_param_grid)
    lp.fit(X_train, y_train)

    X_pruned    = lp.transform(X_train)
    pruned_grid = lp.pruned_grid()
    print(lp.report())

Quick start — Paper 2 (entropy complexity)
───────────────────────────────────────────
    from logiprune import LogiPruneEntropy

    lpe = LogiPruneEntropy(base_grid=my_xgb_grid)
    lpe.fit(X_train, y_train)

    X_pruned    = lpe.transform(X_train)
    pruned_grid = lpe.pruned_grid()
    print(lpe.report())

Quick start — Combined pipeline
─────────────────────────────────
    from logiprune import LogiPrune, LogiPruneEntropy

    # Step 1: propositional structure (Paper 1)
    lp = LogiPrune(base_grid=base_grid)
    lp.fit(X_train, y_train)
    X_p1 = lp.transform(X_train)
    grid_p1 = lp.pruned_grid()

    # Step 2: entropy complexity (Paper 2) on already-pruned features
    lpe = LogiPruneEntropy(base_grid=grid_p1)
    lpe.fit(X_p1, y_train)
    X_final    = lpe.transform(X_p1)
    grid_final = lpe.pruned_grid()

    # Step 3: search (FLAML / Optuna / GridSearch) on grid_final
"""

__version__ = "0.2.4"
__author__  = "Víctor Manuel Peralta Del Riego"
__email__   = "vperalta@ucaribe.edu.mx"
__license__ = "MIT"

# ── Paper 1: propositional structure recovery ─────────────────────────────────
from .core    import LogiPrune
from .relations import (
    BICONDITIONAL, A_IMPLIES_B, B_IMPLIES_A,
    INCOMPATIBLE, A_OR_B, CONTINGENCY,
    classify_pair, RelationResult,
)
from .decomposer import (
    MetricAwareDecomposer,
    DecompositionReport,
    MetricCeiling,
    SubpopulationStrategy,
)
from .contradiction import (
    ContradictionAnalyzer,
    ObservationPruner,
    ViolationSet,
    ContradictionReport,
    PruningResult,
)
from .completeness import (
    PropositionalModel,
    CompletenessAnalyzer,
    ViolationTracker,
    AbductiveProposer,
    Implication,
    CompletenessReport,
    ViolationBatch,
    AbductiveCandidate,
)
from .discretize import (
    AdaptiveDiscretizer,
    AsymmetricSweepAnalyzer,
    ModelLifetimeEstimator,
)
from .sweeper    import SWTSSweeper
from .fuzzy      import FuzzyDisjunctionAnalyzer, TCONORMS

# ── Paper 2: truth table entropy ──────────────────────────────────────────────
from .entropy import (
    EntropyAnalyzer,
    TruthTableProfile,
    EntropyProfile,
    ConditionalEntropyResult,
    truth_table_profile,
    conditional_entropy_gate,
)
from .entropy_pipeline import LogiPruneEntropy

__all__ = [
    # Paper 3 — completeness & open-context
    "PropositionalModel",
    "CompletenessAnalyzer",
    "ViolationTracker",
    "AbductiveProposer",
    "Implication",
    "CompletenessReport",
    "ViolationBatch",
    "AbductiveCandidate",
    # Paper 3 — metric-aware decomposition
    "MetricAwareDecomposer",
    "DecompositionReport",
    "MetricCeiling",
    "SubpopulationStrategy",
    # Paper 3 — contradiction & observation pruning
    "ContradictionAnalyzer",
    "ObservationPruner",
    "ViolationSet",
    "ContradictionReport",
    "PruningResult",
    # Paper 1
    "LogiPrune",
    "AdaptiveDiscretizer",
    "AsymmetricSweepAnalyzer",
    "ModelLifetimeEstimator",
    "SWTSSweeper",
    "FuzzyDisjunctionAnalyzer",
    "TCONORMS",
    "classify_pair",
    "RelationResult",
    "BICONDITIONAL", "A_IMPLIES_B", "B_IMPLIES_A",
    "INCOMPATIBLE", "A_OR_B", "CONTINGENCY",
    # Paper 2
    "LogiPruneEntropy",
    "EntropyAnalyzer",
    "TruthTableProfile",
    "EntropyProfile",
    "ConditionalEntropyResult",
    "truth_table_profile",
    "conditional_entropy_gate",
]
