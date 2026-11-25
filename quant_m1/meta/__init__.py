"""Meta-Intelligence layer.

Organs 5 (meta-weighting), 6 (conditional dependency), and the seed of 9
(lifecycle). Pure computation: no I/O, no vendor coupling, no order path.

Governing law: no model holds permanent authority. Each must demonstrate
positive incremental after-cost marginal contribution, measured with a
confidence penalty, or be down-weighted and eventually retired.
"""
from .state import MarketState, Regime, classify_regime
from .policy import Action, PolicyDecision, decide, RULE_ORDER
from .contribution import (
    ContributionTracker,
    ModelStatus,
    marginal_contribution,
    weights,
)

__all__ = [
    "MarketState", "Regime", "classify_regime",
    "Action", "PolicyDecision", "decide", "RULE_ORDER",
    "ContributionTracker", "ModelStatus", "marginal_contribution", "weights",
]