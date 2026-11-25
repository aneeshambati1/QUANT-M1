"""Total policy: every state has an action, including calm.

This is the executable form of the design doctrine. It is NOT one generalized
strategy. It is a priority-ordered total function from MarketState to Action.
First matching rule wins, so the mapping is deterministic and auditable.

Safety rules precede opportunity rules. A state that is both divergent and
in drawdown resolves to divergence, because unknown broker state dominates.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .state import MarketState, Regime


class Action(Enum):
    LATCH_FREEZE = "latch_freeze"
    FREEZE_NEW_RISK = "freeze_new_risk"
    HALT_LIQUIDITY = "halt_liquidity"
    DERISK_LADDER = "derisk_ladder"
    REDUCE_ALPHA = "reduce_alpha"
    NO_TRADE_BAND = "no_trade_band"
    ALLOCATE_MARGINAL = "allocate_marginal"
    HOLD_CORE_CASH = "hold_core_cash"


# Rule identifiers in evaluation order. Exported so a test can assert totality
# and priority without duplicating the logic.
RULE_ORDER = (
    "R1_EXECUTION_DIVERGENT",
    "R2_DATA_UNKNOWN",
    "R3_DATA_STALE",
    "R4_LIQUIDITY_SHOCK",
    "R5_DRAWDOWN_TRIGGER",
    "R6_REGIME_CRISIS",
    "R7_HIGH_DISAGREEMENT",
    "R8_COST_EXCEEDS_EDGE",
    "R9_REGIME_UNKNOWN",
    "R10_NORMAL_ALLOCATE",
)


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    rule_id: str
    reason: str
    permits_new_risk: bool
    permits_reduce: bool

    def as_dict(self) -> dict:
        return {
            "action": self.action.value,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "permits_new_risk": self.permits_new_risk,
            "permits_reduce": self.permits_reduce,
        }


def decide(state: MarketState) -> PolicyDecision:
    """Total, deterministic, priority-ordered. Exactly one action per state."""

    if state.execution_divergent:
        return PolicyDecision(
            Action.LATCH_FREEZE, "R1_EXECUTION_DIVERGENT",
            "Broker state unknown. No new risk, no reduction until operator clear.",
            False, False,
        )

    if state.data_unknown:
        return PolicyDecision(
            Action.FREEZE_NEW_RISK, "R2_DATA_UNKNOWN",
            "Core observation missing. Hold or reduce only; never guess.",
            False, True,
        )

    if state.data_stale:
        return PolicyDecision(
            Action.FREEZE_NEW_RISK, "R3_DATA_STALE",
            f"Data age {state.data_age_secs}s exceeds tau {state.tau_data_secs}s.",
            False, True,
        )

    if state.liquidity_shock:
        return PolicyDecision(
            Action.HALT_LIQUIDITY, "R4_LIQUIDITY_SHOCK",
            f"Liquidity {state.liquidity_score:.3f} below shock threshold. "
            "Halt new orders; reduce only if executable.",
            False, True,
        )

    if state.drawdown_breached:
        return PolicyDecision(
            Action.DERISK_LADDER, "R5_DRAWDOWN_TRIGGER",
            f"Drawdown {state.drawdown_bps}bp >= trigger "
            f"{state.drawdown_trigger_bps}bp. Ladder, not liquidation.",
            False, True,
        )

    if state.regime is Regime.CRISIS:
        return PolicyDecision(
            Action.REDUCE_ALPHA, "R6_REGIME_CRISIS",
            f"Correlation {state.avg_corr:.3f}: diversification has failed. "
            "Reduce alpha book, keep core within limits.",
            False, True,
        )

    if state.high_disagreement:
        return PolicyDecision(
            Action.REDUCE_ALPHA, "R7_HIGH_DISAGREEMENT",
            f"Model disagreement {state.model_disagreement:.3f}. "
            "Ambiguity is risk; reduce alpha exposure.",
            False, True,
        )

    if state.cost_exceeds_edge:
        return PolicyDecision(
            Action.NO_TRADE_BAND, "R8_COST_EXCEEDS_EDGE",
            "Expected edge does not clear measured cost. Stay inside band.",
            False, False,
        )

    if state.regime is Regime.UNKNOWN:
        return PolicyDecision(
            Action.FREEZE_NEW_RISK, "R9_REGIME_UNKNOWN",
            "Regime unclassifiable. Freeze new risk.",
            False, True,
        )

    return PolicyDecision(
        Action.ALLOCATE_MARGINAL, "R10_NORMAL_ALLOCATE",
        "Normal regime, data fresh, cost clears. Allocate by marginal "
        "after-cost contribution.",
        True, True,
    )


def is_no_trade(action: Action) -> bool:
    """No-trade is an active decision, not an absence of decision."""
    return action in {
        Action.LATCH_FREEZE,
        Action.FREEZE_NEW_RISK,
        Action.HALT_LIQUIDITY,
        Action.NO_TRADE_BAND,
        Action.HOLD_CORE_CASH,
    }