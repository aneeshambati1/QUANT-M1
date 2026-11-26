"""Single-step paper-trading orchestration with a shadow-baseline contribution.

A scheduler calls run_paper_step once per exchange session. It never bypasses
the Risk Kernel: every order passes Kernel.authorize + Gateway.submit, and a
divergence latch blocks new risk. PAPER / RESEARCH ONLY.

CONTRIBUTION CONVENTION (pre-registered default; owner may amend):
  Shadow baseline = BUY-AND-HOLD counterfactual (same starting book, no trades,
  no costs). Both books are marked with the SAME new marks.

    starting_nav   = ledger NAV at old marks (before session advance)
    baseline_after = starting positions * new_marks + starting cash-equivalent
    model_after    = ledger NAV at new marks, after trading + fills + settlement
    ensemble_contribution = (model_after - baseline_after) / starting_nav

  The ledger already deducts costs on fills, so model_after is NET of costs.
  This is the incremental after-cost value of trading over holding. Per-model
  leave-one-out attribution is a documented next step; this function does not
  fabricate per-model numbers.

FAIL-CLOSED: no allocation or a policy block => no orders. A divergence latch
returns a blocked report and does not learn from unverified fills.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from ..meta.state import MarketState
from .cycle import run_cycle
from .execution import generate_intents, dispatch_intents


@dataclass
class PaperStepReport:
    trade_date: int
    action: str
    rule_id: str
    permits_new_risk: bool
    intents_generated: int
    intents_submitted: int
    intents_rejected: int
    filled_orders: int
    starting_nav: int
    baseline_after: int
    model_after: int
    ensemble_contribution: float
    blocked: bool
    note: str = ""
    rejection_reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "trade_date": self.trade_date,
            "action": self.action,
            "rule_id": self.rule_id,
            "permits_new_risk": self.permits_new_risk,
            "intents_generated": self.intents_generated,
            "intents_submitted": self.intents_submitted,
            "intents_rejected": self.intents_rejected,
            "filled_orders": self.filled_orders,
            "starting_nav": self.starting_nav,
            "baseline_after": self.baseline_after,
            "model_after": self.model_after,
            "ensemble_contribution": self.ensemble_contribution,
            "blocked": self.blocked,
            "note": self.note,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _cash_equivalent(ledger) -> int:
    return ledger.state.cash_settled + ledger.state.receivable_total()


def run_paper_step(
    sx: Mapping,
    market_state: MarketState,
    trackers: Mapping,
    forecasts: Mapping,
    *,
    new_trade_date: int,
    new_mark_wall: int,
    new_marks: Mapping[str, int],
    allocator: Callable,
    min_trade_bps: int = 10,
    cost_bps: int = 10,
    attribution: Optional[Callable[[float], Mapping[str, float]]] = None,
) -> PaperStepReport:
    """Run one paper session step and return the ensemble contribution.

    ``sx`` is the dict returned by system.build(). ``attribution`` is an
    optional pre-registered callback mapping the ensemble contribution to
    per-model contributions; if omitted, trackers are NOT mutated (no
    fabricated attribution). The caller may then apply_realized explicitly.
    """
    ledger = sx["ledger"]
    gw = sx["gw"]
    venue = sx["venue"]
    instruments = sx["instruments"]

    if gw._divergence_latched:
        return PaperStepReport(
            trade_date=new_trade_date, action="latch_freeze",
            rule_id="R1_EXECUTION_DIVERGENT", permits_new_risk=False,
            intents_generated=0, intents_submitted=0, intents_rejected=0,
            filled_orders=0, starting_nav=ledger.state.nav(),
            baseline_after=ledger.state.nav(), model_after=ledger.state.nav(),
            ensemble_contribution=0.0, blocked=True,
            note="Divergence latched; no new risk, no learning.",
        )

    starting_positions = dict(ledger.state.positions)
    starting_cash = _cash_equivalent(ledger)
    starting_nav = ledger.state.nav()
    if starting_nav <= 0:
        raise ValueError("starting NAV must be positive")

    for sym, qty in starting_positions.items():
        if qty and sym not in new_marks:
            raise ValueError(f"new_marks missing held symbol {sym}")
    if any(not isinstance(v, int) or v <= 0 for v in new_marks.values()):
        raise ValueError("new_marks must be positive integer cents")

    baseline_after = starting_cash + sum(
        qty * new_marks[sym] for sym, qty in starting_positions.items() if qty
    )

    # Single session-advance authority: settles matured receivables, marks peak.
    ledger.advance_session(new_trade_date, new_mark_wall, dict(new_marks))

    rec = run_cycle(market_state, trackers, forecasts=forecasts, allocator=allocator)

    generated = submitted = rejected = filled = 0
    reasons: list = []
    note = rec.note

    if rec.allocation is not None and rec.permits_new_risk:
        intents, skipped = generate_intents(
            rec.allocation, ledger, dict(new_marks), instruments,
            min_trade_bps=min_trade_bps, cost_bps=cost_bps,
        )
        generated = len(intents)
        report = dispatch_intents(
            intents, sx["kernel"], gw, ledger, sx["flags"], sx["clock"],
            watchdog_stop=False, skipped_dust_trades=skipped,
        )
        submitted = report.intents_submitted
        rejected = report.intents_rejected
        reasons = list(report.rejection_reasons)

        # Paper fill model: fully fill every venue-known working order at its
        # limit price, then process through the Gateway (snapshot-authoritative).
        snap = venue.snapshot() or {}
        venue_orders = snap.get("orders", {})
        for o in list(ledger.state.working):
            if o.terminal or o.remaining <= 0:
                continue
            if o.order_id not in venue_orders:
                continue
            ev = venue.fill(o.order_id, o.qty, price=o.price)
            gw.process(ev)
            filled += 1

    model_after = ledger.state.nav()
    contribution = (model_after - baseline_after) / starting_nav

    if gw._divergence_latched:
        return PaperStepReport(
            trade_date=new_trade_date, action=rec.action, rule_id=rec.rule_id,
            permits_new_risk=rec.permits_new_risk, intents_generated=generated,
            intents_submitted=submitted, intents_rejected=rejected,
            filled_orders=filled, starting_nav=starting_nav,
            baseline_after=baseline_after, model_after=model_after,
            ensemble_contribution=0.0, blocked=True,
            note="Divergence latched during step; contribution not learned.",
            rejection_reasons=reasons,
        )

    if attribution is not None:
        from ..meta.contribution import weights as _w  # noqa: F401
        from ..orchestrator.cycle import apply_realized
        apply_realized(trackers, dict(attribution(contribution)))

    return PaperStepReport(
        trade_date=new_trade_date, action=rec.action, rule_id=rec.rule_id,
        permits_new_risk=rec.permits_new_risk, intents_generated=generated,
        intents_submitted=submitted, intents_rejected=rejected,
        filled_orders=filled, starting_nav=starting_nav,
        baseline_after=baseline_after, model_after=model_after,
        ensemble_contribution=contribution, blocked=False,
        note=note, rejection_reasons=reasons,
    )
