"""Single-step paper-trading orchestration with a shadow-baseline contribution.

A scheduler calls run_paper_step once per exchange session. It never bypasses
the Risk Kernel: every order passes Kernel.authorize + Gateway.submit, and a
divergence latch blocks new risk. PAPER / RESEARCH ONLY.

SESSION TRANSITION (coordinated, crash-safe, idempotent):
  Under the shared lock this step (1) catches the shared Clock up to the new
  mark wall, (2) advances the ledger session (settles matured receivables),
  (3) appends a SESSION_ADVANCE audit event, and (4) re-persists the Gateway's
  signed composite. Because the composite is the authoritative restore source,
  persisting it in the same locked transition prevents a restart from rolling
  the session back. Re-running the same target session is a no-op (idempotent).

CONTRIBUTION CONVENTION (pre-registered default; owner may amend):
  Shadow baseline = BUY-AND-HOLD counterfactual (same starting book, no trades,
  no costs). Both books are marked with the SAME new marks.

    starting_nav   = ledger NAV at old marks (before the session advance)
    baseline_after = starting positions * new_marks + starting cash-equivalent
    model_after    = ledger NAV at new marks, after trading + fills + settlement
    ensemble_contribution = (model_after - baseline_after) / starting_nav

  The ledger already deducts costs on fills, so model_after is NET of costs.
  This is the incremental after-cost value of trading over holding. Per-model
  leave-one-out attribution is a documented next step; this function does not
  fabricate per-model numbers.

FILL MODEL: venue.fill takes an INCREMENTAL quantity, so the step fills each
venue-known working order by its REMAINING quantity and checks the Gateway's
processing outcome. Unconfirmed/blocked outcomes stop the fill loop (fail
closed); only accepted fills are counted.

FAIL-CLOSED: no allocation or a policy block => no orders. A divergence latch
returns a blocked report and does not learn from unverified fills.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from ..meta.state import MarketState
from .cycle import run_cycle, apply_realized
from .execution import generate_intents, dispatch_intents

_ACCEPTED_FILL = frozenset({"APPLIED", "SNAPSHOT_RECONCILED"})


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


def _advance_session(sx, new_trade_date: int, new_mark_wall: int,
                     new_marks: Mapping[str, int]) -> None:
    """Coordinated clock + ledger + signed-composite session transition."""
    ledger, gw, clock = sx["ledger"], sx["gw"], sx["clock"]
    with sx["lock"]:
        # Restart safety: the shared clock may lag the restored mark wall.
        if clock.wall < ledger.state.mark_wall:
            clock.advance_wall(ledger.state.mark_wall - clock.wall)

        if ledger.state.trade_date == new_trade_date:
            if (ledger.state.mark_wall != new_mark_wall
                    or dict(ledger.state.marks) != dict(new_marks)):
                raise ValueError("SESSION_CONFLICT: same date, different marks/wall")
            return  # idempotent re-run of an already-applied session

        if new_trade_date <= ledger.state.trade_date:
            raise ValueError("SESSION_REGRESSION_OR_STASIS")
        if new_mark_wall < ledger.state.mark_wall:
            raise ValueError("MARK_WALL_REGRESSION")

        if clock.wall < new_mark_wall:
            clock.advance_wall(new_mark_wall - clock.wall)
        ledger.advance_session(new_trade_date, new_mark_wall, dict(new_marks))
        gw.log.append("SESSION_ADVANCE",
                      {"trade_date": new_trade_date, "mark_wall": new_mark_wall},
                      clock.next())
        gw._persist()


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
        nav = ledger.state.nav()
        return PaperStepReport(
            trade_date=new_trade_date, action="latch_freeze",
            rule_id="R1_EXECUTION_DIVERGENT", permits_new_risk=False,
            intents_generated=0, intents_submitted=0, intents_rejected=0,
            filled_orders=0, starting_nav=nav, baseline_after=nav,
            model_after=nav, ensemble_contribution=0.0, blocked=True,
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

    _advance_session(sx, new_trade_date, new_mark_wall, new_marks)

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

        # Paper fill model: fill each venue-known working order by its REMAINING
        # quantity (venue.fill is incremental) and verify the Gateway outcome.
        snap = venue.snapshot() or {}
        venue_orders = snap.get("orders", {})
        for o in list(ledger.state.working):
            if o.terminal or o.remaining <= 0:
                continue
            vo = venue_orders.get(o.order_id)
            if vo is None or vo["cum_qty"] >= o.qty:
                continue
            ev = venue.fill(o.order_id, o.remaining, price=o.price)
            outcome = gw.process(ev)
            if outcome in _ACCEPTED_FILL:
                filled += 1
            else:
                note = f"fill processing outcome: {outcome}"
                break  # fail closed on unconfirmed/blocked fills

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
