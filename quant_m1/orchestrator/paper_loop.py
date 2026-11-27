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
  leave-one-out attribution is provided by run_paper_session below.

FILL MODEL: venue.fill takes an INCREMENTAL quantity, so the step fills each
venue-known working order by its REMAINING quantity and checks the Gateway's
processing outcome. Unconfirmed/blocked outcomes stop the fill loop (fail
closed); only accepted fills are counted.

FAIL-CLOSED: no allocation or a policy block => no orders. A divergence latch
returns a blocked report and does not learn from unverified fills.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import numpy as np

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
    
    CRITICAL FIX: Ensures 'forecasts' contains valid numpy arrays before 
    passing to run_cycle/allocator to prevent TypeError.
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

    # --- DATA VALIDATION & EXTRACTION (THE FIX) ---
    # We must ensure the allocator receives numeric arrays, not objects/dicts.
    try:
        mu_raw = forecasts.get("expected_returns")
        cov_raw = forecasts.get("cov")
        
        if mu_raw is None or cov_raw is None:
             raise KeyError("Missing 'expected_returns' or 'cov' in forecasts.")
             
        # Force conversion to float64 numpy arrays
        mu_array = np.asarray(mu_raw, dtype=np.float64)
        cov_array = np.asarray(cov_raw, dtype=np.float64)
        
        # Basic shape sanity check against number of assets in marks
        n_assets = len(new_marks)
        if mu_array.shape != (n_assets,) or cov_array.shape != (n_assets, n_assets):
            raise ValueError(f"Dimension mismatch: mu{mu_array.shape}, cov{cov_array.shape}, expected ({n_assets},)")

        # Update the forecasts dict with cleaned arrays for downstream use
        # Note: We create a shallow copy to avoid mutating original input unexpectedly
        clean_forecasts = dict(forecasts)
        clean_forecasts["expected_returns"] = mu_array
        clean_forecasts["cov"] = cov_array
        
    except Exception as e:
        # Fail Closed: If data is bad, do not trade. Return blocked report.
        return PaperStepReport(
            trade_date=new_trade_date, action="error_data_validation",
            rule_id="R2_INVALID_FORECAST_DATA", permits_new_risk=False,
            intents_generated=0, intents_submitted=0, intents_rejected=0,
            filled_orders=0, starting_nav=starting_nav, baseline_after=baseline_after,
            model_after=starting_nav, ensemble_contribution=0.0, blocked=True,
            note=f"Data Validation Failed: {str(e)}",
        )

    # Run the cycle with CLEANED forecasts
    rec = run_cycle(market_state, trackers, forecasts=clean_forecasts, allocator=allocator)

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


# ---------------------------------------------------------------------------
# Multi-session driver + per-model leave-one-out attribution (O-07 / O-09).
#
# ATTRIBUTION CONVENTION (pre-registered):
#   Model m's 1-session marginal after-cost contribution is
#       (NAV_full_ensemble - NAV_ensemble_without_m) / starting_NAV
#   where the counterfactual "without m" is run in an ISOLATED shadow book that
#   starts from the identical pre-session state and uses the identical config,
#   marks, costs and allocator. Both books pay their own transaction costs, so
#   the difference is net-of-cost. Leave-one-out contributions do NOT sum to the
#   ensemble-vs-hold number (interaction effects); the ensemble-vs-hold figure
#   remains the overall gate, while per-model LOO drives per-model authority.
#   Only horizon_days==1 trackers are attributed per session; other horizons
#   idle-decay (their measurement requires their own holding-period window).
# ---------------------------------------------------------------------------


@dataclass
class PaperSessionReport:
    step: PaperStepReport
    contributions: dict
    attributed: list

    def as_dict(self) -> dict:
        return {"step": self.step.as_dict(),
                "contributions": dict(self.contributions),
                "attributed": list(self.attributed)}


def _shadow_system(real_sx, start: Mapping, tmpdir: str):
    """Isolated counterfactual book faithful to the real pre-session state."""
    from ..system import build as _build
    sx = _build(tmpdir, cash=start["cash_settled"], positions=start["positions"],
                marks=start["marks"], trade_date=start["trade_date"],
                mark_wall=start["mark_wall"])
    for comp in ("kernel", "gw"):
        sx[comp].cfg = real_sx["cfg"]
        sx[comp].instruments = real_sx["instruments"]
        sx[comp].scenarios = real_sx["scenarios"]
    sx["ledger"].state.receivables = tuple(start["receivables"])
    sx["ledger"].state.peak_nav = start["peak_nav"]
    return sx


def run_paper_session(
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
    attribute: bool = True,
) -> PaperSessionReport:
    """One real session step plus per-model leave-one-out attribution."""
    ledger = sx["ledger"]
    start = {
        "cash_settled": ledger.state.cash_settled,
        "positions": dict(ledger.state.positions),
        "marks": dict(ledger.state.marks),
        "receivables": tuple(ledger.state.receivables),
        "peak_nav": ledger.state.peak_nav,
        "trade_date": ledger.state.trade_date,
        "mark_wall": ledger.state.mark_wall,
    }
    step = run_paper_step(sx, market_state, trackers, forecasts,
                          new_trade_date=new_trade_date, new_mark_wall=new_mark_wall,
                          new_marks=new_marks, allocator=allocator,
                          min_trade_bps=min_trade_bps, cost_bps=cost_bps)
    contributions: dict = {}
    attributed: list = []
    if attribute and not step.blocked and step.starting_nav > 0:
        active = [name for name, t in trackers.items()
                  if getattr(t, "eligible", False) and getattr(t, "horizon_days", 1) == 1]
        for m in active:
            without = {k: t for k, t in trackers.items() if k != m}
            tmp = tempfile.mkdtemp(prefix="quant_shadow_")
            try:
                shadow = _shadow_system(sx, start, tmp)
                sh = run_paper_step(shadow, market_state, without, forecasts,
                                    new_trade_date=new_trade_date, new_mark_wall=new_mark_wall,
                                    new_marks=new_marks, allocator=allocator,
                                    min_trade_bps=min_trade_bps, cost_bps=cost_bps)
                if not sh.blocked:
                    contributions[m] = (step.model_after - sh.model_after) / step.starting_nav
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        if contributions:
            apply_realized(trackers, contributions, horizon_days=1)
            attributed = list(contributions)
    return PaperSessionReport(step=step, contributions=contributions, attributed=attributed)


def run_paper_sessions(
    sx: Mapping,
    sessions,
    trackers: Mapping,
    *,
    allocator: Callable,
    min_trade_bps: int = 10,
    cost_bps: int = 10,
    attribute: bool = True,
) -> list:
    """Sequence paper sessions deterministically; stop fail-closed on a block.

    ``sessions`` is an iterable of (market_state, forecasts, new_trade_date,
    new_mark_wall, new_marks). Returns one PaperSessionReport per attempted
    session. This is a deterministic session driver, not a live daemon; the
    scheduled data->decision->execution->reconciliation loop remains O-07.
    """
    reports = []
    for (market_state, forecasts, date, wall, marks) in sessions:
        rep = run_paper_session(sx, market_state, trackers, forecasts,
                                new_trade_date=date, new_mark_wall=wall, new_marks=marks,
                                allocator=allocator, min_trade_bps=min_trade_bps,
                                cost_bps=cost_bps, attribute=attribute)
        reports.append(rep)
        if rep.step.blocked:
            break
    return reports
