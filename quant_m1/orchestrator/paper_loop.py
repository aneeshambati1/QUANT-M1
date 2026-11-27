"""Single-step paper-trading orchestration with a shadow-baseline contribution.

FIX (2026-11-27):
- Removed unsafe allocator wrapper that broke the run_cycle API.
- Removed strict forecast pre-validation that blocked valid empty forecasts
  and model_signals inputs. Delegation to run_cycle is now correct.
- Preserves fail-closed behavior for divergence latches and invalid marks.

PHASE 2 FIX (_shadow_system):
- The shadow is now built with the REAL universe (instruments, marks, allowlist,
  product_classes) at construction, and the REAL cfg object is installed on
  sx["cfg"], kernel.cfg and gw.cfg. ledger.propagate_config binds the shadow
  ledger to the real mandate version/digest, so CONFIG_STATE_MISMATCH cannot
  occur. Instruments/scenarios are installed on every holder. The shadow owns
  its ledger/gateway/venue/workdir; the real system is never mutated.
"""
from __future__ import annotations

import shutil
import tempfile
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
            "trade_date": self.trade_date, "action": self.action,
            "rule_id": self.rule_id, "permits_new_risk": self.permits_new_risk,
            "intents_generated": self.intents_generated,
            "intents_submitted": self.intents_submitted,
            "intents_rejected": self.intents_rejected,
            "filled_orders": self.filled_orders,
            "starting_nav": self.starting_nav,
            "baseline_after": self.baseline_after,
            "model_after": self.model_after,
            "ensemble_contribution": self.ensemble_contribution,
            "blocked": self.blocked, "note": self.note,
            "rejection_reasons": list(self.rejection_reasons),
        }

def _cash_equivalent(ledger) -> int:
    return ledger.state.cash_settled + ledger.state.receivable_total()

def _advance_session(sx, new_trade_date: int, new_mark_wall: int,
                     new_marks: Mapping[str, int]) -> None:
    ledger, gw, clock = sx["ledger"], sx["gw"], sx["clock"]
    with sx["lock"]:
        if clock.wall < ledger.state.mark_wall:
            clock.advance_wall(ledger.state.mark_wall - clock.wall)

        if ledger.state.trade_date == new_trade_date:
            if (ledger.state.mark_wall != new_mark_wall
                    or dict(ledger.state.marks) != dict(new_marks)):
                raise ValueError("SESSION_CONFLICT")
            return

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
    """Run one paper session step. Delegates forecast validation to run_cycle."""
    ledger, gw, venue = sx["ledger"], sx["gw"], sx["venue"]
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
                break

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
# ---------------------------------------------------------------------------

@dataclass
class PaperSessionReport:
    step: PaperStepReport
    contributions: dict
    attributed: list
    def as_dict(self) -> dict:
        return {"step": self.step.as_dict(), "contributions": dict(self.contributions), "attributed": list(self.attributed)}

def _shadow_system(real_sx, start: Mapping, tmpdir: str):
    """Isolated counterfactual system bound to the REAL mandate/universe.

    Phase 2 repair: build with the real instruments/marks/allowlist/product
    classes; share the real cfg object read-only on every holder; bind the
    shadow ledger to the real mandate version/digest via propagate_config.
    The shadow owns its ledger/gateway/venue/workdir; real_sx is never mutated.
    """
    from ..system import build as _build
    real_cfg = real_sx["cfg"]
    sx = _build(
        tmpdir,
        cash=start["cash_settled"],
        positions=dict(start["positions"]),
        marks=dict(start["marks"]),
        trade_date=start["trade_date"],
        mark_wall=start["mark_wall"],
        instruments=real_sx["instruments"],
        scenarios=real_sx["scenarios"],
        allowlist=real_cfg.mandate.allowlist,
        product_classes=real_cfg.mandate.product_classes,
    )
    # Share the real config/universe/scenarios read-only on ALL holders.
    sx["cfg"] = real_cfg
    sx["instruments"] = real_sx["instruments"]
    sx["scenarios"] = real_sx["scenarios"]
    for comp in ("kernel", "gw"):
        sx[comp].cfg = real_cfg
        sx[comp].instruments = real_sx["instruments"]
        sx[comp].scenarios = real_sx["scenarios"]
    # Bind the shadow ledger to the real mandate so authorize() cannot mismatch.
    sx["ledger"].propagate_config(real_cfg.mandate.version, real_cfg.mandate.digest())
    sx["ledger"].state.receivables = tuple(start["receivables"])
    sx["ledger"].state.peak_nav = start["peak_nav"]
    return sx

def run_paper_session(sx, market_state, trackers, forecasts, *, new_trade_date, new_mark_wall, new_marks, allocator, min_trade_bps=10, cost_bps=10, attribute=True):
    ledger = sx["ledger"]
    start = {"cash_settled": ledger.state.cash_settled, "positions": dict(ledger.state.positions), "marks": dict(ledger.state.marks), "receivables": tuple(ledger.state.receivables), "peak_nav": ledger.state.peak_nav, "trade_date": ledger.state.trade_date, "mark_wall": ledger.state.mark_wall}
    step = run_paper_step(sx, market_state, trackers, forecasts, new_trade_date=new_trade_date, new_mark_wall=new_mark_wall, new_marks=new_marks, allocator=allocator, min_trade_bps=min_trade_bps, cost_bps=cost_bps)
    contributions, attributed = {}, []
    if attribute and not step.blocked and step.starting_nav > 0:
        active = [name for name, t in trackers.items() if getattr(t, "eligible", False) and getattr(t, "horizon_days", 1) == 1]
        for m in active:
            without = {k: t for k, t in trackers.items() if k != m}
            tmp = tempfile.mkdtemp(prefix="quant_shadow_")
            try:
                shadow = _shadow_system(sx, start, tmp)
                sh = run_paper_step(shadow, market_state, without, forecasts, new_trade_date=new_trade_date, new_mark_wall=new_mark_wall, new_marks=new_marks, allocator=allocator, min_trade_bps=min_trade_bps, cost_bps=cost_bps)
                if not sh.blocked:
                    contributions[m] = (step.model_after - sh.model_after) / step.starting_nav
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        if contributions:
            apply_realized(trackers, contributions, horizon_days=1)
            attributed = list(contributions)
    return PaperSessionReport(step=step, contributions=contributions, attributed=attributed)

def run_paper_sessions(sx, sessions, trackers, *, allocator, min_trade_bps=10, cost_bps=10, attribute=True):
    reports = []
    for (market_state, forecasts, date, wall, marks) in sessions:
        rep = run_paper_session(sx, market_state, trackers, forecasts, new_trade_date=date, new_mark_wall=wall, new_marks=marks, allocator=allocator, min_trade_bps=min_trade_bps, cost_bps=cost_bps, attribute=attribute)
        reports.append(rep)
        if rep.step.blocked:
            break
    return reports
