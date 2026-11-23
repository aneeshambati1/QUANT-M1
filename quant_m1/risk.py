"""M1.4 deterministic reachable-fill risk evaluation. Per-SimState marks.

M1.4 FIX: risk-reducing exemption.
  A trade whose worst-case usage does not increase ANY constraint is allowed
  even when the base book already breaches a soft concentration limit.
  Without this, a breached position can never be reduced.
  Hard floors (short reachable, negative settled cash, non-positive NAV)
  are NEVER exempted.
  worst_loss / cvar are floored at 0 for the reducing comparison, because
  moving from a stress GAIN to zero is not an increase in risk.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from .core import (Money, Shares, BPS, Mandate, Instrument, WorkingOrder,
                   VerifiedState, ceil_bps, floor_bps)


@dataclass(frozen=True)
class SimState:
    positions: dict[str, Shares]
    cash_settled: Money
    receivable_total: Money
    nav: Money
    marks: dict[str, Money]


def simulate(base: VerifiedState, orders: tuple[WorkingOrder, ...],
             nums: tuple[int, ...], den: int, instruments: dict[str, Instrument],
             settle_lag: int) -> SimState:
    if den <= 0 or len(nums) != len(orders) or any(n < 0 or n > den for n in nums):
        raise ValueError("invalid fill-fraction vector")
    if settle_lag < 1:
        raise ValueError("settle_lag must be >= 1: T+0 would invalidate this model")
    pos = dict(base.positions)
    cash = base.cash_settled
    recv = base.receivable_total()
    for o, num in zip(orders, nums):
        q = o.remaining * num // den
        if q == 0:
            continue
        if o.price <= 0 or o.cost_bps < 0:
            raise ValueError("invalid working order economics")
        notional = q * o.price
        cost = ceil_bps(notional, o.cost_bps)
        if o.side == "BUY":
            pos[o.symbol] = pos.get(o.symbol, 0) + q
            cash -= notional + cost
        elif o.side == "SELL":
            if q > pos.get(o.symbol, 0):
                raise ValueError("reachable fill creates short position")
            pos[o.symbol] = pos.get(o.symbol, 0) - q
            recv += notional - cost
        else:
            raise ValueError("unknown order side")
    if any(s not in base.marks for s, q in pos.items() if q):
        raise ValueError("missing mark for nonzero position")
    nav = cash + recv + sum(q * base.marks[s] for s, q in pos.items() if q)
    return SimState(pos, cash, recv, nav, dict(base.marks))


def vertices(m: int):
    if m < 0:
        raise ValueError("negative order count")
    return itertools.product((0, 1), repeat=m)


def scenario_nav(sim: SimState, marks: dict[str, int]) -> Money:
    if any(s not in marks for s, q in sim.positions.items() if q):
        raise ValueError("scenario missing a held instrument mark")
    return sim.cash_settled + sim.receivable_total + sum(
        q * marks[s] for s, q in sim.positions.items() if q)


@dataclass(frozen=True)
class Usage:
    gross: Money
    issuer_max: Money
    factor_abs: Money
    worst_loss: Money
    cvar: Money
    neg_cash: Money


def usage(sim: SimState, base_nav: Money, instruments: dict[str, Instrument],
          scenarios: tuple[dict[str, int], ...], q_bps: int) -> Usage:
    if not scenarios:
        raise ValueError("at least one stress scenario is required")
    if not 1 <= q_bps <= BPS:
        raise ValueError("CVaR tail fraction must be in (0, 10000] bps")
    gross = 0
    issuer: dict[str, int] = {}
    factor = 0
    for sym, q in sim.positions.items():
        if q == 0:
            continue
        if sym not in instruments or sym not in sim.marks:
            raise ValueError(f"missing instrument or mark: {sym}")
        inst = instruments[sym]
        val = q * sim.marks[sym]
        gross += abs(val)
        for iss, w in inst.lookthrough():
            issuer[iss] = issuer.get(iss, 0) + ceil_bps(abs(val), w)
        factor += inst.beta_bps * val
    losses = sorted((base_nav - scenario_nav(sim, sc) for sc in scenarios), reverse=True)
    k = max(1, (len(losses) * q_bps + BPS - 1) // BPS)
    k = min(k, len(losses))
    cvar = sum(losses[:k]) // k
    return Usage(gross, max(issuer.values(), default=0), abs(factor) // BPS,
                 max(losses), cvar, max(0, -sim.cash_settled))


def set_marks(_marks: dict[str, Money]) -> None:
    """Deprecated compatibility shim. Marks belong to each SimState."""
    return None


CONSTRAINTS = ("gross", "issuer_max", "factor_abs", "worst_loss", "cvar", "neg_cash")
LOSS_CONSTRAINTS = frozenset({"worst_loss", "cvar"})


def is_risk_reducing(worst: dict[str, int], base: dict[str, int]) -> bool:
    """True when no constraint usage increases. Loss metrics floored at 0."""
    for c in CONSTRAINTS:
        w, b = worst[c], base[c]
        if c in LOSS_CONSTRAINTS:
            w, b = max(0, w), max(0, b)
        if w > b:
            return False
    return True


@dataclass(frozen=True)
class Evaluation:
    ok: bool
    reason: str
    max_usage: dict[str, int]
    base_usage: dict[str, int]
    nav_min: Money
    risk_reducing: bool
    n_vertices: int


def evaluate(base: VerifiedState, orders: tuple[WorkingOrder, ...],
             instruments: dict[str, Instrument], mandate: Mandate,
             scenarios: tuple[dict[str, int], ...],
             *, m_vertex: int = 6) -> Evaluation:
    m = len(orders)
    if m > m_vertex:
        return Evaluation(False, "M_VERTEX_EXCEEDED", {}, {}, 0, False, 0)
    if not scenarios:
        return Evaluation(False, "NO_STRESS_SCENARIOS", {}, {}, 0, False, 0)
    try:
        nav0 = base.nav()
        base_sim = simulate(base, orders, (0,) * m, 1, instruments, mandate.settle_lag_days)
        bu = usage(base_sim, nav0, instruments, scenarios, mandate.cvar_q_bps)
        base_d = {c: getattr(bu, c) for c in CONSTRAINTS}
        max_d = {c: -10**30 for c in CONSTRAINTS}
        nav_min = 10**30
        n = 0
        for v in vertices(m):
            n += 1
            sim = simulate(base, orders, v, 1, instruments, mandate.settle_lag_days)
            # HARD FLOORS: never exempted, even for risk-reducing trades.
            if any(q < 0 for q in sim.positions.values()):
                return Evaluation(False, "SHORT_REACHABLE", {}, {}, 0, False, n)
            if sim.cash_settled < 0:
                return Evaluation(False, "SETTLED_CASH_NEGATIVE", {}, {}, 0, False, n)
            nav_min = min(nav_min, sim.nav)
            u = usage(sim, nav0, instruments, scenarios, mandate.cvar_q_bps)
            for c in CONSTRAINTS:
                max_d[c] = max(max_d[c], getattr(u, c))
        if nav_min <= 0:
            return Evaluation(False, "NAV_NONPOSITIVE", max_d, base_d, nav_min, False, n)

        reducing = is_risk_reducing(max_d, base_d)

        checks = (
            ("gross", max_d["gross"], floor_bps(nav_min, mandate.gross_bps), "GROSS"),
            ("issuer_max", max_d["issuer_max"], floor_bps(nav_min, mandate.issuer_bps), "ISSUER"),
            ("factor_abs", max_d["factor_abs"], floor_bps(nav_min, mandate.factor_bps), "FACTOR"),
            ("worst_loss", max_d["worst_loss"], floor_bps(nav0, mandate.worst_loss_bps), "WORST_LOSS"),
            ("cvar", max_d["cvar"], floor_bps(nav0, mandate.cvar_bps), "CVAR"),
        )
        for _, got, lim, name in checks:
            if got > lim:
                if reducing:
                    # Pre-existing breach. Allow de-risking, do not trap the book.
                    continue
                return Evaluation(False, f"LIMIT_{name}", max_d, base_d, nav_min, False, n)
        return Evaluation(True, "OK", max_d, base_d, nav_min, reducing, n)
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        return Evaluation(False, f"INVALID_RISK_INPUT:{type(exc).__name__}", {}, {}, 0, False, 0)


def reservations(ev: Evaluation) -> dict[str, int]:
    if not ev.ok:
        raise ValueError("cannot reserve against a failed risk evaluation")
    return {c: max(0, ev.max_usage[c] - ev.base_usage[c]) for c in CONSTRAINTS}
