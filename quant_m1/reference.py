"""Independent implementations used ONLY by tests. Different code path from risk.py.

M1.2: original four functions preserved EXACTLY (signed, exact-rational).
Added abs/ceil variants so reconciliation can match production semantics.

Production vs reference:
  risk.usage()      -> abs(value)        | ref_gross/ref_issuer -> signed value
  core.ceil_bps()   -> rounds cost UP    | ref_settled          -> exact Fraction

Under the long-only mandate signed == abs. The _abs variants make that explicit.
For cash reconciliation use ref_settled_ceil: production must be <= exact reference.
"""
from __future__ import annotations

from fractions import Fraction

BPS = 10_000


def _ceil_frac(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def ref_nav(cash, recv, positions, marks):
    total = Fraction(cash) + Fraction(recv)
    for s, q in positions.items():
        total += Fraction(q) * Fraction(marks[s])
    return total


def ref_gross(positions, marks):
    return sum(Fraction(q) * Fraction(marks[s]) for s, q in positions.items())


def ref_issuer(positions, marks, instruments):
    acc = {}
    for s, q in positions.items():
        val = Fraction(q) * Fraction(marks[s])
        for iss, w in instruments[s].lookthrough():
            acc[iss] = acc.get(iss, Fraction(0)) + val * Fraction(w, BPS)
    return max(acc.values(), default=Fraction(0))


def ref_settled(cash, orders, fracs):
    """Sells never add settled cash in-window."""
    out = Fraction(cash)
    for o, f in zip(orders, fracs):
        if o.side == "BUY":
            q = Fraction(o.remaining) * f
            n = q * Fraction(o.price)
            c = n * Fraction(o.cost_bps, BPS)
            out -= (n + c)
    return out


def ref_gross_abs(positions, marks):
    """Matches risk.usage(): absolute market value."""
    return sum(abs(Fraction(q) * Fraction(marks[s])) for s, q in positions.items())


def ref_issuer_abs(positions, marks, instruments):
    """Matches risk.usage(): abs value, per-issuer ceil_bps accumulation."""
    acc: dict[str, int] = {}
    for s, q in positions.items():
        val = abs(int(q) * int(marks[s]))
        for iss, w in instruments[s].lookthrough():
            acc[iss] = acc.get(iss, 0) + _ceil_frac(Fraction(val * w, BPS))
    return max(acc.values(), default=0)


def ref_settled_ceil(cash, orders, fracs):
    """Matches production: ceil_bps cost per reachable BUY fill."""
    out = int(cash)
    for o, f in zip(orders, fracs):
        if o.side == "BUY":
            q = int(o.remaining) * int(f) if isinstance(f, int) else Fraction(o.remaining) * f
            if not isinstance(q, int):
                q = int(q)
            n = q * int(o.price)
            out -= n + _ceil_frac(Fraction(n * int(o.cost_bps), BPS))
    return out


def ref_receivable(recv, orders, fracs):
    """Sells add net proceeds to RECEIVABLES, never settled cash."""
    out = Fraction(recv)
    for o, f in zip(orders, fracs):
        if o.side == "SELL":
            q = Fraction(o.remaining) * f
            n = q * Fraction(o.price)
            c = n * Fraction(o.cost_bps, BPS)
            out += (n - c)
    return out
