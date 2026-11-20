"""Independent implementations used ONLY by tests. Different code path from risk.py."""
from __future__ import annotations
from fractions import Fraction


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
            acc[iss] = acc.get(iss, Fraction(0)) + val * Fraction(w, 10_000)
    return max(acc.values(), default=Fraction(0))


def ref_settled(cash, orders, fracs):
    """Sells never add settled cash in-window."""
    out = Fraction(cash)
    for o, f in zip(orders, fracs):
        if o.side == "BUY":
            q = Fraction(o.remaining) * f
            n = q * Fraction(o.price)
            c = n * Fraction(o.cost_bps, 10_000)
            out -= (n + c)
    return out
