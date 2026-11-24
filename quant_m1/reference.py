"""Independent test-only oracle. Mirrors risk.py usage() semantics exactly.

Uses integer ceil_bps aggregation and shared UNK:ALL bucket to match
production Instrument.lookthrough() and usage() behavior.
"""
from __future__ import annotations

from fractions import Fraction

from .core import BPS, ceil_bps


def _ceil_frac(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def ref_nav(cash: int, recv: int, positions: dict[str, int],
            marks: dict[str, int]) -> Fraction:
    total = Fraction(cash) + Fraction(recv)
    for s, q in positions.items():
        total += Fraction(q) * Fraction(marks[s])
    return total


def ref_gross(positions: dict[str, int], marks: dict[str, int]) -> Fraction:
    return sum(Fraction(q) * Fraction(marks[s]) for s, q in positions.items())


def ref_gross_abs(positions: dict[str, int], marks: dict[str, int]) -> int:
    return sum(abs(int(q) * int(marks[s])) for s, q in positions.items())


def ref_issuer(positions: dict[str, int], marks: dict[str, int],
               instruments: dict) -> int:
    """Integer issuer exposure matching production usage().issuer_max.

    Uses ceil_bps and shared UNK:ALL bucket via lookthrough().
    """
    acc: dict[str, int] = {}
    for s, q in positions.items():
        if q == 0:
            continue
        val = abs(int(q) * int(marks[s]))
        for iss, w in instruments[s].lookthrough():
            acc[iss] = acc.get(iss, 0) + ceil_bps(val, w)
    return max(acc.values(), default=0)


def ref_issuer_abs(positions: dict[str, int], marks: dict[str, int],
                   instruments: dict) -> int:
    """Alias kept for backward compatibility. Identical to ref_issuer."""
    return ref_issuer(positions, marks, instruments)


def ref_settled(cash: int, orders: tuple, fracs: tuple) -> Fraction:
    """Sells never add settled cash in-window."""
    out = Fraction(cash)
    for o, f in zip(orders, fracs):
        if o.side == "BUY":
            q = Fraction(o.remaining) * f
            n = q * Fraction(o.price)
            out -= (n + n * Fraction(o.cost_bps, BPS))
    return out


def ref_settled_ceil(cash: int, orders: tuple, fracs: tuple) -> int:
    out = int(cash)
    for o, f in zip(orders, fracs):
        if o.side == "BUY":
            q = int(Fraction(o.remaining) * f)
            n = q * int(o.price)
            out -= n + _ceil_frac(Fraction(n * int(o.cost_bps), BPS))
    return out


def ref_receivable(recv: int, orders: tuple, fracs: tuple) -> Fraction:
    out = Fraction(recv)
    for o, f in zip(orders, fracs):
        if o.side == "SELL":
            q = Fraction(o.remaining) * f
            n = q * Fraction(o.price)
            out += (n - n * Fraction(o.cost_bps, BPS))
    return out
