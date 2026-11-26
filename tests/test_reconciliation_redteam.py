"""Reconciliation red-team: fill-ahead-of-venue and unexpected venue order."""
from __future__ import annotations

import pytest

from quant_m1.authority import Intent
from quant_m1.system import build
from quant_m1.venue import ExecEvent


def _auth(sx, symbol="C", side="BUY", qty=10):
    i = Intent(symbol, side, qty, 10_000, 10)
    return sx["kernel"].authorize(sx["ledger"], i, sx["flags"], sx["clock"], False), i


def test_fill_ahead_of_venue_snapshot_is_blocked(tmp_path):
    sx = build(str(tmp_path))
    tok, intent = _auth(sx); sx["gw"].submit(tok, intent)
    ev = ExecEvent("E-AHEAD", tok.order_id, "FILL", 5, 10_000, 999)
    assert sx["gw"].process(ev) == "FILL_UNCONFIRMED_RESERVED"
    assert sx["gw"]._divergence_latched is True
    assert sx["ledger"].state.positions.get("C", 0) == 0


def test_unexpected_venue_order_latches(tmp_path):
    sx = build(str(tmp_path))
    tok, intent = _auth(sx); sx["gw"].submit(tok, intent)
    sx["venue"].send("rogue", "A", "BUY", 1, 10_000, 0)   # bypasses gateway
    ev = sx["venue"].fill(tok.order_id, 10, price=10_000)
    res = sx["gw"].process(ev)
    assert res == "APPLIED_RECONCILE_DIVERGENT"
    assert sx["gw"]._divergence_latched is True
    assert "NEW_RISK" not in sx["flags"].permissions()


def test_timeout_remains_ambiguous_and_reserved(tmp_path):
    sx = build(str(tmp_path))
    tok, intent = _auth(sx); sx["gw"].submit(tok, intent)
    reserved = dict(sx["ledger"].reserved)
    ev = sx["venue"].timeout(tok.order_id)
    assert sx["gw"].process(ev) == "TIMEOUT_UNRESOLVED_RESERVED"
    o = next(o for o in sx["ledger"].state.working if o.order_id == tok.order_id)
    assert not o.terminal
    assert sx["ledger"].reserved == reserved
