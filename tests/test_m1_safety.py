"""M1 safety tests. Each targets a specific defect in the register.

Updated for M1.5 core/risk/authority:
  - Stale-cum asserts SNAPSHOT_RECONCILED and restores from composite.
  - Settlement fixture sized so the post-trade book is issuer-compliant.
  - G0 detection tests for session authority and UNK:ALL aggregation.
"""
from __future__ import annotations

import pytest

from quant_m1.authority import Intent, Reject
from quant_m1.core import Ledger, ceil_bps
from quant_m1.system import build
from quant_m1.venue import ExecEvent


@pytest.fixture
def sx(tmp_path):
    return build(str(tmp_path))


def _auth(sx, symbol="C", side="BUY", qty=10, price=10_000, cost=10):
    intent = Intent(symbol, side, qty, price, cost)
    tok = sx["kernel"].authorize(sx["ledger"], intent, sx["flags"], sx["clock"], False)
    return tok, intent


def _restore_ledger(sx):
    """Restore ledger from the composite state file written by Gateway._persist."""
    d = sx["gw"].composite_store.load()
    assert d is not None, "composite_state.json must exist after _persist"
    return Ledger.from_dict(d["ledger"], sx["ledger"].store)


# ---- T+1 ----

def test_sell_fill_reconciles_with_t0_venue_cash(sx):
    tok, intent = _auth(sx, symbol="A", side="SELL", qty=10, price=10_000, cost=25)
    sx["gw"].submit(tok, intent)
    ev = sx["venue"].fill(tok.order_id, 10, price=10_000)
    assert sx["gw"].process(ev) == "APPLIED"
    assert sx["flags"].exec == "OK"
    st = sx["ledger"].state
    assert st.cash_settled == 10_000_000
    assert st.cash_unsettled == 100_000 - ceil_bps(100_000, 25)
    assert sx["venue"].cash == st.cash_settled + st.receivable_total()


def test_sell_proceeds_cannot_fund_same_day_buy(tmp_path):
    sx = build(str(tmp_path), cash=0, positions={"A": 10},
               marks={"A": 10_000, "B": 10_000, "C": 10_000})
    tok, intent = _auth(sx, symbol="A", side="SELL", qty=10, price=10_000, cost=0)
    sx["gw"].submit(tok, intent)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 10, price=10_000))
    assert sx["ledger"].state.cash_settled == 0
    assert sx["ledger"].state.cash_unsettled == 100_000
    with pytest.raises(Reject) as e:
        _auth(sx, symbol="C", side="BUY", qty=10, price=10_000, cost=0)
    assert e.value.code == "SETTLED_CASH_NEGATIVE"


def test_settlement_next_session_enables_buy(tmp_path):
    """T+1 gate: sell proceeds unusable same day, usable next session.

    Fixture sized so the traded notional stays inside the issuer cap:
    NAV = 100_000, issuer cap = 40_000, trade notional = 10_000.
    """
    sx = build(str(tmp_path), cash=90_000, positions={"A": 1},
               marks={"A": 10_000, "B": 10_000, "C": 10_000})
    # Drain settled cash so only the receivable can fund the buy.
    sx["ledger"].state.cash_settled = 0
    sx["ledger"].persist()

    tok, intent = _auth(sx, symbol="A", side="SELL", qty=1, price=10_000, cost=0)
    sx["gw"].submit(tok, intent)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 1, price=10_000))
    assert sx["ledger"].state.cash_settled == 0
    assert sx["ledger"].state.cash_unsettled == 10_000

    d = sx["ledger"].state.trade_date
    assert sx["ledger"].settle(d) == 0
    assert sx["ledger"].settle(d + 1) == 10_000
    sx["ledger"].state.trade_date = d + 1
    sx["ledger"].persist()
    _auth(sx, symbol="C", side="BUY", qty=1, price=10_000, cost=0)


# ---- unsubmitted reservations ----

def test_unsubmitted_reservation_does_not_cause_divergent(sx):
    t1, i1 = _auth(sx, qty=5)
    sx["gw"].submit(t1, i1)
    _auth(sx, qty=5)
    ev = sx["venue"].fill(t1.order_id, 5, price=10_000)
    assert sx["gw"].process(ev) == "APPLIED"
    assert sx["flags"].exec == "OK"


def test_recovery_ignores_unsubmitted_reservation(sx):
    t1, i1 = _auth(sx, qty=5)
    sx["gw"].submit(t1, i1)
    _auth(sx, qty=5)
    sx["gw"].process(sx["venue"].fill(t1.order_id, 5, price=10_000))
    assert sx["gw"].recover() == "RECOVERED"


def test_second_authorize_invalidates_first_token(sx):
    t1, i1 = _auth(sx, qty=5)
    _auth(sx, qty=5)
    with pytest.raises(Reject) as e:
        sx["gw"].submit(t1, i1)
    assert e.value.code == "STALE_SNAPSHOT"


# ---- durability ----

def test_stale_cum_event_returns_snapshot_reconciled(sx):
    """Stale cum must return SNAPSHOT_RECONCILED; durable via composite."""
    tok, intent = _auth(sx, qty=10)
    sx["gw"].submit(tok, intent)
    sx["venue"].fill(tok.order_id, 10, price=10_000)
    ev = ExecEvent("E-STALE", tok.order_id, "FILL", 1, 10_000, 999)
    result = sx["gw"].process(ev)
    assert result == "SNAPSHOT_RECONCILED"
    assert ev.exec_id in sx["ledger"].applied_exec_ids
    back = _restore_ledger(sx)
    assert back.state.positions["C"] == 10
    assert sx["gw"].process(ev) == "NOOP_DUPLICATE"


def test_multi_price_fill_uses_authoritative_notional(sx):
    tok, intent = _auth(sx, qty=10, price=10_000, cost=25)
    sx["gw"].submit(tok, intent)
    sx["venue"].fill(tok.order_id, 4, price=10_100)
    ev2 = sx["venue"].fill(tok.order_id, 6, price=10_300)
    assert sx["gw"].process(ev2) == "APPLIED"
    o = next(o for o in sx["ledger"].state.working if o.order_id == tok.order_id)
    assert o.filled_notional == 4 * 10_100 + 6 * 10_300
    assert sx["flags"].exec == "OK"


def test_receivable_uses_current_session(sx):
    tok, intent = _auth(sx, symbol="A", side="SELL", qty=10, price=10_000, cost=0)
    sx["gw"].submit(tok, intent)
    sx["ledger"].state.trade_date = 105
    sx["gw"].process(sx["venue"].fill(tok.order_id, 10, price=10_000))
    due = [d for d, _ in sx["ledger"].state.receivables]
    assert due == [106]


# ---- config binding ----

def test_m_vertex_config_honored(tmp_path):
    sx = build(str(tmp_path), m_vertex=2)
    _auth(sx, qty=1)
    _auth(sx, qty=1)
    with pytest.raises(Reject) as e:
        _auth(sx, qty=1)
    assert e.value.code == "M_VERTEX"


def test_stale_token_rejected_after_mandate_change(sx):
    tok, intent = _auth(sx, qty=5)
    sx["cfg"].apply_config({"mandate": {"gross_bps": 9_000}})
    with pytest.raises(Reject) as e:
        sx["gw"].submit(tok, intent)
    assert e.value.code in {"TOKEN_CONFIG_STALE", "CONFIG_STATE_MISMATCH"}


# ---- durable flags ----

def test_flags_survive_restart(tmp_path):
    sx = build(str(tmp_path))
    sx["flags"].exec = "DIVERGENT"
    sx["gw"]._persist()
    sx2 = build(str(tmp_path))
    assert sx2["flags"].exec == "DIVERGENT"
    assert "NEW_RISK" not in sx2["flags"].permissions()


# ---- timeout ----

def test_timeout_keeps_reservation(sx):
    tok, intent = _auth(sx, qty=10)
    sx["gw"].submit(tok, intent)
    reserved_before = dict(sx["ledger"].reserved)
    ev = sx["venue"].timeout(tok.order_id)
    assert sx["gw"].process(ev) == "TIMEOUT_UNRESOLVED_RESERVED"
    o = next(o for o in sx["ledger"].state.working if o.order_id == tok.order_id)
    assert not o.terminal
    assert sx["ledger"].reserved == reserved_before
    assert "NEW_RISK" not in sx["flags"].permissions()


# ---- preserved guarantees ----

def test_kill_switch_blocks_submit(sx):
    tok, intent = _auth(sx, qty=5)
    open(sx["wd"], "w").close()
    with pytest.raises(Reject) as e:
        sx["gw"].submit(tok, intent)
    assert e.value.code == "WATCHDOG_STOP"


def test_nonce_replay_blocked(sx):
    tok, intent = _auth(sx, qty=5)
    sx["gw"].submit(tok, intent)
    with pytest.raises(Reject) as e:
        sx["gw"].submit(tok, intent)
    assert e.value.code in {"NONCE_REPLAY", "ORDER_NOT_RESERVED"}


def test_intent_mutation_blocked(sx):
    tok, _ = _auth(sx, qty=5, price=10_000)
    with pytest.raises(Reject) as e:
        sx["gw"].submit(tok, Intent("C", "BUY", 5, 10_001, 10))
    assert e.value.code == "INTENT_MUTATED"


def test_live_transport_refused(tmp_path):
    with pytest.raises(ValueError):
        build(str(tmp_path), allow_live=True)


# ---- G0 detection tests ----

def test_session_regression_rejected(tmp_path):
    sx = build(str(tmp_path))
    sx["ledger"].settle(100)
    with pytest.raises(ValueError, match="SETTLEMENT_REGRESSION"):
        sx["ledger"].settle(99)


def test_advance_session_matures_receivable(tmp_path):
    sx = build(str(tmp_path), cash=90_000, positions={"A": 1},
               marks={"A": 10_000, "B": 10_000, "C": 10_000})
    sx["ledger"].state.cash_settled = 0
    sx["ledger"].persist()
    tok, intent = _auth(sx, symbol="A", side="SELL", qty=1, price=10_000, cost=0)
    sx["gw"].submit(tok, intent)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 1, price=10_000))
    d = sx["ledger"].state.trade_date
    released = sx["ledger"].advance_session(
        d + 2, sx["clock"].wall + 86400,
        {"A": 10_000, "B": 10_000, "C": 10_000})
    assert released == 10_000
    assert sx["ledger"].state.trade_date == d + 2
    assert sx["ledger"].state.cash_settled == 10_000


def test_unk_all_aggregation_binds_issuer_cap():
    """Two instruments with missing holdings aggregate into ONE bucket."""
    from quant_m1.core import Instrument
    inst_a = Instrument("X", "PLAIN_ETF", 50_000_000, 10_000, None)
    inst_b = Instrument("Y", "PLAIN_ETF", 50_000_000, 10_000, None)
    assert inst_a.lookthrough() == (("UNK:ALL", 10_000),)
    assert inst_b.lookthrough() == (("UNK:ALL", 10_000),)
    combined = {}
    for iss, w in inst_a.lookthrough():
        combined[iss] = combined.get(iss, 0) + ceil_bps(3_500_000, w)
    for iss, w in inst_b.lookthrough():
        combined[iss] = combined.get(iss, 0) + ceil_bps(3_500_000, w)
    assert combined["UNK:ALL"] == 7_000_000
    assert combined["UNK:ALL"] > 4_000_000  # breaches a 40% cap on 10M NAV


def test_composite_roundtrip_preserves_latch(tmp_path):
    """Ledger AND divergence latch must survive a composite round-trip."""
    sx = build(str(tmp_path))
    tok, intent = _auth(sx, qty=5)
    sx["gw"].submit(tok, intent)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 5, price=10_000))
    sx["gw"].latch_divergence("TEST_LATCH")
    sx["gw"]._persist()

    sx2 = build(str(tmp_path))
    assert sx2["gw"]._divergence_latched is True
    assert sx2["flags"].exec == "DIVERGENT"
    assert sx2["ledger"].state.positions.get("C") == 5