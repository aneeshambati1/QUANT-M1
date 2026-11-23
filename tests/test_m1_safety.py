"""M1.2 safety tests. Each targets a specific defect in the register."""
from __future__ import annotations

import pytest

from quant_m1.authority import Intent, Reject
from quant_m1.core import ceil_bps
from quant_m1.system import build


@pytest.fixture
def sx(tmp_path):
    return build(str(tmp_path))


def _auth(sx, symbol="C", side="BUY", qty=10, price=10_000, cost=10):
    intent = Intent(symbol, side, qty, price, cost)
    tok = sx["kernel"].authorize(sx["ledger"], intent, sx["flags"], sx["clock"], False)
    return tok, intent


# ---- A1: sells must not cause permanent DIVERGENT ----

def test_sell_fill_reconciles_with_t0_venue_cash(sx):
    tok, intent = _auth(sx, symbol="A", side="SELL", qty=10, price=10_000, cost=25)
    sx["gw"].submit(tok, intent)
    ev = sx["venue"].fill(tok.order_id, 10, price=10_000)
    assert sx["gw"].process(ev) == "APPLIED"
    assert sx["flags"].exec == "OK", "T+0 venue cash vs T+1 ledger must reconcile"
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
    sx = build(str(tmp_path), cash=0, positions={"A": 10},
               marks={"A": 10_000, "B": 10_000, "C": 10_000})
    tok, intent = _auth(sx, symbol="A", side="SELL", qty=10, price=10_000, cost=0)
    sx["gw"].submit(tok, intent)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 10, price=10_000))
    assert sx["ledger"].settle(sx["ledger"].state.trade_date) == 0
    assert sx["ledger"].settle(sx["ledger"].state.trade_date + 1) == 100_000
    _auth(sx, symbol="C", side="BUY", qty=10, price=10_000, cost=0)


# ---- A2: unsubmitted reservations must not poison reconciliation ----

def test_unsubmitted_reservation_does_not_cause_divergent(sx):
    t1, i1 = _auth(sx, qty=5)
    _auth(sx, qty=5)                      # second reservation, never submitted
    sx["gw"].submit(t1, i1)
    ev = sx["venue"].fill(t1.order_id, 5, price=10_000)
    assert sx["gw"].process(ev) == "APPLIED"
    assert sx["flags"].exec == "OK"


def test_recovery_ignores_unsubmitted_reservation(sx):
    t1, i1 = _auth(sx, qty=5)
    _auth(sx, qty=5)
    sx["gw"].submit(t1, i1)
    sx["gw"].process(sx["venue"].fill(t1.order_id, 5, price=10_000))
    assert sx["gw"].recover() == "RECOVERED"


# ---- A3: stale-cum path must persist ----

def test_stale_cum_reconciled_is_durable(sx):
    tok, intent = _auth(sx, qty=10)
    sx["gw"].submit(tok, intent)
    sx["venue"].fill(tok.order_id, 10, price=10_000)
    stale = sx["venue"].fill.__self__.outbox[-1]
    from quant_m1.venue import ExecEvent
    ev = ExecEvent("E-STALE", tok.order_id, "FILL", 1, 10_000, 999)
    assert sx["gw"].process(ev) == "SNAPSHOT_RECONCILED"
    assert ev.exec_id in sx["ledger"].applied_exec_ids
    from quant_m1.core import Ledger, Store
    back = Ledger.restore(sx["ledger"].store)
    assert back.state.positions["C"] == 10
    assert sx["gw"].process(ev) == "NOOP_DUPLICATE"


# ---- A4: multi-price fills use venue notional ----

def test_multi_price_fill_uses_authoritative_notional(sx):
    tok, intent = _auth(sx, qty=10, price=10_000, cost=25)
    sx["gw"].submit(tok, intent)
    sx["venue"].fill(tok.order_id, 4, price=10_100)
    ev2 = sx["venue"].fill(tok.order_id, 6, price=10_300)
    assert sx["gw"].process(ev2) == "APPLIED"
    o = next(o for o in sx["ledger"].state.working if o.order_id == tok.order_id)
    assert o.filled_notional == 4 * 10_100 + 6 * 10_300
    assert sx["flags"].exec == "OK"


# ---- A5: receivable dated from fill session, not reservation ----

def test_receivable_uses_current_session(sx):
    tok, intent = _auth(sx, symbol="A", side="SELL", qty=10, price=10_000, cost=0)
    sx["gw"].submit(tok, intent)
    sx["ledger"].state.trade_date = 105          # order rested several sessions
    sx["gw"].process(sx["venue"].fill(tok.order_id, 10, price=10_000))
    due = [d for d, _ in sx["ledger"].state.receivables]
    assert due == [106], "must settle from fill session, not reservation session"


# ---- A6: m_vertex honored end to end ----

def test_m_vertex_config_honored(tmp_path):
    sx = build(str(tmp_path), m_vertex=2)
    t1, i1 = _auth(sx, qty=1)
    _auth(sx, qty=1)
    with pytest.raises(Reject) as e:
        _auth(sx, qty=1)
    assert e.value.code == "M_VERTEX"


# ---- A8: flags are durable ----

def test_flags_survive_restart(tmp_path):
    sx = build(str(tmp_path))
    sx["flags"].exec = "DIVERGENT"
    sx["gw"]._persist()
    sx2 = build(str(tmp_path))
    assert sx2["flags"].exec == "DIVERGENT"
    assert "NEW_RISK" not in sx2["flags"].permissions()


# ---- A9: timeout is unresolved, never terminal ----

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


# ---- preserved original guarantees ----

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
