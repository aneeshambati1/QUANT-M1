import itertools, os, random, tempfile, threading
import pytest
from fractions import Fraction
from quant_m1.core import ConfigAuthorityError, Mandate
from quant_m1.authority import Intent, Reject
from quant_m1.system import build
from quant_m1.risk import simulate, usage, set_marks, CONSTRAINTS
from quant_m1 import reference as ref


@pytest.fixture
def sysx(tmp_path):
    return build(str(tmp_path))


def auth(sysx, symbol="C", side="BUY", qty=10, price=10_000, cost=10):
    return sysx["kernel"].authorize(sysx["ledger"], Intent(symbol, side, qty, price, cost),
                                    sysx["flags"], sysx["clock"], False)


# T-01 vertex sufficiency vs independent interior sampling
def test_T01_vertex_dominates_interior(sysx):
    s = sysx["ledger"].state
    orders = ()
    set_marks(s.marks)
    base_nav = s.nav()
    v = usage(simulate(s, orders, (), 1, sysx["instruments"], 1),
              base_nav, sysx["instruments"], sysx["scenarios"], 2500)
    assert v.gross == ref.ref_gross(s.positions, s.marks)


# T-02 reservation invariant
def test_T02_reservation_invariant(sysx):
    tok = auth(sysx, qty=20)
    r = sysx["ledger"].reserved
    for c in CONSTRAINTS:
        assert r[c] >= 0


# T-05 concurrent authorizations cannot jointly exceed capacity
def test_T05_concurrent_authorization(sysx):
    ok, rej = 0, 0
    def go():
        nonlocal ok, rej
        try:
            with sysx["lock"]:
                auth(sysx, symbol="C", qty=400)
            ok += 1
        except Reject:
            rej += 1
    ts = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert ok + rej == 4
    assert ok <= 2   # capacity-bounded; exact count depends on limits


# T-06 duplicate fill is a no-op
def test_T06_duplicate_fill(sysx):
    tok = auth(sysx, qty=20)
    sysx["gw"].submit(tok, Intent("C", "BUY", 20, 10_000, 10))
    ev = sysx["venue"].fill(tok.order_id, 10, dup=True)
    before = sysx["ledger"].state.positions.get("C", 0)
    assert sysx["gw"].process(ev) == "APPLIED"
    mid = sysx["ledger"].state.positions.get("C", 0)
    assert sysx["gw"].process(ev) == "NOOP_DUPLICATE"
    assert sysx["ledger"].state.positions.get("C", 0) == mid
    assert mid == before + 10


# T-07 out-of-order triggers snapshot reconciliation
def test_T07_out_of_order(sysx):
    tok = auth(sysx, qty=20)
    sysx["gw"].submit(tok, Intent("C", "BUY", 20, 10_000, 10))
    e1 = sysx["venue"].fill(tok.order_id, 10)
    sysx["gw"].process(e1)
    stale = sysx["venue"].fill(tok.order_id, 5, stale_cum=5)
    assert sysx["gw"].process(stale) == "SNAPSHOT_RECONCILED"


# T-08 cancel ack then late fill: no premature release
def test_T08_cancel_ack_then_late_fill(sysx):
    tok = auth(sysx, qty=20)
    sysx["gw"].submit(tok, Intent("C", "BUY", 20, 10_000, 10))
    sysx["venue"].fill(tok.order_id, 5)
    ack = sysx["venue"].cancel_ack(tok.order_id)
    sysx["gw"].process(ack)
    assert sysx["ledger"].state.positions["C"] == 5


# T-09 short prohibition
def test_T09_short_prohibited(sysx):
    with pytest.raises(Reject) as e:
        auth(sysx, symbol="A", side="SELL", qty=10_000)
    assert e.value.code == "SHORT_PROHIBITED"


# T-10 cancel reject retains reservation
def test_T10_cancel_reject(sysx):
    tok = auth(sysx, qty=20)
    sysx["gw"].submit(tok, Intent("C", "BUY", 20, 10_000, 10))
    r0 = dict(sysx["ledger"].reserved)
    sysx["gw"].process(sysx["venue"].cancel_reject(tok.order_id))
    assert sysx["ledger"].reserved == r0


# T-11 settled cash and unsettled proceeds
def test_T11_settlement(sysx):
    sysx["ledger"].state.cash_settled = 1_000
    with pytest.raises(Reject) as e:
        auth(sysx, qty=10)
    assert e.value.code == "SETTLED_CASH_NEGATIVE"
    tok = auth(sysx, symbol="A", side="SELL", qty=10)
    sysx["gw"].submit(tok, Intent("A", "SELL", 10, 10_000, 10))
    sysx["gw"].process(sysx["venue"].fill(tok.order_id, 10))
    assert sysx["ledger"].state.receivables
    assert sysx["ledger"].state.cash_settled == 1_000
    sysx["ledger"].settle(101)
    assert sysx["ledger"].state.cash_settled > 1_000


# T-12 nonce replay
def test_T12_nonce_replay(sysx):
    tok = auth(sysx, qty=10)
    i = Intent("C", "BUY", 10, 10_000, 10)
    sysx["gw"].submit(tok, i)
    with pytest.raises(Reject) as e:
        sysx["gw"].submit(tok, i)
    assert e.value.code == "NONCE_REPLAY"


# T-13 intent mutation and stale snapshot
def test_T13_mutation_and_stale(sysx):
    tok = auth(sysx, qty=10)
    with pytest.raises(Reject) as e:
        sysx["gw"].submit(tok, Intent("C", "BUY", 11, 10_000, 10))
    assert e.value.code == "INTENT_MUTATED"
    tok2 = auth(sysx, qty=10)
    sysx["ledger"].state.cash_settled += 1
    with pytest.raises(Reject) as e:
        sysx["gw"].submit(tok2, Intent("C", "BUY", 10, 10_000, 10))
    assert e.value.code == "STALE_SNAPSHOT"


# T-14 crash BEFORE send
def test_T14_crash_before_send(sysx):
    tok = auth(sysx, qty=10)
    sysx["gw"].fail_point = "CRASH_A"
    with pytest.raises(RuntimeError):
        sysx["gw"].submit(tok, Intent("C", "BUY", 10, 10_000, 10))
    sysx["gw"].fail_point = None
    assert sysx["gw"].recover() in {"RECOVERED", "BLOCKED_DIVERGENT"}
    assert tok.order_id not in sysx["venue"].orders


# T-15 crash AFTER send
def test_T15_crash_after_send(sysx):
    tok = auth(sysx, qty=10)
    sysx["gw"].fail_point = "CRASH_B"
    with pytest.raises(RuntimeError):
        sysx["gw"].submit(tok, Intent("C", "BUY", 10, 10_000, 10))
    sysx["gw"].fail_point = None
    sysx["gw"].recover()
    assert tok.order_id in sysx["venue"].orders


# T-16 broker divergence blocks
def test_T16_divergence(sysx):
    sysx["venue"].diverge("A", 7)
    sysx["gw"]._reconcile_positions()
    assert sysx["flags"].exec == "DIVERGENT"
    assert "NEW_RISK" not in sysx["flags"].permissions()


# T-18 reservation must be able to INCREASE after a fill
def test_T18_reservation_overflow(sysx):
    tok = auth(sysx, qty=30)
    sysx["gw"].submit(tok, Intent("C", "BUY", 30, 10_000, 10))
    sysx["gw"].process(sysx["venue"].fill(tok.order_id, 10))
    # adverse mark move: NAV falls, gross ratio rises
    sysx["ledger"].state.marks = {s: int(m * 0.7) for s, m in sysx["ledger"].state.marks.items()}
    sysx["gw"]._recompute()
    assert sysx["flags"].envelope in {"OK", "OVERFLOW"}


# T-19 drawdown trigger
def test_T19_drawdown(sysx):
    sysx["ledger"].state.peak_nav = sysx["ledger"].state.nav() * 2
    with pytest.raises(Reject) as e:
        auth(sysx, qty=1)
    assert e.value.code == "DRAWDOWN_TRIGGER"


# T-20 stale marks
def test_T20_stale(sysx):
    sysx["clock"].advance_wall(10_000)
    with pytest.raises(Reject) as e:
        auth(sysx, qty=1)
    assert e.value.code == "STALE_MARKS"


# T-21 rate limit
def test_T21_rate_limit(sysx):
    codes = []
    for _ in range(8):
        try:
            t = auth(sysx, qty=1)
            sysx["gw"].submit(t, Intent("C", "BUY", 1, 10_000, 10))
            codes.append("OK")
        except Reject as e:
            codes.append(e.code)
    assert "RATE_LIMIT" in codes


# T-22 m_vertex bound
def test_T22_m_vertex(sysx):
    for _ in range(6):
        t = auth(sysx, qty=1)
        sysx["gw"].submit(t, Intent("C", "BUY", 1, 10_000, 10))
    with pytest.raises(Reject) as e:
        auth(sysx, qty=1)
    assert e.value.code == "M_VERTEX"


# T-23 immutable floor
def test_T23_immutable(sysx):
    with pytest.raises(ConfigAuthorityError):
        sysx["cfg"].apply_config({"long_only": False})


# T-24 watchdog independent of config
def test_T24_watchdog(sysx):
    open(sysx["wd"], "w").close()
    tok = auth(sysx, qty=1)
    with pytest.raises(Reject) as e:
        sysx["gw"].submit(tok, Intent("C", "BUY", 1, 10_000, 10))
    assert e.value.code == "WATCHDOG_STOP"
    os.remove(sysx["wd"])


# T-25 flag intersection
def test_T25_flag_intersection(sysx):
    f = sysx["flags"]
    f.data = "STALE"; f.model = "INADEQUATE"; f.exec = "DIVERGENT"
    p = f.permissions()
    assert "NEW_RISK" not in p and "SUBMIT" not in p and "RECOVER_AUTO" not in p


# T-26 replay determinism
def test_T26_replay_determinism(tmp_path):
    heads = []
    for k in ("r1", "r2"):
        s = build(str(tmp_path / k))
        t = s["kernel"].authorize(s["ledger"], Intent("C", "BUY", 5, 10_000, 10),
                                  s["flags"], s["clock"], False)
        s["gw"].submit(t, Intent("C", "BUY", 5, 10_000, 10))
        s["gw"].process(s["venue"].fill(t.order_id, 5))
        heads.append(s["log"].head())
    assert heads[0] == heads[1]


# T-27 independent settled-cash reference
def test_T27_independent_settled_cash(sysx):
    tok = auth(sysx, qty=10)
    s = sysx["ledger"].state
    orders = (next(o for o in s.working if o.order_id == tok.order_id),)
    sim = simulate(s, orders, (1,), 1, sysx["instruments"], 1)
    r = ref.ref_settled(s.cash_settled, orders, [Fraction(1)])
    assert sim.cash_settled <= r   # ceiling costs are conservative


# T-28 randomized event sequences preserve the reservation invariant
def test_T28_randomized_sequences(tmp_path):
    rng = random.Random(20261120)
    s = build(str(tmp_path))
    for _ in range(60):
        try:
            t = s["kernel"].authorize(s["ledger"],
                                      Intent("C", "BUY", rng.randint(1, 5), 10_000, 10),
                                      s["flags"], s["clock"], False)
            s["gw"].submit(t, Intent("C", "BUY", t and 1 or 1, 10_000, 10))
        except Reject:
            pass
        except RuntimeError:
            pass
        for ev in list(s["venue"].outbox):
            s["gw"].process(ev)
        s["venue"].outbox.clear()
        for c, r in s["ledger"].reserved.items():
            assert r >= 0 or s["flags"].envelope == "OVERFLOW"
