"""M1.2 core/venue/risk/flags/reference acceptance tests.

Runs WITHOUT authority.py. Proves the invariants that were previously untested.

    python -m pytest tests/test_m1_core.py -q
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from quant_m1.core import (
    BPS, Clock, ConfigAuthority, ConfigAuthorityError, EventLog, ImmutableFloor,
    InvariantError, Instrument, Ledger, Mandate, Operational, Store, VerifiedState,
    WorkingOrder, ceil_bps,
)
from quant_m1.flags import Flags, Monitor, may_auto_recover
from quant_m1.reference import (
    ref_gross, ref_gross_abs, ref_receivable, ref_settled, ref_settled_ceil,
)
from quant_m1.risk import evaluate, reservations, simulate
from quant_m1.venue import MockVenue


def _instruments():
    return {
        "A": Instrument("A", "PLAIN_ETF", 50_000_000, 10_000,
                        (("ISS1", 6000), ("ISS2", 4000))),
        "B": Instrument("B", "PLAIN_ETF", 50_000_000, 8_000,
                        (("ISS1", 5000), ("ISS3", 5000))),
        "C": Instrument("C", "PLAIN_ETF", 50_000_000, 9_000, None),
    }


def _ledger(tmp_path, cash=10_000_000, positions=None, marks=None):
    positions = positions if positions is not None else {"A": 300, "B": 300}
    marks = marks or {"A": 10_000, "B": 10_000, "C": 10_000}
    cfg = ConfigAuthority(ImmutableFloor(), Mandate(version=1), Operational())
    nav = cash + sum(q * marks[s] for s, q in positions.items())
    st = VerifiedState(
        0, 100, cash, (), dict(positions), dict(marks),
        1_000_000, (), nav, 1, cfg.mandate.digest(),
    )
    return Ledger(Store(str(tmp_path / "ledger.json")), st), cfg, marks


def _scenarios(marks):
    return (
        {s: int(m * 0.90) for s, m in marks.items()},
        {s: int(m * 0.95) for s, m in marks.items()},
        {s: int(m * 1.02) for s, m in marks.items()},
        {s: int(m * 1.05) for s, m in marks.items()},
    )


# ---------------- T+1: the invariant that was never tested ----------------

def test_sell_receivable_never_touches_settled_cash(tmp_path):
    led, _, _ = _ledger(tmp_path, cash=0, positions={"A": 10})
    before = led.state.cash_settled
    due = led.record_sell_receivable(trade_date=100, amount=100_000, settle_lag_days=1)
    assert due == 101
    assert led.state.cash_settled == before == 0
    assert led.state.cash_unsettled == 100_000
    assert led.state.nav() == 100_000 + 10 * 10_000


def test_settle_same_session_releases_nothing(tmp_path):
    led, _, _ = _ledger(tmp_path, cash=0, positions={"A": 10})
    led.record_sell_receivable(100, 100_000, 1)
    assert led.settle(100) == 0
    assert led.state.cash_settled == 0
    assert led.state.cash_unsettled == 100_000


def test_settle_next_session_releases(tmp_path):
    led, _, _ = _ledger(tmp_path, cash=0, positions={"A": 10})
    led.record_sell_receivable(100, 100_000, 1)
    assert led.settle(101) == 100_000
    assert led.state.cash_settled == 100_000
    assert led.state.cash_unsettled == 0
    assert led.settle(101) == 0  # idempotent


def test_settle_lag_zero_rejected(tmp_path):
    led, _, _ = _ledger(tmp_path)
    with pytest.raises(ValueError):
        led.record_sell_receivable(100, 100_000, 0)
    with pytest.raises(ConfigAuthorityError):
        Mandate(version=1, settle_lag_days=0)


def test_simulate_sell_adds_receivable_not_cash(tmp_path):
    led, cfg, marks = _ledger(tmp_path, cash=0, positions={"A": 10})
    o = WorkingOrder("o1", "A", "SELL", 10, 0, 10_000, 25, "RESERVED", 100, "n")
    sim = simulate(led.state, (o,), (1,), 1, _instruments(), cfg.mandate.settle_lag_days)
    assert sim.cash_settled == 0
    assert sim.receivable_total == 100_000 - ceil_bps(100_000, 25)
    assert sim.positions["A"] == 0


def test_simulate_rejects_t0_settle_lag(tmp_path):
    led, _, _ = _ledger(tmp_path)
    with pytest.raises(ValueError, match="settle_lag"):
        simulate(led.state, (), (), 1, _instruments(), 0)


def test_reference_agrees_sells_never_settle(tmp_path):
    led, _, _ = _ledger(tmp_path, cash=0, positions={"A": 10})
    o = WorkingOrder("o1", "A", "SELL", 10, 0, 10_000, 25, "RESERVED", 100, "n")
    assert ref_settled(0, (o,), (1,)) == Fraction(0)
    assert ref_receivable(0, (o,), (1,)) > 0


# ---------------- immutability / audit ----------------

def test_snapshot_positions_are_read_only(tmp_path):
    led, _, _ = _ledger(tmp_path)
    snap = led.snapshot()
    with pytest.raises(TypeError):
        snap.positions["A"] = 999_999
    with pytest.raises(TypeError):
        snap.marks["A"] = 1
    assert led.state.positions["A"] == 300


def test_snapshot_working_is_deep_copied(tmp_path):
    led, _, _ = _ledger(tmp_path)
    o = WorkingOrder("o1", "C", "BUY", 10, 0, 10_000, 10, "RESERVED", 100, "n")
    led.state.working = (o,)
    snap = led.snapshot()
    snap.working[0].cum_filled = 10
    assert led.state.working[0].cum_filled == 0


def test_digest_changes_with_mark_wall(tmp_path):
    led, _, _ = _ledger(tmp_path)
    d1 = led.state.digest()
    led.state.mark_wall += 1
    assert led.state.digest() != d1


def test_event_log_tamper_detected(tmp_path):
    log = EventLog(Store(str(tmp_path / "events.json")))
    log.append("A", {"x": 1}, 1)
    log.append("B", {"y": 2}, 2)
    assert log.verify()
    log.chain[0]["payload"]["x"] = 999
    assert not log.verify()
    with pytest.raises(ValueError):
        log.restore()


def test_event_log_entries_immutable(tmp_path):
    log = EventLog(Store(str(tmp_path / "e.json")))
    log.append("A", {"x": 1}, 1)
    entries = log.entries()
    entries[0]["payload"]["x"] = 999
    assert log.chain[0]["payload"]["x"] == 1
    assert log.verify()


def test_event_log_redacts_secrets(tmp_path):
    log = EventLog(Store(str(tmp_path / "e.json")))
    log.append("AUTH", {"api_key": "SUPERSECRET", "ok": 1}, 1)
    assert log.chain[0]["payload"]["api_key"] == "[REDACTED]"
    assert log.verify()


# ---------------- config authority ----------------

def test_immutable_floor_change_rejected():
    cfg = ConfigAuthority(ImmutableFloor(), Mandate(version=1), Operational())
    with pytest.raises(ConfigAuthorityError):
        cfg.apply_config({"long_only": False})
    assert cfg.attested()


def test_mandate_range_validated():
    cfg = ConfigAuthority(ImmutableFloor(), Mandate(version=1), Operational())
    with pytest.raises(ConfigAuthorityError):
        cfg.apply_config({"mandate": {"gross_bps": 999_999}})
    with pytest.raises(ConfigAuthorityError):
        cfg.apply_config({"mandate": {"cvar_q_bps": 0}})


def test_unattested_config_rejected():
    cfg = ConfigAuthority(ImmutableFloor(), Mandate(version=1), Operational())
    cfg._floor_digest = "0" * 64
    with pytest.raises(ConfigAuthorityError):
        cfg.apply_config({"mandate": {"gross_bps": 9_000}})


# ---------------- risk config honoring ----------------

def test_evaluate_honors_m_vertex(tmp_path):
    led, cfg, marks = _ledger(tmp_path)
    orders = tuple(
        WorkingOrder(f"o{i}", "C", "BUY", 1, 0, 10_000, 10, "RESERVED", 100, f"n{i}")
        for i in range(3)
    )
    sc = _scenarios(marks)
    assert evaluate(led.state, orders, _instruments(), cfg.mandate, sc,
                    m_vertex=2).reason == "M_VERTEX_EXCEEDED"
    assert evaluate(led.state, orders, _instruments(), cfg.mandate, sc,
                    m_vertex=6).n_vertices == 8


def test_reservations_refuse_failed_evaluation(tmp_path):
    led, cfg, marks = _ledger(tmp_path)
    ev = evaluate(led.state, (), _instruments(), cfg.mandate, ())
    assert not ev.ok
    with pytest.raises(ValueError):
        reservations(ev)


# ---------------- venue ----------------

def test_exec_event_seq_is_monotonic():
    v = MockVenue(positions={"A": 10}, cash=1_000_000)
    v.send("o1", "A", "BUY", 5, 10_000, 10)
    e1 = v.fill("o1", 2, 10_000)
    e2 = v.fill("o1", 3, 10_000)
    seqs = [e.seq for e in v.outbox]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    assert e2.seq > e1.seq > 0


def test_cancel_ack_does_not_relabel_rejected():
    v = MockVenue()
    v.send("o1", "A", "BUY", 5, 10_000)
    v.reject("o1")
    v.cancel_ack("o1")
    assert v.orders["o1"].status == "REJECTED"


def test_timeout_sets_unknown():
    v = MockVenue()
    v.send("o1", "A", "BUY", 5, 10_000)
    ev = v.timeout("o1")
    assert ev.kind == "TIMEOUT"
    assert v.orders["o1"].status == "UNKNOWN"


def test_venue_cash_is_t0_and_declared():
    v = MockVenue(positions={"A": 10}, cash=0)
    v.send("o1", "A", "SELL", 10, 10_000, 0)
    v.fill("o1", 10, 10_000)
    assert v.cash == 100_000            # venue settles immediately
    assert v.snapshot()["cash_is_t0"] is True


def test_venue_client_order_id_reuse_conflict():
    v = MockVenue()
    v.send("o1", "A", "BUY", 5, 10_000)
    with pytest.raises(ValueError):
        v.send("o1", "A", "BUY", 6, 10_000)


# ---------------- flags ----------------

def test_killed_removes_all_permissions():
    assert Flags(killed=True).permissions() == frozenset()


def test_frozen_blocks_new_risk_and_submit():
    p = Flags(frozen=True).permissions()
    assert "NEW_RISK" not in p and "SUBMIT" not in p


def test_disconnect_blocks_submit_and_recovery():
    p = Flags(exec="DISCONNECTED").permissions()
    assert not ({"NEW_RISK", "SUBMIT", "RECOVER_AUTO"} & p)


def test_auto_recovery_requires_independence():
    ok = Monitor("m", "INDEPENDENT", "INDEPENDENT", "INDEPENDENT",
                 "INDEPENDENT", "INDEPENDENT", "INDEPENDENT")
    shared = Monitor("m", "SHARED:data", "INDEPENDENT", "INDEPENDENT",
                     "INDEPENDENT", "INDEPENDENT", "INDEPENDENT")
    assert may_auto_recover(ok, Flags())
    assert not may_auto_recover(shared, Flags())
    assert not may_auto_recover(ok, Flags(killed=True))


# ---------------- reference vs production ----------------

def test_gross_signed_equals_abs_under_long_only(tmp_path):
    led, _, _ = _ledger(tmp_path)
    assert ref_gross(led.state.positions, led.state.marks) == \
           ref_gross_abs(led.state.positions, led.state.marks)


def test_production_cost_is_conservative_vs_exact(tmp_path):
    led, _, _ = _ledger(tmp_path)
    o = WorkingOrder("o1", "C", "BUY", 3, 0, 10_250, 25, "RESERVED", 100, "n")
    exact = ref_settled(led.state.cash_settled, (o,), (1,))
    prod = ref_settled_ceil(led.state.cash_settled, (o,), (1,))
    assert prod <= exact


# ---------------- working order validation ----------------

def test_overfilled_working_order_rejected():
    with pytest.raises(ValueError):
        WorkingOrder("o", "A", "BUY", 5, 6, 10_000, 0, "FILLED", 100, "n")


def test_invalid_side_rejected():
    with pytest.raises(ValueError):
        WorkingOrder("o", "A", "SHORT", 5, 0, 10_000, 0, "RESERVED", 100, "n")


# ---------------- persistence round trip ----------------

def test_persist_restore_preserves_receivables(tmp_path):
    led, _, _ = _ledger(tmp_path, cash=0, positions={"A": 10})
    led.record_sell_receivable(100, 100_000, 1)
    led.persist()
    back = Ledger.restore(Store(str(tmp_path / "ledger.json")))
    assert back is not None
    assert back.state.cash_unsettled == 100_000
    assert back.state.cash_settled == 0
    assert back.state.digest() == led.state.digest()


def test_verify_invariants_catches_negative_cash(tmp_path):
    led, _, _ = _ledger(tmp_path)
    led.state.cash_settled = -1
    with pytest.raises(InvariantError):
        led.verify_invariants()
