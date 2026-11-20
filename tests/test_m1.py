import pytest
from quant_m1.core import WorkingOrder
from quant_m1.authority import Intent, Reject
from quant_m1.venue import ExecEvent
from quant_m1.system import build


@pytest.fixture
def sx(tmp_path):
    return build(str(tmp_path))


def authorize(sx, qty=10, price=10_000, cost=10):
    return sx["kernel"].authorize(
        sx["ledger"],
        Intent("C", "BUY", qty, price, cost),
        sx["flags"],
        sx["clock"],
        False
    )


def test_invalid_intent_is_rejected_before_reservation(sx):
    before = len(sx["ledger"].state.working)
    with pytest.raises(Reject, match="BAD_PRICE"):
        sx["kernel"].authorize(
            sx["ledger"],
            Intent("C", "BUY", 1, 0, 10),
            sx["flags"],
            sx["clock"],
            False
        )
    assert len(sx["ledger"].state.working) == before


def test_cancel_ack_without_snapshot_does_not_release_reservation(sx):
    tok = authorize(sx, qty=20)
    intent = Intent("C", "BUY", 20, 10_000, 10)
    sx["gw"].submit(tok, intent)
    reserved_before = dict(sx["ledger"].reserved)

    ev = sx["venue"].cancel_ack(tok.order_id)
    sx["venue"].snapshot_available = False

    assert sx["gw"].process(ev) == "CANCEL_UNCONFIRMED_RESERVED"
    assert sx["ledger"].reserved == reserved_before

    order = next(
        o for o in sx["ledger"].state.working
        if o.order_id == tok.order_id
    )
    assert order.status != "CANCELLED"


def test_cancel_ack_reconciles_fill_that_arrived_before_cancel(sx):
    tok = authorize(sx, qty=20)
    intent = Intent("C", "BUY", 20, 10_000, 10)
    sx["gw"].submit(tok, intent)

    sx["venue"].fill(tok.order_id, 5, price=10_100)
    cancel = sx["venue"].cancel_ack(tok.order_id)

    assert sx["gw"].process(cancel) == "APPLIED"
    assert sx["ledger"].state.positions["C"] == 5

    order = next(
        o for o in sx["ledger"].state.working
        if o.order_id == tok.order_id
    )
    assert order.cum_filled == 5
    assert order.status == "CANCELLED"
    assert order.filled_notional == 5 * 10_100


def test_execution_price_not_limit_price_drives_cash(sx):
    tok = authorize(sx, qty=10, price=10_000, cost=25)
    intent = Intent("C", "BUY", 10, 10_000, 25)
    sx["gw"].submit(tok, intent)

    before = sx["ledger"].state.cash_settled
    ev = sx["venue"].fill(tok.order_id, 4, price=10_250)

    assert sx["gw"].process(ev) == "APPLIED"

    from quant_m1.core import ceil_bps
    expected = before - 4 * 10_250 - ceil_bps(4 * 10_250, 25)
    assert sx["ledger"].state.cash_settled == expected


def test_recovery_does_not_infer_rejection_from_missing_order(sx):
    tok = authorize(sx, qty=10)
    intent = Intent("C", "BUY", 10, 10_000, 10)

    sx["gw"].fail_point = "CRASH_A"
    with pytest.raises(RuntimeError):
        sx["gw"].submit(tok, intent)

    sx["gw"].fail_point = None
    result = sx["gw"].recover()

    assert result == "BLOCKED_UNRESOLVED"
    order = next(
        o for o in sx["ledger"].state.working
        if o.order_id == tok.order_id
    )
    assert order.status == "SUBMITTING"


def test_conflicting_execution_id_payload_blocks(sx):
    tok = authorize(sx, qty=10)
    intent = Intent("C", "BUY", 10, 10_000, 10)
    sx["gw"].submit(tok, intent)

    ev = sx["venue"].fill(tok.order_id, 2, price=10_000)
    assert sx["gw"].process(ev) == "APPLIED"

    conflict = ExecEvent(
        ev.exec_id,
        ev.order_id,
        "FILL",
        ev.cum_qty,
        ev.price + 1,
        ev.seq
    )

    assert sx["gw"].process(conflict) == "EXEC_ID_PAYLOAD_CONFLICT"
    assert "NEW_RISK" not in sx["flags"].permissions()


def test_failed_recompute_does_not_zero_existing_reservations(sx, monkeypatch):
    authorize(sx, qty=10)
    old = dict(sx["ledger"].reserved)

    import quant_m1.authority as authority
    monkeypatch.setattr(
        authority,
        "evaluate",
        lambda *a, **k: type(
            "Bad", (), {"ok": False, "reason": "INJECTED"}
        )()
    )

    sx["gw"]._recompute()

    assert sx["flags"].envelope == "OVERFLOW"
    assert sx["ledger"].reserved == old


def test_shared_mark_state_is_not_global(sx):
    from dataclasses import replace
    from quant_m1.risk import simulate

    state = sx["ledger"].state
    order = WorkingOrder(
        "x", "C", "BUY", 1, 0, 10_000, 0,
        "RESERVED", 100, "n"
    )

    a = simulate(
        state, (order,), (0,), 1,
        sx["instruments"], 1
    )

    altered = replace(
        state,
        marks={**state.marks, "A": state.marks["A"] * 2}
    )

    b = simulate(
        altered, (order,), (0,), 1,
        sx["instruments"], 1
    )

    assert a.marks["A"] != b.marks["A"]