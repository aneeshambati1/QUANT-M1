"""Red-team detection tests. Each targets a confirmed finding at HEAD 1a86e1d.

Q  — BUY transmitted after NEW_RISK revoked (reachable with no file tampering).
S  — EventLog not restored: spurious latch + audit-chain truncation on restart.
"""
from __future__ import annotations

import pytest

from quant_m1.authority import Intent, Reject
from quant_m1.system import build


def _auth(sx, symbol="C", side="BUY", qty=5, price=10_000, cost=10):
    intent = Intent(symbol, side, qty, price, cost)
    tok = sx["kernel"].authorize(sx["ledger"], intent, sx["flags"], sx["clock"], False)
    return tok, intent


# ---- Q: permission re-check at transmission ----

@pytest.mark.parametrize("field,value", [
    ("market", "RESTRICTED"),
    ("data", "STALE"),
    ("model", "DEGRADED"),
    ("envelope", "OVERFLOW"),
])
def test_buy_submit_blocked_after_new_risk_revoked(tmp_path, field, value):
    """flags.py removes NEW_RISK but keeps SUBMIT for these degradations.
    submit() must re-check, because flags are not in the token's state digest."""
    sx = build(str(tmp_path))
    tok, intent = _auth(sx, qty=5)
    assert "NEW_RISK" in sx["flags"].permissions()
    setattr(sx["flags"], field, value)
    perms = sx["flags"].permissions()
    assert "NEW_RISK" not in perms
    assert "SUBMIT" in perms, "fixture must keep SUBMIT so only NEW_RISK is missing"
    with pytest.raises(Reject) as e:
        sx["gw"].submit(tok, intent)
    assert e.value.code == "MODE_BLOCKS_NEW_RISK"


def test_sell_still_allowed_when_only_new_risk_revoked(tmp_path):
    """Risk-reducing SELL must remain submittable under a NEW_RISK revoke."""
    sx = build(str(tmp_path))
    tok, intent = _auth(sx, symbol="A", side="SELL", qty=5, cost=0)
    sx["flags"].market = "RESTRICTED"
    sx["gw"].submit(tok, intent)  # must not raise


def test_submit_blocked_on_stale_marks(tmp_path):
    """Data freshness must be re-checked at transmission, not only at authorize."""
    sx = build(str(tmp_path))
    tok, intent = _auth(sx, qty=5)
    sx["clock"].advance_wall(sx["cfg"].mandate.tau_data_secs + 1)
    with pytest.raises(Reject) as e:
        sx["gw"].submit(tok, intent)
    assert e.value.code == "STALE_MARKS"


# ---- S: audit chain survives restart ----

def test_restart_after_trading_does_not_latch(tmp_path):
    """HEAD latches AUDIT_HEAD_MISMATCH because the log is never restored."""
    sx = build(str(tmp_path))
    tok, intent = _auth(sx, qty=5)
    sx["gw"].submit(tok, intent)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 5, price=10_000))
    n_before = len(sx["log"].entries())
    assert n_before > 0

    sx2 = build(str(tmp_path))
    assert sx2["gw"]._divergence_latched is False, "spurious latch on clean restart"
    assert sx2["flags"].exec == "OK"
    assert "NEW_RISK" in sx2["flags"].permissions()


def test_audit_chain_not_truncated_on_restart(tmp_path):
    """HEAD's latch append writes a 1-element chain over events.json."""
    sx = build(str(tmp_path))
    tok, intent = _auth(sx, qty=5)
    sx["gw"].submit(tok, intent)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 5, price=10_000))
    before = sx["log"].entries()
    assert len(before) >= 2

    sx2 = build(str(tmp_path))
    after = sx2["log"].entries()
    assert len(after) >= len(before), "audit history lost across restart"
    assert [r["hash"] for r in after[:len(before)]] == [r["hash"] for r in before]
    assert sx2["log"].verify()


def test_latch_survives_restart_for_the_right_reason(tmp_path):
    """The existing roundtrip test passes even under a spurious latch.
    This one distinguishes: no AUDIT_HEAD fault may be recorded."""
    sx = build(str(tmp_path))
    tok, intent = _auth(sx, qty=5)
    sx["gw"].submit(tok, intent)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 5, price=10_000))
    sx["gw"].latch_divergence("GENUINE_TEST_LATCH")
    sx["gw"]._persist()

    sx2 = build(str(tmp_path))
    assert sx2["gw"]._divergence_latched is True
    assert sx2["flags"].exec == "DIVERGENT"
    assert getattr(sx2["gw"], "_restore_fault", None) is None, \
        "latch must come from the persisted flag, not a restore fault"
    kinds = [r["kind"] for r in sx2["log"].entries()]
    assert "GENUINE_TEST_LATCH" in str(sx2["log"].entries())
    assert kinds.count("DIVERGENCE_LATCHED") == 1


def test_lost_audit_event_is_detected(tmp_path):
    """If the chain has LOST a recorded event, that IS a divergence."""
    sx = build(str(tmp_path))
    tok, intent = _auth(sx, qty=5)
    sx["gw"].submit(tok, intent)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 5, price=10_000))
    # Simulate audit loss: truncate events.json to the genesis-free empty chain.
    sx["log"].store.save([])

    sx2 = build(str(tmp_path))
    assert sx2["gw"]._divergence_latched is True
    assert getattr(sx2["gw"], "_restore_fault", None) == "AUDIT_HEAD_MISSING_FROM_CHAIN"