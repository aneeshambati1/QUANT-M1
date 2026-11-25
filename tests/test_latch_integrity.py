"""B15 detection tests: the divergence latch must not be bypassable.

Each test targets a specific bypass path confirmed by static analysis.
"""
from __future__ import annotations

import os

import pytest

from quant_m1.authority import Intent, Reject, operator_clear_body, sign
from quant_m1.system import build


def _auth(sx, symbol="C", side="BUY", qty=5, price=10_000, cost=10):
    intent = Intent(symbol, side, qty, price, cost)
    tok = sx["kernel"].authorize(sx["ledger"], intent, sx["flags"], sx["clock"], False)
    return tok, intent


def _trade(sx):
    tok, intent = _auth(sx)
    sx["gw"].submit(tok, intent)
    sx["gw"].process(sx["venue"].fill(tok.order_id, 5, price=10_000))


# ---- deletion bypass ----

def test_deleting_composite_does_not_clear_latch(tmp_path):
    """Audit chain records the latch; deleting the composite must not erase it."""
    sx = build(str(tmp_path))
    _trade(sx)
    sx["gw"].latch_divergence("GENUINE_FAULT")
    sx["gw"]._persist()
    assert sx["gw"]._divergence_latched is True

    os.remove(sx["gw"].composite_store.path)

    sx2 = build(str(tmp_path))
    assert sx2["gw"]._divergence_latched is True
    assert sx2["flags"].exec == "DIVERGENT"
    assert "NEW_RISK" not in sx2["flags"].permissions()


def test_missing_composite_with_history_is_integrity_fault(tmp_path):
    """Prior activity plus no composite is a fault, not a clean first start."""
    sx = build(str(tmp_path))
    _trade(sx)
    os.remove(sx["gw"].composite_store.path)

    sx2 = build(str(tmp_path))
    assert sx2["gw"]._restore_fault == "COMPOSITE_MISSING_WITH_AUDIT_HISTORY"


def test_first_start_with_no_history_is_clean(tmp_path):
    """A genuinely fresh directory must not latch."""
    sx = build(str(tmp_path))
    assert sx["gw"]._divergence_latched is False
    assert sx["gw"]._restore_fault is None


def test_composite_cannot_override_audit_latch(tmp_path):
    """Tamper the composite to claim a clear latch; audit must win."""
    sx = build(str(tmp_path))
    _trade(sx)
    sx["gw"].latch_divergence("GENUINE_FAULT")
    sx["gw"]._persist()

    d = sx["gw"].composite_store.load()
    d["divergence_latched"] = False
    d["flags"]["exec"] = "OK"
    sx["gw"].composite_store.save(d)

    sx2 = build(str(tmp_path))
    assert sx2["gw"]._divergence_latched is True
    assert sx2["gw"]._restore_fault == "COMPOSITE_LATCH_DISAGREES_WITH_AUDIT"


# ---- unauthenticated clear ----

def test_clear_without_signature_rejected(tmp_path):
    sx = build(str(tmp_path))
    sx["gw"].latch_divergence("FAULT")
    with pytest.raises(Reject) as e:
        sx["gw"].clear_divergence("because I said so")
    assert e.value.code == "OPERATOR_AUTH_REQUIRED"
    assert sx["gw"]._divergence_latched is True


def test_clear_with_empty_reason_rejected(tmp_path):
    sx = build(str(tmp_path))
    sx["gw"].latch_divergence("FAULT")
    with pytest.raises(Reject) as e:
        sx["gw"].clear_divergence("   ")
    assert e.value.code == "OPERATOR_REASON_REQUIRED"


def test_clear_with_bad_signature_rejected(tmp_path):
    sx = build(str(tmp_path))
    sx["gw"].latch_divergence("FAULT")
    gw = sx["gw"]
    with pytest.raises(Reject) as e:
        gw.clear_divergence("resolved", nonce="n1", seq=sx["clock"].seq,
                            operator_sig="deadbeef" * 8)
    assert e.value.code == "OPERATOR_BAD_SIGNATURE"
    assert gw._divergence_latched is True


def test_clear_with_stale_seq_rejected(tmp_path):
    sx = build(str(tmp_path))
    gw = sx["gw"]
    gw.latch_divergence("FAULT")
    stale = sx["clock"].seq
    sx["clock"].next()
    sig = gw.operator_clear_signature("resolved", "n1", stale)
    with pytest.raises(Reject) as e:
        gw.clear_divergence("resolved", nonce="n1", seq=stale, operator_sig=sig)
    assert e.value.code == "OPERATOR_SEQ_STALE"


def test_authenticated_clear_succeeds(tmp_path):
    sx = build(str(tmp_path))
    gw = sx["gw"]
    gw.latch_divergence("FAULT")
    seq = sx["clock"].seq
    sig = gw.operator_clear_signature("reconciled with broker snapshot", "op-1", seq)
    gw.clear_divergence("reconciled with broker snapshot", nonce="op-1",
                        seq=seq, operator_sig=sig)
    assert gw._divergence_latched is False
    assert gw.flags.exec == "OK"
    kinds = [r["kind"] for r in sx["log"].entries()]
    assert kinds.count("DIVERGENCE_CLEARED") == 1


def test_operator_nonce_replay_rejected(tmp_path):
    sx = build(str(tmp_path))
    gw = sx["gw"]
    gw.latch_divergence("FAULT")
    seq = sx["clock"].seq
    sig = gw.operator_clear_signature("first", "op-1", seq)
    gw.clear_divergence("first", nonce="op-1", seq=seq, operator_sig=sig)

    gw.latch_divergence("SECOND_FAULT")
    seq2 = sx["clock"].seq
    sig2 = gw.operator_clear_signature("second", "op-1", seq2)
    with pytest.raises(Reject) as e:
        gw.clear_divergence("second", nonce="op-1", seq=seq2, operator_sig=sig2)
    assert e.value.code == "OPERATOR_NONCE_REPLAY"
    assert gw._divergence_latched is True


def test_reason_tampering_invalidates_signature(tmp_path):
    sx = build(str(tmp_path))
    gw = sx["gw"]
    gw.latch_divergence("FAULT")
    seq = sx["clock"].seq
    sig = gw.operator_clear_signature("approved reason", "op-1", seq)
    with pytest.raises(Reject) as e:
        gw.clear_divergence("different reason", nonce="op-1", seq=seq,
                            operator_sig=sig)
    assert e.value.code == "OPERATOR_BAD_SIGNATURE"


def test_cleared_latch_survives_restart(tmp_path):
    """After an authenticated clear, restart must not re-latch from history."""
    sx = build(str(tmp_path))
    gw = sx["gw"]
    _trade(sx)
    gw.latch_divergence("FAULT")
    seq = sx["clock"].seq
    sig = gw.operator_clear_signature("resolved", "op-1", seq)
    gw.clear_divergence("resolved", nonce="op-1", seq=seq, operator_sig=sig)
    gw._persist()

    sx2 = build(str(tmp_path))
    assert sx2["gw"]._divergence_latched is False
    assert sx2["gw"]._restore_fault is None


def test_operator_body_is_canonical():
    b = operator_clear_body("r", "n", 7)
    assert b == {"action": "CLEAR_DIVERGENCE", "reason": "r", "nonce": "n", "seq": 7}
    s1 = sign(b"k", b)
    s2 = sign(b"k", operator_clear_body("r", "n", 7))
    assert s1 == s2