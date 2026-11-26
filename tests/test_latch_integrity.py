"""B15 + P0 detection tests: the divergence latch must not be bypassable.

Operator clearance now uses an INDEPENDENT one-time key chain (OperatorConsole).
The Gateway holds only the hash of the current operator secret and cannot mint
a clear. Every prior latch-integrity invariant is preserved and strengthened.
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


def _clear(sx, reason, nonce):
    cmd = sx["operator"].make_clear(reason, nonce, sx["clock"].seq)
    sx["gw"].clear_divergence(reason, nonce=cmd["nonce"], seq=cmd["seq"],
                              key_reveal=cmd["key_reveal"], next_hash=cmd["next_hash"],
                              sig=cmd["sig"])


def test_deleting_composite_does_not_clear_latch(tmp_path):
    sx = build(str(tmp_path)); _trade(sx)
    sx["gw"].latch_divergence("GENUINE_FAULT"); sx["gw"]._persist()
    os.remove(sx["gw"].composite_store.path)
    sx2 = build(str(tmp_path))
    assert sx2["gw"]._divergence_latched is True
    assert sx2["flags"].exec == "DIVERGENT"
    assert "NEW_RISK" not in sx2["flags"].permissions()


def test_missing_composite_with_history_is_integrity_fault(tmp_path):
    sx = build(str(tmp_path)); _trade(sx)
    os.remove(sx["gw"].composite_store.path)
    sx2 = build(str(tmp_path))
    assert sx2["gw"]._restore_fault == "COMPOSITE_MISSING_WITH_AUDIT_HISTORY"


def test_first_start_with_no_history_is_clean(tmp_path):
    sx = build(str(tmp_path))
    assert sx["gw"]._divergence_latched is False
    assert sx["gw"]._restore_fault is None


def test_composite_cannot_override_audit_latch(tmp_path):
    sx = build(str(tmp_path)); _trade(sx)
    sx["gw"].latch_divergence("GENUINE_FAULT"); sx["gw"]._persist()
    d = sx["gw"].composite_store.load()
    d["divergence_latched"] = False; d["flags"]["exec"] = "OK"
    body = {k: v for k, v in d.items() if k != "composite_sig"}
    d["composite_sig"] = sign(sx["gw"].key, body)
    sx["gw"].composite_store.save(d)
    sx2 = build(str(tmp_path))
    assert sx2["gw"]._divergence_latched is True
    assert sx2["gw"]._restore_fault == "COMPOSITE_LATCH_DISAGREES_WITH_AUDIT"


def test_clear_without_signature_rejected(tmp_path):
    sx = build(str(tmp_path)); sx["gw"].latch_divergence("FAULT")
    with pytest.raises(Reject) as e:
        sx["gw"].clear_divergence("because I said so")
    assert e.value.code == "OPERATOR_AUTH_REQUIRED"
    assert sx["gw"]._divergence_latched is True


def test_clear_with_empty_reason_rejected(tmp_path):
    sx = build(str(tmp_path)); sx["gw"].latch_divergence("FAULT")
    with pytest.raises(Reject) as e:
        sx["gw"].clear_divergence("   ")
    assert e.value.code == "OPERATOR_REASON_REQUIRED"


def test_clear_with_bad_signature_rejected(tmp_path):
    sx = build(str(tmp_path)); sx["gw"].latch_divergence("FAULT")
    cmd = sx["operator"].make_clear("resolved", "n1", sx["clock"].seq)
    with pytest.raises(Reject) as e:
        sx["gw"].clear_divergence("resolved", nonce="n1", seq=cmd["seq"],
                                  key_reveal=cmd["key_reveal"], next_hash=cmd["next_hash"],
                                  sig="deadbeef" * 8)
    assert e.value.code == "OPERATOR_BAD_SIGNATURE"
    assert sx["gw"]._divergence_latched is True


def test_clear_with_stale_seq_rejected(tmp_path):
    sx = build(str(tmp_path)); sx["gw"].latch_divergence("FAULT")
    stale = sx["clock"].seq
    cmd = sx["operator"].make_clear("resolved", "n1", stale)
    sx["clock"].next()
    with pytest.raises(Reject) as e:
        sx["gw"].clear_divergence("resolved", nonce="n1", seq=stale,
                                  key_reveal=cmd["key_reveal"], next_hash=cmd["next_hash"], sig=cmd["sig"])
    assert e.value.code == "OPERATOR_SEQ_STALE"


def test_authenticated_clear_succeeds(tmp_path):
    sx = build(str(tmp_path)); sx["gw"].latch_divergence("FAULT")
    _clear(sx, "reconciled with broker snapshot", "op-1")
    assert sx["gw"]._divergence_latched is False
    assert sx["gw"].flags.exec == "OK"
    kinds = [r["kind"] for r in sx["log"].entries()]
    assert kinds.count("DIVERGENCE_CLEARED") == 1


def test_operator_nonce_replay_rejected(tmp_path):
    sx = build(str(tmp_path)); sx["gw"].latch_divergence("FAULT")
    _clear(sx, "first", "op-1")
    sx["gw"].latch_divergence("SECOND_FAULT")
    cmd = sx["operator"].make_clear("second", "op-1", sx["clock"].seq)
    with pytest.raises(Reject) as e:
        sx["gw"].clear_divergence("second", nonce="op-1", seq=cmd["seq"],
                                  key_reveal=cmd["key_reveal"], next_hash=cmd["next_hash"], sig=cmd["sig"])
    assert e.value.code == "OPERATOR_NONCE_REPLAY"
    assert sx["gw"]._divergence_latched is True


def test_reason_tampering_invalidates_signature(tmp_path):
    sx = build(str(tmp_path)); sx["gw"].latch_divergence("FAULT")
    cmd = sx["operator"].make_clear("approved reason", "op-1", sx["clock"].seq)
    with pytest.raises(Reject) as e:
        sx["gw"].clear_divergence("different reason", nonce="op-1", seq=cmd["seq"],
                                  key_reveal=cmd["key_reveal"], next_hash=cmd["next_hash"], sig=cmd["sig"])
    assert e.value.code == "OPERATOR_BAD_SIGNATURE"


def test_cleared_latch_survives_restart(tmp_path):
    sx = build(str(tmp_path)); _trade(sx)
    sx["gw"].latch_divergence("FAULT")
    _clear(sx, "resolved", "op-1"); sx["gw"]._persist()
    sx2 = build(str(tmp_path))
    assert sx2["gw"]._divergence_latched is False
    assert sx2["gw"]._restore_fault is None


def test_operator_body_is_canonical():
    b = operator_clear_body("r", "n", 7, "h" * 64)
    assert b == {"action": "CLEAR_DIVERGENCE", "reason": "r", "nonce": "n", "seq": 7, "next_hash": "h" * 64}


def test_gateway_cannot_self_clear(tmp_path):
    sx = build(str(tmp_path)); sx["gw"].latch_divergence("FAULT")
    assert not hasattr(sx["gw"], "operator_clear_signature")
    with pytest.raises(Reject) as e:
        sx["gw"].clear_divergence("self", nonce="x", seq=sx["clock"].seq,
                                  key_reveal="00" * 32, next_hash="11" * 32, sig="00" * 32)
    assert e.value.code == "OPERATOR_BAD_KEY"
    assert sx["gw"]._divergence_latched is True


def test_consumed_operator_key_cannot_be_reused(tmp_path):
    sx = build(str(tmp_path)); sx["gw"].latch_divergence("FAULT")
    cmd = sx["operator"].make_clear("first", "op-1", sx["clock"].seq)
    sx["gw"].clear_divergence("first", nonce="op-1", seq=cmd["seq"],
                              key_reveal=cmd["key_reveal"], next_hash=cmd["next_hash"], sig=cmd["sig"])
    sx["gw"].latch_divergence("AGAIN")
    with pytest.raises(Reject) as e:
        sx["gw"].clear_divergence("again", nonce="op-2", seq=sx["clock"].seq,
                                  key_reveal=cmd["key_reveal"], next_hash="22" * 32, sig=cmd["sig"])
    assert e.value.code == "OPERATOR_BAD_KEY"
