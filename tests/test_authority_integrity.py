"""Authority integrity: ops binding, composite signing, control-state digest,
NaN fail-closed risk. Targets B1/B2/B3 and D-07."""
from __future__ import annotations

import pytest

from quant_m1.authority import Intent, Reject, sign
from quant_m1.core import WorkingOrder
from quant_m1.risk import evaluate
from quant_m1.system import build


def _auth(sx, qty=5):
    i = Intent("C", "BUY", qty, 10_000, 10)
    return sx["kernel"].authorize(sx["ledger"], i, sx["flags"], sx["clock"], False), i


def test_ops_config_change_invalidates_token(tmp_path):
    sx = build(str(tmp_path))
    tok, intent = _auth(sx)
    sx["cfg"].apply_config({"ops": {"auth_ttl_seqs": 40}})
    with pytest.raises(Reject) as e:
        sx["gw"].submit(tok, intent)
    assert e.value.code == "TOKEN_CONFIG_STALE"


def test_unsigned_composite_tamper_detected(tmp_path):
    sx = build(str(tmp_path))
    tok, intent = _auth(sx); sx["gw"].submit(tok, intent)
    d = sx["gw"].composite_store.load()
    d["divergence_latched"] = True   # genuine content change, NOT re-signed
    sx["gw"].composite_store.save(d)
    sx2 = build(str(tmp_path))
    assert sx2["gw"]._restore_fault == "COMPOSITE_SIGNATURE_INVALID"
    assert sx2["gw"]._divergence_latched is True


def test_control_state_tamper_detected_even_when_resigned(tmp_path):
    sx = build(str(tmp_path))
    tok, intent = _auth(sx); sx["gw"].submit(tok, intent)
    d = sx["gw"].composite_store.load()
    d["ledger"]["spent_nonces"] = []   # erase replay protection
    body = {k: v for k, v in d.items() if k != "composite_sig"}
    d["composite_sig"] = sign(sx["gw"].key, body)   # attacker holds key
    sx["gw"].composite_store.save(d)
    sx2 = build(str(tmp_path))
    assert sx2["gw"]._restore_fault == "FLAGS_CONTROL_DIGEST_MISMATCH"
    assert sx2["gw"]._divergence_latched is True


def test_nan_mark_risk_fails_closed(tmp_path):
    sx = build(str(tmp_path))
    sx["ledger"].state.marks["A"] = float("nan")
    ev = evaluate(sx["ledger"].state, (), sx["instruments"], sx["cfg"].mandate, sx["scenarios"])
    assert not ev.ok
    assert ev.reason.startswith("INVALID_RISK_INPUT")


def test_working_order_nan_price_rejected():
    with pytest.raises(ValueError):
        WorkingOrder("o", "A", "BUY", 5, 0, float("nan"), 0, "RESERVED", 100, "n")


def test_working_order_float_qty_rejected():
    with pytest.raises(ValueError):
        WorkingOrder("o", "A", "BUY", 5.0, 0, 10_000, 0, "RESERVED", 100, "n")
