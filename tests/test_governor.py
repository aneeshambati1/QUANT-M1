import os, sys, subprocess, ast
os.chdir("/content/QUANT-M1")

T = r'''"""Governor tests: tamper evidence, pre-registration, and trial-count gate."""
from __future__ import annotations

import json

import pytest

from quant_m1.research import governor
from quant_m1.research.governor import (
    TrialLedger, LedgerError, bonferroni_threshold, ic_gate,
)


def _reg(led, tid="t1"):
    return led.register_trial(
        trial_id=tid, commit="abc", family="momentum",
        hypothesis="21d trailing return predicts next 5d",
        config={"lookback": 21, "horizon": 5}, seed=7,
        data_vintage="yfinance:2026-11-25")


def test_register_and_result(tmp_path):
    led = TrialLedger(str(tmp_path / "ledger.jsonl"))
    _reg(led)
    led.record_result(trial_id="t1", metric_name="t_hac", metric_value=0.26,
                      horizon="5d", gross_net="gross", raw_output_ref="report.json")
    assert led.n_trials() == 1
    assert len(led.results()) == 1


def test_tamper_detected(tmp_path):
    p = tmp_path / "ledger.jsonl"
    led = TrialLedger(str(p))
    _reg(led)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["body"]["seed"] = 999
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError):
        TrialLedger(str(p))


def test_deletion_detected(tmp_path):
    p = tmp_path / "ledger.jsonl"
    led = TrialLedger(str(p))
    _reg(led, "t1")
    _reg(led, "t2")
    lines = p.read_text().splitlines()
    p.write_text(lines[1] + "\n")
    with pytest.raises(LedgerError):
        TrialLedger(str(p))


def test_duplicate_trial_rejected(tmp_path):
    led = TrialLedger(str(tmp_path / "ledger.jsonl"))
    _reg(led)
    with pytest.raises(LedgerError):
        _reg(led)


def test_result_requires_registration(tmp_path):
    led = TrialLedger(str(tmp_path / "ledger.jsonl"))
    with pytest.raises(LedgerError):
        led.record_result(trial_id="ghost", metric_name="t_hac", metric_value=1.0,
                          horizon="5d", gross_net="gross", raw_output_ref="x")


def test_duplicate_result_rejected(tmp_path):
    led = TrialLedger(str(tmp_path / "ledger.jsonl"))
    _reg(led)
    led.record_result(trial_id="t1", metric_name="t_hac", metric_value=0.26,
                      horizon="5d", gross_net="gross", raw_output_ref="x")
    with pytest.raises(LedgerError):
        led.record_result(trial_id="t1", metric_name="t_hac", metric_value=9.0,
                          horizon="5d", gross_net="gross", raw_output_ref="y")


def test_threshold_tightens_with_trials():
    assert bonferroni_threshold(10) > bonferroni_threshold(1)
    assert bonferroni_threshold(100) > bonferroni_threshold(10)


def test_gate_passes_only_above_threshold():
    assert ic_gate(0.26, 1)["passed"] is False
    assert ic_gate(5.0, 1)["passed"] is True
    assert ic_gate(3.0, 100)["passed"] is False


def test_config_order_does_not_change_hash(tmp_path, monkeypatch):
    """Hash must depend on canonical content, not dict insertion order.

    registered_at is part of the hashed body, so freeze it; otherwise two
    registrations microseconds apart differ for a reason unrelated to order.
    """
    monkeypatch.setattr(governor, "_now", lambda: "2026-01-01T00:00:00+00:00")
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    a = TrialLedger(str(p1))
    b = TrialLedger(str(p2))
    a.register_trial(trial_id="t", commit="c", family="f", hypothesis="h",
                     config={"x": 1, "y": 2}, seed=1, data_vintage="v")
    b.register_trial(trial_id="t", commit="c", family="f", hypothesis="h",
                     config={"y": 2, "x": 1}, seed=1, data_vintage="v")
    ha = json.loads(p1.read_text().splitlines()[0])["hash"]
    hb = json.loads(p2.read_text().splitlines()[0])["hash"]
    assert ha == hb


def test_timestamp_is_part_of_tamper_evidence(tmp_path, monkeypatch):
    """Changing registered_at must change the hash."""
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    monkeypatch.setattr(governor, "_now", lambda: "2026-01-01T00:00:00+00:00")
    a = TrialLedger(str(p1))
    a.register_trial(trial_id="t", commit="c", family="f", hypothesis="h",
                     config={"x": 1}, seed=1, data_vintage="v")
    monkeypatch.setattr(governor, "_now", lambda: "2026-01-02T00:00:00+00:00")
    b = TrialLedger(str(p2))
    b.register_trial(trial_id="t", commit="c", family="f", hypothesis="h",
                     config={"x": 1}, seed=1, data_vintage="v")
    ha = json.loads(p1.read_text().splitlines()[0])["hash"]
    hb = json.loads(p2.read_text().splitlines()[0])["hash"]
    assert ha != hb


def test_invalid_gate_inputs(tmp_path):
    with pytest.raises(LedgerError):
        bonferroni_threshold(0)
    with pytest.raises(LedgerError):
        bonferroni_threshold(1, alpha=0.0)
'''

open("tests/test_governor.py", "w").write(T)
ast.parse(T)
c = open("tests/test_governor.py").read()
print("bytes:", len(c))
print("test count:", c.count("\ndef test_"))
print("has monkeypatch:", "monkeypatch.setattr(governor" in c)
assert "monkeypatch.setattr(governor" in c, "WRITE FAILED"

def run(a, l):
    print("\n" + "="*60); print(l); print("="*60)
    p = subprocess.run([sys.executable,"-m","pytest"]+a, capture_output=True, text=True)
    print("RC:", p.returncode); print(p.stdout or "(empty)")
    if p.stderr: print("STDERR:", p.stderr)
    return p

p1 = run(["tests/test_governor.py","-q","--tb=short"], "PHASE 1 governor")
if p1.returncode == 0:
    run(["tests/","-q","--tb=short"], "PHASE 2 full suite")
print("\nNOT PUSHED.")