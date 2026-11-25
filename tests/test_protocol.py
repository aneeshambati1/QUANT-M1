"""Governed protocol tests. SYNTHETIC panels; no edge claim."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_m1.data.loader import panel_from_frame
from quant_m1.research.protocol import (
    run_governed, run_grid, fingerprint, panel_identity, ProtocolError,
)
from quant_m1.research.governor import TrialLedger

LABEL = "SYNTHETIC - NOT AN EDGE CLAIM"


def _panel(n=200, k=5, seed=0, vintage="test-v1"):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((n, k)) * 0.01
    px = 100.0 * np.exp(np.cumsum(r, axis=0))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(px, index=idx, columns=[f"S{i}" for i in range(k)])
    return panel_from_frame(df, source="synthetic", vintage=vintage,
                            pit=False, label=LABEL)


def _run(tmp_path, panel=None, tid="t1", lb=21, hz=5):
    return run_governed(str(tmp_path / "led.jsonl"), panel or _panel(),
                        trial_id=tid, commit="abc123", family="mom",
                        hypothesis="trailing return predicts forward return",
                        lookback=lb, horizon=hz, seed=7,
                        out_dir=str(tmp_path / "raw"))


def test_registration_precedes_measurement(tmp_path):
    """A failing measurement must still leave a registered, counted trial."""
    p = _panel(n=45)  # too few usable sessions for lb=21,hz=5
    with pytest.raises(Exception):
        _run(tmp_path, panel=p)
    led = TrialLedger(str(tmp_path / "led.jsonl"))
    assert led.n_trials() == 1
    assert len(led.results()) == 0


def test_result_and_gate_recorded(tmp_path):
    out = _run(tmp_path)
    led = TrialLedger(str(tmp_path / "led.jsonl"))
    assert len(led.results()) == 1
    assert led.results()[0]["metric_name"] == "t_hac"
    assert out["gate"]["n_trials"] == 1
    assert "required_abs_t" in out["gate"]


def test_raw_output_written_and_referenced(tmp_path):
    out = _run(tmp_path)
    led = TrialLedger(str(tmp_path / "led.jsonl"))
    ref = led.results()[0]["raw_output_ref"]
    assert ref == out["raw_output"]
    payload = json.loads(open(ref).read())
    assert "report" in payload and "panel" in payload


def test_duplicate_hypothesis_rejected(tmp_path):
    _run(tmp_path, tid="t1")
    with pytest.raises(ProtocolError):
        _run(tmp_path, tid="t2")  # same config, same panel identity


def test_redownload_does_not_create_new_trial(tmp_path):
    """Vintage timestamp differs, panel identity does not."""
    _run(tmp_path, panel=_panel(vintage="download-A"), tid="t1")
    with pytest.raises(ProtocolError):
        _run(tmp_path, panel=_panel(vintage="download-B"), tid="t2")


def test_different_config_is_a_new_trial(tmp_path):
    _run(tmp_path, tid="t1", lb=21, hz=5)
    _run(tmp_path, tid="t2", lb=10, hz=5)
    led = TrialLedger(str(tmp_path / "led.jsonl"))
    assert led.n_trials() == 2


def test_commit_required(tmp_path):
    with pytest.raises(ProtocolError):
        run_governed(str(tmp_path / "led.jsonl"), _panel(), trial_id="t",
                     commit="", family="f", hypothesis="h", lookback=21,
                     horizon=5, seed=1, out_dir=str(tmp_path / "raw"))


def test_grid_registers_all_before_measuring(tmp_path):
    out = run_grid(str(tmp_path / "led.jsonl"), _panel(), family="mom",
                   hypothesis="grid search", lookbacks=[5, 10], horizons=[1, 5],
                   commit="abc", seed=7, out_dir=str(tmp_path / "raw"))
    led = TrialLedger(str(tmp_path / "led.jsonl"))
    assert out["n_trials_total"] == 4
    assert led.n_trials() == 4
    assert len(led.results()) == 4
    assert len(out["cells"]) == 4


def test_grid_gate_uses_full_count(tmp_path):
    out = run_grid(str(tmp_path / "led.jsonl"), _panel(), family="mom",
                   hypothesis="grid", lookbacks=[5, 10, 21], horizons=[1, 5],
                   commit="abc", seed=7, out_dir=str(tmp_path / "raw"))
    for c in out["cells"]:
        assert c["gate"]["n_trials"] == 6
    assert out["required_abs_t"] > 1.96


def test_grid_rejects_duplicate_axes(tmp_path):
    with pytest.raises(ProtocolError):
        run_grid(str(tmp_path / "led.jsonl"), _panel(), family="mom",
                 hypothesis="g", lookbacks=[5, 5], horizons=[1],
                 commit="abc", seed=1, out_dir=str(tmp_path / "raw"))


def test_grid_rerun_rejected(tmp_path):
    args = dict(family="mom", hypothesis="g", lookbacks=[5], horizons=[1],
                commit="abc", seed=1, out_dir=str(tmp_path / "raw"))
    run_grid(str(tmp_path / "led.jsonl"), _panel(), **args)
    with pytest.raises(ProtocolError):
        run_grid(str(tmp_path / "led.jsonl"), _panel(), **args)


def test_identity_excludes_vintage():
    a = panel_identity(_panel(vintage="x"))
    b = panel_identity(_panel(vintage="y"))
    assert a == b
    assert fingerprint("f", {"lookback": 5}, _panel(vintage="x")) == \
           fingerprint("f", {"lookback": 5}, _panel(vintage="y"))