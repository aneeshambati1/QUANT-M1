"""Panel freezing: bit-exact round trip and tamper detection."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_m1.data.loader import panel_from_frame, content_hash, DataError
from quant_m1.data.freeze import freeze_panel, load_frozen

LABEL = "SYNTHETIC - NOT AN EDGE CLAIM"


def _panel(seed=0, n=80, k=4):
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.standard_normal((n, k)) * 0.01, axis=0))
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame(px, index=idx, columns=[f"S{i}" for i in range(k)])
    return panel_from_frame(df, source="synthetic", vintage="v1",
                            pit=False, label=LABEL)


def test_panel_carries_content_hash():
    p = _panel()
    assert len(p.content_hash) == 64
    assert p.content_hash == content_hash(p.close)


def test_content_hash_ignores_vintage_but_sees_prices():
    a = _panel(seed=0)
    b = _panel(seed=1)
    assert a.content_hash != b.content_hash


def test_freeze_round_trip_is_bit_exact(tmp_path):
    p = _panel()
    man = freeze_panel(p, str(tmp_path), "panel")
    back = load_frozen(str(tmp_path), "panel")
    assert back.content_hash == p.content_hash == man["content_hash"]
    assert np.array_equal(back.close.to_numpy(), p.close.to_numpy())


def test_tampered_csv_rejected(tmp_path):
    p = _panel()
    freeze_panel(p, str(tmp_path), "panel")
    csv = tmp_path / "panel.csv"
    txt = csv.read_text().splitlines()
    row = txt[5].split(",")
    row[1] = str(float(row[1]) * 1.01)
    txt[5] = ",".join(row)
    csv.write_text("\n".join(txt) + "\n")
    with pytest.raises(DataError):
        load_frozen(str(tmp_path), "panel")


def test_tampered_manifest_rejected(tmp_path):
    p = _panel()
    freeze_panel(p, str(tmp_path), "panel")
    mp = tmp_path / "panel.manifest.json"
    man = json.loads(mp.read_text())
    man["content_hash"] = "0" * 64
    mp.write_text(json.dumps(man))
    with pytest.raises(DataError):
        load_frozen(str(tmp_path), "panel")


def test_missing_artifact_rejected(tmp_path):
    with pytest.raises(DataError):
        load_frozen(str(tmp_path), "absent")