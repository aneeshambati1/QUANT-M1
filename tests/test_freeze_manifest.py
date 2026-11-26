"""Frozen manifest integrity: column_order / start / end must match."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from quant_m1.data.loader import panel_from_frame, DataError
from quant_m1.data.freeze import freeze_panel, load_frozen


def _panel():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2022-01-03", periods=60, tz="UTC")
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(60, 3)), axis=0))
    return panel_from_frame(
        pd.DataFrame(px, index=idx, columns=["SPY", "QQQ", "TLT"]),
        source="fixture", vintage="v1", pit=False,
        label="TEST FIXTURE - no performance claim")


def _freeze(tmp_path):
    freeze_panel(_panel(), str(tmp_path), "panel")
    return tmp_path / "panel.manifest.json"


def test_roundtrip_loads(tmp_path):
    _freeze(tmp_path)
    assert load_frozen(str(tmp_path), "panel").close.shape == (60, 3)


def test_tampered_start_rejected(tmp_path):
    mp = _freeze(tmp_path)
    m = json.loads(mp.read_text())
    m["start"] = "2019-01-01 00:00:00+00:00"
    mp.write_text(json.dumps(m))
    with pytest.raises(DataError):
        load_frozen(str(tmp_path), "panel")


def test_tampered_end_rejected(tmp_path):
    mp = _freeze(tmp_path)
    m = json.loads(mp.read_text())
    m["end"] = "2030-01-01 00:00:00+00:00"
    mp.write_text(json.dumps(m))
    with pytest.raises(DataError):
        load_frozen(str(tmp_path), "panel")


def test_tampered_column_order_rejected(tmp_path):
    mp = _freeze(tmp_path)
    m = json.loads(mp.read_text())
    m["column_order"] = list(reversed(m["column_order"]))
    mp.write_text(json.dumps(m))
    with pytest.raises(DataError):
        load_frozen(str(tmp_path), "panel")
