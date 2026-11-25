"""Panel freezing: canonical order, bit-exact round trip, tamper detection."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_m1.data.loader import (panel_from_frame, content_hash,
                                  column_hashes, DataError)
from quant_m1.data.freeze import freeze_panel, load_frozen

LABEL = "SYNTHETIC - NOT AN EDGE CLAIM"


def _frame(seed=0, n=80, k=4):
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.standard_normal((n, k)) * 0.01, axis=0))
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(px, index=idx, columns=[f"S{i}" for i in range(k)])


def _panel(seed=0):
    return panel_from_frame(_frame(seed), source="synthetic", vintage="v1",
                            pit=False, label=LABEL)


def test_panel_carries_content_hash():
    p = _panel()
    assert len(p.content_hash) == 64
    assert p.content_hash == content_hash(p.close)


def test_column_order_does_not_change_hash():
    """A reordered vendor response is NOT a data revision."""
    df = _frame()
    a = panel_from_frame(df, source="s", vintage="v", pit=False, label=LABEL)
    shuffled = df[list(reversed(df.columns))]
    b = panel_from_frame(shuffled, source="s", vintage="v", pit=False, label=LABEL)
    assert a.content_hash == b.content_hash
    assert list(a.close.columns) == list(b.close.columns)


def test_genuine_price_change_changes_hash():
    df = _frame()
    a = panel_from_frame(df, source="s", vintage="v", pit=False, label=LABEL)
    perturbed = df.copy()
    perturbed.iloc[10, 0] *= 1.0001
    b = panel_from_frame(perturbed, source="s", vintage="v", pit=False, label=LABEL)
    assert a.content_hash != b.content_hash


def test_column_hashes_attribute_a_change():
    df = _frame()
    h0 = column_hashes(df)
    perturbed = df.copy()
    perturbed.iloc[10, 1] *= 1.01
    h1 = column_hashes(perturbed)
    changed = [k for k in h0 if h0[k] != h1[k]]
    assert changed == ["S1"]


def test_duplicate_columns_rejected():
    df = _frame()
    df.columns = ["A", "A", "B", "C"]
    with pytest.raises(DataError):
        panel_from_frame(df, source="s", vintage="v", pit=False, label=LABEL)


def test_freeze_round_trip_is_bit_exact(tmp_path):
    p = _panel()
    man = freeze_panel(p, str(tmp_path), "panel")
    back = load_frozen(str(tmp_path), "panel")
    assert back.content_hash == p.content_hash == man["content_hash"]
    assert np.array_equal(back.close.to_numpy(), p.close.to_numpy())
    assert man["column_order"] == sorted(man["column_order"])
    assert len(man["column_hashes"]) == p.close.shape[1]


def test_tampered_csv_rejected(tmp_path):
    freeze_panel(_panel(), str(tmp_path), "panel")
    csv = tmp_path / "panel.csv"
    txt = csv.read_text().splitlines()
    row = txt[5].split(",")
    row[1] = str(float(row[1]) * 1.01)
    txt[5] = ",".join(row)
    csv.write_text("\n".join(txt) + "\n")
    with pytest.raises(DataError):
        load_frozen(str(tmp_path), "panel")


def test_tampered_manifest_rejected(tmp_path):
    freeze_panel(_panel(), str(tmp_path), "panel")
    mp = tmp_path / "panel.manifest.json"
    man = json.loads(mp.read_text())
    man["content_hash"] = "0" * 64
    mp.write_text(json.dumps(man))
    with pytest.raises(DataError):
        load_frozen(str(tmp_path), "panel")


def test_missing_artifact_rejected(tmp_path):
    with pytest.raises(DataError):
        load_frozen(str(tmp_path), "absent")
