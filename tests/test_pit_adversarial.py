"""Adversarial PIT tests: mixed-offset ordering, NaN, duplicates, lookahead."""
from __future__ import annotations

import pytest

from quant_m1.data.pit import Bar, as_of_bars, query_window_invariant


def _bar(ts, known, close=100.0, vintage="v1"):
    return Bar("SPY", ts, close, close + 1, close - 1, close, 1e6, 1.0, known, "fixture", vintage)


def test_mixed_offset_ordering_is_by_utc_instant():
    b_late = _bar("2024-01-02T23:00:00+00:00", "2024-01-02T23:05:00+00:00")
    b_early = _bar("2024-01-02T00:00:00+02:00", "2024-01-01T22:05:00+00:00")
    out = as_of_bars([b_late, b_early], "SPY", "2024-01-03T00:00:00+00:00")
    assert out[0] is b_early and out[1] is b_late


def test_nan_price_rejected():
    with pytest.raises(ValueError):
        _bar("2024-01-02T21:00:00+00:00", "2024-01-02T21:05:00+00:00", close=float("nan"))


def test_inf_volume_rejected():
    with pytest.raises(ValueError):
        Bar("SPY", "2024-01-02T21:00:00+00:00", 100, 101, 99, 100, float("inf"), 1.0,
            "2024-01-02T21:05:00+00:00", "f", "v1")


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        _bar("2024-01-02T21:00:00", "2024-01-02T21:05:00+00:00")


def test_duplicate_ts_resolves_to_latest_known_vintage():
    ts = "2024-01-02T21:00:00+00:00"
    old = _bar(ts, "2024-01-02T21:05:00+00:00", close=100.0, vintage="v1")
    new = _bar(ts, "2024-01-03T09:00:00+00:00", close=101.0, vintage="v2")
    out = as_of_bars([old, new], "SPY", "2024-01-03T10:00:00+00:00")
    assert len(out) == 1 and out[0].vintage_id == "v2"


def test_future_known_revision_excluded():
    ts = "2024-01-02T21:00:00+00:00"
    old = _bar(ts, "2024-01-02T21:05:00+00:00", close=100.0, vintage="v1")
    future = _bar(ts, "2024-01-04T09:00:00+00:00", close=101.0, vintage="v2")
    out = as_of_bars([old, future], "SPY", "2024-01-03T10:00:00+00:00")
    assert len(out) == 1 and out[0].vintage_id == "v1"


def test_conflicting_same_known_duplicate_raises():
    ts = "2024-01-02T21:00:00+00:00"
    kn = "2024-01-02T21:05:00+00:00"
    a = _bar(ts, kn, close=100.0, vintage="v1")
    b = _bar(ts, kn, close=101.0, vintage="v1")
    with pytest.raises(ValueError):
        as_of_bars([a, b], "SPY", "2024-01-03T00:00:00+00:00")


def test_query_window_invariance_holds():
    bars = [_bar(f"2024-01-0{d}T21:00:00+00:00", f"2024-01-0{d}T21:05:00+00:00") for d in range(2, 6)]
    assert query_window_invariant(bars, "SPY", "2024-01-06T00:00:00+00:00",
                                  "2024-01-02T00:00:00+00:00", "2024-01-04T00:00:00+00:00")
