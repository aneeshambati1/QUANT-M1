"""Archive / drift / delisted-retention tests. No network, no keys."""
from __future__ import annotations
import pytest
from quant_m1.data.multi_source import MultiSourceArchive


def test_new_unchanged_drift_and_no_overwrite(tmp_path):
    a = MultiSourceArchive(str(tmp_path / "arch.json"))
    assert a.ingest("SPY", "2020-01-01", "yf", 100.0, "2020-01-02T00:00:00Z") == "NEW"
    assert a.ingest("SPY", "2020-01-01", "yf", 100.0, "2020-01-03T00:00:00Z") == "UNCHANGED"
    assert a.ingest("SPY", "2020-01-01", "yf", 101.0, "2020-01-04T00:00:00Z") == "DRIFT"
    # Frozen bar is NOT overwritten.
    assert a.close_cents("SPY", "2020-01-01", "yf") == 10000
    assert len(a.drift) == 1
    assert a.drift[0]["old_cents"] == 10000 and a.drift[0]["new_cents"] == 10100


def test_cross_provider_same_price_is_two_bars(tmp_path):
    a = MultiSourceArchive(str(tmp_path / "arch.json"))
    a.ingest("SPY", "2020-01-01", "yf", 100.0, "t0")
    a.ingest("SPY", "2020-01-01", "alpaca", 100.0, "t0")
    assert a.close_cents("SPY", "2020-01-01", "yf") == 10000
    assert a.close_cents("SPY", "2020-01-01", "alpaca") == 10000


def test_delisted_retention_and_reload(tmp_path):
    p = str(tmp_path / "arch.json")
    a = MultiSourceArchive(p)
    a.ingest("XIV", "2018-02-01", "yf", 10.0, "t0")
    assert a.has_symbol("XIV")
    b = MultiSourceArchive(p)          # reload
    assert b.has_symbol("XIV")         # never deleted
    assert b.close_cents("XIV", "2018-02-01", "yf") == 1000


def test_invalid_close_rejected(tmp_path):
    a = MultiSourceArchive(str(tmp_path / "arch.json"))
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            a.ingest("SPY", "2020-01-01", "yf", bad, "t0")


def test_inject_fetcher_counts(tmp_path):
    a = MultiSourceArchive(str(tmp_path / "arch.json"))
    fetcher = lambda sym: [("2020-01-01", 100.0), ("2020-01-02", 101.0)]
    counts = a.ingest_fetcher("SPY", "yf", fetcher, "t0")
    assert counts == {"NEW": 2, "UNCHANGED": 0, "DRIFT": 0}
    counts2 = a.ingest_fetcher("SPY", "yf", fetcher, "t1")
    assert counts2 == {"NEW": 0, "UNCHANGED": 2, "DRIFT": 0}


def test_chain_tamper_detected(tmp_path):
    p = str(tmp_path / "arch.json")
    a = MultiSourceArchive(p)
    a.ingest("SPY", "2020-01-01", "yf", 100.0, "t0")
    a.chain[0]["payload"]["close_cents"] = 99999   # tamper
    a.store.save(a.chain)
    with pytest.raises(ValueError):
        MultiSourceArchive(p)
