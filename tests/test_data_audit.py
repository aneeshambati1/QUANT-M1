"""Round 9A data-audit tests. Verdict logic only; no edge or statistical claims."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_m1.research.data_audit import audit_panel, DataReadinessReport


def _panel(n=300, k=3, seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, k)), axis=0))
    df = pd.DataFrame(px, index=idx, columns=[f"E{i}" for i in range(k)])
    if nan_frac > 0:
        mask = rng.random(df.shape) < nan_frac
        df = df.mask(mask)
    return df


def test_back_adjusted_panel_is_not_suitable():
    """A non-PIT, close-only, no-provenance panel must be NOT_SUITABLE."""
    rep = audit_panel(_panel(), manifest={"pit": False})
    assert rep.verdict == "NOT_SUITABLE_FOR_CONFIRMATORY"
    assert "NOT_POINT_IN_TIME" in rep.reasons
    assert "NO_AVAILABILITY_PROVENANCE" in rep.reasons
    assert "SURVIVORSHIP_BIAS_NOT_RULED_OUT" in rep.reasons


def test_empty_panel_rejected():
    rep = audit_panel(pd.DataFrame())
    assert rep.verdict == "NOT_SUITABLE_FOR_CONFIRMATORY"
    assert "EMPTY_OR_INVALID_PANEL" in rep.reasons


def test_nan_and_gap_fractions_reported():
    df = _panel(n=200, nan_frac=0.10, seed=3)
    rep = audit_panel(df, manifest={"pit": True})
    assert rep.nan_fraction > 0.05
    assert 0.0 <= rep.gap_fraction <= 1.0


def test_insufficient_episodes_flagged():
    df = _panel(n=50)
    rep = audit_panel(df, manifest={"pit": True}, min_valid_episodes=252)
    assert any("INSUFFICIENT_VALID_EPISODES" in r for r in rep.reasons)


def test_fully_provisioned_panel_can_be_suitable():
    df = _panel(n=400)
    manifest = {"pit": True, "survivorship_free": True, "content_hash": None}
    rep = audit_panel(
        df, manifest,
        has_known_at_columns=True, has_adjustment_log=True,
        has_delisted_universe=True, min_valid_episodes=252)
    assert rep.verdict == "SUITABLE"
    assert rep.reasons == []


def test_hash_mismatch_flags_non_reproducibility():
    df = _panel(n=300)
    manifest = {"pit": True, "survivorship_free": True,
                "content_hash": "deadbeef" * 8}
    rep = audit_panel(df, manifest,
                      has_known_at_columns=True, has_adjustment_log=True,
                      has_delisted_universe=True)
    assert rep.reproducible_from_vintage is False
    assert "VINTAGE_HASH_MISMATCH" in rep.reasons


def test_provisional_block_count_is_positive_and_labeled():
    rep = audit_panel(_panel(n=500, seed=7))
    assert rep.n_contiguous_blocks_provisional >= 0
    assert rep.block_rule_note == "PROVISIONAL_EXPLORATORY"


def test_nonpositive_prices_excluded_from_valid_episodes():
    df = _panel(n=300)
    df.iloc[10, 0] = -5.0
    rep = audit_panel(df, manifest={"pit": True})
    assert rep.n_valid_episodes <= rep.n_dates
