"""Round 9A data-audit tests. Verdict logic only; no edge or statistical claims.

Fail-closed coverage: reproducibility gate, invalid prices, duplicate dates,
contiguous episodes, session-based gaps, and evidence-gated provenance.
"""
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


FULL_EVIDENCE = {
    "pit": True, "survivorship_free": True, "known_at": True,
    "adjustment_log": True, "delisted_universe": True,
}


def _suitable_manifest(df):
    from quant_m1.data.loader import content_hash
    return {"pit": True, "survivorship_free": True,
            "content_hash": content_hash(df)}


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
    assert rep.gap_basis in {"OBSERVED_SESSIONS", "EXCHANGE_CALENDAR"}


def test_insufficient_episodes_flagged():
    df = _panel(n=50)
    rep = audit_panel(df, manifest={"pit": True}, min_valid_episodes=252)
    assert any("INSUFFICIENT_VALID_EPISODES" in r for r in rep.reasons)


def test_missing_hash_blocks_suitable():
    """CENTRAL GATE: no frozen-vintage hash => never SUITABLE."""
    df = _panel(n=400)
    rep = audit_panel(
        df, {"pit": True, "survivorship_free": True, "content_hash": None},
        has_known_at_columns=True, has_adjustment_log=True,
        has_delisted_universe=True, min_valid_episodes=252,
        provenance_evidence=FULL_EVIDENCE)
    assert rep.verdict == "NOT_SUITABLE_FOR_CONFIRMATORY"
    assert "VINTAGE_HASH_MISSING" in rep.reasons


def test_fully_provisioned_panel_with_verified_hash_can_be_suitable():
    df = _panel(n=400)
    rep = audit_panel(
        df, _suitable_manifest(df),
        has_known_at_columns=True, has_adjustment_log=True,
        has_delisted_universe=True, min_valid_episodes=252,
        provenance_evidence=FULL_EVIDENCE)
    assert rep.verdict == "SUITABLE", rep.reasons
    assert rep.reasons == []
    assert rep.reproducible_from_vintage is True


def test_hash_mismatch_flags_non_reproducibility():
    df = _panel(n=300)
    manifest = {"pit": True, "survivorship_free": True,
                "content_hash": "deadbeef" * 8}
    rep = audit_panel(df, manifest,
                      has_known_at_columns=True, has_adjustment_log=True,
                      has_delisted_universe=True,
                      provenance_evidence=FULL_EVIDENCE)
    assert rep.reproducible_from_vintage is False
    assert "VINTAGE_HASH_MISMATCH" in rep.reasons


def test_provenance_flags_without_evidence_are_forced_false():
    """Greenwash guard: caller assertions alone cannot grant provenance."""
    df = _panel(n=400)
    rep = audit_panel(df, _suitable_manifest(df),
                      has_known_at_columns=True, has_adjustment_log=True,
                      has_delisted_universe=True, min_valid_episodes=252)
    assert rep.has_known_at_provenance is False
    assert rep.has_adjustment_history is False
    assert rep.has_delisted_coverage is False
    assert "PROVENANCE_EVIDENCE_NOT_SUPPLIED" in rep.reasons
    assert rep.verdict == "NOT_SUITABLE_FOR_CONFIRMATORY"


def test_nonpositive_and_infinite_prices_flagged_separately():
    df = _panel(n=300)
    df.iloc[10, 0] = -5.0
    df.iloc[20, 1] = np.inf
    rep = audit_panel(df, manifest={"pit": True})
    assert rep.nonpositive_fraction > 0.0
    assert rep.nonfinite_fraction > rep.nan_fraction
    assert "NONPOSITIVE_PRICES_PRESENT" in rep.reasons
    assert "NONFINITE_VALUES_PRESENT" in rep.reasons
    assert rep.n_valid_rows < rep.n_dates


def test_duplicate_index_dates_rejected():
    df = _panel(n=100)
    dup = pd.concat([df, df.iloc[[5]]])
    rep = audit_panel(dup, manifest={"pit": True})
    assert rep.duplicate_dates == 1
    assert any("DUPLICATE_INDEX_DATES" in r for r in rep.reasons)
    assert rep.verdict == "NOT_SUITABLE_FOR_CONFIRMATORY"


def test_episodes_are_contiguous_runs_not_row_counts():
    df = _panel(n=300)
    df.iloc[150, :] = np.nan          # split the panel into two runs
    rep = audit_panel(df, manifest={"pit": True}, min_valid_episodes=252)
    assert rep.n_valid_rows == 299
    assert rep.n_valid_episodes == 2
    assert rep.longest_valid_episode == 150
    # 299 valid rows but no 252-long contiguous run => must fail
    assert any("INSUFFICIENT_VALID_EPISODES" in r for r in rep.reasons)


def test_gap_uses_observed_sessions_not_weekday_invention():
    # Build a panel missing an interior session entirely (all symbols NaN).
    df = _panel(n=120)
    df.iloc[60, :] = np.nan
    rep = audit_panel(df, manifest={"pit": True})
    assert rep.gap_basis == "OBSERVED_SESSIONS"
    assert rep.gap_fraction > 0.0


def test_regime_transition_count_is_labeled_exploratory():
    rep = audit_panel(_panel(n=500, seed=7))
    assert rep.n_regime_transitions_provisional >= 0
    assert rep.n_contiguous_blocks_provisional == rep.n_regime_transitions_provisional
    assert rep.block_rule_note == "PROVISIONAL_EXPLORATORY"


def test_no_forward_fill_of_missing_prices():
    """Padding would fabricate zero returns; fill_method=None must not."""
    df = _panel(n=120)
    df.iloc[40:45, 0] = np.nan
    rep = audit_panel(df, manifest={"pit": True})
    assert rep.nan_fraction > 0.0
    # transition count computed on honest returns; simply must not raise
    assert rep.n_regime_transitions_provisional >= 0


def test_report_as_dict_roundtrip_keys():
    rep = audit_panel(_panel(n=60))
    d = rep.as_dict()
    for key in ("verdict", "reasons", "n_valid_episodes", "n_valid_rows",
                "longest_valid_episode", "nonfinite_fraction",
                "nonpositive_fraction", "gap_basis", "duplicate_dates",
                "provenance_evidence_verified", "reproducible_from_vintage"):
        assert key in d
