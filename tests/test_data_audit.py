%%writefile quant_m1/research/data_audit.py
"""Round 9A — PIT Data Readiness Audit (fail-closed revision).

Mechanically evaluates whether a price panel is suitable for CONFIRMATORY
strategy testing. This module makes NO statistical claim, NO acceptance
decision, and NO edge assertion. Its output is a structured readiness verdict.

A back-adjusted, non-PIT, close-only panel is EXPECTED to be reported
NOT_SUITABLE_FOR_CONFIRMATORY. That is the correct, honest result and is the
point of the audit: it converts an unknown into a measurable data gap.

Fail-closed rules added in this revision (see docs/PIT_DATA_ACCEPTANCE_SPEC.md):
  * SUITABLE requires ESTABLISHED reproducibility from a frozen vintage. A
    missing content_hash, an unavailable rehash, or a mismatch all block it.
  * Invalid data is measured separately from missing data: non-finite values
    and non-positive prices are counted and flagged, not silently coerced.
  * Duplicate index dates are rejected as invalid structure.
  * Gap measurement uses actual observed sessions by default; an exchange
    calendar may be supplied explicitly. Generic weekday ranges are labelled
    approximate and never presented as exchange-session truth.
  * Valid episodes are CONTIGUOUS runs of fully valid rows, reported alongside
    the raw valid-row count so the two can never be confused.
  * The regime-transition count is renamed to what it measures. The legacy
    field name is retained as a deprecated alias for interface compatibility.
  * Caller-supplied provenance booleans remain honor-system inputs; an
    optional evidence mapping is required to convert them into verified flags.
    Without evidence the flags are forced False (fail-closed greenwash guard).

The block/episode definitions here remain PROVISIONAL and tagged exploratory.
Confirmatory definitions belong to the pre-registered protocol, not this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class DataReadinessReport:
    verdict: str                                   # SUITABLE / NOT_SUITABLE_FOR_CONFIRMATORY
    reasons: List[str] = field(default_factory=list)
    n_symbols: int = 0
    n_dates: int = 0
    n_valid_episodes: int = 0                      # contiguous fully-valid runs
    n_valid_rows: int = 0                          # raw fully-valid row count
    longest_valid_episode: int = 0
    valid_fraction: float = 0.0
    nan_fraction: float = 0.0
    nonfinite_fraction: float = 0.0
    nonpositive_fraction: float = 0.0
    gap_fraction: float = 0.0
    gap_basis: str = "NONE"                        # EXCHANGE_CALENDAR / OBSERVED_SESSIONS / NONE
    duplicate_dates: int = 0
    is_pit_flag: bool = False
    has_known_at_provenance: bool = False
    has_adjustment_history: bool = False
    has_delisted_coverage: bool = False
    survivorship_free: bool = False
    provenance_evidence_verified: bool = False
    reproducible_from_vintage: Optional[bool] = None
    n_contiguous_blocks_provisional: int = 0       # deprecated alias of regime transitions
    n_regime_transitions_provisional: int = 0
    block_rule_note: str = "PROVISIONAL_EXPLORATORY"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "n_symbols": self.n_symbols,
            "n_dates": self.n_dates,
            "n_valid_episodes": self.n_valid_episodes,
            "n_valid_rows": self.n_valid_rows,
            "longest_valid_episode": self.longest_valid_episode,
            "valid_fraction": self.valid_fraction,
            "nan_fraction": self.nan_fraction,
            "nonfinite_fraction": self.nonfinite_fraction,
            "nonpositive_fraction": self.nonpositive_fraction,
            "gap_fraction": self.gap_fraction,
            "gap_basis": self.gap_basis,
            "duplicate_dates": self.duplicate_dates,
            "is_pit_flag": self.is_pit_flag,
            "has_known_at_provenance": self.has_known_at_provenance,
            "has_adjustment_history": self.has_adjustment_history,
            "has_delisted_coverage": self.has_delisted_coverage,
            "survivorship_free": self.survivorship_free,
            "provenance_evidence_verified": self.provenance_evidence_verified,
            "reproducible_from_vintage": self.reproducible_from_vintage,
            "n_contiguous_blocks_provisional": self.n_contiguous_blocks_provisional,
            "n_regime_transitions_provisional": self.n_regime_transitions_provisional,
            "block_rule_note": self.block_rule_note,
        }


def _recompute_content_hash(close: pd.DataFrame) -> Optional[str]:
    """Reproduce the frozen-manifest content hash if the loader exposes it."""
    try:
        from ..data.loader import content_hash
        return content_hash(close)
    except Exception:
        return None


def _provisional_regime_transitions(close: pd.DataFrame, window: int = 21) -> int:
    """PROVISIONAL, EXPLORATORY regime-transition count (NOT data blocks).

    Counts sign-changes of a rolling realized-volatility regime indicator on the
    equal-weight index. Placeholder for sample-size discussion ONLY; the
    confirmatory block rule must be pre-registered separately.
    """
    # fill_method=None: never forward-fill missing prices before computing
    # returns. Padding fabricates zero returns across gaps.
    rets = close.pct_change(fill_method=None).dropna(how="all")
    if rets.empty:
        return 0
    idx_ret = rets.mean(axis=1)
    vol = idx_ret.rolling(window).std()
    med = vol.median()
    if not np.isfinite(med) or med <= 0:
        return 0
    regime = (vol > med).astype(int).dropna()
    if regime.empty:
        return 0
    return int((regime.diff().abs() > 0).sum()) + 1


# Backwards-compatible private alias (older callers/tests may import it).
_provisional_block_count = _provisional_regime_transitions


def _contiguous_episodes(valid_mask: np.ndarray) -> tuple:
    """Return (n_runs, longest_run) of consecutive True values."""
    n_runs = 0
    longest = 0
    run = 0
    for ok in valid_mask:
        if ok:
            run += 1
            if run == 1:
                n_runs += 1
            longest = max(longest, run)
        else:
            run = 0
    return n_runs, longest


def _session_index(idx: pd.DatetimeIndex, exchange_calendar=None) -> tuple:
    """Session basis for gap measurement.

    Explicit exchange calendar wins. Otherwise use the panel's own observed
    sessions (labelled OBSERVED_SESSIONS) rather than inventing holidays.
    """
    if exchange_calendar is not None:
        sessions = exchange_calendar.sessions_in_range(idx.min(), idx.max())
        return pd.DatetimeIndex(sessions), "EXCHANGE_CALENDAR"
    return pd.DatetimeIndex(sorted(idx.unique())), "OBSERVED_SESSIONS"


def audit_panel(
    close: pd.DataFrame,
    manifest: Optional[Mapping] = None,
    *,
    min_valid_episodes: int = 252,
    has_known_at_columns: bool = False,
    has_adjustment_log: bool = False,
    has_delisted_universe: bool = False,
    provenance_evidence: Optional[Mapping[str, bool]] = None,
    exchange_calendar=None,
    require_reproducibility: bool = True,
) -> DataReadinessReport:
    """Audit a price panel for confirmatory-readiness. Fail closed.

    Args:
        close: wide-form close panel (dates x symbols).
        manifest: frozen manifest dict (optional). Supplies pit flag,
            survivorship_free, and content_hash for the reproducibility gate.
        min_valid_episodes: minimum CONTIGUOUS valid-episode length required.
        has_known_at_columns / has_adjustment_log / has_delisted_universe:
            honor-system assertions. Ignored (forced False) unless
            provenance_evidence verifies them.
        provenance_evidence: mapping of flag name -> bool indicating that a
            documentary evidence bundle exists (vendor doc reference, sample
            extract, event-log hash). Keys: known_at, adjustment_log,
            delisted_universe, pit, survivorship_free. When omitted, all
            caller-supplied provenance flags are forced False.
        exchange_calendar: optional exchange_calendars calendar for exact
            session-based gap measurement.
        require_reproducibility: when True (default) SUITABLE additionally
            requires an established vintage-hash match.
    """
    rep = DataReadinessReport(verdict="NOT_SUITABLE_FOR_CONFIRMATORY")
    manifest = dict(manifest or {})

    if not isinstance(close, pd.DataFrame) or close.empty:
        rep.reasons.append("EMPTY_OR_INVALID_PANEL")
        return rep

    rep.n_symbols = int(close.shape[1])
    rep.n_dates = int(close.shape[0])

    idx = pd.DatetimeIndex(close.index)
    rep.duplicate_dates = int(idx.duplicated().sum())

    numeric = close.apply(pd.to_numeric, errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    total = max(arr.size, 1)
    rep.nan_fraction = float(np.isnan(arr).sum() / total)
    rep.nonfinite_fraction = float((~np.isfinite(arr)).sum() / total)
    finite = np.isfinite(arr)
    rep.nonpositive_fraction = float((finite & (arr <= 0)).sum() / total)

    # --- gap measurement on an explicit session basis --------------------
    if len(idx) >= 2:
        sessions, basis = _session_index(idx, exchange_calendar)
        rep.gap_basis = basis
        expected = max(len(sessions), 1)
        observed = set(idx.normalize())
        gaps = []
        for col in numeric.columns:
            present = numeric[col].notna()
            have = int(present.sum())
            gaps.append(max(0.0, 1.0 - have / expected))
        rep.gap_fraction = float(np.mean(gaps)) if gaps else 0.0
        missing_sessions = len([s for s in sessions.normalize() if s not in observed])
        if missing_sessions:
            rep.gap_fraction = max(rep.gap_fraction, missing_sessions / expected)
    else:
        rep.gap_basis = "NONE"
        rep.gap_fraction = 0.0

    # --- valid rows and CONTIGUOUS episodes ------------------------------
    valid_row_mask = np.isfinite(arr).all(axis=1) & (arr > 0).all(axis=1)
    rep.n_valid_rows = int(valid_row_mask.sum())
    rep.n_valid_episodes, rep.longest_valid_episode = _contiguous_episodes(valid_row_mask)
    rep.valid_fraction = float(rep.n_valid_rows / max(rep.n_dates, 1))

    # --- provenance: evidence-gated, fail closed -------------------------
    ev = dict(provenance_evidence or {})
    rep.provenance_evidence_verified = bool(provenance_evidence)
    rep.is_pit_flag = bool(manifest.get("pit", False)) and bool(ev.get("pit", False))
    rep.survivorship_free = bool(manifest.get("survivorship_free", False)) and bool(ev.get("survivorship_free", False))
    rep.has_known_at_provenance = bool(has_known_at_columns) and bool(ev.get("known_at", False))
    rep.has_adjustment_history = bool(has_adjustment_log) and bool(ev.get("adjustment_log", False))
    rep.has_delisted_coverage = bool(has_delisted_universe) and bool(ev.get("delisted_universe", False))

    # --- reproducibility from frozen vintage (fail closed) ---------------
    expected_hash = manifest.get("content_hash")
    if expected_hash:
        recomputed = _recompute_content_hash(numeric)
        rep.reproducible_from_vintage = (
            None if recomputed is None else bool(recomputed == expected_hash)
        )
    else:
        rep.reproducible_from_vintage = None

    # --- provisional regime transitions (exploratory only) ---------------
    rep.n_regime_transitions_provisional = _provisional_regime_transitions(numeric)
    rep.n_contiguous_blocks_provisional = rep.n_regime_transitions_provisional

    # --- verdict ----------------------------------------------------------
    if rep.duplicate_dates:
        rep.reasons.append(f"DUPLICATE_INDEX_DATES:{rep.duplicate_dates}")
    if rep.nonfinite_fraction > rep.nan_fraction:
        rep.reasons.append("NONFINITE_VALUES_PRESENT")
    if rep.nonpositive_fraction > 0:
        rep.reasons.append("NONPOSITIVE_PRICES_PRESENT")
    if not rep.is_pit_flag:
        rep.reasons.append("NOT_POINT_IN_TIME")
    if not rep.has_known_at_provenance:
        rep.reasons.append("NO_AVAILABILITY_PROVENANCE")
    if not rep.has_adjustment_history:
        rep.reasons.append("NO_CORPORATE_ACTION_EVENT_LOG")
    if not rep.has_delisted_coverage:
        rep.reasons.append("NO_DELISTED_COVERAGE")
    if not rep.survivorship_free:
        rep.reasons.append("SURVIVORSHIP_BIAS_NOT_RULED_OUT")
    if provenance_evidence is None:
        rep.reasons.append("PROVENANCE_EVIDENCE_NOT_SUPPLIED")
    if rep.longest_valid_episode < min_valid_episodes:
        rep.reasons.append(
            f"INSUFFICIENT_VALID_EPISODES:{rep.longest_valid_episode}<{min_valid_episodes}")
    if require_reproducibility:
        if not expected_hash:
            rep.reasons.append("VINTAGE_HASH_MISSING")
        elif rep.reproducible_from_vintage is None:
            rep.reasons.append("VINTAGE_HASH_UNVERIFIABLE")
        elif rep.reproducible_from_vintage is False:
            rep.reasons.append("VINTAGE_HASH_MISMATCH")

    if not rep.reasons:
        rep.verdict = "SUITABLE"
    return rep


def audit_from_frozen(path: str, name: str, **kwargs) -> DataReadinessReport:
    """Convenience wrapper: load a frozen panel and audit it.

    A back-adjusted close-only frozen panel has no availability provenance, so
    the known_at/adjustment/delisted flags default to False unless the caller
    supplies verified evidence otherwise.
    """
    from ..data.freeze import load_frozen
    import json, os
    panel = load_frozen(path, name)
    manifest_path = os.path.join(path, f"{name}.manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    return audit_panel(panel.close, manifest, **kwargs)
