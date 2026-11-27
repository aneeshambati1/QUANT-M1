"""Round 9A — Point-in-time data readiness audit.

This module evaluates mechanical data readiness only. It does not establish
strategy performance, statistical validity, or the existence of trading edge.

The audit is fail-closed for confirmatory readiness:
- Frozen-vintage reproducibility must be established when required.
- Invalid prices, missing values, duplicate dates, and malformed timestamps
  block confirmatory suitability.
- Provenance assertions require a complete evidence mapping.
- Regime-transition counts are exploratory and are NOT independent data blocks.

Important:
The provenance evidence mapping is supplied by the caller. This module checks
that the required keys are present and true; it does not independently inspect
vendor documents or evidence artifacts. Those artifacts must be reviewed and
recorded by the research process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional

import numpy as np
import pandas as pd


_REQUIRED_PROVENANCE_KEYS = {
    "pit",
    "survivorship_free",
    "known_at",
    "adjustment_log",
    "delisted_universe",
}


@dataclass
class DataReadinessReport:
    verdict: str
    reasons: List[str] = field(default_factory=list)

    n_symbols: int = 0
    n_dates: int = 0

    n_valid_episodes: int = 0
    n_valid_rows: int = 0
    longest_valid_episode: int = 0

    valid_fraction: float = 0.0
    nan_fraction: float = 0.0
    nonfinite_fraction: float = 0.0
    nonpositive_fraction: float = 0.0
    gap_fraction: float = 0.0
    gap_basis: str = "NONE"

    duplicate_dates: int = 0

    is_pit_flag: bool = False
    has_known_at_provenance: bool = False
    has_adjustment_history: bool = False
    has_delisted_coverage: bool = False
    survivorship_free: bool = False

    provenance_evidence_verified: bool = False
    reproducible_from_vintage: Optional[bool] = None

    # Deprecated compatibility field. This is not a count of data blocks.
    n_contiguous_blocks_provisional: int = 0
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
            "provenance_evidence_verified": (
                self.provenance_evidence_verified
            ),
            "reproducible_from_vintage": (
                self.reproducible_from_vintage
            ),
            "n_contiguous_blocks_provisional": (
                self.n_contiguous_blocks_provisional
            ),
            "n_regime_transitions_provisional": (
                self.n_regime_transitions_provisional
            ),
            "block_rule_note": self.block_rule_note,
        }


def _recompute_content_hash(
    close: pd.DataFrame,
) -> Optional[str]:
    """Recompute the frozen-panel hash using the repository loader."""
    try:
        from ..data.loader import content_hash

        return content_hash(close)
    except Exception:
        return None


def _provisional_regime_transitions(
    close: pd.DataFrame,
    window: int = 21,
) -> int:
    """Return exploratory volatility-regime transitions.

    This is not an estimate of independent blocks and must not be used as
    confirmatory sample size.

    Missing rolling-volatility observations are removed before converting
    the regime indicator to integers. This prevents NaN comparisons from
    being incorrectly converted into low-volatility regime values.
    """
    if (
        not isinstance(close, pd.DataFrame)
        or close.empty
        or close.shape[1] == 0
        or window < 2
    ):
        return 0

    returns = close.pct_change(fill_method=None)
    index_returns = returns.mean(axis=1, skipna=True)

    volatility = index_returns.rolling(
        window=window,
        min_periods=window,
    ).std()

    volatility = volatility.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if volatility.empty:
        return 0

    median_volatility = volatility.median()

    if (
        not np.isfinite(median_volatility)
        or median_volatility <= 0
    ):
        return 0

    regime = (volatility > median_volatility).astype("int8")

    if regime.empty:
        return 0

    transitions = int(regime.diff().abs().fillna(0).sum())

    # Number of observed regime segments, not independent data blocks.
    return transitions + 1


# Backwards-compatible private alias for existing callers.
_provisional_block_count = _provisional_regime_transitions


def _contiguous_episodes(
    valid_mask: np.ndarray,
) -> tuple[int, int]:
    """Return the number and maximum length of consecutive valid runs."""
    n_runs = 0
    longest_run = 0
    current_run = 0

    for is_valid in valid_mask:
        if bool(is_valid):
            current_run += 1

            if current_run == 1:
                n_runs += 1

            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    return n_runs, longest_run


def _session_index(
    idx: pd.DatetimeIndex,
    exchange_calendar=None,
) -> tuple[pd.DatetimeIndex, str]:
    """Return expected sessions and the basis used for gap measurement."""
    if exchange_calendar is not None:
        sessions = exchange_calendar.sessions_in_range(
            idx.min(),
            idx.max(),
        )
        return pd.DatetimeIndex(sessions), "EXCHANGE_CALENDAR"

    # Without a calendar, absent dates cannot be detected reliably.
    # The report labels this basis explicitly.
    return (
        pd.DatetimeIndex(sorted(idx.unique())),
        "OBSERVED_SESSIONS",
    )


def _invalid_report(reason: str) -> DataReadinessReport:
    report = DataReadinessReport(
        verdict="NOT_SUITABLE_FOR_CONFIRMATORY",
    )
    report.reasons.append(reason)
    return report


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
    """Audit a close-price panel for confirmatory data readiness.

    Args:
        close:
            Wide-form price panel. Rows are dates; columns are symbols.
        manifest:
            Frozen manifest with PIT, survivorship, and content-hash fields.
        min_valid_episodes:
            Minimum required length of the longest contiguous fully-valid run.
        has_known_at_columns:
            Caller assertion that availability timestamps are present.
        has_adjustment_log:
            Caller assertion that corporate-action history is available.
        has_delisted_universe:
            Caller assertion that delisted instruments are represented.
        provenance_evidence:
            Caller-supplied mapping containing the required provenance keys.
            This is not independent document verification.
        exchange_calendar:
            Optional exchange_calendars-compatible calendar.
        require_reproducibility:
            Require a verified frozen-vintage content-hash match.

    Returns:
        DataReadinessReport. SUITABLE means only that these mechanical gates
        passed; it does not prove that the dataset is economically correct.
    """
    report = DataReadinessReport(
        verdict="NOT_SUITABLE_FOR_CONFIRMATORY",
    )

    if not isinstance(close, pd.DataFrame) or close.empty:
        return _invalid_report("EMPTY_OR_INVALID_PANEL")

    if close.shape[0] < 1 or close.shape[1] < 1:
        return _invalid_report("EMPTY_OR_INVALID_PANEL")

    if not isinstance(close.index, pd.DatetimeIndex):
        return _invalid_report("INDEX_NOT_DATETIME")

    # Record dimensions and duplicate dates before rejecting malformed
    # ordering. This allows the report to expose both problems when present.
    report.n_symbols = int(close.shape[1])
    report.n_dates = int(close.shape[0])

    idx = pd.DatetimeIndex(close.index)
    report.duplicate_dates = int(idx.duplicated().sum())

    if report.duplicate_dates:
        report.reasons.append(
            f"DUPLICATE_INDEX_DATES:{report.duplicate_dates}"
        )

    if close.index.hasnans:
        report.reasons.append("NAT_TIMESTAMP_PRESENT")
        return report

    if not close.index.is_monotonic_increasing:
        report.reasons.append("UNSORTED_TIMESTAMP_INDEX")
        return report

    # Convert nonnumeric values to NaN so they are counted and rejected.
    numeric = close.apply(pd.to_numeric, errors="coerce")
    arr = numeric.to_numpy(dtype=float)

    total_cells = max(int(arr.size), 1)

    is_nan = np.isnan(arr)
    is_finite = np.isfinite(arr)
    is_nonpositive = is_finite & (arr <= 0)

    report.nan_fraction = float(is_nan.sum() / total_cells)
    report.nonfinite_fraction = float(
        (~is_finite).sum() / total_cells
    )
    report.nonpositive_fraction = float(
        is_nonpositive.sum() / total_cells
    )

    if report.nan_fraction > 0:
        report.reasons.append("MISSING_VALUES_PRESENT")

    if report.nonfinite_fraction > report.nan_fraction:
        report.reasons.append("NONFINITE_VALUES_PRESENT")

    if report.nonpositive_fraction > 0:
        report.reasons.append("NONPOSITIVE_PRICES_PRESENT")

    # Gap measurement. Observed-session mode cannot detect a date absent
    # from the index entirely; use an exchange calendar for that.
    if len(idx) >= 2:
        try:
            sessions, basis = _session_index(
                idx,
                exchange_calendar=exchange_calendar,
            )
            report.gap_basis = basis

            expected_sessions = max(len(sessions), 1)
            observed_dates = set(idx.normalize())

            per_symbol_gaps = []

            for column in numeric.columns:
                present_count = int(numeric[column].notna().sum())

                per_symbol_gaps.append(
                    max(
                        0.0,
                        1.0 - present_count / expected_sessions,
                    )
                )

            report.gap_fraction = (
                float(np.mean(per_symbol_gaps))
                if per_symbol_gaps
                else 0.0
            )

            missing_calendar_sessions = sum(
                session.normalize() not in observed_dates
                for session in sessions
            )

            if missing_calendar_sessions:
                report.gap_fraction = max(
                    report.gap_fraction,
                    missing_calendar_sessions / expected_sessions,
                )

        except Exception:
            report.gap_basis = "NONE"
            report.gap_fraction = 0.0
            report.reasons.append("SESSION_GAP_MEASUREMENT_FAILED")
    else:
        report.gap_basis = "NONE"
        report.gap_fraction = 0.0

    # A valid row requires every symbol to have a finite, positive price.
    valid_row_mask = (
        np.isfinite(arr).all(axis=1)
        & (arr > 0).all(axis=1)
    )

    report.n_valid_rows = int(valid_row_mask.sum())

    (
        report.n_valid_episodes,
        report.longest_valid_episode,
    ) = _contiguous_episodes(valid_row_mask)

    report.valid_fraction = float(
        report.n_valid_rows / max(report.n_dates, 1)
    )

    # Provenance assertions are accepted only with all required evidence
    # keys present and true. This verifies mapping completeness only;
    # the evidence itself must be reviewed externally.
    evidence = dict(provenance_evidence or {})

    report.provenance_evidence_verified = (
        _REQUIRED_PROVENANCE_KEYS.issubset(evidence.keys())
        and all(
            evidence.get(key) is True
            for key in _REQUIRED_PROVENANCE_KEYS
        )
    )

    report.is_pit_flag = (
        bool((manifest or {}).get("pit", False))
        and evidence.get("pit") is True
    )

    report.survivorship_free = (
        bool((manifest or {}).get("survivorship_free", False))
        and evidence.get("survivorship_free") is True
    )

    report.has_known_at_provenance = (
        bool(has_known_at_columns)
        and evidence.get("known_at") is True
    )

    report.has_adjustment_history = (
        bool(has_adjustment_log)
        and evidence.get("adjustment_log") is True
    )

    report.has_delisted_coverage = (
        bool(has_delisted_universe)
        and evidence.get("delisted_universe") is True
    )

    if not report.is_pit_flag:
        report.reasons.append("NOT_POINT_IN_TIME")

    if not report.has_known_at_provenance:
        report.reasons.append("NO_AVAILABILITY_PROVENANCE")

    if not report.has_adjustment_history:
        report.reasons.append("NO_CORPORATE_ACTION_EVENT_LOG")

    if not report.has_delisted_coverage:
        report.reasons.append("NO_DELISTED_COVERAGE")

    if not report.survivorship_free:
        report.reasons.append("SURVIVORSHIP_BIAS_NOT_RULED_OUT")

    if not report.provenance_evidence_verified:
        if not evidence:
            report.reasons.append(
                "PROVENANCE_EVIDENCE_NOT_SUPPLIED"
            )
        else:
            report.reasons.append(
                "PROVENANCE_EVIDENCE_INCOMPLETE_OR_UNVERIFIED"
            )

    if report.longest_valid_episode < min_valid_episodes:
        report.reasons.append(
            "INSUFFICIENT_VALID_EPISODES:"
            f"{report.longest_valid_episode}"
            f"<{min_valid_episodes}"
        )

    # Reproducibility gate.
    manifest = dict(manifest or {})
    expected_hash = manifest.get("content_hash")

    if expected_hash:
        recomputed_hash = _recompute_content_hash(numeric)

        report.reproducible_from_vintage = (
            None
            if recomputed_hash is None
            else bool(recomputed_hash == expected_hash)
        )
    else:
        report.reproducible_from_vintage = None

    if require_reproducibility:
        if not expected_hash:
            report.reasons.append("VINTAGE_HASH_MISSING")
        elif report.reproducible_from_vintage is None:
            report.reasons.append("VINTAGE_HASH_UNVERIFIABLE")
        elif report.reproducible_from_vintage is False:
            report.reasons.append("VINTAGE_HASH_MISMATCH")

    # Exploratory diagnostic only. Never interpret as independent blocks.
    report.n_regime_transitions_provisional = (
        _provisional_regime_transitions(numeric)
    )

    # Preserve legacy field for compatibility; explicitly not true blocks.
    report.n_contiguous_blocks_provisional = (
        report.n_regime_transitions_provisional
    )

    if not report.reasons:
        report.verdict = "SUITABLE"

    return report


def audit_from_frozen(
    path: str,
    name: str,
    **kwargs,
) -> DataReadinessReport:
    """Load a frozen panel and run the readiness audit."""
    import json
    import os

    from ..data.freeze import load_frozen

    panel = load_frozen(path, name)

    manifest_path = os.path.join(
        path,
        f"{name}.manifest.json",
    )

    manifest = {}

    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)

    return audit_panel(
        panel.close,
        manifest,
        **kwargs,
    )
