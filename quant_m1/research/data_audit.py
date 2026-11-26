"""Round 9A — PIT Data Readiness Audit.

Mechanically evaluates whether a price panel is suitable for CONFIRMATORY
strategy testing. This module makes NO statistical claim, NO acceptance
decision, and NO edge assertion. Its output is a structured readiness verdict.

A back-adjusted, non-PIT, close-only panel is EXPECTED to be reported
NOT_SUITABLE_FOR_CONFIRMATORY. That is the correct, honest result and is the
point of the audit: it converts an unknown into a measurable data gap.

The audit deliberately keeps the block-count PROVISIONAL and tagged exploratory.
Confirmatory block definitions belong to the pre-registered protocol, not here.
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
    n_valid_episodes: int = 0
    valid_fraction: float = 0.0
    nan_fraction: float = 0.0
    gap_fraction: float = 0.0
    is_pit_flag: bool = False
    has_known_at_provenance: bool = False
    has_adjustment_history: bool = False
    has_delisted_coverage: bool = False
    survivorship_free: bool = False
    reproducible_from_vintage: Optional[bool] = None
    n_contiguous_blocks_provisional: int = 0
    block_rule_note: str = "PROVISIONAL_EXPLORATORY"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "n_symbols": self.n_symbols,
            "n_dates": self.n_dates,
            "n_valid_episodes": self.n_valid_episodes,
            "valid_fraction": self.valid_fraction,
            "nan_fraction": self.nan_fraction,
            "gap_fraction": self.gap_fraction,
            "is_pit_flag": self.is_pit_flag,
            "has_known_at_provenance": self.has_known_at_provenance,
            "has_adjustment_history": self.has_adjustment_history,
            "has_delisted_coverage": self.has_delisted_coverage,
            "survivorship_free": self.survivorship_free,
            "reproducible_from_vintage": self.reproducible_from_vintage,
            "n_contiguous_blocks_provisional": self.n_contiguous_blocks_provisional,
            "block_rule_note": self.block_rule_note,
        }


def _recompute_content_hash(close: pd.DataFrame) -> Optional[str]:
    """Reproduce the frozen-manifest content hash if the loader exposes it."""
    try:
        from ..data.loader import content_hash
        return content_hash(close)
    except Exception:
        return None


def _provisional_block_count(close: pd.DataFrame, window: int = 21) -> int:
    """PROVISIONAL, EXPLORATORY block count.

    Counts sign-changes of a rolling realized-volatility regime indicator on the
    equal-weight index. This is a placeholder for sample-size estimation ONLY;
    the confirmatory block rule must be pre-registered separately.
    """
    rets = close.pct_change().dropna(how="all")
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
    changes = int((regime.diff().abs() > 0).sum())
    return changes + 1


def audit_panel(
    close: pd.DataFrame,
    manifest: Optional[Mapping] = None,
    *,
    min_valid_episodes: int = 252,
    has_known_at_columns: bool = False,
    has_adjustment_log: bool = False,
    has_delisted_universe: bool = False,
) -> DataReadinessReport:
    """Audit a price panel for confirmatory-readiness.

    Args:
        close: wide-form close panel (dates x symbols), finite prices.
        manifest: frozen manifest dict (optional). Used for pit flag + hash check.
        min_valid_episodes: minimum valid episodes to be considered.
        has_known_at_columns: True if the SOURCE data carried availability
            timestamps. A back-adjusted close-only panel must pass False.
        has_adjustment_log: True if corporate actions are stored as dated events.
        has_delisted_universe: True if the universe includes delisted funds.
    """
    rep = DataReadinessReport(verdict="NOT_SUITABLE_FOR_CONFIRMATORY")
    manifest = dict(manifest or {})

    if not isinstance(close, pd.DataFrame) or close.empty:
        rep.reasons.append("EMPTY_OR_INVALID_PANEL")
        return rep

    rep.n_symbols = int(close.shape[1])
    rep.n_dates = int(close.shape[0])

    # --- Q6: data loss to missing/invalid -------------------------------
    numeric = close.apply(pd.to_numeric, errors="coerce")
    rep.nan_fraction = float(numeric.isna().to_numpy().mean())

    # gap fraction: missing business days between first/last date per symbol
    idx = pd.DatetimeIndex(close.index)
    if len(idx) >= 2:
        full = pd.bdate_range(idx.min(), idx.max())
        expected = len(full)
        gaps = []
        for col in close.columns:
            present = numeric[col].notna()
            gaps.append(1.0 - float(present.sum()) / max(expected, 1))
        rep.gap_fraction = float(np.mean(gaps)) if gaps else 0.0
    else:
        rep.gap_fraction = 0.0

    # --- Q2: valid episodes after quality filters ------------------------
    valid_rows = numeric.dropna(how="all")
    valid_rows = valid_rows[(valid_rows > 0).all(axis=1)]
    rep.n_valid_episodes = int(len(valid_rows))
    rep.valid_fraction = float(rep.n_valid_episodes / max(rep.n_dates, 1))

    # --- Q1: PIT provenance ----------------------------------------------
    rep.is_pit_flag = bool(manifest.get("pit", False))
    rep.has_known_at_provenance = bool(has_known_at_columns)
    rep.has_adjustment_history = bool(has_adjustment_log)
    rep.has_delisted_coverage = bool(has_delisted_universe)
    rep.survivorship_free = bool(manifest.get("survivorship_free", False))

    # --- Q5: reproducibility from frozen vintage -------------------------
    expected_hash = manifest.get("content_hash")
    if expected_hash:
        recomputed = _recompute_content_hash(numeric)
        rep.reproducible_from_vintage = (
            None if recomputed is None else bool(recomputed == expected_hash)
        )

    # --- Q3/Q4: provisional blocks & corporate actions -------------------
    rep.n_contiguous_blocks_provisional = _provisional_block_count(numeric)

    # --- verdict ----------------------------------------------------------
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
    if rep.n_valid_episodes < min_valid_episodes:
        rep.reasons.append(
            f"INSUFFICIENT_VALID_EPISODES:{rep.n_valid_episodes}<{min_valid_episodes}")
    if rep.reproducible_from_vintage is False:
        rep.reasons.append("VINTAGE_HASH_MISMATCH")

    if not rep.reasons:
        rep.verdict = "SUITABLE"
    return rep


def audit_from_frozen(path: str, name: str, **kwargs) -> DataReadinessReport:
    """Convenience wrapper: load a frozen panel and audit it.

    A back-adjusted close-only frozen panel has no availability provenance, so
    the known_at/adjustment/delisted flags default to False unless the caller
    supplies evidence otherwise.
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
