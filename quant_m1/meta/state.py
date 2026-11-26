"""Market state classification. Seed of Organ 1 (Unified Market World Model).

Every field is Optional because UNKNOWN must be representable. A missing
observation is not zero; treating it as zero is failure mode D7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Regime(Enum):
    UNKNOWN = "unknown"
    CALM = "calm"
    ELEVATED = "elevated"
    RISK_OFF = "risk_off"
    CRISIS = "crisis"


# POLICY assumptions, not empirically calibrated thresholds.
CORR_ELEVATED = 0.60
CORR_RISK_OFF = 0.75
CORR_CRISIS = 0.85

LIQUIDITY_SHOCK = 0.35
DISAGREEMENT_HIGH = 0.60


def _finite_number(value: object) -> bool:
    """True only for finite real numeric values; bool is not a number here."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("as_of must be a non-empty ISO-8601 timestamp")

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("as_of must be a valid ISO-8601 timestamp") from exc

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware UTC")

    if dt.utcoffset().total_seconds() != 0:
        raise ValueError("as_of must be expressed in UTC")

    return dt.astimezone(timezone.utc)


def classify_regime(avg_corr: Optional[float]) -> Regime:
    """Correlation-driven regime. UNKNOWN is never silently mapped to CALM."""
    if avg_corr is None:
        return Regime.UNKNOWN

    if not _finite_number(avg_corr) or not -1.0 <= avg_corr <= 1.0:
        return Regime.UNKNOWN

    if avg_corr >= CORR_CRISIS:
        return Regime.CRISIS
    if avg_corr >= CORR_RISK_OFF:
        return Regime.RISK_OFF
    if avg_corr >= CORR_ELEVATED:
        return Regime.ELEVATED
    return Regime.CALM


def effective_breadth_var(n_assets: int, avg_corr: float) -> float:
    """Variance-based effective breadth.

    N_eff = N / (1 + (N-1) rho).

    Returns 0.0 for invalid input rather than fabricating usable breadth.
    """
    if not isinstance(n_assets, int) or isinstance(n_assets, bool):
        return 0.0
    if n_assets < 1:
        return 0.0
    if not _finite_number(avg_corr):
        return 0.0
    if not -1.0 <= avg_corr <= 1.0:
        return 0.0

    lo = -1.0 / max(n_assets - 1, 1)
    if not lo < avg_corr <= 1.0:
        return 0.0

    denom = 1.0 + (n_assets - 1) * avg_corr
    if not math.isfinite(denom) or denom <= 0.0:
        return 0.0

    result = n_assets / denom
    if not math.isfinite(result):
        return 0.0

    return float(min(result, n_assets))


@dataclass(frozen=True)
class MarketState:
    """Immutable observation bundle. One instance per decision cycle.

    as_of must be a timezone-aware UTC ISO-8601 timestamp.
    The caller remains responsible for point-in-time data correctness.
    """
    as_of: str
    avg_corr: Optional[float] = None
    vol_ann: Optional[float] = None
    liquidity_score: Optional[float] = None
    data_age_secs: Optional[int] = None
    model_disagreement: Optional[float] = None
    execution_divergent: bool = False
    drawdown_bps: int = 0
    drawdown_trigger_bps: int = 1_000
    tau_data_secs: int = 120
    cost_exceeds_edge: bool = False
    n_assets: int = 0

    def __post_init__(self) -> None:
        _parse_utc(self.as_of)

        if self.avg_corr is not None:
            if not _finite_number(self.avg_corr):
                raise ValueError("avg_corr must be finite")
            if not -1.0 <= self.avg_corr <= 1.0:
                raise ValueError("avg_corr must be in [-1, 1]")

        if self.vol_ann is not None:
            if not _finite_number(self.vol_ann) or self.vol_ann < 0.0:
                raise ValueError("vol_ann must be finite and >= 0")

        if self.liquidity_score is not None:
            if not _finite_number(self.liquidity_score):
                raise ValueError("liquidity_score must be finite")
            if not 0.0 <= self.liquidity_score <= 1.0:
                raise ValueError("liquidity_score must be in [0, 1]")

        if self.model_disagreement is not None:
            if not _finite_number(self.model_disagreement):
                raise ValueError("model_disagreement must be finite")
            if not 0.0 <= self.model_disagreement <= 1.0:
                raise ValueError("model_disagreement must be in [0, 1]")

        if self.data_age_secs is not None:
            if not _nonnegative_int(self.data_age_secs):
                raise ValueError("data_age_secs must be a nonnegative integer")

        if not isinstance(self.tau_data_secs, int) or isinstance(
            self.tau_data_secs, bool
        ):
            raise ValueError("tau_data_secs must be an integer")
        if self.tau_data_secs < 1:
            raise ValueError("tau_data_secs must be >= 1")

        if not isinstance(self.drawdown_bps, int) or isinstance(
            self.drawdown_bps, bool
        ):
            raise ValueError("drawdown_bps must be an integer")

        if not isinstance(self.drawdown_trigger_bps, int) or isinstance(
            self.drawdown_trigger_bps, bool
        ):
            raise ValueError("drawdown_trigger_bps must be an integer")
        if self.drawdown_trigger_bps < 1:
            raise ValueError("drawdown_trigger_bps must be >= 1")

        if not isinstance(self.n_assets, int) or isinstance(self.n_assets, bool):
            raise ValueError("n_assets must be an integer")
        if self.n_assets < 0:
            raise ValueError("n_assets must be >= 0")

        if not isinstance(self.execution_divergent, bool):
            raise ValueError("execution_divergent must be bool")

        if not isinstance(self.cost_exceeds_edge, bool):
            raise ValueError("cost_exceeds_edge must be bool")

    @property
    def regime(self) -> Regime:
        return classify_regime(self.avg_corr)

    @property
    def data_unknown(self) -> bool:
        """Volatility and liquidity are required for allocation decisions."""
        return self.vol_ann is None or self.liquidity_score is None

    @property
    def data_stale(self) -> bool:
        if self.data_age_secs is None:
            return True
        return self.data_age_secs > self.tau_data_secs

    @property
    def liquidity_shock(self) -> bool:
        if self.liquidity_score is None:
            return False
        return self.liquidity_score < LIQUIDITY_SHOCK

    @property
    def high_disagreement(self) -> bool:
        if self.model_disagreement is None:
            return False
        return self.model_disagreement >= DISAGREEMENT_HIGH

    @property
    def drawdown_breached(self) -> bool:
        return self.drawdown_bps >= self.drawdown_trigger_bps

    @property
    def breadth(self) -> float:
        if self.avg_corr is None or self.n_assets < 1:
            return 0.0
        return effective_breadth_var(self.n_assets, self.avg_corr)
