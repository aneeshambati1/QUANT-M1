"""Market state classification. Seed of Organ 1 (Unified Market World Model).

Every field is Optional because UNKNOWN must be representable. A missing
observation is not zero; treating it as zero is failure mode D7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Regime(Enum):
    UNKNOWN = "unknown"
    CALM = "calm"
    ELEVATED = "elevated"
    RISK_OFF = "risk_off"
    CRISIS = "crisis"


# ASSUMPTION: these thresholds are POLICY choices, taken from the dossier's
# correlation ladder (Part 4.6). They are not measured calibration.
CORR_ELEVATED = 0.60
CORR_RISK_OFF = 0.75
CORR_CRISIS = 0.85

# ASSUMPTION: liquidity below this fraction of normal is a shock.
LIQUIDITY_SHOCK = 0.35

# ASSUMPTION: dispersion of model forecasts above this is high disagreement.
DISAGREEMENT_HIGH = 0.60


def classify_regime(avg_corr: Optional[float]) -> Regime:
    """Correlation-driven regime. UNKNOWN is never silently mapped to CALM."""
    if avg_corr is None:
        return Regime.UNKNOWN
    if not math.isfinite(avg_corr):
        return Regime.UNKNOWN
    if not -1.0 <= avg_corr <= 1.0:
        return Regime.UNKNOWN
    if avg_corr >= CORR_CRISIS:
        return Regime.CRISIS
    if avg_corr >= CORR_RISK_OFF:
        return Regime.RISK_OFF
    if avg_corr >= CORR_ELEVATED:
        return Regime.ELEVATED
    return Regime.CALM


def effective_breadth_var(n_assets: int, avg_corr: float) -> float:
    """Variance-based effective breadth. Adopted conservative measure.

    N_eff = N / (1 + (N-1) rho). Returns 0.0 for invalid input rather than
    guessing, so a caller cannot accidentally trade on a fabricated breadth.
    """
    if n_assets < 1:
        return 0.0
    if not math.isfinite(avg_corr):
        return 0.0
    lo = -1.0 / max(n_assets - 1, 1)
    if not lo < avg_corr <= 1.0:
        return 0.0
    denom = 1.0 + (n_assets - 1) * avg_corr
    if denom <= 0:
        return 0.0
    return float(min(n_assets / denom, n_assets))


@dataclass(frozen=True)
class MarketState:
    """Immutable observation bundle. One instance per decision cycle.

    as_of must be tz-aware UTC ISO-8601. The caller is responsible for PIT
    correctness; this class does not fetch data and cannot introduce lookahead.
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
        if not self.as_of:
            raise ValueError("as_of required")
        if self.avg_corr is not None and not -1.0 <= self.avg_corr <= 1.0:
            raise ValueError("avg_corr must be in [-1, 1]")
        if self.vol_ann is not None and self.vol_ann < 0:
            raise ValueError("vol_ann must be >= 0")
        if self.liquidity_score is not None and not 0.0 <= self.liquidity_score <= 1.0:
            raise ValueError("liquidity_score must be in [0, 1]")
        if self.model_disagreement is not None and not 0.0 <= self.model_disagreement <= 1.0:
            raise ValueError("model_disagreement must be in [0, 1]")
        if self.data_age_secs is not None and self.data_age_secs < 0:
            raise ValueError("data_age_secs must be >= 0")
        if self.tau_data_secs < 1:
            raise ValueError("tau_data_secs must be >= 1")
        if self.drawdown_trigger_bps < 1:
            raise ValueError("drawdown_trigger_bps must be >= 1")
        if self.n_assets < 0:
            raise ValueError("n_assets must be >= 0")

    @property
    def regime(self) -> Regime:
        return classify_regime(self.avg_corr)

    @property
    def data_unknown(self) -> bool:
        """Volatility and liquidity are required for any allocation decision.

        Correlation is regime-classification input, not a gating observation.
        A state with known vol/liquidity but unknown correlation reaches R9
        (REGIME_UNKNOWN) rather than being blocked at R2. This preserves the
        policy's ability to distinguish 'world partially observed' from
        'world unobserved'.
        """
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