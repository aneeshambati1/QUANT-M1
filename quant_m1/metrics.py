"""
Metric schema: no unitless, no horizon-ambiguous, no gross/net-ambiguous metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Unit(Enum):
    BPS = "bps"
    USD = "usd"
    RATIO = "ratio"
    COUNT = "count"
    PERCENT = "percent"


class Horizon(Enum):
    PER_BAR = "per_bar"
    DAILY = "daily"
    ANNUAL_252 = "annual_252_daily"
    FULL_PATH = "full_path"
    PER_TRADE = "per_trade"


class GrossNet(Enum):
    GROSS = "gross"
    NET = "net"


RETURN_LIKE = {"return", "sharpe", "cagr", "sortino", "calmar", "alpha"}


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    unit: Unit
    horizon: Horizon
    n_obs: int
    gross_net: Optional[GrossNet] = None
    n_eff: Optional[int] = None
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None
    config_hash: str = ""
    periods_per_year: Optional[int] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"Metric {self.name}: non-finite value")
        if self.n_obs < 0:
            raise ValueError("n_obs must be >= 0")
        if self.n_eff is not None and self.n_eff <= 0:
            raise ValueError("n_eff must be positive")
        if self.name.lower() in RETURN_LIKE and self.gross_net is None:
            raise ValueError(f"Metric {self.name} must declare gross/net")
        if self.horizon == Horizon.ANNUAL_252 and not self.periods_per_year:
            raise ValueError("annualized metric must declare periods_per_year")
        if self.ci_lo is not None and self.ci_hi is not None and self.ci_lo > self.ci_hi:
            raise ValueError("ci_lo > ci_hi")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.value,
            "horizon": self.horizon.value,
            "n_obs": self.n_obs,
            "gross_net": self.gross_net.value if self.gross_net else None,
            "n_eff": self.n_eff,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "config_hash": self.config_hash,
            "periods_per_year": self.periods_per_year,
        }


def assert_no_ambiguous_metrics(metrics: list[Metric]) -> None:
    for m in metrics:
        if m.name.lower() in RETURN_LIKE and m.gross_net is None:
            raise ValueError(f"ambiguous gross/net metric: {m.name}")
        if m.horizon == Horizon.ANNUAL_252 and not m.periods_per_year:
            raise ValueError(f"ambiguous annualization: {m.name}")
