"""Sleeve → ProbeJournal feedback loop.

Connects the bounded exploration sleeve to the ProbeJournal so provisional
models can earn authority through observed performance.

Uses the REAL ProbeJournal API: journal.record(candidate, horizon, value).
Sleeve observations are stored as numeric return values keyed by signal name
and horizon. Metadata is embedded in the candidate key for auditability.

Safety guarantees:
- Paper positions only (no live execution)
- Observations tagged via candidate key prefix "sleeve:"
- Authority still requires passing promotion thresholds
- Risk Kernel veto still applies to all positions
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np


def record_sleeve_observation(
    journal: Any,
    signal_name: str,
    net_return: float,
    horizon_days: int = 1,
    selector_weight: float = 0.0,
    regime: str = "unknown",
) -> Dict[str, Any]:
    """Record a single sleeve observation in the ProbeJournal.

    Uses the real journal.record(candidate, horizon, value) API.
    The candidate key is prefixed with 'sleeve:' for audit trail.

    Args:
        journal: ProbeJournal instance (must have .record() method)
        signal_name: Name of the signal generating this position
        net_return: Realized return after costs (decimal, e.g. 0.02 for 2%)
        horizon_days: Holding period in days (default 1)
        selector_weight: Weight assigned by meta-selector (0-1)
        regime: Market regime when position was taken

    Returns:
        Dict with observation details and confirmation
    """
    # Build auditable candidate key
    candidate = f"sleeve:{signal_name}:{regime}"

    # Record using real ProbeJournal API
    journal.record(candidate, horizon_days, float(net_return))

    return {
        "candidate": candidate,
        "signal_name": signal_name,
        "horizon_days": horizon_days,
        "net_return": float(net_return),
        "selector_weight": float(selector_weight),
        "regime": regime,
        "recorded": True,
    }


def process_sleeve_positions(
    journal: Any,
    sleeve_positions: Dict[str, Dict[str, Any]],
    realized_returns: Dict[str, float],
    cost_bps: float = 5.0,
    regime: str = "unknown",
    horizon_days: int = 1,
) -> Dict[str, Any]:
    """Process all sleeve positions and record observations.

    Main integration point called after each sleeve execution cycle.
    Takes positions the sleeve decided to take, the returns those positions
    actually realized, and records them in the ProbeJournal.

    Args:
        journal: ProbeJournal instance
        sleeve_positions: Dict mapping signal_name → position details
            Each entry should have: {'position_usd': float, 'selector_weight': float}
        realized_returns: Dict mapping signal_name → realized gross return
        cost_bps: Transaction cost in basis points (default 5.0)
        regime: Current market regime (default 'unknown')
        horizon_days: Holding period in days (default 1)

    Returns:
        Summary dict with counts and statistics
    """
    observations = []
    total_position = 0.0
    weighted_return = 0.0

    for signal_name, pos_details in sleeve_positions.items():
        position_usd = pos_details.get("position_usd", 0.0)
        selector_weight = pos_details.get("selector_weight", 0.0)

        if position_usd == 0.0:
            continue

        gross_return = realized_returns.get(signal_name, 0.0)
        cost_frac = cost_bps / 10000.0
        net_return = gross_return - cost_frac

        obs = record_sleeve_observation(
            journal=journal,
            signal_name=signal_name,
            net_return=net_return,
            horizon_days=horizon_days,
            selector_weight=selector_weight,
            regime=regime,
        )

        observations.append(obs)
        total_position += position_usd
        weighted_return += position_usd * net_return

    portfolio_return = weighted_return / total_position if total_position > 0 else 0.0

    return {
        "observations_recorded": len(observations),
        "total_position_usd": float(total_position),
        "portfolio_net_return": float(portfolio_return),
        "regime": regime,
        "horizon_days": horizon_days,
        "observations": observations,
    }


def get_sleeve_performance(journal: Any, signal_name: str, horizon_days: int = 1) -> Dict[str, Any]:
    """Retrieve performance statistics for a signal from sleeve observations.

    Queries the ProbeJournal for observations with candidate key matching
    the given signal name.

    Args:
        journal: ProbeJournal instance (must have .n_obs() and .certificate_tracker())
        signal_name: Name of the signal to query
        horizon_days: Horizon to query (default 1)

    Returns:
        Dict with performance statistics
    """
    candidate_pattern = f"sleeve:{signal_name}:"

    try:
        n = journal.n_obs(candidate_pattern, horizon_days)
    except Exception:
        n = 0

    if n == 0:
        return {
            "signal_name": signal_name,
            "n_observations": 0,
            "mean_return": 0.0,
            "std_return": 0.0,
            "sharpe": 0.0,
        }

    try:
        cert = journal.certificate_tracker(candidate_pattern, horizon_days)
        mean_ret = float(cert.mean) if hasattr(cert, "mean") else 0.0
        std_ret = float(cert.std) if hasattr(cert, "std") else 0.0
    except Exception:
        mean_ret = 0.0
        std_ret = 0.0

    ann_factor = 252.0 / max(horizon_days, 1)
    sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor) if std_ret > 0 else 0.0

    return {
        "signal_name": signal_name,
        "n_observations": int(n),
        "mean_return": float(mean_ret),
        "std_return": float(std_ret),
        "sharpe": float(sharpe),
    }
