"""Sleeve → ProbeJournal feedback loop.

Connects the bounded exploration sleeve to the ProbeJournal so provisional
models can earn authority through observed performance.

The sleeve generates paper positions based on signal strength. These positions
are recorded as observations in the ProbeJournal, which tracks whether the
sleeve's signals are predictive. If the sleeve demonstrates consistent
outperformance, the underlying signal models can earn authority.

Safety guarantees:
- Paper positions only (no live execution)
- Observations are tagged as "sleeve_exploration" for audit
- Authority still requires passing promotion thresholds
- Risk Kernel veto still applies to all positions
"""

from typing import Dict, Any
import numpy as np
from datetime import datetime


def record_sleeve_observation(
    journal: Any,
    signal_name: str,
    net_return: float,
    horizon_days: int = 1,
    selector_weight: float = 0.0,
    regime: str = "unknown",
) -> Dict[str, Any]:
    """Record a single sleeve observation in the ProbeJournal.

    Uses the real ProbeJournal API:
    1. journal.set_family(candidate, family) - register the candidate
    2. journal.record(candidate, horizon, value) - record the observation

    Args:
        journal: ProbeJournal instance (must have set_family, record methods)
        signal_name: Name of the signal generating this position
        net_return: Realized return after costs (decimal, e.g. 0.02 for 2%)
        horizon_days: Holding period in days (default 1)
        selector_weight: Weight assigned by meta-selector (0-1)
        regime: Market regime when position was taken

    Returns:
        Dict with observation details and journal confirmation

    Safety:
        - Does not grant authority directly
        - Observation is tagged for audit trail
        - Authority promotion still requires threshold crossing
    """
    # Build candidate key
    candidate = f"sleeve:{signal_name}:{regime}"
    family = "sleeve_exploration"

    # Register candidate family (required by ProbeJournal)
    try:
        journal.set_family(candidate, family)
    except Exception as e:
        # If set_family fails, journal might not support it - continue anyway
        pass

    # Record observation using real ProbeJournal API
    journal.record(candidate, horizon_days, float(net_return))

    return {
        "candidate": candidate,
        "family": family,
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

    This is the main integration point called after each sleeve execution cycle.
    It takes the positions the sleeve decided to take, the returns those positions
    actually realized, and records them in the ProbeJournal.

    Args:
        journal: ProbeJournal instance
        sleeve_positions: Dict mapping signal_name → position details
            Each entry should have: {'position_usd': float, 'selector_weight': float}
        realized_returns: Dict mapping signal_name → realized return
        cost_bps: Transaction cost in basis points (default 5.0)
        regime: Current market regime (default 'unknown')
        horizon_days: Holding period in days (default 1)

    Returns:
        Summary dict with counts and statistics

    Example:
        sleeve_positions = {
            'momentum_12_1': {'position_usd': 1000.0, 'selector_weight': 0.3},
            'mean_reversion': {'position_usd': 500.0, 'selector_weight': 0.2},
        }
        realized_returns = {
            'momentum_12_1': 0.02,
            'mean_reversion': -0.01,
        }
        result = process_sleeve_positions(journal, sleeve_positions, realized_returns)
    """
    observations = []
    total_position = 0.0
    weighted_return = 0.0

    for signal_name, pos_details in sleeve_positions.items():
        position_usd = pos_details.get('position_usd', 0.0)
        selector_weight = pos_details.get('selector_weight', 0.0)

        if position_usd == 0.0:
            continue

        gross_return = realized_returns.get(signal_name, 0.0)

        # Calculate net return after costs
        cost_frac = cost_bps / 10000.0
        net_return = gross_return - cost_frac

        # Record this observation
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

    # Calculate portfolio-level statistics
    portfolio_return = weighted_return / total_position if total_position > 0 else 0.0

    summary = {
        'observations_recorded': len(observations),
        'total_position_usd': total_position,
        'portfolio_net_return': portfolio_return,
        'regime': regime,
        'horizon_days': horizon_days,
        'observations': observations,
    }

    return summary
