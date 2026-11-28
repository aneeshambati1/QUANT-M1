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
from quant_m1.research.probe import ProbeJournal


def record_sleeve_observation(
    journal: ProbeJournal,
    signal_name: str,
    position: float,
    realized_return: float,
    cost_bps: float,
    regime: str,
    selector_weight: float,
) -> Dict[str, Any]:
    """Record a single sleeve observation in the ProbeJournal.

    Args:
        journal: ProbeJournal instance to record into
        signal_name: Name of the signal generating this position
        position: Dollar amount of the paper position
        realized_return: Actual return realized over the holding period
        cost_bps: Transaction cost in basis points
        regime: Market regime when position was taken
        selector_weight: Weight assigned by meta-selector (0-1)

    Returns:
        Dict with observation details and journal confirmation

    Safety:
        - Does not grant authority directly
        - Observation is tagged for audit trail
        - Authority promotion still requires threshold crossing
    """
    # Calculate net return after costs
    cost_frac = cost_bps / 10000.0
    net_return = realized_return - cost_frac

    # Prepare observation metadata
    metadata = {
        'source': 'sleeve_exploration',
        'signal_name': signal_name,
        'regime': regime,
        'selector_weight': float(selector_weight),
        'position_usd': float(position),
        'gross_return': float(realized_return),
        'net_return': float(net_return),
        'cost_bps': float(cost_bps),
        'recorded_at': datetime.utcnow().isoformat() + 'Z',
    }

    # Record in ProbeJournal
    # The journal tracks these observations and computes running statistics
    observation_id = journal.record_observation(
        signal_name=signal_name,
        return_value=net_return,
        metadata=metadata,
    )

    return {
        'observation_id': observation_id,
        'signal_name': signal_name,
        'net_return': net_return,
        'recorded': True,
        'metadata': metadata,
    }


def process_sleeve_positions(
    journal: ProbeJournal,
    sleeve_positions: Dict[str, Dict[str, Any]],
    realized_returns: Dict[str, float],
    cost_bps: float = 5.0,
    regime: str = 'unknown',
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

        realized_return = realized_returns.get(signal_name, 0.0)

        # Record this observation
        obs = record_sleeve_observation(
            journal=journal,
            signal_name=signal_name,
            position=position_usd,
            realized_return=realized_return,
            cost_bps=cost_bps,
            regime=regime,
            selector_weight=selector_weight,
        )

        observations.append(obs)
        total_position += position_usd
        weighted_return += (position_usd * obs['net_return'])

    # Calculate portfolio-level statistics
    portfolio_return = weighted_return / total_position if total_position > 0 else 0.0

    summary = {
        'observations_recorded': len(observations),
        'total_position_usd': total_position,
        'portfolio_net_return': portfolio_return,
        'regime': regime,
        'observations': observations,
    }

    return summary


def get_sleeve_performance(journal: ProbeJournal, signal_name: str) -> Dict[str, Any]:
    """Retrieve performance statistics for a signal from sleeve observations.

    Queries the ProbeJournal for all observations tagged as 'sleeve_exploration'
    for the given signal and computes performance metrics.

    Args:
        journal: ProbeJournal instance
        signal_name: Name of the signal to query

    Returns:
        Dict with performance statistics:
        - n_observations: Number of sleeve observations
        - mean_return: Average net return
        - std_return: Standard deviation of returns
        - sharpe: Annualized Sharpe ratio
        - hit_rate: Fraction of positive returns
        - total_pnl: Sum of all position-weighted returns

    This data is used to determine if the signal has earned authority.
    """
    # Query journal for sleeve observations of this signal
    # Note: Actual query method depends on ProbeJournal implementation
    # This is a placeholder that should be implemented based on your journal API

    observations = journal.query_observations(
        signal_name=signal_name,
        source='sleeve_exploration',
    )

    if not observations:
        return {
            'n_observations': 0,
            'mean_return': 0.0,
            'std_return': 0.0,
            'sharpe': 0.0,
            'hit_rate': 0.0,
            'total_pnl': 0.0,
        }

    returns = [obs['net_return'] for obs in observations]
    positions = [obs['position_usd'] for obs in observations]

    returns_array = np.array(returns)
    positions_array = np.array(positions)

    mean_return = float(np.mean(returns_array))
    std_return = float(np.std(returns_array))
    hit_rate = float(np.mean(returns_array > 0))

    # Annualized Sharpe (assuming daily observations, 252 trading days)
    if std_return > 0:
        sharpe = (mean_return / std_return) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Total P&L weighted by position size
    total_pnl = float(np.sum(positions_array * returns_array))

    return {
        'n_observations': len(observations),
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe': sharpe,
        'hit_rate': hit_rate,
        'total_pnl': total_pnl,
    }
