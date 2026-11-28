"""Fail-closed sleeve -> ProbeJournal research adapter.

Records realized, after-cost sleeve observations in the hash-chained
ProbeJournal. This adapter does not grant authority or enable live trading.

Call only after the stated holding period has actually completed.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from collections.abc import Mapping
from typing import Any


SLEEVE_FAMILY = "sleeve_exploration"


def _finite_number(value: Any, name: str) -> float:
    """Convert a finite real number, rejecting booleans and invalid values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def _validate_horizon(horizon_days: int) -> None:
    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, Integral)
        or horizon_days < 1
    ):
        raise ValueError("horizon_days must be an integer >= 1")


def record_sleeve_observation(
    journal: Any,
    signal_name: str,
    net_return: float,
    horizon_days: int = 1,
    selector_weight: float = 0.0,
    regime: str = "unknown",
) -> dict:
    """Register and record one completed sleeve observation.

    The candidate key is stable for each signal/regime pair.
    Family registration errors are deliberately not swallowed.
    """
    if not isinstance(signal_name, str) or not signal_name.strip():
        raise ValueError("signal_name must be non-empty")

    if not isinstance(regime, str) or not regime.strip():
        raise ValueError("regime must be non-empty")

    _validate_horizon(horizon_days)

    realized = _finite_number(net_return, "net_return")
    weight = _finite_number(selector_weight, "selector_weight")

    if not 0.0 <= weight <= 1.0:
        raise ValueError("selector_weight must be in [0, 1]")

    candidate = f"sleeve:{signal_name}:{regime}"

    # ProbeJournal requires immutable family registration before recording.
    # Do not suppress errors: a family conflict must stop the observation.
    journal.set_family(candidate, SLEEVE_FAMILY)
    journal.record(candidate, int(horizon_days), realized)

    return {
        "candidate": candidate,
        "family": SLEEVE_FAMILY,
        "signal_name": signal_name,
        "horizon_days": int(horizon_days),
        "net_return": realized,
        "selector_weight": weight,
        "regime": regime,
        "recorded": True,
    }


def process_sleeve_positions(
    journal: Any,
    sleeve_positions: Mapping[str, Mapping[str, Any]],
    realized_returns: Mapping[str, float],
    cost_bps: float = 5.0,
    regime: str = "unknown",
    horizon_days: int = 1,
) -> dict:
    """Record completed, nonzero sleeve positions.

    Missing realized returns are errors, NOT zero-return observations.

    cost_bps is the total transaction-cost drag in basis points applicable
    to each supplied realized position return. The caller is responsible
    for supplying the correct cost basis for the completed trade.
    """
    if not isinstance(sleeve_positions, Mapping):
        raise ValueError("sleeve_positions must be a mapping")

    if not isinstance(realized_returns, Mapping):
        raise ValueError("realized_returns must be a mapping")

    _validate_horizon(horizon_days)

    if not isinstance(regime, str) or not regime.strip():
        raise ValueError("regime must be non-empty")

    costs = _finite_number(cost_bps, "cost_bps")
    if costs < 0:
        raise ValueError("cost_bps must be >= 0")

    cost_frac = costs / 10_000.0

    # Validate every position before writing anything to the journal.
    # This prevents a missing result halfway through the batch from
    # silently producing a partially recorded batch.
    prepared = []

    for signal_name, details in sleeve_positions.items():
        if not isinstance(signal_name, str) or not signal_name.strip():
            raise ValueError("position signal names must be non-empty")

        if not isinstance(details, Mapping):
            raise ValueError(f"position details invalid for {signal_name}")

        position_usd = _finite_number(
            details.get("position_usd", 0.0),
            f"{signal_name}.position_usd",
        )

        if position_usd < 0:
            raise ValueError("negative positions are not allowed")

        # No position means no observation.
        if position_usd == 0:
            continue

        if signal_name not in realized_returns:
            raise ValueError(
                f"missing realized return for active position: {signal_name}"
            )

        gross_return = _finite_number(
            realized_returns[signal_name],
            f"{signal_name}.realized_return",
        )

        selector_weight = _finite_number(
            details.get("selector_weight", 0.0),
            f"{signal_name}.selector_weight",
        )

        if not 0.0 <= selector_weight <= 1.0:
            raise ValueError(
                f"{signal_name}.selector_weight must be in [0, 1]"
            )

        net_return = gross_return - cost_frac

        prepared.append({
            "signal_name": signal_name,
            "position_usd": position_usd,
            "selector_weight": selector_weight,
            "net_return": net_return,
        })

    # Only write after all batch inputs have passed validation.
    observations = []
    total_position = 0.0
    weighted_return = 0.0

    for item in prepared:
        obs = record_sleeve_observation(
            journal=journal,
            signal_name=item["signal_name"],
            net_return=item["net_return"],
            horizon_days=int(horizon_days),
            selector_weight=item["selector_weight"],
            regime=regime,
        )

        observations.append(obs)
        total_position += item["position_usd"]
        weighted_return += (
            item["position_usd"] * item["net_return"]
        )

    portfolio_return = (
        weighted_return / total_position
        if total_position > 0
        else 0.0
    )

    return {
        "observations_recorded": len(observations),
        "total_position_usd": float(total_position),
        "portfolio_net_return": float(portfolio_return),
        "regime": regime,
        "horizon_days": int(horizon_days),
        "observations": observations,
    }
