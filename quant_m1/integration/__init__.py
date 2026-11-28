"""Integration bridges between QUANT-M1 subsystems.

This module connects isolated components that need to work together:
- sleeve_probe: Feeds sleeve observations into ProbeJournal for authority earning
- selector_executor: Ensures meta-selector weights influence actual allocation decisions

All functions maintain safety guarantees: no live trading, no authority bypass,
paper-only execution with Risk Kernel veto.
"""

from quant_m1.integration.sleeve_probe import (
    record_sleeve_observation,
    process_sleeve_positions,
)
from quant_m1.integration.selector_executor import (
    apply_selector_weights,
    validate_selector_integration,
)

__all__ = [
    'record_sleeve_observation',
    'process_sleeve_positions',
    'apply_selector_weights',
    'validate_selector_integration',
]
