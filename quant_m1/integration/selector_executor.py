"""Meta-selector → Cost-aware allocator integration.

Ensures that the meta-selector's regime-dependent weights actually influence
the cost-aware allocator's position sizing decisions.

The selector determines which signal families should be active in each regime.
The executor translates those weights into concrete position sizing parameters
for the cost-aware allocator.

Safety guarantees:
- Weights are normalized to sum to 1.0
- Zero-weight families receive zero allocation
- Risk Kernel veto still applies
- All positions remain paper-only
"""

from typing import Dict, List, Tuple
import numpy as np
from quant_m1.allocate.cost_aware import CostAwareAllocator


def apply_selector_weights(
    allocator: CostAwareAllocator,
    selector_weights: Dict[str, float],
    signal_families: Dict[str, List[str]],
) -> Dict[str, float]:
    """Apply selector weights to signal family allocations.

    The meta-selector outputs weights for signal families (e.g., momentum, reversal).
    This function translates those weights into allocation parameters that the
    cost-aware allocator uses when sizing positions.

    Args:
        allocator: CostAwareAllocator instance
        selector_weights: Dict mapping family_name → weight (0-1, should sum to 1.0)
            Example: {'momentum': 0.6, 'reversal': 0.3, 'carry': 0.1}
        signal_families: Dict mapping family_name → list of signal names
            Example: {'momentum': ['mom_12_1', 'mom_6_1'], 'reversal': ['rev_5']}

    Returns:
        Dict mapping signal_name → effective_weight for use in allocation

    Process:
        1. Normalize selector weights to sum to 1.0
        2. Distribute each family's weight equally among its member signals
        3. Return effective weights for each individual signal

    Example:
        selector_weights = {'momentum': 0.6, 'reversal': 0.4}
        signal_families = {
            'momentum': ['mom_12_1', 'mom_6_1'],
            'reversal': ['rev_5']
        }
        result = apply_selector_weights(allocator, selector_weights, signal_families)
        # result = {'mom_12_1': 0.3, 'mom_6_1': 0.3, 'rev_5': 0.4}
    """
    # Normalize weights
    total_weight = sum(selector_weights.values())
    if total_weight <= 0:
        # No active families, return zero weights for all signals
        all_signals = []
        for signals in signal_families.values():
            all_signals.extend(signals)
        return {signal: 0.0 for signal in all_signals}

    normalized = {k: v / total_weight for k, v in selector_weights.items()}

    # Distribute weights to individual signals
    effective_weights = {}
    for family_name, weight in normalized.items():
        signals = signal_families.get(family_name, [])
        if not signals:
            continue

        # Equal distribution within family
        per_signal_weight = weight / len(signals)
        for signal in signals:
            effective_weights[signal] = per_signal_weight

    return effective_weights


def validate_selector_integration(
    selector_weights: Dict[str, float],
    signal_families: Dict[str, List[str]],
    allocator_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate that selector weights are being properly integrated.

    Performs sanity checks on the selector → allocator integration to ensure
    weights are being applied correctly.

    Args:
        selector_weights: Weights from meta-selector
        signal_families: Signal family definitions
        allocator_state: Current state of the allocator (positions, weights, etc.)

    Returns:
        Validation report with status and any issues found

    Checks performed:
        1. Weights sum to ~1.0 (within floating point tolerance)
        2. All families in selector_weights exist in signal_families
        3. No negative weights
        4. Zero-weight families have zero allocation in allocator state
        5. High-weight families have proportionally higher allocation
    """
    issues = []
    warnings = []

    # Check 1: Weights sum to 1.0
    total_weight = sum(selector_weights.values())
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        issues.append(f"Weights sum to {total_weight:.6f}, expected 1.0")

    # Check 2: All families exist
    for family in selector_weights.keys():
        if family not in signal_families:
            issues.append(f"Family '{family}' in selector_weights but not in signal_families")

    # Check 3: No negative weights
    for family, weight in selector_weights.items():
        if weight < 0:
            issues.append(f"Family '{family}' has negative weight: {weight}")

    # Check 4: Zero-weight families have zero allocation
    # This requires inspecting allocator_state for actual allocations
    # Placeholder implementation - should be adapted to your allocator state structure
    allocations = allocator_state.get('allocations', {})
    for family, weight in selector_weights.items():
        if np.isclose(weight, 0.0, atol=1e-6):
            # Check that all signals in this family have zero allocation
            signals = signal_families.get(family, [])
            for signal in signals:
                if signal in allocations and allocations[signal] > 0:
                    issues.append(
                        f"Family '{family}' has zero weight but signal '{signal}' "
                        f"has allocation {allocations[signal]}"
                    )

    # Check 5: Proportional allocation (soft check)
    # Higher-weight families should generally have higher total allocation
    family_allocations = {}
    for family, signals in signal_families.items():
        total_alloc = sum(allocations.get(s, 0.0) for s in signals)
        family_allocations[family] = total_alloc

    # Compare highest-weight family vs lowest-weight family
    if len(selector_weights) >= 2:
        sorted_families = sorted(selector_weights.items(), key=lambda x: x[1], reverse=True)
        top_family = sorted_families[0]
        bottom_family = sorted_families[-1]

        if (top_family[1] > bottom_family[1] * 2 and  # Top should be significantly higher
            family_allocations.get(top_family[0], 0) < family_allocations.get(bottom_family[0], 0)):
            warnings.append(
                f"Highest-weight family '{top_family[0]}' (weight={top_family[1]:.3f}) "
                f"has lower allocation than lowest-weight family '{bottom_family[0]}' "
                f"(weight={bottom_family[1]:.3f})"
            )

    # Determine overall status
    if issues:
        status = 'FAILED'
    elif warnings:
        status = 'WARNING'
    else:
        status = 'PASSED'

    return {
        'status': status,
        'issues': issues,
        'warnings': warnings,
        'total_weight': total_weight,
        'family_allocations': family_allocations,
    }


def compute_regime_allocation(
    selector_weights: Dict[str, float],
    signal_families: Dict[str, List[str]],
    signal_strengths: Dict[str, float],
    risk_budget: float,
) -> Dict[str, float]:
    """Compute final allocation combining selector weights and signal strengths.

    This is the complete integration function that combines:
    1. Meta-selector regime weights (which families to use)
    2. Signal strengths within each family (which signals are strong)
    3. Risk budget (how much total capital to deploy)

    Args:
        selector_weights: Family weights from meta-selector (sum to 1.0)
        signal_families: Family → list of signals mapping
        signal_strengths: Signal → strength score (higher = stronger signal)
        risk_budget: Total capital available for allocation

    Returns:
        Dict mapping signal_name → dollar allocation

    Algorithm:
        1. For each family, compute weighted signal strengths
        2. Normalize within family so strongest signal gets most allocation
        3. Multiply by family weight from selector
        4. Scale by risk budget

    Example:
        selector_weights = {'momentum': 0.7, 'reversal': 0.3}
        signal_families = {
            'momentum': ['mom_12_1', 'mom_6_1'],
            'reversal': ['rev_5']
        }
        signal_strengths = {
            'mom_12_1': 2.5,
            'mom_6_1': 1.0,
            'rev_5': 1.5
        }
        risk_budget = 100000.0

        result = compute_regime_allocation(
            selector_weights, signal_families, signal_strengths, risk_budget
        )
        # result might be:
        # {'mom_12_1': 50000, 'mom_6_1': 20000, 'rev_5': 30000}
    """
    allocations = {}

    for family_name, signals in signal_families.items():
        family_weight = selector_weights.get(family_name, 0.0)

        if family_weight <= 0 or not signals:
            # Zero allocation for this family
            for signal in signals:
                allocations[signal] = 0.0
            continue

        # Get strengths for signals in this family
        family_strengths = {s: signal_strengths.get(s, 0.0) for s in signals}

        # Normalize strengths to probabilities (softmax-like)
        total_strength = sum(family_strengths.values())
        if total_strength <= 0:
            # Equal allocation if no strengths
            per_signal = family_weight * risk_budget / len(signals)
            for signal in signals:
                allocations[signal] = per_signal
        else:
            # Weighted allocation based on strength
            for signal, strength in family_strengths.items():
                weight_in_family = strength / total_strength
                allocation = family_weight * weight_in_family * risk_budget
                allocations[signal] = allocation

    return allocations
