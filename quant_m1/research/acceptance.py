"""Acceptance gate: wires DSR + PBO into the trial ledger as DIAGNOSTICS
alongside the existing Bonferroni/HAC t-stat gate.

DSR and PBO are NOT replacements for the primary chronological test. They are
selection-accounting diagnostics that quantify how much of an observed Sharpe
ratio could be explained by multiple testing (DSR) and how unstable the best
config is across CSCV splits (PBO). Neither repairs non-PIT data.

The existing ic_gate remains the PRIMARY acceptance criterion. This module
adds a secondary diagnostic layer that must also be satisfied before promotion.

FAMILY COUNTING: the family size for DSR comes from the IMMUTABLE trial
ledger depth, not from a user-supplied number. Every registered trial counts,
including failed ones. You cannot reduce the family by discarding losers.

NO DEFAULTS: thresholds, score function and embargo are injected. This module
makes no acceptance decision on its own.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

import numpy as np

from .dsr import deflated_sharpe
from .governor import TrialLedger
from .pbo import pbo_cscv


def acceptance_diagnostics(
    ledger: TrialLedger,
    returns_by_config: Dict[str, np.ndarray],
    *,
    selected_config: str,
    n_obs: int,
    periods_per_year: int = 252,
    dsr_threshold: float = 0.05,
    pbo_threshold: float = 0.50,
    sigma_sr: float = 1.0,
    n_groups: int = 10,
    embargo: int = 0,
    label_horizon: int = 1,
    score_fn: Optional[Callable] = None,
    family_override: Optional[int] = None,
) -> Dict:
    """Compute DSR and PBO diagnostics for a selected config.

    Args:
        ledger: TrialLedger instance (family count derived from registry depth).
        returns_by_config: {config_name: 1-D return array} for ALL configs tried.
        selected_config: name of the config being evaluated.
        n_obs: number of return observations at observation frequency.
        periods_per_year: annualization factor (252 for daily).
        dsr_threshold: minimum DSR p-value to pass (default 0.05).
        pbo_threshold: maximum PBO to pass (default 0.50).
        sigma_sr: cross-trial Sharpe std for DSR expected-max calculation.
        n_groups: CSCV group count (must be even).
        embargo: embargo rows after test block in CSCV.
        label_horizon: forward-return horizon for CSCV purge.
        score_fn: scoring function for PBO (default: annualized Sharpe).
        family_override: if set, use this instead of ledger.n_trials(). ONLY for
            pre-registered sub-family analyses. Default None = use full ledger.

    Returns dict with:
        dsr: full DSR output dict
        pbo: full PBO output dict
        family_size: number of trials counted
        dsr_passed: bool
        pbo_passed: bool
        passed: bool (both must pass)
    """
    if selected_config not in returns_by_config:
        raise ValueError(f"selected_config '{selected_config}' not in returns_by_config")
    if len(returns_by_config) < 2:
        raise ValueError("need >= 2 configs for PBO")

    family_size = family_override if family_override is not None else ledger.n_trials()
    if family_size < 1:
        raise ValueError("family_size must be >= 1")

    sel_returns = np.asarray(returns_by_config[selected_config], dtype=float)
    if len(sel_returns) != n_obs:
        raise ValueError(f"selected config has {len(sel_returns)} obs, expected {n_obs}")
    sr_ann = float(np.mean(sel_returns) / np.std(sel_returns, ddof=1)
                   * np.sqrt(periods_per_year)) if np.std(sel_returns, ddof=1) > 1e-12 else 0.0
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurt
    sk = float(sp_skew(sel_returns, bias=False))
    ku = float(sp_kurt(sel_returns, bias=False)) + 3.0

    dsr_result = deflated_sharpe(
        sr_ann=sr_ann, n_obs=n_obs, n_trials=family_size,
        skew=sk, kurtosis_raw=ku, periods_per_year=periods_per_year,
        sigma_sr=sigma_sr,
    )

    pbo_result = pbo_cscv(
        returns_by_config, n_groups=n_groups, label_horizon=label_horizon,
        embargo=embargo, score_fn=score_fn,
    )

    dsr_passed = dsr_result["dsr"] >= dsr_threshold
    pbo_passed = pbo_result["pbo"] <= pbo_threshold

    return {
        "dsr": dsr_result,
        "pbo": pbo_result,
        "family_size": family_size,
        "dsr_threshold": dsr_threshold,
        "pbo_threshold": pbo_threshold,
        "dsr_passed": dsr_passed,
        "pbo_passed": pbo_passed,
        "passed": dsr_passed and pbo_passed,
    }
