"""PFCS scaffolding tests: governance correctness, NOT edge detection."""
from __future__ import annotations

import numpy as np
import pytest

from quant_m1.research.pfcs import (
    CaseDefinition, CaseRegistry, CaseStats, build_case_policy_weights,
    classify_episode, hierarchical_shrinkage, power_gate,
)


def _case(cid="high_vol", min_obs=16, min_eff=0.01):
    return CaseDefinition(
        case_id=cid, description="test case",
        observable_fn_id="vol_above_median_v1",
        min_independent_obs=min_obs, min_effect_size=min_eff,
        created_at="2026-11-26T00:00:00Z",
    )


def _registry(cases=None, alpha=0.05):
    return CaseRegistry(cases=tuple(cases or [_case()]), family_alpha=alpha)


def test_duplicate_case_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        CaseRegistry(cases=(_case("a"), _case("a")))


def test_bonferroni_threshold_increases_with_cases():
    r1 = _registry([_case("a")])
    r5 = _registry([_case(f"c{i}") for i in range(5)])
    r50 = _registry([_case(f"c{i}") for i in range(50)])
    assert r5.bonferroni_threshold > r1.bonferroni_threshold
    assert r50.bonferroni_threshold > r5.bonferroni_threshold


def test_registry_digest_changes_with_case_set():
    d1 = _registry([_case("a")]).compute_registry_digest()
    d2 = _registry([_case("a"), _case("b")]).compute_registry_digest()
    assert d1 != d2


def test_insufficient_obs_blocks_activation():
    cd = _case(min_obs=16)
    stats = CaseStats(n_independent_obs=10, mean_effect=0.5, var_effect=0.01)
    ok, reason = power_gate(stats, cd, family_threshold_z=2.89)
    assert not ok and "INSUFFICIENT_OBS" in reason


def test_below_family_threshold_blocks():
    cd = _case(min_obs=5, min_eff=0.001)
    stats = CaseStats(n_independent_obs=20, mean_effect=0.001, var_effect=1.0)
    ok, reason = power_gate(stats, cd, family_threshold_z=2.89)
    assert not ok and "BELOW_FAMILY_THRESHOLD" in reason


def test_below_min_effect_blocks():
    cd = _case(min_obs=5, min_eff=0.1)
    stats = CaseStats(n_independent_obs=100, mean_effect=0.001, var_effect=1e-8)
    ok, reason = power_gate(stats, cd, family_threshold_z=1.96)
    assert not ok and "BELOW_MIN_EFFECT" in reason


def test_all_gates_pass():
    cd = _case(min_obs=5, min_eff=0.01)
    stats = CaseStats(n_independent_obs=50, mean_effect=0.5, var_effect=0.01)
    ok, reason = power_gate(stats, cd, family_threshold_z=2.58)
    assert ok and reason == "PASSED"


def test_zero_variance_blocks():
    cd = _case(min_obs=5)
    stats = CaseStats(n_independent_obs=50, mean_effect=0.5, var_effect=0.0)
    ok, reason = power_gate(stats, cd, family_threshold_z=2.0)
    assert not ok and "INSUFFICIENT_VARIANCE" in reason


def test_small_tau_shrinks_to_pooled():
    stats = {"a": CaseStats(mean_effect=1.0, var_effect=0.01, pooled_mean=0.0)}
    result = hierarchical_shrinkage(stats, tau_sq=1e-10)
    assert abs(result["a"]) < 0.01


def test_large_tau_preserves_case_estimate():
    stats = {"a": CaseStats(mean_effect=1.0, var_effect=0.01, pooled_mean=0.0)}
    result = hierarchical_shrinkage(stats, tau_sq=1e6)
    assert abs(result["a"] - 1.0) < 0.01


def test_high_within_variance_shrinks_more():
    stats = {
        "precise": CaseStats(mean_effect=1.0, var_effect=0.001, pooled_mean=0.0),
        "noisy":   CaseStats(mean_effect=1.0, var_effect=1.0,   pooled_mean=0.0),
    }
    result = hierarchical_shrinkage(stats, tau_sq=0.1)
    assert abs(result["precise"]) > abs(result["noisy"])


def test_negative_tau_rejected():
    with pytest.raises(ValueError):
        hierarchical_shrinkage({}, tau_sq=-1.0)


def test_no_active_case_returns_pooled():
    reg = _registry()
    pooled = np.array([0.5, 0.3, 0.2])
    w, diag = build_case_policy_weights(reg, {}, None, tau_sq=0.1,
                                         pooled_weights=pooled, case_weight_deltas={})
    assert np.allclose(w, pooled)
    assert diag["_active"] == "POOLED_DEFAULT"


def test_unknown_case_returns_pooled():
    reg = _registry()
    pooled = np.array([0.5, 0.3, 0.2])
    w, diag = build_case_policy_weights(reg, {}, "nonexistent", tau_sq=0.1,
                                         pooled_weights=pooled, case_weight_deltas={})
    assert np.allclose(w, pooled)
    assert "UNKNOWN_CASE" in diag["_active"]


def test_insufficient_power_returns_pooled():
    reg = _registry([_case(min_obs=100)])
    stats = {"high_vol": CaseStats(n_independent_obs=5, mean_effect=0.5, var_effect=0.01)}
    pooled = np.array([0.5, 0.3, 0.2])
    delta = {"high_vol": np.array([0.1, -0.05, -0.05])}
    w, diag = build_case_policy_weights(reg, stats, "high_vol", tau_sq=0.1,
                                         pooled_weights=pooled, case_weight_deltas=delta)
    assert np.allclose(w, pooled)
    assert "INSUFFICIENT_OBS" in diag["high_vol"]


def test_activated_case_applies_delta():
    cd = _case(min_obs=5, min_eff=0.01)
    reg = _registry([cd])
    stats = {"high_vol": CaseStats(n_independent_obs=50, mean_effect=0.5,
                                    var_effect=0.01, pooled_mean=0.0)}
    pooled = np.array([0.4, 0.3, 0.3])
    delta = {"high_vol": np.array([0.2, -0.1, -0.1])}
    w, diag = build_case_policy_weights(reg, stats, "high_vol", tau_sq=1.0,
                                         pooled_weights=pooled, case_weight_deltas=delta)
    assert diag["high_vol"] == "ACTIVATED"
    assert w[0] > pooled[0]
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w >= 0)


def test_adjusted_weights_clipped_and_renormalized():
    cd = _case(cid="clip_test", min_obs=5, min_eff=0.001)
    reg = _registry([cd])
    stats = {"clip_test": CaseStats(n_independent_obs=50, mean_effect=1.0,
                                     var_effect=0.001, pooled_mean=0.0)}
    pooled = np.array([0.1, 0.1, 0.8])
    delta = {"clip_test": np.array([-0.5, 0.3, 0.2])}
    w, diag = build_case_policy_weights(reg, stats, "clip_test", tau_sq=10.0,
                                         pooled_weights=pooled, case_weight_deltas=delta)
    assert diag["clip_test"] == "ACTIVATED"
    assert np.all(w >= 0)
    assert abs(w.sum() - 1.0) < 1e-9


def test_current_panel_is_not_identifiable():
    """With ~16 independent regimes and min_obs=16 per case, at most ONE case
    can qualify. Multi-case specialization is NOT IDENTIFIABLE."""
    n_regimes = 16
    cases = [_case(f"c{i}", min_obs=16) for i in range(5)]
    total_needed = sum(c.min_independent_obs for c in cases)
    assert total_needed > n_regimes
    max_possible_qualified = n_regimes // min(c.min_independent_obs for c in cases)
    assert max_possible_qualified <= 1
