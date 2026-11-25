"""Meta-layer acceptance tests.

Covers: regime boundaries, policy totality and priority, EWMA half-life,
LCB gating, probation, retirement, idle decay additivity, all-zero weights.
"""
from __future__ import annotations

import itertools
import math

import pytest

from quant_m1.meta import (
    MarketState, Regime, classify_regime,
    Action, decide, RULE_ORDER,
    ContributionTracker, ModelStatus, marginal_contribution, weights,
)
from quant_m1.meta.contribution import authority_report, _ewma_lambda
from quant_m1.meta.state import (
    CORR_CRISIS, CORR_ELEVATED, CORR_RISK_OFF,
    DISAGREEMENT_HIGH, LIQUIDITY_SHOCK, effective_breadth_var,
)


def _state(**kw) -> MarketState:
    base = dict(as_of="2026-11-25T14:30:00+00:00", avg_corr=0.436,
                vol_ann=0.155, liquidity_score=1.0, data_age_secs=10,
                model_disagreement=0.1, n_assets=46)
    base.update(kw)
    return MarketState(**base)


# ---- regime ----

@pytest.mark.parametrize("corr,expected", [
    (None, Regime.UNKNOWN),
    (float("nan"), Regime.UNKNOWN),
    (1.5, Regime.UNKNOWN),
    (0.436, Regime.CALM),
    (CORR_ELEVATED - 1e-9, Regime.CALM),
    (CORR_ELEVATED, Regime.ELEVATED),
    (CORR_RISK_OFF, Regime.RISK_OFF),
    (CORR_CRISIS, Regime.CRISIS),
    (0.95, Regime.CRISIS),
])
def test_regime_boundaries(corr, expected):
    assert classify_regime(corr) is expected


def test_breadth_matches_adopted_conservative_measure():
    # Dossier: effective_breadth(46, 0.436) = 2.2308 (variance-based).
    assert effective_breadth_var(46, 0.436) == pytest.approx(2.2308, abs=1e-3)


def test_breadth_collapses_in_crisis():
    calm = effective_breadth_var(46, 0.436)
    crisis = effective_breadth_var(46, 0.95)
    assert crisis < calm
    assert crisis == pytest.approx(1.05, abs=0.02)


def test_breadth_invalid_returns_zero_not_guess():
    assert effective_breadth_var(0, 0.5) == 0.0
    assert effective_breadth_var(10, float("nan")) == 0.0


def test_unknown_is_not_calm():
    s = _state(avg_corr=None)
    assert s.regime is Regime.UNKNOWN
    assert s.data_unknown is True


# ---- policy totality and priority ----

def test_policy_is_total_over_state_grid():
    """Every reachable state combination yields exactly one action."""
    seen = set()
    for corr, liq, age, dis, div, dd, cost in itertools.product(
        [None, 0.436, 0.65, 0.80, 0.90],
        [None, 0.1, 1.0],
        [None, 10, 500],
        [None, 0.1, 0.9],
        [False, True],
        [0, 2000],
        [False, True],
    ):
        s = _state(avg_corr=corr, liquidity_score=liq, data_age_secs=age,
                   model_disagreement=dis, execution_divergent=div,
                   drawdown_bps=dd, cost_exceeds_edge=cost)
        d = decide(s)
        assert isinstance(d.action, Action)
        assert d.rule_id in RULE_ORDER
        seen.add(d.rule_id)
    assert seen == set(RULE_ORDER), f"unreachable rules: {set(RULE_ORDER) - seen}"


def test_divergence_dominates_every_other_condition():
    s = _state(execution_divergent=True, drawdown_bps=5000,
               avg_corr=0.99, liquidity_score=0.0, data_age_secs=9999)
    d = decide(s)
    assert d.action is Action.LATCH_FREEZE
    assert d.permits_new_risk is False
    assert d.permits_reduce is False


def test_unknown_data_freezes_but_allows_reduce():
    d = decide(_state(avg_corr=None))
    assert d.action is Action.FREEZE_NEW_RISK
    assert d.permits_new_risk is False
    assert d.permits_reduce is True


def test_stale_data_freezes():
    d = decide(_state(data_age_secs=121, tau_data_secs=120))
    assert d.rule_id == "R3_DATA_STALE"


def test_liquidity_shock_halts():
    d = decide(_state(liquidity_score=LIQUIDITY_SHOCK - 0.01))
    assert d.action is Action.HALT_LIQUIDITY


def test_drawdown_uses_ladder_not_liquidation():
    d = decide(_state(drawdown_bps=1000, drawdown_trigger_bps=1000))
    assert d.action is Action.DERISK_LADDER
    assert d.permits_reduce is True


def test_cost_exceeding_edge_is_no_trade():
    d = decide(_state(cost_exceeds_edge=True))
    assert d.action is Action.NO_TRADE_BAND
    assert d.permits_new_risk is False


def test_calm_fresh_state_allocates():
    d = decide(_state())
    assert d.action is Action.ALLOCATE_MARGINAL
    assert d.permits_new_risk is True


def test_high_disagreement_reduces_alpha():
    d = decide(_state(model_disagreement=DISAGREEMENT_HIGH))
    assert d.action is Action.REDUCE_ALPHA


# ---- marginal contribution ----

def test_marginal_is_paired_difference_not_raw_return():
    c = marginal_contribution(0.0020, 0.0025, 0.0003)
    assert c == pytest.approx(-0.0008)
    assert c < 0


def test_marginal_rejects_negative_cost():
    with pytest.raises(ValueError):
        marginal_contribution(0.01, 0.0, -0.001)


def test_marginal_rejects_non_finite():
    with pytest.raises(ValueError):
        marginal_contribution(float("nan"), 0.0, 0.0)


# ---- EWMA mechanics ----

def test_lambda_matches_half_life_definition():
    h = 60.0
    lam = _ewma_lambda(h)
    assert math.pow(1.0 - lam, h) == pytest.approx(0.5, abs=1e-12)


def test_ewma_converges_to_constant_input():
    t = ContributionTracker(half_life_obs=10.0, probation_obs=1, prior_std=0.01)
    for _ in range(400):
        t.update(0.001)
    assert t.mean == pytest.approx(0.001, abs=1e-9)
    assert t.std < 1e-4


def test_unobserved_model_has_negative_lcb():
    t = ContributionTracker()
    assert t.n == 0
    assert t.lcb < 0
    assert t.eligible is False
    assert t.raw_weight() == 0.0


def test_probation_blocks_weight_even_with_positive_mean():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=50, prior_std=0.001)
    for _ in range(10):
        t.update(0.01)
    assert t.mean > 0
    assert t.lcb > 0
    assert t.status is ModelStatus.PROBATION
    assert t.raw_weight() == 0.0


def test_positive_contribution_earns_weight_after_probation():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=20, prior_std=0.001)
    for _ in range(200):
        t.update(0.01)
    assert t.status is ModelStatus.ACTIVE
    assert t.raw_weight() > 0


def test_negative_contribution_demotes_to_zero_weight():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=20,
                            prior_std=0.001, retire_min_obs=10_000)
    for _ in range(200):
        t.update(-0.001)
    assert t.status is ModelStatus.DEMOTED
    assert t.raw_weight() == 0.0


def test_sustained_negative_retires_permanently():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=10,
                            prior_std=0.001, retire_min_obs=50,
                            retire_band=0.001)
    for _ in range(100):
        t.update(-0.01)
    assert t.status is ModelStatus.RETIRED
    before = t.mean
    t.update(0.05)  # retirement is absorbing
    assert t.mean == before
    assert t.raw_weight() == 0.0


def test_idle_decay_erodes_authority():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=10,
                            prior_std=0.001, idle_grace=0)
    for _ in range(100):
        t.update(0.01)
    peak = t.mean
    assert t.eligible
    t.decay_idle(50)
    assert t.mean < peak
    assert t.idle == 50


def test_idle_grace_preserves_authority_briefly():
    t = ContributionTracker(half_life_obs=5.0, probation_obs=10,
                            prior_std=0.001, idle_grace=5)
    for _ in range(100):
        t.update(0.01)
    peak = t.mean
    t.decay_idle(3)
    assert t.mean == pytest.approx(peak)


def test_update_resets_idle():
    t = ContributionTracker(probation_obs=1, idle_grace=0)
    t.decay_idle(10)
    assert t.idle == 10
    t.update(0.001)
    assert t.idle == 0


# ---- idle decay additivity (locks the grace-accounting fix) ----

def _trained(**kw):
    params = dict(half_life_obs=5.0, probation_obs=10, prior_std=0.001)
    params.update(kw)
    t = ContributionTracker(**params)
    for _ in range(100):
        t.update(0.01)
    return t


@pytest.mark.parametrize("grace", [0, 3, 5, 12])
def test_idle_decay_is_additive_across_call_splitting(grace):
    """decay_idle(1) x N must equal decay_idle(N).

    The pre-fix version recomputed the whole past-grace span each call, so
    repeated small calls over-decayed. This test fails on that version.
    """
    one_shot = _trained(idle_grace=grace)
    many_calls = _trained(idle_grace=grace)
    assert one_shot.mean == pytest.approx(many_calls.mean)

    one_shot.decay_idle(20)
    for _ in range(20):
        many_calls.decay_idle(1)

    assert many_calls.idle == one_shot.idle == 20
    assert many_calls.mean == pytest.approx(one_shot.mean, rel=1e-12)


def test_idle_decay_crossing_grace_decays_only_the_excess():
    """With grace=5, decay_idle(8) must decay exactly 3 steps, not 8."""
    t = _trained(idle_grace=5)
    peak = t.mean
    lam = _ewma_lambda(t.half_life_obs)
    t.decay_idle(8)
    expected = peak * math.pow(1.0 - lam, 3)
    assert t.mean == pytest.approx(expected, rel=1e-12)


def test_idle_decay_two_stage_crossing_matches_single_call():
    """4 then 4 with grace=5 must equal 8 in one call (3 effective steps)."""
    single = _trained(idle_grace=5)
    staged = _trained(idle_grace=5)
    single.decay_idle(8)
    staged.decay_idle(4)   # entirely within grace, no decay
    staged.decay_idle(4)   # crosses grace, 3 effective steps
    assert staged.mean == pytest.approx(single.mean, rel=1e-12)


# ---- weight aggregation ----

def test_all_ineligible_yields_all_zero_not_equal_weight():
    ts = {"a": ContributionTracker(), "b": ContributionTracker()}
    w = weights(ts)
    assert w == {"a": 0.0, "b": 0.0}
    assert sum(w.values()) == 0.0


def test_weights_normalise_over_eligible_only():
    good = ContributionTracker(half_life_obs=5.0, probation_obs=10, prior_std=0.001)
    bad = ContributionTracker(half_life_obs=5.0, probation_obs=10,
                              prior_std=0.001, retire_min_obs=10_000)
    for _ in range(200):
        good.update(0.01)
        bad.update(-0.001)
    w = weights({"good": good, "bad": bad})
    assert w["good"] == pytest.approx(1.0)
    assert w["bad"] == 0.0


def test_stronger_contribution_gets_more_weight():
    strong = ContributionTracker(half_life_obs=5.0, probation_obs=10, prior_std=0.001)
    weak = ContributionTracker(half_life_obs=5.0, probation_obs=10, prior_std=0.001)
    for _ in range(200):
        strong.update(0.010)
        weak.update(0.002)
    w = weights({"strong": strong, "weak": weak})
    assert w["strong"] > w["weak"] > 0
    assert sum(w.values()) == pytest.approx(1.0)


def test_authority_report_carries_units_and_convention():
    t = ContributionTracker(probation_obs=1)
    t.update(0.001)
    r = authority_report({"m": t})
    assert r["unit"] == "fractional_return_per_observation"
    assert r["cost_basis"] == "net_after_cost"
    assert "weight_convention" in r
    assert "m" in r["models"]
    assert "weight" in r["models"]["m"]


# ---- validation ----

@pytest.mark.parametrize("kw", [
    {"avg_corr": 1.5}, {"vol_ann": -0.1}, {"liquidity_score": 1.5},
    {"model_disagreement": -0.1}, {"data_age_secs": -1},
    {"tau_data_secs": 0}, {"n_assets": -1},
])
def test_state_rejects_invalid(kw):
    with pytest.raises(ValueError):
        _state(**kw)


@pytest.mark.parametrize("kw", [
    {"half_life_obs": 0}, {"kappa": -1}, {"probation_obs": 0},
    {"prior_std": 0}, {"retire_band": 0}, {"idle_grace": -1},
])
def test_tracker_rejects_invalid(kw):
    with pytest.raises(ValueError):
        ContributionTracker(**kw)