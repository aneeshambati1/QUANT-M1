"""Fast Phase 2 mechanics tests. Statistical calibration belongs in null suite."""
from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from quant_m1.system import build
from quant_m1.universe import etf_instruments
from quant_m1.meta.contribution import (
    ContributionTracker,
    se_weights,
    se_raw_weights,
)
from quant_m1.meta.probe import (
    ProbeJournal,
    assign_families,
    probe_coefficient,
    _ProbeGate,
)
from quant_m1.orchestrator.paper_loop import _shadow_system
from quant_m1.research.promotion_ledger import PromotionLedger


SYMS = ["SPY", "QQQ", "TLT"]
MARKS = {"SPY": 50_000, "QQQ": 45_000, "TLT": 9_000}


def _sx(tmp_path):
    return build(
        str(tmp_path),
        cash=10_000_000,
        positions={},
        marks=dict(MARKS),
        instruments=etf_instruments(SYMS),
    )


def test_shadow_config_and_universe_sync(tmp_path):
    sx = _sx(tmp_path)
    state = sx["ledger"].state
    start = {
        "cash_settled": state.cash_settled,
        "positions": dict(state.positions),
        "marks": dict(state.marks),
        "receivables": tuple(state.receivables),
        "peak_nav": state.peak_nav,
        "trade_date": state.trade_date,
        "mark_wall": state.mark_wall,
    }
    shadow = _shadow_system(sx, start, tempfile.mkdtemp())
    assert shadow["cfg"] is sx["cfg"]
    assert shadow["kernel"].cfg is sx["cfg"]
    assert shadow["gw"].cfg is sx["cfg"]
    assert shadow["ledger"].state.config_digest == sx["cfg"].mandate.digest()
    assert shadow["ledger"].state.mandate_version == sx["cfg"].mandate.version
    assert set(SYMS) <= set(shadow["instruments"])
    for scenario in shadow["scenarios"]:
        assert set(SYMS) <= set(scenario)


def test_se_stats_fail_closed_and_finite():
    tracker = ContributionTracker()
    assert tracker.se_stats(2.0)["valid"] is False

    for _ in range(130):
        tracker.update(0.0)
    stats = tracker.se_stats(2.0)
    assert stats["valid"] is False
    assert stats["reason"] == "nonpositive_gamma0"

    tracker2 = ContributionTracker()
    rng = np.random.default_rng(0)
    for _ in range(399):
        tracker2.update(float(rng.normal(0.001, 0.01)))
    stats2 = tracker2.se_stats(2.865)
    assert stats2["valid"]
    assert math.isfinite(stats2["se_lcb"])
    assert stats2["n_eff_corr"] > 0


def test_original_rule_unchanged():
    tracker = ContributionTracker()
    for _ in range(300):
        tracker.update(0.02)
    assert tracker.eligible
    assert tracker.kappa == 1.0
    assert tracker.lcb == pytest.approx(tracker.mean - tracker.std)


def test_duplicate_family_guard():
    signals = {
        "a": np.array([1.0, 2.0, 3.0]),
        "b": np.array([2.0, 4.0, 6.0]),
        "c": np.array([3.0, 1.0, 2.0]),
    }
    families = assign_families(signals)
    assert families["a"] == "a"
    assert families["b"] == "a"
    assert probe_coefficient("a", families, {}) == 0.05
    assert probe_coefficient("b", families, {}) == 0.0
    assert probe_coefficient("a", families, {"a": 0.5}) == 0.0


def test_probe_coefficient_rejects_invalid_pi():
    with pytest.raises(ValueError):
        probe_coefficient("a", {"a": "a"}, {}, pi=float("nan"))
    with pytest.raises(ValueError):
        probe_coefficient("a", {"a": "a"}, {}, pi=1.1)


def test_se_weights_fail_closed_without_ledger():
    tracker = ContributionTracker()
    rng = np.random.default_rng(1)
    for _ in range(399):
        tracker.update(float(rng.normal(0.01, 0.001)))

    # Computable statistics are not themselves promotion authority.
    assert se_raw_weights({"m1": tracker}, 2.865)["m1"] >= 0.0
    weights = se_weights({"m1": tracker}, 2.865)
    assert weights == {"m1": 0.0}


def test_se_weights_require_certificate():
    tracker = ContributionTracker()
    rng = np.random.default_rng(11)
    for _ in range(399):
        tracker.update(float(rng.normal(0.01, 0.001)))

    class NoCertificate:
        def is_promoted(self, candidate, horizon):
            return False

    class Certificate:
        def is_promoted(self, candidate, horizon):
            return candidate == "m1" and horizon == 1

    denied = se_weights({"m1": tracker}, 2.865, ledger=NoCertificate())
    assert denied == {"m1": 0.0}

    allowed = se_weights(
        {"m1": tracker},
        2.865,
        c_model=0.40,
        ledger=Certificate(),
    )
    assert 0.0 <= allowed["m1"] <= 0.40 + 1e-9


def test_se_weights_all_zero_and_caps():
    trackers = {
        "m1": ContributionTracker(),
        "m2": ContributionTracker(),
    }
    assert se_weights(trackers, 2.865) == {"m1": 0.0, "m2": 0.0}

    rng = np.random.default_rng(1)
    big = ContributionTracker()
    for _ in range(399):
        big.update(float(rng.normal(0.01, 0.001)))

    class Certificate:
        def is_promoted(self, candidate, horizon):
            return True

    capped = se_weights(
        {"m1": big},
        2.865,
        c_model=0.40,
        ledger=Certificate(),
    )
    assert capped["m1"] <= 0.40 + 1e-9


def test_ledger_one_test_and_lifetime_closure(tmp_path):
    ledger = PromotionLedger(str(tmp_path / "ledger.json"))
    assert ledger.z_cell == pytest.approx(2.128, abs=0.01)
    ledger.register("m1", 1, "famA")

    tracker = ContributionTracker()
    rng = np.random.default_rng(2)
    for _ in range(399):
        tracker.update(float(rng.normal(0.0, 0.01)))

    report = ledger.evaluate("m1", 1, tracker)
    assert report["promoted"] is False
    with pytest.raises(ValueError):
        ledger.evaluate("m1", 1, tracker)
    with pytest.raises(ValueError):
        ledger.register("m1_dup", 1, "famA")

    ledger.register("m2", 1, "famB")
    fresh = ContributionTracker()
    with pytest.raises(ValueError):
        ledger.evaluate("m2", 1, fresh)

    restored = PromotionLedger(str(tmp_path / "ledger.json"))
    assert "famA" in restored.closed_families


def test_probe_gate_is_research_only():
    gate = _ProbeGate(1)
    assert gate.eligible
    assert gate.raw_weight() == 1.0
    assert gate.snapshot()["probe_gate"] is True


def test_ledger_program_and_lifetime_budgets(tmp_path):
    path = str(tmp_path / "ledger.json")
    ledger = PromotionLedger(
        path,
        g_max=2,
        g_lifetime=4,
        alpha_global=0.05,
        alpha_lifetime=0.10,
    )
    ledger.register("a", 1, "fa")
    ledger.register("b", 1, "fb")
    with pytest.raises(ValueError):
        ledger.register("c", 1, "fc")

    ledger.new_program("program_2")
    ledger.register("c", 1, "fc")
    ledger.register("d", 1, "fd")
    with pytest.raises(ValueError):
        ledger.new_program("program_3")


def test_ledger_config_frozen_on_restore(tmp_path):
    path = str(tmp_path / "ledger.json")
    PromotionLedger(path, g_max=3, g_lifetime=6)
    with pytest.raises(ValueError):
        PromotionLedger(path, g_max=2, g_lifetime=6)
    with pytest.raises(ValueError):
        PromotionLedger(path, g_max=3, g_lifetime=6, alpha_global=0.10)


def test_ledger_retirement_closes_family(tmp_path):
    path = str(tmp_path / "ledger.json")
    ledger = PromotionLedger(path)
    ledger.register("r1", 1, "famR")

    tracker = ContributionTracker()
    for _ in range(120):
        tracker.update(-0.01)
    assert tracker.status.value == "retired"

    ledger.record_terminal("r1", 1, "RETIRED")
    with pytest.raises(ValueError):
        ledger.register("r2", 1, "famR")

    restored = PromotionLedger(path)
    assert "famR" in restored.closed_families


def test_ledger_certificate_gates_se_weights(tmp_path):
    tracker = ContributionTracker()
    rng = np.random.default_rng(3)
    for _ in range(120):
        tracker.update(float(rng.normal(0.01, 0.001)))

    ledger = PromotionLedger(str(tmp_path / "ledger.json"))
    ledger.register("m", 1, "famM")
    with pytest.raises(ValueError):
        ledger.evaluate("m", 1, tracker)

    assert ledger.is_promoted("m", 1) is False
    assert se_weights({"m": tracker}, 2.865, ledger=ledger) == {"m": 0.0}


def test_probe_journal_family_is_immutable_and_reloads(tmp_path):
    path = str(tmp_path / "probe.json")
    journal = ProbeJournal(path)
    journal.set_family("model_a", "family_a")
    journal.record("model_a", 1, 0.001)

    with pytest.raises(ValueError):
        journal.set_family("model_a", "family_b")

    restored = ProbeJournal(path)
    assert restored.families["model_a"] == "family_a"
    assert restored.n_obs("model_a", 1) == 1
    assert restored.certificate_tracker("model_a", 1).n == 1


def test_probe_journal_rejects_unregistered_candidate(tmp_path):
    journal = ProbeJournal(str(tmp_path / "probe.json"))
    with pytest.raises(ValueError):
        journal.record("unknown", 1, 0.001)
