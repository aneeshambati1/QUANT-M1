"""Acceptance gate tests: DSR + PBO diagnostics wired to trial ledger."""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

from quant_m1.research.acceptance import acceptance_diagnostics
from quant_m1.research.governor import TrialLedger


def _ledger_with_trials(n: int, tmp_path) -> TrialLedger:
    from pathlib import Path
    p = Path(tmp_path) if not isinstance(tmp_path, Path) else tmp_path
    p.mkdir(parents=True, exist_ok=True)
    led = TrialLedger(str(p / "trials.jsonl"))
    for i in range(n):
        led.register_trial(
            trial_id=f"t{i}", commit="abc1234", family="test",
            hypothesis=f"h{i}", config={"k": i}, seed=i,
            data_vintage="v1",
        )
    return led


def _returns(n_configs: int, n_obs: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {f"c{i}": rng.normal(0.0003, 0.01, n_obs) for i in range(n_configs)}


def test_family_count_from_ledger(tmp_path):
    led = _ledger_with_trials(13, tmp_path)
    ret = _returns(3, 252)
    res = acceptance_diagnostics(led, ret, selected_config="c0", n_obs=252)
    assert res["family_size"] == 13


def test_dsr_passes_for_strong_signal(tmp_path):
    led = _ledger_with_trials(5, tmp_path)
    rng = np.random.default_rng(42)
    strong = rng.normal(0.005, 0.01, 504)
    weak = rng.normal(0.0, 0.01, 504)
    ret = {"strong": strong, "weak": weak}
    res = acceptance_diagnostics(led, ret, selected_config="strong", n_obs=504,
                                 dsr_threshold=0.05)
    assert res["dsr"]["dsr"] > 0.0


def test_pbo_high_for_unstable_best(tmp_path):
    led = _ledger_with_trials(5, tmp_path)
    rng = np.random.default_rng(7)
    ret = {f"c{i}": rng.normal(0, 0.01, 504) for i in range(6)}
    res = acceptance_diagnostics(led, ret, selected_config="c0", n_obs=504,
                                 pbo_threshold=0.50)
    assert res["pbo"]["pbo"] > 0.3


def test_selected_config_must_exist(tmp_path):
    led = _ledger_with_trials(3, tmp_path)
    ret = _returns(2, 100)
    with pytest.raises(ValueError, match="not in returns_by_config"):
        acceptance_diagnostics(led, ret, selected_config="nonexistent", n_obs=100)


def test_need_at_least_two_configs(tmp_path):
    led = _ledger_with_trials(3, tmp_path)
    ret = {"only": np.zeros(100)}
    with pytest.raises(ValueError, match=">= 2 configs"):
        acceptance_diagnostics(led, ret, selected_config="only", n_obs=100)


def test_family_override(tmp_path):
    led = _ledger_with_trials(50, tmp_path)
    ret = _returns(3, 252)
    res = acceptance_diagnostics(led, ret, selected_config="c0", n_obs=252,
                                 family_override=10)
    assert res["family_size"] == 10


def test_passed_requires_both(tmp_path):
    led = _ledger_with_trials(3, tmp_path)
    rng = np.random.default_rng(99)
    ret = {f"c{i}": rng.normal(0.005, 0.01, 504) for i in range(3)}
    res = acceptance_diagnostics(led, ret, selected_config="c0", n_obs=504,
                                 dsr_threshold=0.99, pbo_threshold=0.01)
    assert res["passed"] is False


def test_dsr_increases_with_fewer_trials(tmp_path):
    """Fewer trials in family -> lower hurdle -> higher DSR for same Sharpe."""
    rng = np.random.default_rng(42)
    signal = rng.normal(0.003, 0.01, 504)
    noise = rng.normal(0.0, 0.01, 504)
    ret = {"sig": signal, "noise": noise}

    led_small = _ledger_with_trials(5, tmp_path / "small")
    led_large = _ledger_with_trials(100, tmp_path / "large")

    res_small = acceptance_diagnostics(led_small, ret, selected_config="sig", n_obs=504)
    res_large = acceptance_diagnostics(led_large, ret, selected_config="sig", n_obs=504)

    assert res_small["dsr"]["dsr"] >= res_large["dsr"]["dsr"]
