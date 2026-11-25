"""Positive / negative controls for the assembled measurement instrument.

Unit tests validate components. None validates that the ASSEMBLED pipeline
detects a real edge end-to-end at the real panel shape and the real trial
count. These controls do that.

Three controls at the exact market-panel shape (2516 sessions x 10 assets,
lookback 21 -> horizon 5, seed 1000), calibrated by execution:

  zero          edge_k 0.000  -> IC ~ 0.001, t ~ 0.08   must NOT be detected
  tradeable_003 edge_k 0.015  -> IC ~ 0.030, t ~ 2.31   NOT detected at N=13
  strong        edge_k 0.050  -> IC ~ 0.096, t ~ 7.42   MUST be detected

The middle control is the point: a genuine 0.030 IC lies inside this
project's own stated tradeable band, yet the instrument cannot resolve it
at N=13. That makes the market null INCONCLUSIVE rather than "no edge".

Each control is a DISTINCT dataset and carries a distinct source string, so
the content-drift check does not mistake three intentional panels for one
silently revised panel.

These are SYNTHETIC instrument checks. They use a separate ledger and a
distinct family. They are NOT market hypotheses and must never be counted
against the market trial ledger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import PricePanel, panel_from_frame
from .governor import ic_gate
from .protocol import run_governed

CONTROL_SEED = 1000
N_SESSIONS = 2516
N_ASSETS = 10
LOOKBACK = 21
HORIZON = 5
MARKET_TRIAL_COUNT = 13
FAMILY = "CONTROL_SYNTHETIC"
LABEL = "SYNTHETIC INSTRUMENT CONTROL - NOT AN EDGE CLAIM"

CONTROLS = {"zero": 0.000, "tradeable_003": 0.015, "strong": 0.050}


def synthetic_edge_panel(edge_k: float, *, n_sessions: int = N_SESSIONS,
                         n_assets: int = N_ASSETS, lookback: int = LOOKBACK,
                         seed: int = CONTROL_SEED) -> PricePanel:
    """Prices with a KNOWN cross-sectional predictive edge.

    Next-session return = edge_k * z(trailing return) * sigma + noise.
    edge_k = 0 gives a pure random walk with no predictive structure.
    """
    if n_sessions < lookback + 40:
        raise ValueError("n_sessions too small for lookback")
    if n_assets < 2:
        raise ValueError("need >= 2 assets")
    rng = np.random.default_rng(seed)
    px = np.full((n_sessions, n_assets), 100.0)
    for t in range(0, lookback):
        px[t + 1] = px[t] * np.exp(0.01 * rng.standard_normal(n_assets))
    for t in range(lookback, n_sessions - 1):
        tr = px[t] / px[t - lookback] - 1.0
        sd = tr.std()
        z = (tr - tr.mean()) / sd if sd > 0 else np.zeros(n_assets)
        px[t + 1] = px[t] * np.exp(edge_k * z * 0.01
                                   + 0.01 * rng.standard_normal(n_assets))
    idx = pd.bdate_range("2015-01-02", periods=n_sessions, tz="UTC")
    df = pd.DataFrame(px, index=idx, columns=[f"S{i}" for i in range(n_assets)])
    return panel_from_frame(df, source=f"synthetic_control:edge_k={edge_k}",
                            vintage=f"edge_k={edge_k};seed={seed}",
                            pit=False, label=LABEL)


def run_control(name: str, ledger_path: str, out_dir: str, commit: str) -> dict:
    if name not in CONTROLS:
        raise KeyError(f"unknown control {name}")
    k = CONTROLS[name]
    panel = synthetic_edge_panel(k)
    out = run_governed(ledger_path, panel, trial_id=f"ctrl_{name}",
                       commit=commit, family=FAMILY,
                       hypothesis=f"injected edge_k={k} at lb{LOOKBACK}->hz{HORIZON}",
                       lookback=LOOKBACK, horizon=HORIZON, seed=CONTROL_SEED,
                       out_dir=out_dir)
    out["edge_k"] = k
    out["gate_at_market_N"] = ic_gate(out["report"].t_hac, MARKET_TRIAL_COUNT)
    return out


def run_all_controls(ledger_path: str, out_dir: str, commit: str) -> dict:
    rows = [run_control(n, ledger_path, out_dir, commit) for n in CONTROLS]
    return {"market_trial_count": MARKET_TRIAL_COUNT,
            "required_abs_t": rows[0]["gate_at_market_N"]["required_abs_t"],
            "controls": [{"name": r["trial_id"], "edge_k": r["edge_k"],
                          "mean_ic": r["report"].mean_ic,
                          "t_hac": r["report"].t_hac,
                          "detected_at_market_N":
                              r["gate_at_market_N"]["passed"]} for r in rows],
            "label": LABEL}