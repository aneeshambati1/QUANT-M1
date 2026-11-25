"""Pre-registered momentum grid. Search space frozen at import time.

The grid is declared BEFORE any measurement. It excludes (lookback=21,
horizon=5) because that cell was already run as trial001 and failed
(t_hac = 0.26). Re-running it would not create a new trial.

DATA STATUS: yfinance back-adjusted closes, NON-PIT, gross of cost.
Any result here is PROVISIONAL and cannot support capital allocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_m1.data.loader import load_yfinance
from quant_m1.research.baseline import UNIVERSE
from quant_m1.research.governor import TrialLedger
from quant_m1.research.protocol import run_grid, fingerprint

FAMILY = "momentum"
HYPOTHESIS = "cross-sectional trailing return predicts forward return"
GRID_LOOKBACKS = (5, 10, 63)
GRID_HORIZONS = (1, 5, 10, 21)
ALREADY_RUN = (21, 5)


def register_prior(ledger_path: str, panel, report_path: str, commit: str) -> bool:
    """Register trial001 retrospectively, with an honest note. Returns False if present."""
    led = TrialLedger(ledger_path)
    tid = "trial001_lb21_hz5"
    if any(t["trial_id"] == tid for t in led.trials()):
        return False
    payload = json.loads(Path(report_path).read_text())
    rep = payload.get("report", payload)
    cfg = {"lookback": ALREADY_RUN[0], "horizon": ALREADY_RUN[1]}
    cfg["fingerprint"] = fingerprint(FAMILY, cfg, panel)
    led.register_trial(
        trial_id=tid, commit=commit, family=FAMILY, hypothesis=HYPOTHESIS,
        config=cfg, seed=0, data_vintage=panel.vintage,
        note="declared pre-run in session; ledger entry created post-run")
    led.record_result(
        trial_id=tid, metric_name="t_hac", metric_value=float(rep["t_hac"]),
        horizon=rep.get("horizon", "5d"), gross_net="gross",
        raw_output_ref=str(report_path),
        note="retrospective registration of executed trial")
    return True


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--ledger", default="evidence/trials.jsonl")
    ap.add_argument("--out-dir", default="evidence/grid001")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--prior-report", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    panel = load_yfinance(UNIVERSE, a.start, a.end)
    print("PANEL:", json.dumps(panel.as_dict(), indent=2))

    if a.prior_report:
        added = register_prior(a.ledger, panel, a.prior_report, a.commit)
        print("prior trial registered:", added)

    out = run_grid(a.ledger, panel, family=FAMILY, hypothesis=HYPOTHESIS,
                   lookbacks=GRID_LOOKBACKS, horizons=GRID_HORIZONS,
                   commit=a.commit, seed=7, out_dir=a.out_dir, alpha=a.alpha,
                   id_prefix="grid001")

    summary = {
        "family": FAMILY, "hypothesis": HYPOTHESIS,
        "grid_lookbacks": list(GRID_LOOKBACKS),
        "grid_horizons": list(GRID_HORIZONS),
        "n_trials_total": out["n_trials_total"],
        "required_abs_t": out["required_abs_t"],
        "survivors": out["survivors"],
        "cells": [{"trial_id": c["trial_id"], "lookback": c["lookback"],
                   "horizon": c["horizon"], "mean_ic": c["mean_ic"],
                   "t_hac": c["t_hac"], "passed": c["gate"]["passed"]}
                  for c in out["cells"]],
        "panel": panel.as_dict(),
        "label": "PROVISIONAL - NON-PIT - gross of cost - NOT AN EDGE CLAIM",
    }
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.out_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()