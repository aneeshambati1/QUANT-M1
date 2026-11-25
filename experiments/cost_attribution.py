"""Pre-declared cost attribution on the frozen real panel. All cells reported.

Registers in a SEPARATE ledger (evidence/cost_accounting.jsonl). These are
deterministic measurements, NOT hypothesis tests, so they do NOT increment
the market trial count used by the Bonferroni gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_m1.data.loader import load_yfinance
from quant_m1.data.freeze import freeze_panel, load_frozen
from quant_m1.research.baseline import UNIVERSE
from quant_m1.research.cost_accounting import attribute, POLICIES, COST_SCENARIOS_BP
from quant_m1.research.governor import TrialLedger


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--panel-dir", default="evidence")
    ap.add_argument("--panel-name", default="panel_real_v1")
    ap.add_argument("--out-dir", default="evidence/cost_accounting")
    ap.add_argument("--ledger", default="evidence/cost_accounting.jsonl")
    a = ap.parse_args()

    try:
        panel = load_frozen(a.panel_dir, a.panel_name)
        src = "frozen"
    except Exception:
        panel = load_yfinance(UNIVERSE, "2015-01-01", "2025-01-01")
        freeze_panel(panel, a.panel_dir, a.panel_name)
        src = "downloaded_and_frozen"
    print(f"panel source: {src}  hash: {panel.content_hash[:16]}...  "
          f"rows: {panel.close.shape[0]}")

    returns = panel.close.pct_change().dropna(how="any")
    out = attribute(returns)
    out["panel"] = panel.as_dict()
    out["panel_source"] = src

    led = TrialLedger(a.ledger)
    tid = f"cost_attr_{panel.content_hash[:12]}"
    if not any(t["trial_id"] == tid for t in led.trials()):
        led.register_trial(
            trial_id=tid, commit=a.commit, family="cost_accounting",
            hypothesis="deterministic turnover/cost attribution; NOT a hypothesis test",
            config={"policies": list(POLICIES),
                    "cost_bp_oneway": list(COST_SCENARIOS_BP)},
            seed=0, data_vintage=panel.vintage,
            note="accounting measurement; does not increment market trial count",
            content_hash=panel.content_hash)
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        ref = str(Path(a.out_dir) / "summary.json")
        Path(ref).write_text(json.dumps(out, indent=2, sort_keys=True))
        led.record_result(trial_id=tid, metric_name="annual_cost_bp_range",
                          metric_value=float(max(r["annual_cost_bp"] for r in out["rows"])),
                          horizon="full_panel", gross_net="net", raw_output_ref=ref,
                          note="max annual cost across all cells; descriptive")

    print(f"\n{'policy':16} {'ann_traded':>11} " +
          " ".join(f"{c:>9.0f}bp" for c in COST_SCENARIOS_BP) +
          f" {'net_ann@3bp':>12} {'vol':>7} {'sharpe*':>8} {'maxDD*':>8}")
    print("-" * 92)
    for p in POLICIES:
        rs = {r["cost_bp_oneway"]: r for r in out["rows"] if r["policy"] == p}
        base = rs[3.0]
        print(f"{p:16} {base['annual_traded_notional']:>11.3f} " +
              " ".join(f"{rs[c]['annual_cost_bp']:>11.1f}" for c in COST_SCENARIOS_BP) +
              f" {base['net_ann_return']*100:>11.2f}% {base['net_ann_vol']*100:>6.1f}%"
              f" {base['net_sharpe_descriptive']:>8.2f} {base['net_max_dd_descriptive']*100:>7.1f}%")
    print("\n* DESCRIPTIVE ONLY. One realised history. Sampling uncertainty NOT")
    print("  quantified. No significance claimed. Risk-free assumed 0.")
    print("  Turnover and cost columns are mechanical given the cost assumption.")
    return out


if __name__ == "__main__":
    main()