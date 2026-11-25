"""Governed measurement protocol. Registration BEFORE execution.

Two entry points:

  run_governed  - one hypothesis, registered then measured then gated.
  run_grid      - a FULL search grid registered up front, then measured.
                  Declaring the grid before seeing results is what makes
                  the multiple-testing correction honest. Sequential
                  "try, look, try again" search is not supported here.

Duplicate control: a hypothesis fingerprint is derived from family, config,
and panel IDENTITY (source, date range, symbols) - not from the download
timestamp. Re-downloading the same range does not create a new trial.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..data.loader import PricePanel
from .baseline import run_baseline
from .governor import TrialLedger, LedgerError, ic_gate


class ProtocolError(Exception):
    pass


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def panel_identity(panel: PricePanel) -> dict:
    """Content identity, deliberately excluding the download timestamp."""
    return {"source": panel.source,
            "start": str(panel.close.index[0]),
            "end": str(panel.close.index[-1]),
            "symbols": sorted(panel.close.columns.tolist()),
            "n_rows": int(panel.close.shape[0])}


def fingerprint(family: str, config: dict, panel: PricePanel) -> str:
    body = {"family": family, "config": config, "identity": panel_identity(panel)}
    return hashlib.sha256(_canon(body).encode()).hexdigest()


def _check_duplicate(led: TrialLedger, fp: str) -> None:
    for t in led.trials():
        if t.get("config", {}).get("fingerprint") == fp:
            raise ProtocolError(
                f"hypothesis already registered as {t['trial_id']}; "
                "re-running it does not create a new trial")


def _write_raw(out_dir: Path, trial_id: str, payload: dict) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{trial_id}.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(p)


def run_governed(ledger_path: str, panel: PricePanel, *, trial_id: str,
                 commit: str, family: str, hypothesis: str,
                 lookback: int, horizon: int, seed: int,
                 out_dir: str, alpha: float = 0.05) -> dict:
    if not commit or not commit.strip():
        raise ProtocolError("commit SHA required")
    led = TrialLedger(ledger_path)
    cfg = {"lookback": int(lookback), "horizon": int(horizon)}
    fp = fingerprint(family, cfg, panel)
    _check_duplicate(led, fp)
    cfg["fingerprint"] = fp

    # REGISTER BEFORE MEASURING. If measurement raises, the trial still counts.
    led.register_trial(trial_id=trial_id, commit=commit, family=family,
                       hypothesis=hypothesis, config=cfg, seed=int(seed),
                       data_vintage=panel.vintage)

    rep = run_baseline(panel, lookback=lookback, horizon=horizon)
    raw = _write_raw(Path(out_dir), trial_id,
                     {"panel": panel.as_dict(), "report": rep.as_dict()})
    led.record_result(trial_id=trial_id, metric_name="t_hac",
                      metric_value=rep.t_hac, horizon=rep.horizon,
                      gross_net="gross", raw_output_ref=raw)
    gate = ic_gate(rep.t_hac, led.n_trials(), alpha)
    return {"trial_id": trial_id, "report": rep, "gate": gate, "raw_output": raw}


def run_grid(ledger_path: str, panel: PricePanel, *, family: str,
             hypothesis: str, lookbacks, horizons, commit: str, seed: int,
             out_dir: str, alpha: float = 0.05,
             id_prefix: str = "grid") -> dict:
    """Pre-register the FULL grid, then measure every cell, then gate at N.

    No result is inspected until every cell is registered. The gate uses the
    final trial count, so adding cells later cannot retroactively loosen it.
    """
    if not commit or not commit.strip():
        raise ProtocolError("commit SHA required")
    lbs = [int(x) for x in lookbacks]
    hzs = [int(x) for x in horizons]
    if not lbs or not hzs:
        raise ProtocolError("grid must be non-empty")
    if len(set(lbs)) != len(lbs) or len(set(hzs)) != len(hzs):
        raise ProtocolError("grid axes must not contain duplicates")

    led = TrialLedger(ledger_path)
    cells = []
    for lb in lbs:
        for hz in hzs:
            cfg = {"lookback": lb, "horizon": hz}
            fp = fingerprint(family, cfg, panel)
            _check_duplicate(led, fp)
            cells.append((lb, hz, fp))

    # Phase 1: register everything before any measurement.
    ids = []
    for lb, hz, fp in cells:
        tid = f"{id_prefix}_lb{lb}_hz{hz}"
        cfg = {"lookback": lb, "horizon": hz, "fingerprint": fp}
        led.register_trial(trial_id=tid, commit=commit, family=family,
                           hypothesis=hypothesis, config=cfg, seed=int(seed),
                           data_vintage=panel.vintage,
                           note="pre-registered grid cell")
        ids.append(tid)

    n_after_register = led.n_trials()

    # Phase 2: measure and record.
    rows = []
    for (lb, hz, fp), tid in zip(cells, ids):
        rep = run_baseline(panel, lookback=lb, horizon=hz)
        raw = _write_raw(Path(out_dir), tid,
                         {"panel": panel.as_dict(), "report": rep.as_dict()})
        led.record_result(trial_id=tid, metric_name="t_hac",
                          metric_value=rep.t_hac, horizon=rep.horizon,
                          gross_net="gross", raw_output_ref=raw)
        rows.append({"trial_id": tid, "lookback": lb, "horizon": hz,
                     "mean_ic": rep.mean_ic, "t_hac": rep.t_hac,
                     "report": rep, "raw_output": raw})

    # Phase 3: gate every cell at the full trial count.
    for r in rows:
        r["gate"] = ic_gate(r["t_hac"], n_after_register, alpha)
    rows.sort(key=lambda r: abs(r["t_hac"]), reverse=True)
    survivors = [r["trial_id"] for r in rows if r["gate"]["passed"]]
    return {"n_trials_total": n_after_register,
            "required_abs_t": rows[0]["gate"]["required_abs_t"] if rows else None,
            "cells": rows, "survivors": survivors}