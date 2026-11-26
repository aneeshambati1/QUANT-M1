#!/usr/bin/env python3
"""Vendor trial audit — QUANT-M1 PIT data acceptance diligence.

Evaluation tooling ONLY. Measures data readiness; runs NO strategy hypothesis,
consumes ZERO trial-registry (Bonferroni) budget. Fail-closed: any flag that
lacks documentary evidence is forced False. Reports only what was measured.

Modes:
  raw     : --panel wide CSV (date index x symbols) + --manifest JSON.
            Preliminary verdict; vintage-hash check unavailable unless the
            repo loader can rehash the panel.
  frozen  : --frozen DIR --name NAME  -> audit_from_frozen() on a panel
            frozen with the repo's existing data/freeze.py tooling.
            Acceptance-grade mode (hash reproducibility is live).

Usage (from repo root):
  python experiments/vendor_trial_audit.py --panel trial.csv \
      --manifest trial.manifest.json --evidence ./evidence_bundle \
      --canaries both --out vendor_audit_report.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

# Approximate terminal windows — CONFIRM against exchange notices / vendor docs.
# A last date outside the window => MANUAL_REVIEW (fail-closed), never auto-pass.
ETP_CANARIES = {
    "XIV":  ("2018-02-01", "2018-03-31"),
    "TVIX": ("2020-06-15", "2020-08-15"),
    "UGAZ": ("2020-07-01", "2020-10-31"),
    "DGAZ": ("2020-07-01", "2020-10-31"),
}
EQUITY_CANARIES = {  # LEH may continue OTC as LEHMQ; verify via security master.
    "LEH": ("2008-09-01", "2008-12-31"),
    "BSC": ("2008-05-01", "2008-06-30"),
    "ENE": ("2001-11-01", "2002-01-31"),
}
EVIDENCE_FILES = {
    "pit":              ["vendor_vintage_policy"],
    "known_at_rows":    ["known_at_sample.csv"],
    "known_at_protocol":["known_at_protocol"],
    "adjustment_log":   ["corporate_actions.csv"],
    "delisted":         ["delisted_coverage_policy"],
    "survivorship_free":["survivorship_policy", "constituents_asof.csv"],
}
DOC_EXT = (".md", ".txt", ".pdf")

def find_evidence(evdir: Path, stem: str) -> str | None:
    if stem.endswith(".csv"):
        p = evdir / stem
        return str(p) if p.is_file() else None
    for ext in DOC_EXT:
        p = evdir / (stem + ext)
        if p.is_file():
            return str(p)
    return None

def evidence_present(evdir: Path, key: str) -> tuple[bool, list[str]]:
    found, missing = [], []
    for stem in EVIDENCE_FILES[key]:
        hit = find_evidence(evdir, stem)
        (found if hit else missing).append(hit or stem)
    return (not missing), found + [f"MISSING:{m}" for m in missing]

def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").sort_index()
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        raise ValueError("panel must be a non-empty wide CSV with a date index")
    return df

def check_known_at(sample_csv: str) -> dict:
    """known_at >= observation date (a close cannot be known before it exists).
    Usage-side no-lookahead (decision_time >= known_at) is enforced by the
    pre-registered protocol, not here."""
    out = {"pass": False, "n_rows": 0, "violations": 0, "detail": ""}
    try:
        df = pd.read_csv(sample_csv)
        need = {"symbol", "date", "known_at"}
        if not need.issubset(df.columns):
            out["detail"] = f"missing columns: {sorted(need - set(df.columns))}"
            return out
        d = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        k = pd.to_datetime(df["known_at"], errors="coerce").dt.normalize()
        bad = (k < d) | d.isna() | k.isna()
        out.update(n_rows=len(df), violations=int(bad.sum()),
                   pass_=(len(df) > 0 and not bad.any()))
        out["pass"] = out.pop("pass_")
        out["detail"] = "ok" if out["pass"] else "known_at precedes observation or unparseable dates"
    except Exception as e:
        out["detail"] = f"unreadable sample: {e}"
    return out

def check_actions(actions_csv: str) -> dict:
    """Dated corporate-action event log, separate from adjusted prices."""
    out = {"pass": False, "n_events": 0, "detail": ""}
    try:
        df = pd.read_csv(actions_csv)
        need = {"symbol", "event_type", "ex_date"}
        if not need.issubset(df.columns):
            out["detail"] = f"missing columns: {sorted(need - set(df.columns))}"
            return out
        ex = pd.to_datetime(df["ex_date"], errors="coerce")
        ok = ex.notna() & df["event_type"].isin(["split", "dividend"])
        splits = df["event_type"].eq("split")
        if splits.any():
            if "ratio" not in df.columns:
                out["detail"] = "split events require a 'ratio' column"; return out
            r = pd.to_numeric(df.loc[splits, "ratio"], errors="coerce")
            ok &= ~splits | (r > 0).reindex(df.index, fill_value=False)
        divs = df["event_type"].eq("dividend")
        if divs.any():
            if "amount" not in df.columns:
                out["detail"] = "dividend events require an 'amount' column"; return out
            a = pd.to_numeric(df.loc[divs, "amount"], errors="coerce")
            ok &= ~divs | (a >= 0).reindex(df.index, fill_value=False)
        out.update(n_events=int(ok.sum()), pass_=(int(ok.sum()) > 0 and bool(ok.all())))
        out["pass"] = out.pop("pass_")
        out["detail"] = "ok" if out["pass"] else "invalid/unparseable events present"
    except Exception as e:
        out["detail"] = f"unreadable log: {e}"
    return out

def check_canaries(close: pd.DataFrame, required: dict) -> dict:
    res = {}
    for sym, (lo, hi) in required.items():
        if sym not in close.columns:
            res[sym] = {"status": "FAIL_ABSENT"}; continue
        s = close[sym].dropna()
        s = s[s > 0]
        if s.empty:
            res[sym] = {"status": "FAIL_NO_VALID_HISTORY"}; continue
        last = s.index.max()
        status = "PASS" if pd.Timestamp(lo) <= last <= pd.Timestamp(hi) else "MANUAL_REVIEW"
        res[sym] = {"status": status, "first": str(s.index.min().date()),
                    "last": str(last.date()), "window": [lo, hi],
                    "note": "verify identity via security master (ticker reuse risk)"}
    return res

def coverage_stats(close: pd.DataFrame) -> dict:
    """Diagnostic: audit_panel's 'valid episode' requires ALL symbols positive
    on a row. Broad survivorship-free panels are non-rectangular by design, so
    the raw verdict may report INSUFFICIENT_VALID_EPISODES spuriously. The
    confirmatory episode definition must be PRE-REGISTERED before acceptance."""
    valid = close.gt(0)
    row_cov = valid.mean(axis=1)
    per_sym = valid.sum()
    span_days = (close.index.max() - close.index.min()).days
    return {
        "n_symbols": int(close.shape[1]),
        "n_rows": int(close.shape[0]),
        "span_years": round(span_days / 365.25, 2),
        "rows_all_symbols_valid": int(valid.all(axis=1).sum()),
        "rows_ge_90pct_coverage": int((row_cov >= 0.90).sum()),
        "symbols_ge_252_obs": int((per_sym >= 252).sum()),
        "symbols_ge_5000_obs": int((per_sym >= 5000).sum()),
    }

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel"); ap.add_argument("--manifest")
    ap.add_argument("--frozen"); ap.add_argument("--name")
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--canaries", choices=["etp", "equity", "both"], default="both")
    ap.add_argument("--known-at-mode", choices=["rows", "protocol"], default="rows")
    ap.add_argument("--min-episodes", type=int, default=5000)   # PROPOSED — pre-register
    ap.add_argument("--min-years", type=float, default=20.0)    # PROPOSED — pre-register
    ap.add_argument("--min-breadth", type=int, default=300)     # PROPOSED — pre-register
    ap.add_argument("--out", default="vendor_audit_report.json")
    a = ap.parse_args()
    evdir = Path(a.evidence)
    notes, ev = [], {}

    if a.frozen:
        from quant_m1.data.freeze import load_frozen
        close = load_frozen(a.frozen, a.name).close
        manifest = json.loads((Path(a.frozen) / f"{a.name}.manifest.json").read_text())
        mode = "frozen"
    else:
        if not (a.panel and a.manifest):
            print("raw mode requires --panel and --manifest"); return 2
        close = load_panel(a.panel)
        manifest = json.loads(Path(a.manifest).read_text())
        mode = "raw"
        notes.append("RAW mode: acceptance-grade verdict requires a frozen-vintage rerun.")

    # Evidence-gated flags (fail-closed).
    for key in ("pit", "delisted", "survivorship_free"):
        ok, files = evidence_present(evdir, key); ev[key] = files
        if not ok:
            notes.append(f"evidence missing for {key}; related flag forced False")
    pit_flag = bool(manifest.get("pit", False)) and evidence_present(evdir, "pit")[0]
    surv_flag = bool(manifest.get("survivorship_free", False)) and evidence_present(evdir, "survivorship_free")[0]
    manifest = dict(manifest, pit=pit_flag, survivorship_free=surv_flag)

    ka_key = "known_at_rows" if a.known_at_mode == "rows" else "known_at_protocol"
    ka_ev_ok, ka_files = evidence_present(evdir, ka_key); ev["known_at"] = ka_files
    ka = check_known_at(str(evdir / "known_at_sample.csv")) if a.known_at_mode == "rows" else {"pass": ka_ev_ok, "detail": "protocol mode: requires pre-registered operationalization + hash"}
    known_at_flag = bool(ka["pass"] and ka_ev_ok)

    ac = check_actions(str(evdir / "corporate_actions.csv"))
    ac_ev_ok, ac_files = evidence_present(evdir, "adjustment_log"); ev["adjustment_log"] = ac_files
    actions_flag = bool(ac["pass"] and ac_ev_ok)
    notes.append("Corporate-action REPLAY test (reconstruct PIT total returns) is vendor-format-specific: PENDING until sample format known.")

    required = {}
    if a.canaries in ("etp", "both"): required.update(ETP_CANARIES)
    if a.canaries in ("equity", "both"): required.update(EQUITY_CANARIES)
    can = check_canaries(close, required)
    can_pass = all(v["status"] == "PASS" for v in can.values())
    delisted_flag = can_pass and evidence_present(evdir, "delisted")[0]

    from quant_m1.research.data_audit import audit_panel, audit_from_frozen
    kw = dict(min_valid_episodes=a.min_episodes, has_known_at_columns=known_at_flag,
              has_adjustment_log=actions_flag, has_delisted_universe=delisted_flag)
    rep = (audit_from_frozen(a.frozen, a.name, **kw) if mode == "frozen"
           else audit_panel(close, manifest, **kw))

    cov = coverage_stats(close)
    breadth_pass = cov["symbols_ge_252_obs"] >= a.min_breadth
    history_pass = cov["span_years"] >= a.min_years
    if "INSUFFICIENT_VALID_EPISODES" in " ".join(rep.reasons) and cov["rows_ge_90pct_coverage"] >= a.min_episodes:
        notes.append("Episode shortfall may be the rectangular-panel artifact: rows with >=90% coverage meet the floor. Pre-register the confirmatory episode rule; do NOT edit audit code ad hoc.")
    overall = (rep.verdict == "SUITABLE" and can_pass and breadth_pass and history_pass
               and known_at_flag and actions_flag and mode == "frozen")

    report = {"mode": mode, "overall_gate": "PASS" if overall else "NOT_ACCEPTED",
              "audit": rep.as_dict(), "canaries": can, "coverage": cov,
              "known_at_check": ka, "actions_check": ac, "evidence": ev,
              "breadth_pass": breadth_pass, "history_pass": history_pass,
              "thresholds": {"min_episodes": a.min_episodes, "min_years": a.min_years,
                             "min_breadth": a.min_breadth, "status": "PROPOSED — pre-register before confirmatory use"},
              "notes": notes}
    Path(a.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("mode", "overall_gate", "breadth_pass", "history_pass")}, indent=2))
    print("audit verdict:", rep.verdict); print("reasons:", rep.reasons)
    print("canaries:", {k: v["status"] for k, v in can.items()})
    print("full report ->", a.out)
    return 0 if overall else 1

if __name__ == "__main__":
    sys.exit(main())
