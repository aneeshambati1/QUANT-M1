"""Governed optimization: hash-chained trial ledger + multiple-testing gate.

Parameter search is only legitimate when every trial is counted. Append-only,
tamper-evident ledger plus a conservative Bonferroni gate on the HAC t-stat.
DSR is NOT implemented.

RULES:
  - A trial must be registered BEFORE its result is recorded.
  - Editing or deleting any ledger line breaks verification.
  - The significance threshold tightens as the trial count grows.

NOTE: the chain is unkeyed SHA-256. It detects accidental alteration, not a
determined adversary who recomputes every hash. External anchoring is a
known open gap.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import norm

GENESIS = "0" * 64


class LedgerError(Exception):
    pass


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(body: dict) -> str:
    return hashlib.sha256(_canon(body).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrialLedger:
    def __init__(self, path: str):
        self.path = Path(path)
        self._entries: list[dict] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._entries = []
            return
        entries: list[dict] = []
        prev = GENESIS
        for i, line in enumerate(self.path.read_text().splitlines()):
            if not line.strip():
                raise LedgerError(f"blank line at {i}")
            rec = json.loads(line)
            if rec.get("seq") != i:
                raise LedgerError(f"sequence gap at {i}")
            if rec.get("prev_hash") != prev:
                raise LedgerError(f"broken chain at {i}")
            body = {k: rec[k] for k in ("seq", "kind", "prev_hash", "body")}
            if _hash(body) != rec.get("hash"):
                raise LedgerError(f"tampered record at {i}")
            entries.append(rec)
            prev = rec["hash"]
        self._entries = entries

    def _append(self, kind: str, body: dict) -> dict:
        seq = len(self._entries)
        prev = self._entries[-1]["hash"] if self._entries else GENESIS
        rec = {"seq": seq, "kind": kind, "prev_hash": prev, "body": body}
        rec["hash"] = _hash(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(_canon(rec) + "\n")
        self._entries.append(rec)
        return rec

    def register_trial(self, *, trial_id: str, commit: str, family: str,
                       hypothesis: str, config: dict, seed: int,
                       data_vintage: str, note: str = "",
                       panel_key: str = "", content_hash: str = "") -> dict:
        if not all([trial_id, commit, family, hypothesis, data_vintage]):
            raise LedgerError("trial_id, commit, family, hypothesis, vintage required")
        if any(e["kind"] == "register" and e["body"]["trial_id"] == trial_id
               for e in self._entries):
            raise LedgerError("duplicate trial_id")
        body = {"trial_id": trial_id, "commit": commit, "family": family,
                "hypothesis": hypothesis, "config": config, "seed": int(seed),
                "data_vintage": data_vintage, "note": note,
                "panel_key": panel_key, "content_hash": content_hash,
                "registered_at": _now()}
        return self._append("register", body)

    def record_result(self, *, trial_id: str, metric_name: str,
                      metric_value: float, horizon: str, gross_net: str,
                      raw_output_ref: str, note: str = "") -> dict:
        regs = [e for e in self._entries
                if e["kind"] == "register" and e["body"]["trial_id"] == trial_id]
        if not regs:
            raise LedgerError("result requires prior registration")
        if any(e["kind"] == "result" and e["body"]["trial_id"] == trial_id
               for e in self._entries):
            raise LedgerError("duplicate result for trial_id")
        if gross_net not in {"gross", "net"}:
            raise LedgerError("gross_net must be 'gross' or 'net'")
        body = {"trial_id": trial_id, "metric_name": metric_name,
                "metric_value": float(metric_value), "horizon": horizon,
                "gross_net": gross_net, "raw_output_ref": raw_output_ref,
                "note": note, "result_at": _now()}
        return self._append("result", body)

    def trials(self) -> list[dict]:
        return [e["body"] for e in self._entries if e["kind"] == "register"]

    def results(self) -> list[dict]:
        return [e["body"] for e in self._entries if e["kind"] == "result"]

    def n_trials(self) -> int:
        return len(self.trials())


def bonferroni_threshold(n_trials: int, alpha: float = 0.05) -> float:
    if n_trials < 1:
        raise LedgerError("n_trials must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise LedgerError("alpha must be in (0,1)")
    return float(norm.ppf(1.0 - alpha / (2.0 * n_trials)))


def ic_gate(t_hac: float, n_trials: int, alpha: float = 0.05) -> dict:
    thr = bonferroni_threshold(n_trials, alpha)
    return {"t_hac": float(t_hac), "n_trials": int(n_trials),
            "alpha": alpha, "required_abs_t": thr,
            "passed": abs(float(t_hac)) >= thr}