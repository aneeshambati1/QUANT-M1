"""Phase 2 global promotion-testing budget (v3 Amendment A + audit repairs).

Repairs over the first implementation:
  A. Explicit PROGRAM identity. Each program has its own roster (<= g_max) and
     its own alpha_global. Lifetime caps (alpha_lifetime, g_lifetime) bound ALL
     programs. A new program is opened explicitly via new_program(); it never
     resets lifetime spend or reopens closed families.
  B. Frozen configuration. The first record in the chain is a CONFIG record
     binding alpha_global/g_max/alpha_lifetime/g_lifetime/n_eval/se_min_obs/
     lag_cap. Reopening with different parameters is rejected. Certificates
     are replayed using the alpha_cell recorded IN EACH CERTIFICATE, not a
     freshly supplied one.
  C. Terminal retirement/failure events. record_terminal() closes a family
     durably even when a candidate never reaches n_eval (e.g. retirement at
     120 observations under the unchanged tracker rule).
  D. Single-writer lock. An exclusive OS file lock (fcntl.flock) is held for
     every mutation and reload; atomic replacement alone is not sufficient.

Hash-chained, deterministic, fail-closed. Kappa and the tracker authority
rule are NOT touched here.
"""
from __future__ import annotations

import fcntl
import math
import os
from contextlib import contextmanager

from ..core import Store, sha


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def nppf(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _ncdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


_CONFIG_KEYS = ("alpha_global", "g_max", "alpha_lifetime", "g_lifetime",
                "n_eval", "se_min_obs", "lag_cap")


class PromotionLedger:
    def __init__(self, path: str, *, alpha_global: float = 0.05, g_max: int = 24,
                 alpha_lifetime: float = 0.10, g_lifetime: int = 48,
                 n_eval: int = 399, se_min_obs: int = 120, lag_cap: int = 21):
        if not (0 < alpha_global <= alpha_lifetime <= 1):
            raise ValueError("require 0 < alpha_global <= alpha_lifetime <= 1")
        if not (1 <= g_max <= g_lifetime):
            raise ValueError("require 1 <= g_max <= g_lifetime")
        if not (n_eval >= se_min_obs >= 2):
            raise ValueError("require n_eval >= se_min_obs >= 2")
        self.store = Store(path)
        self.lock_path = path + ".lock"
        self._lock_depth = 0
        self._params = {"alpha_global": alpha_global, "g_max": g_max,
                        "alpha_lifetime": alpha_lifetime, "g_lifetime": g_lifetime,
                        "n_eval": n_eval, "se_min_obs": se_min_obs, "lag_cap": lag_cap}
        with self._locked():
            self.chain = self.store.load() or []
            self._verify()
            if not self.chain:
                self._append("CONFIG", dict(self._params))
            else:
                cfg = self.chain[0]
                if cfg["kind"] != "CONFIG":
                    raise ValueError("PROMOTION_LEDGER_MISSING_CONFIG")
                for k in _CONFIG_KEYS:
                    if cfg["payload"][k] != self._params[k]:
                        raise ValueError(f"PROMOTION_LEDGER_CONFIG_MISMATCH:{k}")
            self._replay()

    # ---------------- concurrency ----------------
    @contextmanager
    def _locked(self):
        # Reentrant within this instance: only the outermost context holds the
        # OS lock. Nested calls (register -> _reload, etc.) reuse it. Cross-
        # process writers are still serialized by flock on the lock file.
        if self._lock_depth > 0:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
        f = open(self.lock_path, "a+")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            self._lock_depth = 1
            yield
        finally:
            self._lock_depth = 0
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            f.close()

    def _reload(self):
        self.chain = self.store.load() or []
        self._verify()
        self._replay()

    # ---------------- chain ----------------
    def _verify(self) -> None:
        prev = "0" * 64
        for rec in self.chain:
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("prev") != prev or rec.get("hash") != sha(body):
                raise ValueError("PROMOTION_LEDGER_CHAIN_INVALID")
            prev = rec["hash"]

    def _append(self, kind: str, payload: dict) -> None:
        prev = self.chain[-1]["hash"] if self.chain else "0" * 64
        rec = {"kind": kind, "payload": payload, "prev": prev}
        rec["hash"] = sha(rec)
        self.chain.append(rec)
        self.store.save(self.chain)

    def _replay(self) -> None:
        cfg = self.chain[0]["payload"]
        self.alpha_global = cfg["alpha_global"]
        self.g_max = cfg["g_max"]
        self.alpha_lifetime = cfg["alpha_lifetime"]
        self.g_lifetime = cfg["g_lifetime"]
        self.n_eval = cfg["n_eval"]
        self.se_min_obs = cfg["se_min_obs"]
        self.lag_cap = cfg["lag_cap"]
        self.alpha_cell = self.alpha_global / self.g_max
        self.z_cell = nppf(1.0 - self.alpha_cell)
        self.programs: dict = {}
        self.current_program = None
        self.cells: dict = {}            # (candidate,horizon) -> {family, program}
        self.closed_families: set = set()
        self.tested: set = set()
        self.alpha_spent_lifetime = 0.0
        self.cells_spent_lifetime = 0
        for rec in self.chain[1:]:
            p = rec["payload"]
            k = rec["kind"]
            if k == "PROGRAM_OPEN":
                pid = p["program_id"]
                if pid in self.programs:
                    raise ValueError("DUPLICATE_PROGRAM_ID")
                self.programs[pid] = {"cells": {}, "alpha_spent": 0.0}
                self.current_program = pid
            elif k == "REGISTER":
                pid = p["program_id"]
                key = (p["candidate"], p["horizon"])
                self.cells[key] = {"family": p["family"], "program": pid}
                self.programs[pid]["cells"][key] = p["family"]
            elif k == "CERTIFICATE":
                key = (p["candidate"], p["horizon"])
                self.tested.add(key)
                # Replay with the RECORDED alpha_cell, not a recomputed one.
                self.alpha_spent_lifetime += p["alpha_cell"]
                self.programs[p["program_id"]]["alpha_spent"] += p["alpha_cell"]
                self.cells_spent_lifetime += 1
                if not p["promoted"]:
                    self.closed_families.add(p["family"])
            elif k == "TERMINAL":
                self.closed_families.add(p["family"])
                key = (p["candidate"], p["horizon"])
                self.tested.add(key)  # terminal: never promotable/testable again
        if self.current_program is None:
            # First program is opened implicitly by the CONFIG record; append
            # it inline (no reload) to avoid recursion.
            self._append("PROGRAM_OPEN", {"program_id": "program_1"})
            self.programs["program_1"] = {"cells": {}, "alpha_spent": 0.0}
            self.current_program = "program_1"

    # ---------------- programs ----------------
    def new_program(self, program_id: str) -> None:
        with self._locked():
            self._reload()
            if program_id in self.programs:
                raise ValueError("DUPLICATE_PROGRAM_ID")
            pending = sum(1 for c in self.cells if c not in self.tested)
            if self.cells_spent_lifetime + pending + 1 > self.g_lifetime:
                raise ValueError("LIFETIME_CELL_BUDGET_EXHAUSTED")
            if self.alpha_spent_lifetime + (pending + 1) * self.alpha_cell > self.alpha_lifetime + 1e-12:
                raise ValueError("LIFETIME_ALPHA_BUDGET_EXHAUSTED")
            self._append("PROGRAM_OPEN", {"program_id": program_id})
            self._replay()

    # ---------------- cells ----------------
    def register(self, candidate: str, horizon: int, family: str,
                 program_id: str | None = None) -> None:
        with self._locked():
            self._reload()
            pid = program_id or self.current_program
            if pid not in self.programs:
                raise ValueError("UNKNOWN_PROGRAM")
            key = (candidate, horizon)
            if key in self.cells:
                raise ValueError("CELL_ALREADY_REGISTERED")
            if family in self.closed_families:
                raise ValueError("FAMILY_CLOSED_FOR_LIFETIME")
            prog = self.programs[pid]
            if len(prog["cells"]) >= self.g_max:
                raise ValueError("PROGRAM_ROSTER_EXHAUSTED")
            pending = len(prog["cells"]) - sum(
                1 for c in prog["cells"] if c in self.tested)
            if self.cells_spent_lifetime + pending + 1 > self.g_lifetime:
                raise ValueError("LIFETIME_CELL_BUDGET_EXHAUSTED")
            if self.alpha_spent_lifetime + (pending + 1) * self.alpha_cell > self.alpha_lifetime + 1e-12:
                raise ValueError("LIFETIME_ALPHA_BUDGET_EXHAUSTED")
            self._append("REGISTER", {"candidate": candidate, "horizon": horizon,
                                      "family": family, "program_id": pid})
            self._replay()

    def evaluate(self, candidate: str, horizon: int, tracker) -> dict:
        with self._locked():
            self._reload()
            key = (candidate, horizon)
            if key not in self.cells:
                raise ValueError("CELL_NOT_REGISTERED")
            if key in self.tested:
                raise ValueError("CELL_ALREADY_TESTED_NO_RETEST")
            if tracker.n != self.n_eval:
                raise ValueError(f"EVALUATION_REQUIRES_EXACTLY_{self.n_eval}_OBS")
            st = tracker.se_stats(self.z_cell, self.lag_cap)
            promoted = bool(st["valid"] and st["se_lcb"] > 0.0)
            info = self.cells[key]
            payload = {"candidate": candidate, "horizon": horizon,
                       "family": info["family"], "program_id": info["program"],
                       "promoted": promoted, "z_cell": self.z_cell,
                       "alpha_cell": self.alpha_cell, "se": st}
            self._append("CERTIFICATE", payload)
            self._replay()
            return payload

    def record_terminal(self, candidate: str, horizon: int, reason: str) -> None:
        """Durable terminal event (e.g. RETIRED at 120 obs). Closes the family
        for the system lifetime even though no certificate was issued."""
        if reason not in {"RETIRED", "WITHDRAWN", "INVALID"}:
            raise ValueError("BAD_TERMINAL_REASON")
        with self._locked():
            self._reload()
            key = (candidate, horizon)
            if key not in self.cells:
                raise ValueError("CELL_NOT_REGISTERED")
            if key in self.tested:
                raise ValueError("CELL_ALREADY_TERMINAL")
            info = self.cells[key]
            self._append("TERMINAL", {"candidate": candidate, "horizon": horizon,
                                      "family": info["family"],
                                      "program_id": info["program"],
                                      "reason": reason})
            self._replay()

    def is_promoted(self, candidate: str, horizon: int) -> bool:
        for rec in self.chain:
            if rec["kind"] == "CERTIFICATE":
                p = rec["payload"]
                if (p["candidate"], p["horizon"]) == (candidate, horizon):
                    return bool(p["promoted"])
        return False

    def report(self) -> dict:
        return {"alpha_global": self.alpha_global, "g_max": self.g_max,
                "alpha_cell": self.alpha_cell, "z_cell": self.z_cell,
                "current_program": self.current_program,
                "programs": {pid: {"cells": len(pr["cells"]),
                                   "alpha_spent": pr["alpha_spent"]}
                             for pid, pr in self.programs.items()},
                "alpha_spent_lifetime": self.alpha_spent_lifetime,
                "cells_spent_lifetime": self.cells_spent_lifetime,
                "closed_families": sorted(self.closed_families),
                "tested": sorted(self.tested)}
