"""Phase 2 global promotion-testing budget (v3 Amendment A + clarifications).

One finite program budget alpha_global=0.05 (one-sided), Bonferroni-partitioned
over a pre-declared roster of G_max candidate-horizon cells. Each cell is
tested EXACTLY ONCE at N_eval=399. Failed cells close their FAMILY for the
system lifetime. Lifetime caps (alpha_lifetime, G_lifetime) bound all programs.
Hash-chained, deterministic, fail-closed. No dependency on trial_registry.
"""
from __future__ import annotations

import math
import os
from typing import Optional

from ..core import Store, canonical, sha


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


class PromotionLedger:
    def __init__(self, path: str, *, alpha_global: float = 0.05, g_max: int = 24,
                 alpha_lifetime: float = 0.10, g_lifetime: int = 48,
                 n_eval: int = 399, se_min_obs: int = 120, lag_cap: int = 21,
                 alpha_spent_lifetime: float = 0.0, cells_spent_lifetime: int = 0):
        if not (0 < alpha_global <= alpha_lifetime <= 1):
            raise ValueError("require 0 < alpha_global <= alpha_lifetime <= 1")
        if not (1 <= g_max <= g_lifetime):
            raise ValueError("require 1 <= g_max <= g_lifetime")
        if n_eval < se_min_obs >= 2:
            raise ValueError("require n_eval >= se_min_obs >= 2")
        self.store = Store(path)
        self.alpha_global = alpha_global
        self.g_max = g_max
        self.alpha_lifetime = alpha_lifetime
        self.g_lifetime = g_lifetime
        self.n_eval = n_eval
        self.se_min_obs = se_min_obs
        self.lag_cap = lag_cap
        self.alpha_cell = alpha_global / g_max
        self.z_cell = nppf(1.0 - self.alpha_cell)
        self.alpha_spent_lifetime = alpha_spent_lifetime
        self.cells_spent_lifetime = cells_spent_lifetime
        self.chain = self.store.load() or []
        self._verify()
        self.cells: dict = {}
        self.closed_families: set = set()
        self.tested: set = set()
        self.alpha_spent = 0.0
        for rec in self.chain:
            p = rec["payload"]
            if rec["kind"] == "REGISTER":
                self.cells[(p["candidate"], p["horizon"])] = p["family"]
            elif rec["kind"] == "CERTIFICATE":
                key = (p["candidate"], p["horizon"])
                self.tested.add(key)
                self.alpha_spent += self.alpha_cell
                self.alpha_spent_lifetime += self.alpha_cell
                self.cells_spent_lifetime += 1
                if not p["promoted"]:
                    self.closed_families.add(p["family"])

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

    def register(self, candidate: str, horizon: int, family: str) -> None:
        key = (candidate, horizon)
        if key in self.cells:
            raise ValueError("CELL_ALREADY_REGISTERED")
        if family in self.closed_families:
            raise ValueError("FAMILY_CLOSED_FOR_LIFETIME")
        if len(self.cells) >= self.g_max:
            raise ValueError("PROGRAM_ROSTER_EXHAUSTED")
        if self.cells_spent_lifetime + (len(self.cells) - len(self.tested)) + 1 > self.g_lifetime:
            raise ValueError("LIFETIME_CELL_BUDGET_EXHAUSTED")
        if self.alpha_spent_lifetime + (len(self.cells) - len(self.tested) + 1) * self.alpha_cell > self.alpha_lifetime + 1e-12:
            raise ValueError("LIFETIME_ALPHA_BUDGET_EXHAUSTED")
        self.cells[key] = family
        self._append("REGISTER", {"candidate": candidate, "horizon": horizon,
                                  "family": family})

    def evaluate(self, candidate: str, horizon: int, tracker) -> dict:
        key = (candidate, horizon)
        if key not in self.cells:
            raise ValueError("CELL_NOT_REGISTERED")
        if key in self.tested:
            raise ValueError("CELL_ALREADY_TESTED_NO_RETEST")
        if tracker.n != self.n_eval:
            raise ValueError(f"EVALUATION_REQUIRES_EXACTLY_{self.n_eval}_OBS")
        st = tracker.se_stats(self.z_cell, self.lag_cap)
        promoted = bool(st["valid"] and st["se_lcb"] > 0.0)
        family = self.cells[key]
        self.tested.add(key)
        self.alpha_spent += self.alpha_cell
        self.alpha_spent_lifetime += self.alpha_cell
        self.cells_spent_lifetime += 1
        if not promoted:
            self.closed_families.add(family)
        payload = {"candidate": candidate, "horizon": horizon, "family": family,
                   "promoted": promoted, "z_cell": self.z_cell,
                   "alpha_cell": self.alpha_cell, "se": st}
        self._append("CERTIFICATE", payload)
        return payload

    def report(self) -> dict:
        return {"alpha_global": self.alpha_global, "g_max": self.g_max,
                "alpha_cell": self.alpha_cell, "z_cell": self.z_cell,
                "alpha_spent": self.alpha_spent,
                "alpha_spent_lifetime": self.alpha_spent_lifetime,
                "cells_spent_lifetime": self.cells_spent_lifetime,
                "closed_families": sorted(self.closed_families),
                "tested": sorted(self.tested)}
