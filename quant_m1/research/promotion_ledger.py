"""Fail-closed Phase 2 promotion ledger.

Research-only. Lifetime registration and alpha reservations are never recycled.
Certificates are one-shot; terminal events can close a previously promoted cell.
"""
from __future__ import annotations

import fcntl
import math
import os
import threading
from contextlib import contextmanager

from ..core import Store, sha


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def nppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _ncdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


_CONFIG_KEYS = (
    "alpha_global", "g_max", "alpha_lifetime", "g_lifetime",
    "n_eval", "se_min_obs", "lag_cap",
)


class PromotionLedger:
    """Hash-chained, single-writer promotion ledger.

    A registered cell permanently consumes one lifetime roster slot and one
    alpha_cell reservation. A certificate or terminal event does not release
    either reservation. Family closure is permanent.
    """

    def __init__(
        self,
        path: str,
        *,
        alpha_global: float = 0.05,
        g_max: int = 24,
        alpha_lifetime: float = 0.10,
        g_lifetime: int = 48,
        n_eval: int = 399,
        se_min_obs: int = 120,
        lag_cap: int = 21,
    ):
        if not (0 < alpha_global <= alpha_lifetime <= 1):
            raise ValueError("require 0 < alpha_global <= alpha_lifetime <= 1")
        if isinstance(g_max, bool) or not isinstance(g_max, int) or g_max < 1:
            raise ValueError("g_max must be a positive integer")
        if isinstance(g_lifetime, bool) or not isinstance(g_lifetime, int):
            raise ValueError("g_lifetime must be an integer")
        if g_lifetime < g_max:
            raise ValueError("require g_lifetime >= g_max")
        if not (n_eval >= se_min_obs >= 2):
            raise ValueError("require n_eval >= se_min_obs >= 2")
        if isinstance(lag_cap, bool) or not isinstance(lag_cap, int) or lag_cap < 0:
            raise ValueError("lag_cap must be a nonnegative integer")

        self.store = Store(path)
        self.lock_path = path + ".lock"
        self._thread_lock = threading.RLock()
        self._local = threading.local()
        self._params = {
            "alpha_global": float(alpha_global),
            "g_max": g_max,
            "alpha_lifetime": float(alpha_lifetime),
            "g_lifetime": g_lifetime,
            "n_eval": n_eval,
            "se_min_obs": se_min_obs,
            "lag_cap": lag_cap,
        }

        with self._locked():
            self.chain = self.store.load() or []
            self._verify()
            if not self.chain:
                self._append("CONFIG", dict(self._params))
            else:
                cfg = self.chain[0]
                if cfg.get("kind") != "CONFIG":
                    raise ValueError("PROMOTION_LEDGER_MISSING_CONFIG")
                for key in _CONFIG_KEYS:
                    if cfg["payload"].get(key) != self._params[key]:
                        raise ValueError(f"PROMOTION_LEDGER_CONFIG_MISMATCH:{key}")
            self._replay()
            if not self.programs:
                self._append("PROGRAM_OPEN", {"program_id": "program_1"})
                self._replay()

    @contextmanager
    def _locked(self):
        """Thread- and process-safe reentrant writer lock."""
        with self._thread_lock:
            depth = getattr(self._local, "depth", 0)
            if depth:
                self._local.depth = depth + 1
                try:
                    yield
                finally:
                    self._local.depth -= 1
                return

            os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
            with open(self.lock_path, "a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                self._local.depth = 1
                try:
                    yield
                finally:
                    self._local.depth = 0
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

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

    def _reload(self) -> None:
        self.chain = self.store.load() or []
        self._verify()
        self._replay()

    def _replay(self) -> None:
        if not self.chain or self.chain[0].get("kind") != "CONFIG":
            raise ValueError("PROMOTION_LEDGER_MISSING_CONFIG")

        cfg = self.chain[0]["payload"]
        for key in _CONFIG_KEYS:
            if key not in cfg:
                raise ValueError(f"PROMOTION_LEDGER_CONFIG_MISSING:{key}")

        self.alpha_global = cfg["alpha_global"]
        self.g_max = cfg["g_max"]
        self.alpha_lifetime = cfg["alpha_lifetime"]
        self.g_lifetime = cfg["g_lifetime"]
        self.n_eval = cfg["n_eval"]
        self.se_min_obs = cfg["se_min_obs"]
        self.lag_cap = cfg["lag_cap"]
        self.alpha_cell = self.alpha_global / self.g_max
        self.z_cell = nppf(1.0 - self.alpha_cell)

        self.programs = {}
        self.current_program = None
        self.cells = {}             # (candidate, horizon) -> family/program
        self.closed_families = set()
        self.tested = set()
        self.terminal = {}          # key -> reason
        self.certificates = {}      # key -> certificate payload
        self.alpha_spent_lifetime = 0.0

        for rec in self.chain[1:]:
            kind, p = rec.get("kind"), rec.get("payload", {})
            if kind == "PROGRAM_OPEN":
                pid = p.get("program_id")
                if not isinstance(pid, str) or not pid.strip() or pid in self.programs:
                    raise ValueError("PROMOTION_LEDGER_INVALID_PROGRAM")
                self.programs[pid] = {"cells": {}, "alpha_spent": 0.0}
                self.current_program = pid

            elif kind == "REGISTER":
                pid = p.get("program_id")
                key = (p.get("candidate"), p.get("horizon"))
                family = p.get("family")
                if pid not in self.programs:
                    raise ValueError("PROMOTION_LEDGER_REGISTER_UNKNOWN_PROGRAM")
                if key in self.cells:
                    raise ValueError("PROMOTION_LEDGER_DUPLICATE_CELL")
                if not isinstance(key[0], str) or not key[0].strip():
                    raise ValueError("PROMOTION_LEDGER_INVALID_CANDIDATE")
                if isinstance(key[1], bool) or not isinstance(key[1], int) or key[1] < 1:
                    raise ValueError("PROMOTION_LEDGER_INVALID_HORIZON")
                if not isinstance(family, str) or not family.strip():
                    raise ValueError("PROMOTION_LEDGER_INVALID_FAMILY")
                self.cells[key] = {"family": family, "program": pid}
                self.programs[pid]["cells"][key] = family

            elif kind == "CERTIFICATE":
                key = (p.get("candidate"), p.get("horizon"))
                if key not in self.cells or key in self.tested:
                    raise ValueError("PROMOTION_LEDGER_INVALID_CERTIFICATE_ORDER")
                info = self.cells[key]
                if p.get("program_id") != info["program"] or p.get("family") != info["family"]:
                    raise ValueError("PROMOTION_LEDGER_CERTIFICATE_BINDING_MISMATCH")
                alpha = p.get("alpha_cell")
                if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
                    raise ValueError("PROMOTION_LEDGER_INVALID_CERTIFICATE_ALPHA")
                self.tested.add(key)
                self.certificates[key] = p
                self.alpha_spent_lifetime += alpha
                self.programs[info["program"]]["alpha_spent"] += alpha
                if not bool(p.get("promoted")):
                    self.closed_families.add(info["family"])

            elif kind == "TERMINAL":
                key = (p.get("candidate"), p.get("horizon"))
                if key not in self.cells or key in self.terminal:
                    raise ValueError("PROMOTION_LEDGER_INVALID_TERMINAL_ORDER")
                info = self.cells[key]
                if p.get("program_id") != info["program"] or p.get("family") != info["family"]:
                    raise ValueError("PROMOTION_LEDGER_TERMINAL_BINDING_MISMATCH")
                reason = p.get("reason")
                if reason not in {"RETIRED", "WITHDRAWN", "INVALID"}:
                    raise ValueError("PROMOTION_LEDGER_BAD_TERMINAL_REASON")
                self.terminal[key] = reason
                self.tested.add(key)
                self.closed_families.add(info["family"])

            else:
                raise ValueError(f"PROMOTION_LEDGER_UNKNOWN_EVENT:{kind}")

        if len(self.cells) > self.g_lifetime:
            raise ValueError("PROMOTION_LEDGER_LIFETIME_ROSTER_OVERFLOW")
        for pid, program in self.programs.items():
            if len(program["cells"]) > self.g_max:
                raise ValueError(f"PROMOTION_LEDGER_PROGRAM_ROSTER_OVERFLOW:{pid}")
        if len(self.cells) * self.alpha_cell > self.alpha_lifetime + 1e-12:
            raise ValueError("PROMOTION_LEDGER_LIFETIME_ALPHA_OVERFLOW")

    def new_program(self, program_id: str) -> None:
        if not isinstance(program_id, str) or not program_id.strip():
            raise ValueError("INVALID_PROGRAM_ID")
        with self._locked():
            self._reload()
            if program_id in self.programs:
                raise ValueError("DUPLICATE_PROGRAM_ID")
            if len(self.cells) >= self.g_lifetime:
                raise ValueError("LIFETIME_CELL_BUDGET_EXHAUSTED")
            if (len(self.cells) + 1) * self.alpha_cell > self.alpha_lifetime + 1e-12:
                raise ValueError("LIFETIME_ALPHA_BUDGET_EXHAUSTED")
            self._append("PROGRAM_OPEN", {"program_id": program_id})
            self._replay()

    def register(
        self, candidate: str, horizon: int, family: str,
        program_id: str | None = None,
    ) -> None:
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
            if len(self.cells) >= self.g_lifetime:
                raise ValueError("LIFETIME_CELL_BUDGET_EXHAUSTED")
            if (len(self.cells) + 1) * self.alpha_cell > self.alpha_lifetime + 1e-12:
                raise ValueError("LIFETIME_ALPHA_BUDGET_EXHAUSTED")
            if len(self.programs[pid]["cells"]) >= self.g_max:
                raise ValueError("PROGRAM_ROSTER_EXHAUSTED")

            self._append("REGISTER", {
                "candidate": candidate, "horizon": horizon,
                "family": family, "program_id": pid,
            })
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

            stats = tracker.se_stats(self.z_cell, self.lag_cap)
            promoted = bool(stats["valid"] and stats["se_lcb"] > 0.0)
            info = self.cells[key]
            payload = {
                "candidate": candidate, "horizon": horizon,
                "family": info["family"], "program_id": info["program"],
                "promoted": promoted, "z_cell": self.z_cell,
                "alpha_cell": self.alpha_cell, "se": stats,
            }
            self._append("CERTIFICATE", payload)
            self._replay()
            return payload

    def record_terminal(self, candidate: str, horizon: int, reason: str) -> None:
        if reason not in {"RETIRED", "WITHDRAWN", "INVALID"}:
            raise ValueError("BAD_TERMINAL_REASON")
        with self._locked():
            self._reload()
            key = (candidate, horizon)
            if key not in self.cells:
                raise ValueError("CELL_NOT_REGISTERED")
            if key in self.terminal:
                raise ValueError("CELL_ALREADY_TERMINAL")
            info = self.cells[key]
            self._append("TERMINAL", {
                "candidate": candidate, "horizon": horizon,
                "family": info["family"], "program_id": info["program"],
                "reason": reason,
            })
            self._replay()

    def is_promoted(self, candidate: str, horizon: int) -> bool:
        key = (candidate, horizon)
        return (
            key in self.certificates
            and bool(self.certificates[key].get("promoted"))
            and key not in self.terminal
        )

    def report(self) -> dict:
        return {
            "alpha_global": self.alpha_global,
            "g_max": self.g_max,
            "alpha_cell": self.alpha_cell,
            "z_cell": self.z_cell,
            "current_program": self.current_program,
            "programs": {
                pid: {
                    "cells": len(pr["cells"]),
                    "alpha_spent": pr["alpha_spent"],
                }
                for pid, pr in self.programs.items()
            },
            "registered_cells_lifetime": len(self.cells),
            "alpha_reserved_lifetime": len(self.cells) * self.alpha_cell,
            "alpha_spent_lifetime": self.alpha_spent_lifetime,
            "cells_spent_lifetime": len(self.cells),
            "closed_families": sorted(self.closed_families),
            "tested": sorted(self.tested),
            "terminal": dict(self.terminal),
        }
