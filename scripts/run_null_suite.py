#!/usr/bin/env python3
"""Frozen Phase 2 calibration suite; research only.

Simulates the certificate statistic at n=399 and evaluates WHOLE programs:
a replicate fails if ANY registered cell is falsely promoted. Bounded-memory
chunks keep the run feasible. This script never changes alpha, z, null
assumptions, or acceptance limits to force a pass. Any failed check exits
nonzero and Phase 2 stays disabled.

Corrections over the prior revision (verified against the frozen protocol
evidence/phase2_probe_protocol_v3.md):
  1. Adds the protocol-declared PER-CELL acceptance check
     (alpha_cell + 0.01 = 0.0121) that was missing.
  2. Removes the undeclared LIFETIME_BOUND=0.10. The frozen protocol declares
     only the program-wide family-wise bound 0.07 and the per-cell bound
     0.0121. The lifetime procedure is therefore tested against the declared
     0.07 bound. A lifetime-specific acceptance bound requires an APPROVED
     protocol amendment; it is not invented here.
  3. Freezes the replication count (default 5000) and all seeds explicitly,
     resolving the prior unspecified-replication gap. Ratification of these
     values in the protocol document is a pending amendment item.
  4. The positive-control power check remains INFORMATIONAL because the frozen
     protocol declares no numeric power floor. It is flagged
     amendment_required=True so the gap cannot be missed; a numeric power gate
     requires an approved amendment (see proposal). It is NOT silently added.

Run:
    python scripts/run_null_suite.py --reps 5000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_m1.core import sha
from quant_m1.research.promotion_ledger import nppf

HL = 60.0
LAM = 1.0 - 2.0 ** (-1.0 / HL)
N_EVAL = 399
LAG = 21
G_PROGRAM = 24
G_LIFETIME = 48
ALPHA_GLOBAL = 0.05
ALPHA_LIFETIME = 0.10
ALPHA_CELL = ALPHA_GLOBAL / G_PROGRAM
Z_CELL = nppf(1.0 - ALPHA_CELL)
PROGRAM_BOUND = 0.07          # frozen: 0.05 + 0.02
PER_CELL_BOUND = ALPHA_CELL + 0.01   # frozen: 0.0121
LIFETIME_BOUND = PROGRAM_BOUND       # declared family-wise bound; amendment pending
RETIRE_MIN = 120
RETIRE_BAND = -0.002
RETIRE_DELTA = 21
WEAK_MEAN = -0.0002
COST_MEAN = -0.0005
STRONG_MEAN = -0.005
POWER_MEAN = 0.001
DEFAULT_REPS = 5000
SEEDS = {
    "program_iid": 101, "program_ar03": 102, "program_ar05": 103,
    "per_cell": 104, "lifetime": 201, "weak": 301, "cost": 401,
    "retire": 501, "power": 601,
}


def wilson_upper(rate: float, reps: int, z: float = 1.6449) -> float:
    if reps < 1 or not 0.0 <= rate <= 1.0:
        raise ValueError("invalid Wilson inputs")
    den = 1.0 + z * z / reps
    center = (rate + z * z / (2.0 * reps)) / den
    half = z * math.sqrt(rate * (1.0 - rate) / reps + z * z / (4.0 * reps * reps)) / den
    return center + half


def cert_stats(x: np.ndarray) -> np.ndarray:
    """Vectorized mirror of ContributionTracker.se_stats + ledger decision."""
    if x.ndim != 2 or x.shape[1] != N_EVAL:
        raise ValueError(f"expected shape (batch, {N_EVAL})")
    n = x.shape[1]
    w = LAM * (1.0 - LAM) ** np.arange(n - 1, -1, -1)
    W = float(w.sum())
    mhat = (x @ w) / W
    d = x - x.mean(axis=1, keepdims=True)
    gamma0 = (d * d).mean(axis=1)
    acc = gamma0 * float(w @ w)
    for k in range(1, min(LAG, n - 1) + 1):
        gamma_k = (d[:, k:] * d[:, :-k]).mean(axis=1)
        cross = float(w[:-k] @ w[k:])
        acc += 2.0 * (1.0 - k / (LAG + 1)) * gamma_k * cross
    var_mean = acc / (W * W)
    valid = (np.isfinite(var_mean) & (var_mean > 0.0)
             & np.isfinite(gamma0) & (gamma0 > 0.0))
    lcb = np.full(len(x), -np.inf)
    lcb[valid] = mhat[valid] - Z_CELL * np.sqrt(var_mean[valid])
    return valid & (lcb > 0.0)


def generate(rng, batch, cells, mu, rho, sigma=0.01):
    if not -0.999 <= rho <= 0.999:
        raise ValueError("rho outside supported range")
    if rho == 0.0:
        return rng.normal(mu, sigma, (batch, cells, N_EVAL))
    x = np.empty((batch, cells, N_EVAL), dtype=np.float64)
    x[:, :, 0] = rng.normal(mu, sigma, (batch, cells))
    innov = rng.normal(0.0, sigma * math.sqrt(1.0 - rho * rho),
                       (batch, cells, N_EVAL))
    for t in range(1, N_EVAL):
        x[:, :, t] = mu * (1.0 - rho) + rho * x[:, :, t - 1] + innov[:, :, t]
    return x


def program_false_promotion_rate(reps, cells, mu, rho, seed, chunk):
    rng = np.random.default_rng(seed)
    false_programs = 0
    done = 0
    while done < reps:
        b = min(chunk, reps - done)
        x = generate(rng, b, cells, mu, rho).reshape(b * cells, N_EVAL)
        promoted = cert_stats(x).reshape(b, cells)
        false_programs += int(promoted.any(axis=1).sum())
        done += b
    return false_programs / reps


def per_cell_false_rate(reps, mu, rho, seed, chunk):
    rng = np.random.default_rng(seed)
    false_cells = 0
    total = 0
    while total < reps:
        b = min(chunk, reps - total)
        x = generate(rng, b, 1, mu, rho).reshape(b, N_EVAL)
        false_cells += int(cert_stats(x).sum())
        total += b
    return false_cells / reps


def retirement_detection(reps, seed, chunk):
    rng = np.random.default_rng(seed)
    length = RETIRE_MIN + RETIRE_DELTA + 40
    detected = 0
    total = 0
    while total < reps:
        b = min(chunk, reps - total)
        x = generate(rng, b, 1, STRONG_MEAN, 0.0)[:, 0, :length]
        m = np.zeros(b)
        first = np.full(b, 10**9)
        for t in range(length):
            m = (1.0 - LAM) * m + LAM * x[:, t]
            hit = (t + 1 >= RETIRE_MIN) & (m < RETIRE_BAND) & (first > 10**8)
            first[hit] = t + 1
        detected += int((first <= RETIRE_MIN + RETIRE_DELTA).sum())
        total += b
    return detected / reps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--out", default="evidence/null_suite_report.json")
    args = parser.parse_args()
    if args.reps < 100 or args.chunk < 1:
        parser.error("--reps must be >=100 and --chunk must be >=1")

    checks = []

    def add_rate(name, rate, bound, reps):
        upper = wilson_upper(rate, reps)
        checks.append({"check": name, "rate": rate, "wilson95_upper": upper,
                       "bound": bound, "pass": bool(upper <= bound)})

    for rho, key in ((0.0, "program_iid"), (0.3, "program_ar03"), (0.5, "program_ar05")):
        rate = program_false_promotion_rate(args.reps, G_PROGRAM, 0.0, rho,
                                            SEEDS[key], args.chunk)
        add_rate(f"program24_iid_or_ar{rho}", rate, PROGRAM_BOUND, args.reps)

    per_cell = per_cell_false_rate(args.reps, 0.0, 0.0, SEEDS["per_cell"], args.chunk)
    add_rate("per_cell_zero_edge", per_cell, PER_CELL_BOUND, args.reps)

    lifetime = program_false_promotion_rate(args.reps, G_LIFETIME, 0.0, 0.0,
                                            SEEDS["lifetime"], args.chunk)
    add_rate("lifetime48_iid", lifetime, LIFETIME_BOUND, args.reps)

    weak = program_false_promotion_rate(args.reps, G_PROGRAM, WEAK_MEAN, 0.3,
                                        SEEDS["weak"], args.chunk)
    add_rate("program24_weak_negative", weak, PROGRAM_BOUND, args.reps)

    cost = program_false_promotion_rate(args.reps, G_PROGRAM, COST_MEAN, 0.0,
                                        SEEDS["cost"], args.chunk)
    add_rate("program24_cost_domination", cost, PROGRAM_BOUND, args.reps)

    retirement = retirement_detection(args.reps, SEEDS["retire"], args.chunk)
    checks.append({"check": "strong_negative_retirement_by_141",
                   "rate": retirement, "bound_min": 0.95,
                   "pass": bool(retirement >= 0.95)})

    power = program_false_promotion_rate(args.reps, 1, POWER_MEAN, 0.3,
                                         SEEDS["power"], args.chunk)
    checks.append({"check": "positive_control_single_cell_power",
                   "rate": power, "informational": True,
                   "amendment_required": True, "pass": True})

    all_pass = all(item["pass"] for item in checks)
    report = {
        "protocol": "phase2_v3_frozen",
        "parameters": {
            "z_cell": Z_CELL, "alpha_cell": ALPHA_CELL,
            "alpha_global": ALPHA_GLOBAL, "alpha_lifetime": ALPHA_LIFETIME,
            "n_eval": N_EVAL, "lag_cap": LAG,
            "program_cells": G_PROGRAM, "lifetime_cells": G_LIFETIME,
            "program_bound": PROGRAM_BOUND, "per_cell_bound": PER_CELL_BOUND,
            "lifetime_bound": LIFETIME_BOUND, "reps": args.reps,
            "chunk": args.chunk, "seeds": SEEDS,
            "weak_mean": WEAK_MEAN, "cost_mean": COST_MEAN,
            "strong_mean": STRONG_MEAN, "power_mean": POWER_MEAN,
        },
        "amendment_pending": [
            "lifetime acceptance bound (currently the declared program bound 0.07)",
            "numeric positive-control power floor",
            "ratification of frozen reps=5000 and seeds in the protocol document",
        ],
        "checks": checks, "all_pass": all_pass,
    }
    report["sha"] = sha({k: v for k, v in report.items() if k != "sha"})
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    for c in checks:
        tag = "PASS" if c["pass"] else "FAIL"
        extra = " [amendment_required]" if c.get("amendment_required") else ""
        print(f"{tag}  {c['check']}: {c['rate']:.4f}{extra}")
    print("ALL PASS" if all_pass else "SUITE FAILED — KEEP PHASE 2 DISABLED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
