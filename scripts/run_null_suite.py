#!/usr/bin/env python3
"""Frozen Phase 2 calibration suite; research only.

Simulates the certificate statistic at n=399. Tests program-wide 24-cell
false promotion and lifetime 48-cell false promotion using bounded-memory
chunks. This script does not change alpha, z, or acceptance limits to force
a pass. Any failed check exits nonzero.

Run:
    python scripts/run_null_suite.py --reps 2000
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
PROGRAM_BOUND = 0.07
LIFETIME_BOUND = 0.10
RETIRE_MIN = 120
RETIRE_BAND = -0.002
RETIRE_DELTA = 21


def wilson_upper(rate: float, reps: int, z: float = 1.6449) -> float:
    if reps < 1 or not 0.0 <= rate <= 1.0:
        raise ValueError("invalid Wilson inputs")
    den = 1.0 + z * z / reps
    center = (rate + z * z / (2.0 * reps)) / den
    half = z * math.sqrt(
        rate * (1.0 - rate) / reps + z * z / (4.0 * reps * reps)
    ) / den
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
    valid = (
        np.isfinite(var_mean)
        & (var_mean > 0.0)
        & np.isfinite(gamma0)
        & (gamma0 > 0.0)
    )
    lcb = np.full(len(x), -np.inf)
    lcb[valid] = mhat[valid] - Z_CELL * np.sqrt(var_mean[valid])
    return valid & (lcb > 0.0)


def generate(
    rng: np.random.Generator,
    batch: int,
    cells: int,
    mu: float,
    rho: float,
    sigma: float = 0.01,
) -> np.ndarray:
    """Generate independent candidate series within each replicate."""
    if not -0.999 <= rho <= 0.999:
        raise ValueError("rho outside supported range")
    x = np.empty((batch, cells, N_EVAL), dtype=np.float64)
    x[:, :, 0] = rng.normal(mu, sigma, (batch, cells))
    if rho == 0.0:
        x[:, :, :] = rng.normal(mu, sigma, (batch, cells, N_EVAL))
        return x

    innovations = rng.normal(
        0.0, sigma * math.sqrt(1.0 - rho * rho),
        (batch, cells, N_EVAL),
    )
    for t in range(1, N_EVAL):
        x[:, :, t] = mu * (1.0 - rho) + rho * x[:, :, t - 1] + innovations[:, :, t]
    return x


def program_false_promotion_rate(
    reps: int,
    cells: int,
    mu: float,
    rho: float,
    seed: int,
    chunk: int,
) -> float:
    rng = np.random.default_rng(seed)
    false_programs = 0
    done = 0
    while done < reps:
        b = min(chunk, reps - done)
        x = generate(rng, b, cells, mu, rho)
        promoted = cert_stats(x.reshape(b * cells, N_EVAL)).reshape(b, cells)
        false_programs += int(promoted.any(axis=1).sum())
        done += b
    return false_programs / reps


def retirement_detection(reps: int, seed: int, chunk: int) -> float:
    rng = np.random.default_rng(seed)
    total = 0
    detected = 0
    length = RETIRE_MIN + RETIRE_DELTA + 40
    while total < reps:
        b = min(chunk, reps - total)
        x = rng.normal(-0.005, 0.01, (b, length))
        mean = np.zeros(b)
        first = np.full(b, length + 1, dtype=np.int32)
        for t in range(length):
            mean = (1.0 - LAM) * mean + LAM * x[:, t]
            hit = (
                (t + 1 >= RETIRE_MIN)
                & (mean < RETIRE_BAND)
                & (first > length)
            )
            first[hit] = t + 1
        detected += int((first <= RETIRE_MIN + RETIRE_DELTA).sum())
        total += b
    return detected / reps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--out", default="evidence/null_suite_report.json")
    args = parser.parse_args()

    if args.reps < 100 or args.chunk < 1:
        parser.error("--reps must be >=100 and --chunk must be >=1")

    checks = []

    def add_rate(name: str, rate: float, bound: float, reps: int) -> None:
        upper = wilson_upper(rate, reps)
        checks.append({
            "check": name,
            "rate": rate,
            "wilson95_upper": upper,
            "bound": bound,
            "pass": bool(upper <= bound),
        })

    # One 24-cell program: reject the replicate if ANY cell promotes.
    for rho, seed in ((0.0, 101), (0.3, 102), (0.5, 103)):
        rate = program_false_promotion_rate(
            args.reps, G_PROGRAM, 0.0, rho, seed, args.chunk
        )
        add_rate(f"program24_iid_or_ar{rho}", rate, PROGRAM_BOUND, args.reps)

    # Lifetime roster: one 48-cell family of registered opportunities.
    lifetime_rate = program_false_promotion_rate(
        args.reps, G_LIFETIME, 0.0, 0.0, 201, args.chunk
    )
    add_rate("lifetime48_iid", lifetime_rate, LIFETIME_BOUND, args.reps)

    weak_negative = program_false_promotion_rate(
        args.reps, G_PROGRAM, -0.0002, 0.3, 301, args.chunk
    )
    add_rate("program24_weak_negative", weak_negative, PROGRAM_BOUND, args.reps)

    cost_domination = program_false_promotion_rate(
        args.reps, G_PROGRAM, -0.0005, 0.0, 401, args.chunk
    )
    add_rate("program24_cost_domination", cost_domination, PROGRAM_BOUND, args.reps)

    retirement = retirement_detection(args.reps, 501, args.chunk)
    checks.append({
        "check": "strong_negative_retirement_by_141",
        "rate": retirement,
        "bound_min": 0.95,
        "pass": bool(retirement >= 0.95),
    })

    # Positive control is disclosure only, not a calibration pass condition.
    power = program_false_promotion_rate(
        args.reps, 1, 0.001, 0.3, 601, args.chunk
    )
    checks.append({
        "check": "positive_control_single_cell_power",
        "rate": power,
        "informational": True,
        "pass": True,
    })

    all_pass = all(item["pass"] for item in checks)
    report = {
        "protocol": "phase2_v3_frozen",
        "parameters": {
            "z_cell": Z_CELL,
            "alpha_cell": ALPHA_CELL,
            "alpha_global": ALPHA_GLOBAL,
            "alpha_lifetime": ALPHA_LIFETIME,
            "n_eval": N_EVAL,
            "lag_cap": LAG,
            "program_cells": G_PROGRAM,
            "lifetime_cells": G_LIFETIME,
            "program_bound": PROGRAM_BOUND,
            "lifetime_bound": LIFETIME_BOUND,
            "reps": args.reps,
            "chunk": args.chunk,
        },
        "checks": checks,
        "all_pass": all_pass,
    }
    report["sha"] = sha({k: v for k, v in report.items() if k != "sha"})

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    for item in checks:
        state = "PASS" if item["pass"] else "FAIL"
        print(f"{state} {item['check']}: {item['rate']:.4f}")
    print("ALL PASS" if all_pass else "SUITE FAILED — KEEP PHASE 2 DISABLED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
