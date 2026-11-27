#!/usr/bin/env python3
"""Phase 2 calibration suite (v3 Amendments D/E). Offline, deterministic seed.

Simulates the ACTUAL certificate procedure (renormalized EWMA mean +
Bartlett-tapered weighted-covariance variance at n=399, z_cell from the global
Bonferroni budget). Acceptance: one-sided Wilson 95% upper <= nominal +
tolerance. Program-wide bound 0.07; per-cell bound alpha_cell + 0.01.
Strong-negative retirement: unchanged rule, frozen Delta=21 (deadline 141).
Writes a hashed JSON report; exits non-zero on any FAIL. Run BEFORE enabling.
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_m1.core import sha
from quant_m1.research.promotion_ledger import nppf

HL = 60.0; LAM = 1 - 2 ** (-1 / HL); N_EVAL = 399; SE_MIN = 120; LAG = 21
ALPHA_CELL = 0.05 / 24; Z_CELL = nppf(1 - ALPHA_CELL)
RETIRE_MIN = 120; RETIRE_BAND = -0.002; DELTA = 21

def wilson_upper(phat, B, z=1.6449):
    den = 1 + z * z / B
    c = (phat + z * z / (2 * B)) / den
    h = z * math.sqrt(phat * (1 - phat) / B + z * z / (4 * B * B)) / den
    return c + h

def cert_stats(X):
    """Vectorized mirror of ContributionTracker.se_stats at n=N_EVAL."""
    B, n = X.shape
    w = LAM * (1 - LAM) ** np.arange(n - 1, -1, -1)
    W = w.sum()
    mhat = (X @ w) / W
    d = X - X.mean(axis=1, keepdims=True)
    g0 = (d * d).mean(axis=1)
    acc = g0 * (w * w).sum()
    for k in range(1, min(LAG, n - 1) + 1):
        gk = (d[:, k:] * d[:, :-k]).mean(axis=1)
        cross = (w[:-k] * w[k:]).sum()
        acc += 2 * (1 - k / (LAG + 1)) * gk * cross
    var = acc / (W * W)
    ok = np.isfinite(var) & (var > 0) & (g0 > 0)
    lcb = np.where(ok, mhat - Z_CELL * np.sqrt(np.maximum(var, 0)), -np.inf)
    return ok & (lcb > 0)

def gen(B, n, mu, rho, sig=0.01, seed=0):
    rng = np.random.default_rng(seed)
    if rho == 0:
        return rng.normal(mu, sig, (B, n))
    X = np.zeros((B, n)); X[:, 0] = rng.normal(mu, sig, B)
    e = rng.normal(0, sig * math.sqrt(1 - rho * rho), (B, n))
    for t in range(1, n):
        X[:, t] = mu * (1 - rho) + rho * X[:, t - 1] + e[:, t]
    return X

def retirement_detection(B, seed=5):
    X = gen(B, RETIRE_MIN + DELTA + 40, -0.005, 0.0, seed=seed)
    m = np.zeros(B); first = np.full(B, 10**9)
    for t in range(X.shape[1]):
        m = (1 - LAM) * m + LAM * X[:, t]
        hit = (t + 1 >= RETIRE_MIN) & (m < RETIRE_BAND) & (first > 10**8)
        first[hit] = t + 1
    return float((first <= RETIRE_MIN + DELTA).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="evidence/null_suite_report.json")
    args = ap.parse_args()
    B = args.reps
    checks = []
    def add(name, rate, bound, extra=None):
        u = wilson_upper(rate, B)
        checks.append({"check": name, "rate": rate, "wilson95_upper": u,
                       "bound": bound, "pass": bool(u <= bound), **(extra or {})})
    # (a,b) zero-edge nulls, program-wide target 5% -> bound 0.07
    for rho in (0.0, 0.3, 0.5):
        add(f"zero_edge_rho_{rho}", float(cert_stats(gen(B, N_EVAL, 0.0, rho, seed=1)).mean()), 0.07)
    # per-cell bound alpha_cell + 0.01 (single-cell nominal alpha_cell)
    add("per_cell_zero_edge", float(cert_stats(gen(B, N_EVAL, 0.0, 0.0, seed=2)).mean()), ALPHA_CELL + 0.01)
    # weak-negative: rate-based, same program-wide rule
    add("weak_negative", float(cert_stats(gen(B, N_EVAL, -0.0002, 0.3, seed=3)).mean()), 0.07)
    # cost-domination: gross edge below cost -> after-cost series negative
    add("cost_domination", float(cert_stats(gen(B, N_EVAL, -0.0005, 0.0, seed=4)).mean()), 0.07)
    # strong-negative retirement, frozen Delta=21, require >= 0.95
    det = retirement_detection(B)
    checks.append({"check": "strong_negative_retirement_by_141", "rate": det,
                   "bound_min": 0.95, "pass": bool(det >= 0.95)})
    # positive control: power disclosure (no pass gate; reported)
    pw = float(cert_stats(gen(B, N_EVAL, 0.001, 0.3, seed=6)).mean())
    checks.append({"check": "positive_control_power", "rate": pw, "informational": True, "pass": True})
    ok = all(c["pass"] for c in checks)
    report = {"params": {"z_cell": Z_CELL, "n_eval": N_EVAL, "lag": LAG,
                         "delta": DELTA, "reps": B}, "checks": checks, "all_pass": ok}
    report["sha"] = sha({k: v for k, v in report.items() if k != "sha"})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    for c in checks:
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']}: rate={c['rate']:.4f}")
    print("ALL PASS" if ok else "SUITE FAILED — DO NOT ENABLE")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
