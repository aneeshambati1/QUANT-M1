#!/usr/bin/env python3
"""Phase 2 calibration suite — PROTOCOL v4 (3/6 roster; owner-selected design).

Frozen design (evidence/phase2_probe_protocol_v4.md):
  G_program=3, G_lifetime=6, alpha_global=0.05, alpha_lifetime=0.10,
  alpha_cell = alpha_lifetime/G_lifetime = 0.10/6 = 0.016667, z_cell=2.1280,
  N_eval=399, half_life=60. Program FWER <= 3*alpha_cell ~ 0.049; lifetime
  FWER <= 6*alpha_cell = 0.10 BY CONSTRUCTION. Acceptance: program 0.07,
  per-cell alpha_cell+0.01=0.026667, lifetime Wilson-upper 0.12 (0.10 + 0.02
  Monte Carlo tolerance; does NOT raise the nominal 0.10 budget).

Positive-control power is INFORMATIONAL (measured ~0.14). It is disclosed as
weak, not claimed adequate. No threshold is tuned to force a pass.

Reuses the VERIFIED prewhitened-AR(1) HAC estimator from run_null_suite.py.
This file does NOT enable Phase 2; activation requires owner approval + green
5000-rep CI run on the ratified commit.

Run:
    python scripts/run_null_suite_v4.py --reps 5000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_m1.core import sha

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "quant_m1_null_estimator", os.path.join(_ROOT, "scripts", "run_null_suite.py"))
est = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(est)

# ---- v4 frozen protocol parameters (3/6 roster) ------------------------
HL = 60.0
N_EVAL = 399
G_PROGRAM = 3
G_LIFETIME = 6
ALPHA_GLOBAL = 0.05
ALPHA_LIFETIME = 0.10
ALPHA_CELL = ALPHA_LIFETIME / G_LIFETIME          # 0.016667
Z_CELL = est.nppf(1.0 - ALPHA_CELL)               # 2.1280
PROGRAM_BOUND = 0.07                              # 0.05 + 0.02
PER_CELL_BOUND = ALPHA_CELL + 0.01                # 0.026667
LIFETIME_BOUND = 0.12                             # 0.10 + 0.02 (declared)
RETIRE_MIN = 120
RETIRE_BAND = -0.002
RETIRE_DELTA = 21
WEAK_MEAN = -0.0002
COST_MEAN = -0.0005
STRONG_MEAN = -0.005
POWER_MEAN = 0.001
DEFAULT_REPS = 5000
SEEDS = {"prog_iid": 101, "prog_ar03": 102, "prog_ar05": 103,
         "per_cell": 104, "lifetime": 201, "weak": 301, "cost": 401,
         "retire": 501, "power": 601}


def _configure_estimator() -> None:
    est.HL = HL
    est.LAM = 1.0 - 2.0 ** (-1.0 / HL)
    est.N_EVAL = N_EVAL
    est.G_PROGRAM = G_PROGRAM
    est.G_LIFETIME = G_LIFETIME
    est.ALPHA_CELL = ALPHA_CELL
    est.Z_CELL = Z_CELL


def wilson_upper(rate: float, reps: int, z: float = 1.6449) -> float:
    if reps < 1 or not 0.0 <= rate <= 1.0:
        raise ValueError("invalid Wilson inputs")
    den = 1.0 + z * z / reps
    center = (rate + z * z / (2.0 * reps)) / den
    half = z * math.sqrt(rate * (1.0 - rate) / reps + z * z / (4.0 * reps * reps)) / den
    return center + half


def retirement_detection(reps: int, seed: int, chunk: int) -> float:
    import numpy as np
    rng = np.random.default_rng(seed)
    length = RETIRE_MIN + RETIRE_DELTA + 40
    detected = 0
    total = 0
    lam = 1.0 - 2.0 ** (-1.0 / 60.0)
    while total < reps:
        b = min(chunk, reps - total)
        x = est.generate(rng, b, 1, STRONG_MEAN, 0.0)[:, 0, :length]
        m = np.zeros(b)
        first = np.full(b, 10**9)
        for t in range(length):
            m = (1.0 - lam) * m + lam * x[:, t]
            hit = (t + 1 >= RETIRE_MIN) & (m < RETIRE_BAND) & (first > 10**8)
            first[hit] = t + 1
        detected += int((first <= RETIRE_MIN + RETIRE_DELTA).sum())
        total += b
    return detected / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--out", default="evidence/null_suite_v4_report.json")
    args = ap.parse_args()
    if args.reps < 100 or args.chunk < 1:
        ap.error("--reps must be >=100 and --chunk >=1")

    _configure_estimator()
    checks = []

    def add_rate(name, rate, bound):
        upper = wilson_upper(rate, args.reps)
        checks.append({"check": name, "rate": rate, "wilson95_upper": upper,
                       "bound": bound, "pass": bool(upper <= bound)})

    for rho, key in ((0.0, "prog_iid"), (0.3, "prog_ar03"), (0.5, "prog_ar05")):
        r = est.program_false_promotion_rate(args.reps, G_PROGRAM, 0.0, rho,
                                             SEEDS[key], args.chunk)
        add_rate(f"program3_rho{rho}", r, PROGRAM_BOUND)

    pc = est.per_cell_false_rate(args.reps, 0.0, 0.0, SEEDS["per_cell"], args.chunk)
    add_rate("per_cell_zero_edge", pc, PER_CELL_BOUND)

    lt = est.program_false_promotion_rate(args.reps, G_LIFETIME, 0.0, 0.0,
                                          SEEDS["lifetime"], args.chunk)
    add_rate("lifetime6_iid", lt, LIFETIME_BOUND)

    wk = est.program_false_promotion_rate(args.reps, G_PROGRAM, WEAK_MEAN, 0.3,
                                          SEEDS["weak"], args.chunk)
    add_rate("program3_weak_negative", wk, PROGRAM_BOUND)

    ct = est.program_false_promotion_rate(args.reps, G_PROGRAM, COST_MEAN, 0.0,
                                          SEEDS["cost"], args.chunk)
    add_rate("program3_cost_domination", ct, PROGRAM_BOUND)

    ret = retirement_detection(args.reps, SEEDS["retire"], args.chunk)
    checks.append({"check": "strong_negative_retirement_by_141", "rate": ret,
                   "bound_min": 0.95, "pass": bool(ret >= 0.95)})

    import numpy as np
    rng = np.random.default_rng(SEEDS["power"])
    x = est.generate(rng, args.reps, 1, POWER_MEAN, 0.3).reshape(args.reps, N_EVAL)
    pw = float(est.cert_stats(x).mean())
    checks.append({"check": "positive_control_power", "rate": pw,
                   "informational": True, "pass": True,
                   "note": "WEAK (~0.14); disclosed, not claimed adequate"})

    all_pass = all(c["pass"] for c in checks)
    report = {
        "protocol": "phase2_v4_3of6",
        "parameters": {"hl": HL, "n_eval": N_EVAL, "g_program": G_PROGRAM,
                       "g_lifetime": G_LIFETIME, "alpha_global": ALPHA_GLOBAL,
                       "alpha_lifetime": ALPHA_LIFETIME, "alpha_cell": ALPHA_CELL,
                       "z_cell": Z_CELL, "program_bound": PROGRAM_BOUND,
                       "per_cell_bound": PER_CELL_BOUND,
                       "lifetime_bound": LIFETIME_BOUND, "reps": args.reps,
                       "seeds": SEEDS},
        "checks": checks, "all_pass": all_pass,
        "activation": "BLOCKED — requires owner approval + green 5000-rep CI",
    }
    report["sha"] = sha({k: v for k, v in report.items() if k != "sha"})
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    for c in checks:
        tag = "PASS" if c["pass"] else "FAIL"
        extra = " [informational]" if c.get("informational") else ""
        print(f"{tag}  {c['check']}: {c['rate']:.4f}{extra}")
    print("ALL PASS — v4 (3/6) calibration satisfied (activation still gated)"
          if all_pass else "SUITE FAILED — KEEP PHASE 2 DISABLED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
