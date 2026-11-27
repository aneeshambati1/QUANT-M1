#!/usr/bin/env python3
"""Phase 2 calibration suite — PROTOCOL v4 (PROPOSED, pending owner ratification).

Redesign over v3 (see evidence/phase2_probe_protocol_v4.md):
  * Small pre-registered roster: 6 cells/program, 12 lifetime cells.
  * Per-cell alpha = alpha_lifetime / G_lifetime = 0.10/12 = 0.008333,
    z_cell = 2.394. Lifetime FWER <= 0.10 BY CONSTRUCTION (finite Bonferroni),
    program FWER <= 0.05. No undefined acceptance bound.
  * Longer evaluation window N_eval=1008, half_life=252 (n_eff ~ 724) to buy
    statistical power honestly. Power is a GATE (floor 0.25), not informational.
  * Reuses the VERIFIED prewhitened-AR(1) HAC estimator from run_null_suite.py
    (the v3 module). No estimator is re-implemented here; only the protocol
    parameters and acceptance bounds change.

This file does NOT enable Phase 2. It exits nonzero on any failed check.
Activation additionally requires owner ratification of v4 and a green CI run.

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

# Load the verified estimator module (v3 suite) without executing its main().
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "quant_m1_null_estimator", os.path.join(_ROOT, "scripts", "run_null_suite.py"))
est = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(est)

# ---- v4 frozen protocol parameters -------------------------------------
HL = 252.0
N_EVAL = 1008
G_PROGRAM = 6
G_LIFETIME = 12
ALPHA_LIFETIME = 0.10
ALPHA_CELL = ALPHA_LIFETIME / G_LIFETIME          # 0.008333
Z_CELL = est.nppf(1.0 - ALPHA_CELL)               # 2.394
PROGRAM_BOUND = 0.07                              # 0.05 + 0.02
PER_CELL_BOUND = ALPHA_CELL + 0.01                # 0.018333
LIFETIME_BOUND = 0.12                             # 0.10 + 0.02 (declared)
POWER_FLOOR = 0.25                                # GATE, not informational
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
    """Point the verified estimator at the v4 protocol parameters."""
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
    lam = 1.0 - 2.0 ** (-1.0 / 60.0)   # retirement uses the unchanged tracker rule
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
        add_rate(f"program6_rho{rho}", r, PROGRAM_BOUND)

    pc = est.per_cell_false_rate(args.reps, 0.0, 0.0, SEEDS["per_cell"], args.chunk)
    add_rate("per_cell_zero_edge", pc, PER_CELL_BOUND)

    lt = est.program_false_promotion_rate(args.reps, G_LIFETIME, 0.0, 0.0,
                                          SEEDS["lifetime"], args.chunk)
    add_rate("lifetime12_iid", lt, LIFETIME_BOUND)

    wk = est.program_false_promotion_rate(args.reps, G_PROGRAM, WEAK_MEAN, 0.3,
                                          SEEDS["weak"], args.chunk)
    add_rate("program6_weak_negative", wk, PROGRAM_BOUND)

    ct = est.program_false_promotion_rate(args.reps, G_PROGRAM, COST_MEAN, 0.0,
                                          SEEDS["cost"], args.chunk)
    add_rate("program6_cost_domination", ct, PROGRAM_BOUND)

    ret = retirement_detection(args.reps, SEEDS["retire"], args.chunk)
    checks.append({"check": "strong_negative_retirement_by_141", "rate": ret,
                   "bound_min": 0.95, "pass": bool(ret >= 0.95)})

    import numpy as np
    rng = np.random.default_rng(SEEDS["power"])
    x = est.generate(rng, args.reps, 1, POWER_MEAN, 0.3).reshape(args.reps, N_EVAL)
    pw = float(est.cert_stats(x).mean())
    checks.append({"check": "positive_control_power", "rate": pw,
                   "bound_min": POWER_FLOOR, "gate": True,
                   "pass": bool(pw >= POWER_FLOOR)})

    all_pass = all(c["pass"] for c in checks)
    report = {
        "protocol": "phase2_v4_PROPOSED",
        "parameters": {"hl": HL, "n_eval": N_EVAL, "g_program": G_PROGRAM,
                       "g_lifetime": G_LIFETIME, "alpha_cell": ALPHA_CELL,
                       "z_cell": Z_CELL, "program_bound": PROGRAM_BOUND,
                       "per_cell_bound": PER_CELL_BOUND,
                       "lifetime_bound": LIFETIME_BOUND,
                       "power_floor": POWER_FLOOR, "reps": args.reps,
                       "seeds": SEEDS},
        "checks": checks, "all_pass": all_pass,
        "activation": "BLOCKED — requires owner ratification of v4 + green CI",
    }
    report["sha"] = sha({k: v for k, v in report.items() if k != "sha"})
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    for c in checks:
        tag = "PASS" if c["pass"] else "FAIL"
        print(f"{tag}  {c['check']}: {c['rate']:.4f}")
    print("ALL PASS — v4 calibration satisfied (activation still gated)"
          if all_pass else "SUITE FAILED — KEEP PHASE 2 DISABLED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
