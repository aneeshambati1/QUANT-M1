#!/usr/bin/env python3
"""Proposed Phase 2 prewhitened-HAC calibration suite.

Research-only. This script simulates certificate decisions for complete
programs, not isolated candidate cells alone.

The estimator is:
    prewhitened AR(1) + Bartlett HAC on filtered residuals

The exact estimator implementation is mirrored against
ContributionTracker.se_stats before any simulation is accepted.

IMPORTANT:
- This is a proposed estimator amendment, not the currently frozen v3 method.
- The lifetime acceptance rule is undefined in the frozen v3 protocol.
- An unresolved protocol check keeps the script nonzero and Phase 2 disabled.
- This script never changes statistical limits to force a pass.

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


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from quant_m1.core import sha
from quant_m1.meta.contribution import (
    ContributionTracker,
    SE_ESTIMATOR_ID,
)
from quant_m1.research.promotion_ledger import nppf


# ---------------------------------------------------------------------------
# Frozen design parameters retained from Phase 2 Protocol v3.
# ---------------------------------------------------------------------------

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
PER_CELL_BOUND = ALPHA_CELL + 0.01

# No lifetime acceptance limit is defined in the frozen v3 document.
LIFETIME_BOUND = None

RETIRE_MIN = 120
RETIRE_BAND = -0.002
RETIRE_DELTA = 21

WEAK_MEAN = -0.0002
COST_MEAN = -0.0005
STRONG_MEAN = -0.005
POWER_MEAN = 0.001

DEFAULT_REPS = 5000
DEFAULT_CHUNK = 32

PHI_ABS_LIMIT = 0.995

SEEDS = {
    "program_iid": 101,
    "program_ar03": 102,
    "program_ar05": 103,
    "per_cell": 104,
    "lifetime": 201,
    "weak": 301,
    "cost": 401,
    "retire": 501,
    "power": 601,
}


# ---------------------------------------------------------------------------
# Statistical helpers.
# ---------------------------------------------------------------------------

def wilson_upper(
    rate: float,
    reps: int,
    z: float = 1.6449,
) -> float:
    """One-sided Wilson confidence upper bound."""
    if (
        reps < 1
        or not math.isfinite(rate)
        or not 0.0 <= rate <= 1.0
    ):
        raise ValueError("invalid Wilson inputs")

    denominator = 1.0 + z * z / reps

    center = (
        rate + z * z / (2.0 * reps)
    ) / denominator

    half = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / reps
            + z * z / (4.0 * reps * reps)
        )
        / denominator
    )

    return center + half


def _normalized_ewma_weights(n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be >= 2")

    raw = LAM * (
        (1.0 - LAM)
        ** np.arange(n - 1, -1, -1, dtype=np.float64)
    )

    total = float(raw.sum())

    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("invalid EWMA weight sum")

    return raw / total


def cert_components(x: np.ndarray) -> dict:
    """Vectorized mirror of ContributionTracker.se_stats.

    Input shape:
        (number_of_series, N_EVAL)

    Returns estimator components and the final lower confidence bound.
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2 or x.shape[1] != N_EVAL:
        raise ValueError(
            f"expected shape (rows, {N_EVAL})"
        )

    if x.shape[0] < 1:
        raise ValueError("at least one series is required")

    if not np.isfinite(x).all():
        raise ValueError(
            "null suite input contains nonfinite values"
        )

    n = x.shape[1]
    residual_count = n - 1

    weights = _normalized_ewma_weights(n)
    mean_hat = x @ weights

    centered = x - x.mean(
        axis=1,
        keepdims=True,
    )

    denominator = np.sum(
        centered[:, :-1] * centered[:, :-1],
        axis=1,
    )

    numerator = np.sum(
        centered[:, 1:] * centered[:, :-1],
        axis=1,
    )

    phi_hat = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )

    phi_valid = (
        np.isfinite(phi_hat)
        & (np.abs(phi_hat) < PHI_ABS_LIMIT)
        & (denominator > 0.0)
    )

    # Invalid phi values are replaced only for safe intermediate arithmetic.
    # They remain invalid in the final certificate decision.
    phi_safe = np.where(
        phi_valid,
        phi_hat,
        0.0,
    )

    residuals = (
        centered[:, 1:]
        - phi_safe[:, None] * centered[:, :-1]
    )

    residuals = (
        residuals
        - residuals.mean(
            axis=1,
            keepdims=True,
        )
    )

    gamma0 = np.mean(
        residuals * residuals,
        axis=1,
    )

    # Coefficient on the initial state:
    # c0 = sum_t a[t] * phi**t.
    powers = np.arange(
        n,
        dtype=np.float64,
    )

    c0 = np.sum(
        weights[None, :]
        * np.power(
            phi_safe[:, None],
            powers[None, :],
        ),
        axis=1,
    )

    # Coefficients on innovations e[1], ..., e[n-1]:
    # c[j] = a[j+1] + phi*c[j+1].
    residual_weights = np.empty(
        (x.shape[0], residual_count),
        dtype=np.float64,
    )

    next_weight = np.zeros(
        x.shape[0],
        dtype=np.float64,
    )

    for j in range(residual_count - 1, -1, -1):
        next_weight = (
            weights[j + 1]
            + phi_safe * next_weight
        )

        residual_weights[:, j] = next_weight

    initial_state_variance = np.divide(
        gamma0,
        1.0 - phi_safe * phi_safe,
        out=np.zeros_like(gamma0),
        where=(1.0 - phi_safe * phi_safe) > 0.0,
    )

    var_mean = (
        initial_state_variance * c0 * c0
        + gamma0
        * np.sum(
            residual_weights * residual_weights,
            axis=1,
        )
    )

    # Bartlett HAC on prewhitened residuals.
    effective_lag = min(
        LAG,
        residual_count - 1,
    )

    for k in range(1, effective_lag + 1):
        gamma_k = np.sum(
            residuals[:, k:] * residuals[:, :-k],
            axis=1,
        ) / residual_count

        cross_weight = np.sum(
            residual_weights[:, :-k]
            * residual_weights[:, k:],
            axis=1,
        )

        bartlett_weight = (
            1.0 - k / (effective_lag + 1.0)
        )

        var_mean += (
            2.0
            * bartlett_weight
            * gamma_k
            * cross_weight
        )

    valid = (
        phi_valid
        & np.isfinite(gamma0)
        & (gamma0 > 0.0)
        & np.isfinite(var_mean)
        & (var_mean > 0.0)
    )

    standard_error = np.full(
        x.shape[0],
        np.nan,
        dtype=np.float64,
    )

    standard_error[valid] = np.sqrt(
        var_mean[valid]
    )

    se_lcb = np.full(
        x.shape[0],
        -np.inf,
        dtype=np.float64,
    )

    se_lcb[valid] = (
        mean_hat[valid]
        - Z_CELL * standard_error[valid]
    )

    n_eff_corr = np.full(
        x.shape[0],
        np.nan,
        dtype=np.float64,
    )

    n_eff_corr[valid] = (
        gamma0[valid] / var_mean[valid]
    )

    return {
        "valid": valid,
        "mean": mean_hat,
        "var_mean": var_mean,
        "se": standard_error,
        "se_lcb": se_lcb,
        "phi_hat": phi_hat,
        "gamma0": gamma0,
        "n_eff_corr": n_eff_corr,
        "lag_cap": effective_lag,
    }


def cert_stats(x: np.ndarray) -> np.ndarray:
    """Return promotion decisions for a batch of candidate series."""
    result = cert_components(x)

    return (
        result["valid"]
        & (result["se_lcb"] > 0.0)
    )


def verify_production_mirror() -> dict:
    """Check production/simulator parity using identical full-length samples.

    Retirement is deliberately delayed only in this parity fixture.

    Without this setting, a random sample can trigger the tracker retirement
    rule before observation 399. The production tracker then correctly stops
    recording new observations, while the vectorized statistical mirror
    continues using all 399 observations. Comparing those two different
    histories creates a false parity failure.

    This does not change production retirement settings or production logic.
    """
    rng = np.random.default_rng(9001)

    scenarios = (
        (0.0, 0.0),
        (0.3, 0.0),
        (0.5, 0.0),
        (0.3, WEAK_MEAN),
    )

    observations = []

    for rho, mean in scenarios:
        batch = generate(
            rng,
            batch=2,
            cells=1,
            mu=mean,
            rho=rho,
        )[:, 0, :]

        observations.extend(batch)

    comparisons = 0

    for row in observations:
        # Important: the parity fixture must retain all 399 observations.
        # Production trackers keep their normal retirement settings.
        tracker = ContributionTracker(
            se_min_obs=120,
            se_max_obs=N_EVAL,
            retire_min_obs=N_EVAL + 1,
        )

        for value in row:
            tracker.update(float(value))

        if tracker.n != N_EVAL:
            raise RuntimeError(
                "PARITY_FIXTURE_DID_NOT_RETAIN_ALL_OBSERVATIONS"
            )

        production = tracker.se_stats(
            Z_CELL,
            LAG,
        )

        simulated = cert_components(
            row.reshape(1, -1)
        )

        simulated_valid = bool(
            simulated["valid"][0]
        )

        if bool(production["valid"]) != simulated_valid:
            raise RuntimeError(
                "PRODUCTION_SIMULATOR_VALIDITY_MISMATCH"
            )

        if simulated_valid:
            comparisons_to_check = (
                (
                    "mean",
                    production["mean"],
                    float(simulated["mean"][0]),
                ),
                (
                    "var_mean",
                    production["var_mean"],
                    float(simulated["var_mean"][0]),
                ),
                (
                    "se",
                    production["se"],
                    float(simulated["se"][0]),
                ),
                (
                    "se_lcb",
                    production["se_lcb"],
                    float(simulated["se_lcb"][0]),
                ),
                (
                    "phi_hat",
                    production["phi_hat"],
                    float(simulated["phi_hat"][0]),
                ),
                (
                    "gamma0",
                    production["gamma0"],
                    float(simulated["gamma0"][0]),
                ),
            )

            for name, actual, expected in comparisons_to_check:
                if not math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        f"PRODUCTION_SIMULATOR_MISMATCH:{name}"
                    )

            production_decision = (
                production["se_lcb"] > 0.0
            )

            simulated_decision = bool(
                simulated["se_lcb"][0] > 0.0
            )

            if production_decision != simulated_decision:
                raise RuntimeError(
                    "PRODUCTION_SIMULATOR_DECISION_MISMATCH"
                )

        comparisons += 1

    return {
        "passed": True,
        "comparisons": comparisons,
        "estimator_id": SE_ESTIMATOR_ID,
        "retirement_disabled_in_fixture": True,
    }


# ---------------------------------------------------------------------------
# Null generation.
# ---------------------------------------------------------------------------

def generate(
    rng: np.random.Generator,
    batch: int,
    cells: int,
    mu: float,
    rho: float,
    sigma: float = 0.01,
) -> np.ndarray:
    if (
        isinstance(batch, bool)
        or not isinstance(batch, int)
        or batch < 1
    ):
        raise ValueError(
            "batch must be a positive integer"
        )

    if (
        isinstance(cells, bool)
        or not isinstance(cells, int)
        or cells < 1
    ):
        raise ValueError(
            "cells must be a positive integer"
        )

    if not math.isfinite(mu):
        raise ValueError("mu must be finite")

    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            "sigma must be positive and finite"
        )

    if not -0.999 <= rho <= 0.999:
        raise ValueError(
            "rho outside supported range"
        )

    if rho == 0.0:
        return rng.normal(
            mu,
            sigma,
            (batch, cells, N_EVAL),
        )

    x = np.empty(
        (batch, cells, N_EVAL),
        dtype=np.float64,
    )

    x[:, :, 0] = rng.normal(
        mu,
        sigma,
        (batch, cells),
    )

    innovation_sigma = sigma * math.sqrt(
        1.0 - rho * rho
    )

    innovations = rng.normal(
        0.0,
        innovation_sigma,
        (batch, cells, N_EVAL),
    )

    for t in range(1, N_EVAL):
        x[:, :, t] = (
            mu * (1.0 - rho)
            + rho * x[:, :, t - 1]
            + innovations[:, :, t]
        )

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
        batch = min(
            chunk,
            reps - done,
        )

        x = generate(
            rng,
            batch,
            cells,
            mu,
            rho,
        ).reshape(
            batch * cells,
            N_EVAL,
        )

        promoted = cert_stats(x).reshape(
            batch,
            cells,
        )

        false_programs += int(
            promoted.any(axis=1).sum()
        )

        done += batch

    return false_programs / reps


def per_cell_false_rate(
    reps: int,
    mu: float,
    rho: float,
    seed: int,
    chunk: int,
) -> float:
    rng = np.random.default_rng(seed)

    false_cells = 0
    total = 0

    while total < reps:
        batch = min(
            chunk,
            reps - total,
        )

        x = generate(
            rng,
            batch,
            1,
            mu,
            rho,
        ).reshape(
            batch,
            N_EVAL,
        )

        false_cells += int(
            cert_stats(x).sum()
        )

        total += batch

    return false_cells / reps


def retirement_detection(
    reps: int,
    seed: int,
    chunk: int,
) -> float:
    rng = np.random.default_rng(seed)

    length = (
        RETIRE_MIN
        + RETIRE_DELTA
        + 40
    )

    detected = 0
    total = 0

    while total < reps:
        batch = min(
            chunk,
            reps - total,
        )

        x = generate(
            rng,
            batch,
            1,
            STRONG_MEAN,
            0.0,
        )[:, 0, :length]

        mean = np.zeros(batch)
        first_hit = np.full(
            batch,
            10**9,
        )

        for t in range(length):
            mean = (
                (1.0 - LAM) * mean
                + LAM * x[:, t]
            )

            hit = (
                (t + 1 >= RETIRE_MIN)
                & (mean < RETIRE_BAND)
                & (first_hit > 10**8)
            )

            first_hit[hit] = t + 1

        detected += int(
            (
                first_hit
                <= RETIRE_MIN + RETIRE_DELTA
            ).sum()
        )

        total += batch

    return detected / reps


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Proposed prewhitened-HAC Phase 2 null calibration. "
            "This does not authorize Phase 2 activation."
        )
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK,
    )

    parser.add_argument(
        "--out",
        default="evidence/null_suite_report.json",
    )

    args = parser.parse_args()

    if args.reps < 100:
        parser.error("--reps must be >= 100")

    if args.chunk < 1:
        parser.error("--chunk must be >= 1")

    # A statistical result is not accepted unless production and simulator
    # agree on identical complete observation histories.
    try:
        parity = verify_production_mirror()

    except Exception as exc:
        print(
            f"FAIL  production/simulator parity: {exc}"
        )
        print("PHASE 2 MUST REMAIN DISABLED")
        return 2

    checks = []

    def add_rate(
        name: str,
        rate: float,
        bound: float,
        reps: int,
    ) -> None:
        upper = wilson_upper(
            rate,
            reps,
        )

        checks.append({
            "check": name,
            "rate": rate,
            "wilson95_upper": upper,
            "bound": bound,
            "pass": bool(upper <= bound),
            "status": (
                "PASS"
                if upper <= bound
                else "FAIL"
            ),
        })

    # Whole-program false-promotion rate.
    for rho, key in (
        (0.0, "program_iid"),
        (0.3, "program_ar03"),
        (0.5, "program_ar05"),
    ):
        rate = program_false_promotion_rate(
            args.reps,
            G_PROGRAM,
            0.0,
            rho,
            SEEDS[key],
            args.chunk,
        )

        add_rate(
            f"program24_iid_or_ar{rho}",
            rate,
            PROGRAM_BOUND,
            args.reps,
        )

    # Single-cell false-promotion rate.
    per_cell = per_cell_false_rate(
        args.reps,
        0.0,
        0.0,
        SEEDS["per_cell"],
        args.chunk,
    )

    add_rate(
        "per_cell_zero_edge",
        per_cell,
        PER_CELL_BOUND,
        args.reps,
    )

    # Lifetime rate is measured, but the frozen protocol does not state a
    # lifetime acceptance limit. Do not invent one or count this as a pass.
    lifetime = program_false_promotion_rate(
        args.reps,
        G_LIFETIME,
        0.0,
        0.0,
        SEEDS["lifetime"],
        args.chunk,
    )

    lifetime_upper = wilson_upper(
        lifetime,
        args.reps,
    )

    checks.append({
        "check": "lifetime48_iid",
        "rate": lifetime,
        "wilson95_upper": lifetime_upper,
        "bound": LIFETIME_BOUND,
        "pass": False,
        "status": "BLOCKED_PROTOCOL_ACCEPTANCE_UNDEFINED",
        "blocking": True,
        "reason": (
            "The frozen v3 protocol specifies alpha_lifetime=0.10 "
            "but does not define a lifetime Wilson-upper acceptance limit."
        ),
    })

    # Negative-edge robustness.
    weak = program_false_promotion_rate(
        args.reps,
        G_PROGRAM,
        WEAK_MEAN,
        0.3,
        SEEDS["weak"],
        args.chunk,
    )

    add_rate(
        "program24_weak_negative",
        weak,
        PROGRAM_BOUND,
        args.reps,
    )

    # Costs dominate the assumed small positive gross edge.
    cost = program_false_promotion_rate(
        args.reps,
        G_PROGRAM,
        COST_MEAN,
        0.0,
        SEEDS["cost"],
        args.chunk,
    )

    add_rate(
        "program24_cost_domination",
        cost,
        PROGRAM_BOUND,
        args.reps,
    )

    # Strong-negative retirement detection.
    retirement = retirement_detection(
        args.reps,
        SEEDS["retire"],
        args.chunk,
    )

    checks.append({
        "check": "strong_negative_retirement_by_141",
        "rate": retirement,
        "bound_min": 0.95,
        "pass": bool(retirement >= 0.95),
        "status": (
            "PASS"
            if retirement >= 0.95
            else "FAIL"
        ),
    })

    # Positive-control power remains informational because the frozen
    # protocol did not specify a numeric power floor.
    power = program_false_promotion_rate(
        args.reps,
        1,
        POWER_MEAN,
        0.3,
        SEEDS["power"],
        args.chunk,
    )

    checks.append({
        "check": "positive_control_single_cell_power",
        "rate": power,
        "informational": True,
        "amendment_required": True,
        "pass": True,
        "status": "INFORMATIONAL_NO_FROZEN_POWER_FLOOR",
    })

    all_pass = all(
        bool(item["pass"])
        for item in checks
    )

    report = {
        "protocol": "phase2_v3_proposed_prewhitened_hac",
        "active_protocol": "phase2_v3_frozen",
        "protocol_status": "PROPOSED_AMENDMENT_NOT_APPROVED",
        "activation_ready": False,
        "estimator_id": SE_ESTIMATOR_ID,
        "production_simulator_parity": parity,
        "parameters": {
            "half_life_obs": HL,
            "lambda": LAM,
            "z_cell": Z_CELL,
            "alpha_cell": ALPHA_CELL,
            "alpha_global": ALPHA_GLOBAL,
            "alpha_lifetime": ALPHA_LIFETIME,
            "n_eval": N_EVAL,
            "lag_cap": LAG,
            "program_cells": G_PROGRAM,
            "lifetime_cells": G_LIFETIME,
            "program_bound": PROGRAM_BOUND,
            "per_cell_bound": PER_CELL_BOUND,
            "lifetime_bound": LIFETIME_BOUND,
            "phi_abs_limit": PHI_ABS_LIMIT,
            "reps": args.reps,
            "chunk": args.chunk,
            "seeds": SEEDS,
            "weak_mean": WEAK_MEAN,
            "cost_mean": COST_MEAN,
            "strong_mean": STRONG_MEAN,
            "power_mean": POWER_MEAN,
        },
        "amendment_pending": [
            (
                "Approve or reject the prewhitened AR(1) plus Bartlett "
                "HAC residual estimator and re-freeze the protocol."
            ),
            (
                "Define the lifetime acceptance rule for G_lifetime=48; "
                "the active v3 document does not specify one."
            ),
            (
                "Ratify the 5000-replication requirement and the fixed "
                "simulation seeds in the protocol document."
            ),
            (
                "Define a numeric positive-control power floor, or "
                "explicitly retain power as informational."
            ),
        ],
        "checks": checks,
        "all_pass": all_pass,
        "activation_ready": False,
    }

    report["sha"] = sha({
        key: value
        for key, value in report.items()
        if key != "sha"
    })

    output_dir = os.path.dirname(args.out)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        args.out,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            sort_keys=True,
        )

    for item in checks:
        status = item["status"]
        extra = ""

        if item.get("informational"):
            extra = " [informational]"
        elif item.get("blocking"):
            extra = " [protocol decision required]"

        print(
            f"{status:>8}  "
            f"{item['check']}: "
            f"{item['rate']:.4f}"
            f"{extra}"
        )

    if all_pass:
        print(
            "CALIBRATION CHECKS PASS; "
            "PROTOCOL RATIFICATION STILL REQUIRED"
        )
    else:
        print(
            "SUITE BLOCKED/FAILED — "
            "KEEP PHASE 2 DISABLED"
        )

    print(f"Report: {args.out}")
    print(f"Report SHA: {report['sha']}")

    # Even a passing simulation cannot activate Phase 2 until the owner
    # approves and re-freezes the protocol amendment.
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
