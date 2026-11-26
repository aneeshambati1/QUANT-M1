# QUANT-M1: Research & Infrastructure Memo

**Date:** 2026-11-26  
**Status:** NO TRADEABLE EDGE DEMONSTRATED. INFRASTRUCTURE VALIDATED — PAPER / RESEARCH ONLY.  
**Panel:** 10 ETFs (SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, USO, UUP), 2015-2024, Daily Closes.

---

## 1. Executive Summary

This project set out to build an institutional-grade, autonomous trading infrastructure and discover a cross-sectional momentum edge. 

**The verdict:** The infrastructure was successfully built, validated by an automated test suite and external red-team audits. The authoritative test count is the GitHub Actions `tests` workflow run at the current HEAD (414 passing at the verified commit); historical counts quoted in earlier memos are superseded. However, the research phase yielded **zero survivors** across 13 pre-registered market trials. Subsequent power analysis proved the dataset and time horizon are fundamentally insufficient to detect realistic edge sizes. The search was halted based on pre-declared statistical stopping criteria, not frustration.

## 2. The Core Finding: Inconclusive, Not Negative

It is vital to distinguish between "no edge exists" and "we cannot see the edge." 

Our instrument controls proved the measurement pipeline works: it successfully detects an injected IC of 0.096 and correctly ignores an IC of 0.000. However, it **cannot resolve** an IC of 0.030 (the lower bound of our target tradeable band) at N=13 trials. 

The minimum detectable effect (MDE) for volatility conditioning on this specific 10-year panel is **44% MSE reduction**, due to an effective sample size of only ~16 independent volatility regimes. Because realistic financial effects are an order of magnitude smaller than our detection threshold, the null result is strictly **inconclusive**.

## 3. Infrastructure Built

The system implements controls rarely seen outside institutional environments:

| Component | Implementation |
|---|---|
| **Execution Kernel** | Integer-money accounting, reachable-fill risk enumeration, fail-closed divergence latches. |
| **Data Provenance** | Canonical, order-invariant SHA-256 content hashing. Per-symbol attribution for vendor revisions. |
| **Reproducibility** | Frozen price panel (`evidence/panel_real_v1.csv`) with bit-exact manifest verification. |
| **Governance** | Hash-chained, HMAC-signed trial ledger. Bonferroni multiple-testing gates. |
| **Settlement** | Unified authority modeling the historical T+2 to T+1 transition (effective 2024-05-28). |
| **Cost Accounting** | Deterministic turnover attribution separating mechanical costs from path-dependent returns. |

## 4. Research Results

### 4.1 Market Trials (13 Registered, 0 Survivors)
- **Hypothesis:** Cross-sectional trailing-return momentum.
- **Grid:** 1 baseline + 12 pre-registered cells (lookbacks 5/10/63, horizons 1/5/10/21).
- **Result:** Best absolute HAC t-statistic was 1.51. Bonferroni threshold at N=13 requires |t| ≥ 2.89. 
- **Conclusion:** No cell cleared the gate. 

### 4.2 Mechanical Findings (Cost Attribution)
While statistical alpha was absent, deterministic cost accounting yielded a definitive mechanical finding regarding portfolio turnover:

| Policy | Ann. Traded | Cost @ 3bp | Cost @ 10bp |
|---|---|---|---|
| Buy & Hold | 0.00x | 0.0 bp | 0.0 bp |
| **Band Rebalance (25%)** | **0.17x** | **0.5 bp** | **1.7 bp** |
| Monthly (21d) | 0.38x | 1.1 bp | 3.8 bp |
| Daily | 1.78x | 5.3 bp | 17.8 bp |
| Chase Top 3 (Weekly) | 32.60x | 97.8 bp | 326.0 bp |

**Finding:** Concentrated, high-frequency chasing consumes 196x more capital in friction than disciplined threshold-band rebalancing, for no statistically significant improvement in descriptive Sharpe ratios.

## 5. Why the Search Stopped

Continuing to test new hypotheses on this specific dataset violates the project's pre-declared stopping criterion. 

1. **Time-Series Power:** 2,516 daily rows of 10 highly correlated ETFs do not provide 25,160 independent observations. Forward-return overlap and serial dependence reduce the effective sample size drastically.
2. **Regime Starvation:** 10 years contains too few independent volatility regimes to validate conditional-exposure strategies (e.g., volatility targeting).
3. **Data Limitations:** The panel relies on back-adjusted, non-Point-In-Time (PIT) closes, introducing unavoidable survivorship and look-ahead biases that cannot be patched via software.

## 6. Engineering Defect Status (Red Team Findings — updated)

External adversarial audits identified architectural gaps. Current status at the verified HEAD:

- **[P0] Operator Auth Bypass — RESOLVED.** Operator clearance now uses an independent one-time key hash chain (`OperatorConsole`); the Gateway stores only a hash and cannot self-clear. Proven by `test_latch_integrity`.
- **[P1] Execution State Machine — PARTIALLY RESOLVED.** Explicit `UNKNOWN` state, timeout-stays-ambiguous, snapshot-authoritative fills and partial-fill handling are implemented and tested. A durable venue outbox and restart venue reconciliation (persistent broker exec_ids) remain open (register O-08).
- **[P1] DSR Integration — OPEN.** DSR and PBO exist in the codebase but are not wired into the primary trial acceptance gate; they require a pre-registered return-series protocol (register O-03). Do not apply DSR to HAC t-stats.
- **[P2] Horizon Mixing — RESOLVED.** `ContributionTracker` carries a fixed `horizon_days`; mixed-horizon weight resolution fails closed, and per-session attribution only updates horizon-1 trackers. Proven by `test_contribution_types` and the paper-loop driver tests.

The paper loop now includes a coordinated, crash-safe session transition and a multi-session driver with per-model leave-one-out after-cost attribution (`run_paper_session`, `run_paper_sessions`). It remains a deterministic session driver, not a live scheduled daemon (register O-07).

## 7. Next Steps & Handoff

This repository represents a complete, negative research result backed by production-grade plumbing. To advance from paper to live deployment, the following sequence is required:

1. **Acquire PIT Data:** Replace yfinance with a survivorship-free, point-in-time OHLCV database.
2. **Wire DSR/PBO:** Integrate the deflated Sharpe gate into the return-series acceptance protocol to prevent multiple-testing artifacts.
3. **Expand Universe:** Move from 10 ETFs to 300+ equities to achieve the cross-sectional breadth required for the fundamental law of active management.
4. **Broker Integration:** Replace `MockVenue` with a state-machine-driven adapter (e.g., Interactive Brokers) that handles partial fills and rejects.

---

*Risk Disclosure: This repository contains no investment advice, no profitability claims, and no live capital routing. All historical backtests are provisional, gross of realistic market impact, and subject to unquantified sampling uncertainty. The system is a research and measurement instrument, not a deployed trading bot.*
