# Batch A Pre-Registration: Cost-Aware Allocation Ablation (MECHANICAL)

Status: FROZEN before execution. Owner approved Batch A. This is a research-only
allocation-level study; it grants NO model authority, routes NO orders, and
changes NO live default. The runner switch is a separate owner decision gated
on the acceptance rule below.

## Hypothesis (mechanical, not alpha)
Putting transaction cost inside the allocation objective (solve_cost_aware)
reduces realized friction versus the mean-variance engine plus a post-hoc
no-trade band, UNDER IDENTICAL FORECASTS. Both arms consume the same
uncalibrated momentum-rank placeholder expected returns and the same realized
Ledoit-Wolf covariance. Any after-cost return difference is therefore a
cost/timing MECHANICAL effect, not evidence of trading edge.

## Frozen parameters (taken from the runner / memo BEFORE running; not tuned)
panel = evidence/panel_real_v1.csv (frozen, non-PIT)
decision window = last 252 days (sandbox may run fewer; disclosed)
cov_window = 63 (Ledoit-Wolf constant-correlation, same as runner)
momentum lookback = 21; rank cuts >0.8 / <0.2; ER magnitude +/-0.001 (placeholder)
risk_aversion = 2.0; band_frac = 0.25; cost_bps = 5; min_trade_bps = 5
max_gross = 1.0 (both arms); long-only; cash yield = 0 (disclosed)

## Arms (paired, path-dependent, identical inputs each day)
CONTROL   : engine.solve_allocation(er, cov, ra=2.0, w0, band=0.25)
TREATMENT : cost_aware_allocator(er, cov, ra=2.0, w0, band=0.25, cost=5bp)
Each arm starts flat and evolves its OWN executed weights with buy-and-hold
drift between rebalances. Solver failure -> hold current weights (fail-closed).

## Execution friction model (allocation-level; disclosed approximation)
Per-asset dust filter: |delta_w| < 5bp of NAV -> skipped.
Cost = 5bp x sum(|executed delta_w|), charged at the rebalance.
Gross guard: if executed gross > 1, scale proportionally to 1 (counted).
Next-day realized return applied to executed weights; cash earns 0.
This mirrors the kernel's cost/dust/gross rules at fractional-weight level;
it is NOT integer-cent kernel execution and NOT a live ledger.

## Metrics
Per arm: total cost (bp), annualized one-way turnover (x), mean gross,
rebalance days, solver-failure days, guardrail events, net & gross cumulative
return, max drawdown of the net path. Paired: net_T - net_C; cost_C - cost_T.

## Acceptance rule (pre-registered; NO return-based selection)
PRIMARY   : total_cost_T <= total_cost_C  (non-inferiority on friction)
SECONDARY : turnover_T <= turnover_C
GUARDRAILS: both arms gross <= 1 every day; long-only; failures fail-closed.
DECISION  : switch the runner default to cost-aware ONLY IF primary AND
            secondary pass AND guardrails hold AND the owner explicitly
            approves. Otherwise KEEP the control. Returns are NOT a criterion.

## Disclosures
Non-PIT panel; placeholder forecasts; no alpha or profitability claim.
Deterministic solvers (SLSQP); no random seed needed. One frozen run; no
parameter search. Synthetic/placeholder inputs are not market validation.
