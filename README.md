# QUANT-M1 — Institutional Paper-Trading Research System

A safety-first, evidence-driven ETF portfolio research system. **Paper / research
only — no live capital, no live order routing.**

## What it is

An institutional-grade framework that separates three concerns:

1. **Adaptive intelligence proposes** — models generate allocation ideas.
2. **Independent validation evaluates** — after-cost contribution is measured
   against counterfactual baselines.
3. **Risk authority decides** — an independent Risk Kernel holds FINAL VETO over
   every order. Execution happens only through the authorized paper path.

## Architecture at a glance

| Layer | Module | Role |
|---|---|---|
| Safety kernel | `core.py`, `authority.py`, `risk.py` | Integer-money ledger, HMAC authorization, reachable-fill risk veto |
| Allocation | `allocate/` | Cost-aware long-only optimizer (gross ≤ 1, cash valid) |
| Meta / learning | `meta/`, `orchestrator/` | Contribution trackers, policy gates, paper loop |
| Measurement | `research/paired_decision.py` | Paired counterfactual decision ledger (PCDL) |
| Research governance | `research/governor.py`, `acceptance.py`, `pfcs.py` | Trial registry, Bonferroni + DSR/PBO diagnostics |
| Data integrity | `data/`, `research/data_audit.py` | Point-in-time validation, frozen vintages |
| AI boundary | `research/ai_advisory.py`, `tests/test_ai_hfpc.py` | Advisory-only LLM role, zero authority |
| Execution venues | `venue.py`, `venue_adapters/alpaca_paper.py` | MockVenue + Alpaca Paper (hardcoded paper=True) |

## Hard safety rules (non-negotiable)

- Long-only, unleveraged, USD, cash is a valid allocation, T+1 settlement.
- **No live capital.** Alpaca adapter is hardcoded to `paper=True`.
- The LLM may propose and critique. It may NEVER set weights, override risk, or trade.
- No component gets permanent authority — everything must keep re-earning it.

## Quick start

```bash
pip install -r quant_m1/requirements.txt
python -m pytest tests -q          # expect ~509 passed
```

## Running the paper loop

See `docs/HOW_TO_RUN.md`. The loop is a bounded, restartable batch runner —
not a live daemon. It never invents market data and stops fail-closed on any
divergence.

## Honest research status

**No tradeable edge is demonstrated.** The existing 10-ETF panel is
back-adjusted and non-PIT; the data audit returns `NOT_SUITABLE_FOR_CONFIRMATORY`.
Results run on yfinance are **exploratory only**. Confirmatory testing requires
genuine point-in-time data. See `docs/ROUND_9A_PIT_DATA_READINESS_SPEC.md`.
