# QUANT-M1 — Paper-Trading Research System

**Research and paper trading only. No live capital and no live order routing.**

QUANT-M1 is a safety-first ETF research system. It separates proposal,
measurement, portfolio construction, risk authority, and paper execution.

## Core design

- Adaptive models propose; they do not own allocation or execution authority.
- Contribution is measured against paired counterfactuals, after costs.
- The independent Risk Kernel has final veto over every order.
- Cash is a valid portfolio state.
- Evidence must be reproducible, versioned, and tied to the exact code/config.
- Synthetic fixtures test mechanics; they are not market validation.
- No profitability or tradeable-edge claim is established.

## Scope

- US-listed liquid ETFs
- Long-only, unleveraged
- USD base currency
- T+1 settlement logic
- Paper/research execution only
- Point-in-time data required for confirmatory research

## Architecture

| Area | Module | Purpose |
|---|---|---|
| Safety and authority | `core.py`, `authority.py`, `risk.py` | Ledger, authorization, risk veto |
| Allocation | `allocate/` | Long-only portfolio optimization |
| Portfolio loop | `orchestrator/` | Decision and paper execution path |
| Contribution | `meta/contribution.py` | Horizon-isolated after-cost attribution |
| Probe research | `meta/probe.py` | Disposable paired shadow probes |
| Promotion | `research/promotion_ledger.py` | One-shot certificates and lifetime budgets |
| Paired measurement | `research/paired_decision.py` | Counterfactual decision accounting |
| Research governance | `research/governor.py`, `acceptance.py`, `pfcs.py` | Trial tracking and validation |
| Data integrity | `data/`, `research/data_audit.py` | PIT checks and frozen data |
| AI boundary | `research/ai_advisory.py` | Advisory-only analysis |
| Paper venues | `venue.py`, `venue_adapters/` | Simulated/paper execution |

## Safety invariants

1. No live order routing or live capital.
2. AI output cannot set portfolio weights, bypass risk, or execute orders.
3. Risk authority is independent of adaptive model output.
4. Phase 2 promotion is one-shot and certificate-gated.
5. A failed or terminal family cannot be reopened by correlation drift.
6. Lifetime registrations and alpha reservations are not recycled.
7. Missing, invalid, or incomplete evidence fails closed to cash.
8. No threshold or protocol change is made solely to obtain a passing result.

## Quick start

Install project dependencies:

```bash
pip install -r quant_m1/requirements.txt
