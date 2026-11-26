# Round 9A — PIT Data Readiness Specification

Status: DRAFT for owner approval. No confirmatory decisions are made here.
Scope: Defines what data can support a confirmatory strategy evaluation, and
provides a mechanical audit against that definition.

## 1. Purpose

Before selecting statistical thresholds or running any confirmatory experiment,
establish what data can actually support the experiment. The current panel
(`evidence/panel_real_v1.csv`) is a back-adjusted, non-PIT close series and is
INELIGIBLE for confirmatory strategy testing. This document specifies what a
usable panel requires and provides an automated audit that reports, mechanically,
whether a candidate panel meets the bar.

## 2. Definition of Point-In-Time (PIT)

An observation is PIT if, at the historical decision time `decision_ts`, a
market participant could have known exactly that value and no later revision.

Formally, every bar carries two timestamps:
- `ts` — the market instant the bar describes.
- `known_at` — the instant the value became available (publication/availability).

A bar is usable at `decision_ts` iff `ts <= decision_ts` AND `known_at <= decision_ts`.

Back-adjusted prices FAIL this test: a split or distribution that occurred after
`decision_ts` retroactively changes all prior prices, so the value known at
`decision_ts` differs from the stored value. This is lookahead. Statistical
corrections (deflators, survivorship weights) do NOT restore PIT validity; they
only change the bias. The panel must be rebuilt from genuine vintages.

## 3. Required data properties

A usable panel MUST provide:

| Property | Requirement | Checked by |
|---|---|---|
| PIT universe & membership | Historical constituents incl. delisted funds; membership as-of each date | manifest flag + membership table |
| Availability provenance | `known_at` timestamp for every observation | provenance columns / vintage log |
| Corporate-action history | Splits, distributions, adjustments as dated events, not pre-applied | adjustment event log |
| Frozen vintages + checksums | Immutable snapshots with content hash | manifest `content_hash` + recompute |
| Missing-data handling | Explicit policy for missing bars, stale prices, holidays | gap report |
| Settlement consistency | T+1 aware; no same-day availability of sale proceeds | settlement model (existing) |
| Raw vs derived separation | Raw prices stored separately from computed features | schema separation |

## 4. The audit answers six questions mechanically

1. Which symbols/dates are genuinely PIT? (`pit` flag + provenance presence)
2. How many valid evaluation episodes survive data-quality filters?
3. How many independent contiguous blocks exist? (provisional rule; exploratory)
4. Are corporate actions and delistings represented correctly?
5. Can every episode be reconstructed from its frozen vintage? (hash recompute)
6. What fraction of the intended sample is lost to missing/invalid data?

The audit returns a verdict and reasons. It makes NO statistical claim and NO
acceptance decision. `NOT_SUITABLE_FOR_CONFIRMATORY` is a valid and expected
output for a back-adjusted panel.

## 5. What the existing panel lacks (anticipated audit findings)

- `pit` flag is `False`.
- No `known_at` availability provenance.
- Back-adjusted closes → survivorship + lookahead bias (unrepairable by stats).
- No delisted-fund coverage.
- No corporate-action event log.

Therefore the current panel is suitable ONLY for deterministic software tests
(kernel, ledger, reconciliation, governance) — NOT for strategy confirmation.

## 6. Acquisition requirements (next step after this spec is approved)

- Survivorship-free US ETF universe with listing/delisting history.
- PIT OHLCV + total-return with per-observation availability timestamps.
- Corporate-action event log (dated, not pre-applied).
- Frozen vintages with checksums and explicit revision handling.
- Minimum target: ≥30 independent volatility-regime episodes after filtering
  (current panel has ~16; this is a data problem, not a method problem).
