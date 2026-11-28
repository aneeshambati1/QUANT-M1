# Free-Data Track Protocol (EXPLORATORY — NOT confirmatory)

Status: ACTIVE for exploratory research only. This document does NOT amend or
weaken docs/PIT_DATA_ACCEPTANCE_SPEC.md, which remains the confirmatory bar.
The free-data panel is PIT-ADJACENT and must never be used to make a
confirmatory alpha claim.

## What the multi-source archive delivers
- Cross-provider DRIFT detection (frozen bars never overwritten).
- Self-stamped received_at (a weaker proxy for vendor known_at).
- Delisted retention by discipline (downloaded symbols never deleted).
- A reproducible, hash-chained vintage of everything received.

## Acceptance-spec scorecard (honest)
| Requirement | Free stack | Note |
|---|---|---|
| NOT_POINT_IN_TIME | PARTIAL | self-stamped receipt, not vendor PIT |
| NO_AVAILABILITY_PROVENANCE | PARTIAL | received_at proxy only |
| NO_CORPORATE_ACTION_EVENT_LOG | NOT MET | EDGAR parsing out of scope |
| NO_DELISTED_COVERAGE | MET (forward) | only for symbols already downloaded |
| SURVIVORSHIP_BIAS_NOT_RULED_OUT | NOT MET | requires paid universe membership |
| INSUFFICIENT_VALID_EPISODES | MET | years of daily bars available |
| VINTAGE_HASH_MISMATCH | MET (self) | detects provider restatement vs own archive |

## Consequence
Trials on this panel are EXPLORATORY. The 13-trial market result remains
INCONCLUSIVE. Phase 2 remains DISABLED. No profitability claim.

## Providers
Two sources (Alpaca + yfinance) give genuine triangulation. A third adds
operational drag without closing the survivorship gap. Real fetchers are
version-sensitive adapters requiring network + keys; the tested core is the
archive/drift logic with injected fetchers.
