# PIT Data Acceptance Specification

Acceptance rule: `audit_from_frozen()` returns `SUITABLE` on the delivered panel.

## Audit failure → vendor requirement → verification

| Audit failure reason | Vendor must deliver | Verification (evidence, not assertions) |
|---|---|---|
| `NOT_POINT_IN_TIME` | Vintage-dated data, `pit: true` in manifest | Vendor vintage documentation; no retroactively restated series |
| `NO_AVAILABILITY_PROVENANCE` | Known-at / as-of availability timestamps per observation | Sample rows showing availability dates ≤ observation dates |
| `NO_CORPORATE_ACTION_EVENT_LOG` | Dividends/splits as dated events, not back-adjusted closes | Event log with ex/pay dates; replay test reconstructing total returns PIT |
| `NO_DELISTED_COVERAGE` | Dead symbols retained | Canary test: vendor must return history for known delisted ETPs (XIV †2018, TVIX †2020, UGAZ/DGAZ †2020); if breadth expands to equities: LEH, BSC, ENE |
| `SURVIVORSHIP_BIAS_NOT_RULED_OUT` | Point-in-time universe membership | Index/eligible-universe constituents as-of each rebalance date |
| `INSUFFICIENT_VALID_EPISODES` | Code floor is 252 rows; proposed spec: ≥5,000 daily observations and ≥20 years | Justified by §2 power analysis (~16 regimes in 10y is starvation); threshold must be pre-registered, not tuned |
| `VINTAGE_HASH_MISMATCH` | Immutable delivery vintage | Freeze on receipt, SHA-256 content hash, reload-and-rehash must match |

## Governance notes

1. **Audit flags are honor-system inputs.** `audit_panel()` accepts `has_known_at_columns`, `has_adjustment_log`, `has_delisted_universe` as caller-supplied booleans. Someone could pass `True` and greenwash the verdict. The acceptance spec therefore requires a documentary evidence bundle per flag (vendor doc reference, sample extract, event-log file hash) stored alongside the manifest. The verdict is only as honest as its inputs.

2. **Breadth is part of acceptance, not an afterthought.** Per §7 and the power analysis: ≥300 names for the cross-sectional track, or a pre-registered pivot to the TSMOM track with its own (different) breadth justification. A PIT-clean 10-ETF panel still fails the *purpose* even if it passes the *audit*. The confirmatory block rule for power claims must also be pre-registered — the audit's block count is explicitly tagged `PROVISIONAL_EXPLORATORY`.

## Disclosures and assumptions

The audit mechanics above are quoted from the repo's actual `data_audit.py`. The ≥5,000-observation / ≥20-year thresholds are proposals requiring pre-registration, not existing rules. Vendor capabilities (Norgate, CRSP, QuantConnect, Sharadar/Tiingo) are unverified — the checklist exists precisely so vendor evaluation becomes pass/fail rather than sales-pitch judgment. Nothing here is investment advice or a profitability claim; the expected outcome of the full sequence remains "possibly no tradeable edge, measured honestly."
