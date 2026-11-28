# Pre-Registration: 12-1 Momentum & 5-Day Reversal (EXPLORATORY)

Status: FROZEN before execution. EXPLORATORY only (non-PIT panel). This is a
NEW budget family, separate from the 13 market trials. It does not enable
Phase 2 and does not alter any frozen Phase 2 parameter.

## Hypotheses
H1: cross-sectional 12-1 momentum (252d formation, 21d skip, 21d hold, long top
    ranks) has positive mean hold return.
H2: 5-day reversal (5d formation, 5d hold, long bottom ranks) has positive mean
    hold return.

## Test statistic & acceptance
Newey-West HAC t (lag 3) on non-overlapping hold returns. Two cells; Bonferroni
alpha 0.05/2 = 0.025 two-sided -> |t| >= 2.24 (normal approx). A cell "survives"
only if |t| >= 2.24 AND mean hold return > 0. Survival on this non-PIT panel is
SUGGESTIVE, not confirmatory.

## Constraints
Long-only is the deployable book; no short leg is traded. No parameter tuning
after seeing results. One frozen run. Seeds fixed in tests. Panel remains
non-PIT; the 13-trial result remains INCONCLUSIVE; no profitability claim.
