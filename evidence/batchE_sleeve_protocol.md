# Batch E Pre-Registration: Exploration Sleeve + Meta-Selector + Signal Families

Status: FROZEN before execution. RESEARCH ONLY. Does not route orders, does not
grant authority, does not enable Phase 2, and does not alter any frozen Phase 2
parameter or Risk Kernel limit.

## Goal
Break the authority deadlock (a zero-weight model can never earn observations)
with a BOUNDED paper exploration sleeve, and add regime-aware multi-signal
combination. This is an optimization of the research layer, not a removal of
verification.

## Frozen parameters (not tuned post-hoc)
SLEEVE_CAP_FRAC=0.10; MIN_PROBE_WEIGHT=0.005; MAX_MODEL_WEIGHT=0.05;
KELLY_FRACTION=0.25. Selector regime map fixed in selector.py. Signal
magnitudes +/-0.001 placeholder (matches runner convention).

## Sleeve sizing rule (evidence-proportional)
- retired -> 0; already-eligible -> excluded (full-weight path)
- n==0 -> exploration floor MIN_PROBE_WEIGHT
- else -> fractional-Kelly on positive EWMA evidence, capped at MAX_MODEL_WEIGHT
- total capped at SLEEVE_CAP_FRAC (proportional shrink)

## Signal families
- momentum (21d trend, long winners) — computable from prices.
- reversal (5d, long losers) — computable from prices.
- carry — NOT computable from the price-only panel; reported unavailable, not
  fabricated. Requires real yield data (separate data step).

## Activation gate (unchanged)
The sleeve is wired to REPORT ONLY. Feeding sleeve observations into the
ProbeJournal (so provisional models can earn authority) is a SEPARATE activation
step requiring owner approval AND the Phase 2 v4 null suite to remain passing.
No live routing. No profitability claim. Panel non-PIT; prior result inconclusive.
