# Phase 2 Protocol v3 — Amendment PROPOSAL (NOT APPROVED / NOT IMPLEMENTED)

Status: PROPOSED for owner decision. The frozen `phase2_probe_protocol_v3.md`
remains the active specification. No parameter below is changed until the owner
explicitly approves, the document is re-frozen, and the null suite is re-run at
5000 reps and passes. Phase 2 stays DISABLED throughout.

## Verified trigger (Colab, 5000 reps, HEAD b23f8fe, 2026-11-27)
- program24_iid_or_ar0.5 = 0.0738 (Wilson95 upper 0.0801) > 0.07  -> FAIL
- lifetime48_iid         = 0.0974 (Wilson95 upper 0.1045) > 0.07  -> FAIL
- positive_control power = 0.0444 (no declared floor; informational)
- All other checks PASS. Full test suite: 558 passed.

## Inconsistency 1 — lifetime acceptance bound is undefined
Per-cell alpha = 0.05/24 = 0.002083. Over G_lifetime=48 independent cells the
theoretical any-false rate is 1-(1-0.002083)^48 = 0.0953. This is incompatible
with the declared program acceptance bound 0.07, and its Wilson upper (0.1045)
also exceeds the declared budget alpha_lifetime=0.10. The protocol declares a
lifetime BUDGET (0.10) but no lifetime ACCEPTANCE bound.

Options (owner must choose one):
  L1. Declare lifetime acceptance = alpha_lifetime + tolerance = 0.12.
      Accepts ~9.5% lifetime family-wise error (~1-in-10 chance of >=1 false
      promotion over the system lifetime). Simplest; passes current suite.
      Requires explicit owner acceptance of that error budget.
  L2. Reduce G_lifetime to 34 so 1-(1-alpha_cell)^G <= 0.07. Keeps strict
      error control; cuts lifetime candidate capacity 48 -> 34. Re-freeze/re-run.
  L3. Replace per-cell Bonferroni with a sequential alpha-spending procedure
      (e.g. Lan-DeMets) that controls lifetime FWER at 0.10 across looks.
      Most rigorous; requires a new calibration suite and re-freeze.

## Inconsistency 2 — HAC under-correction at AR(1) rho=0.5
Variance inflation ~2.96x vs IID; Bartlett L=21 under-corrects, giving program
FWER 0.0738 > 0.07. L is frozen; it may not be tuned to pass.

Options:
  A1. Improve the variance estimator (prewhitening or data-dependent bandwidth)
      and re-calibrate against the SAME 0.07 bound. Re-freeze/re-run.
  A2. Declare rho=0.5 out of calibration scope. NOT recommended: the ETF panel
      has high-correlation regimes; this would weaken the null suite.
  A3. Relax the program bound. NOT recommended: violates error-control intent.

## Inconsistency 3 — no numeric power floor; power is 0.044
The procedure detects a +10bp/day edge only 4.4% of the time. Options:
  P1. Declare a numeric power floor (e.g. >=0.20) as an acceptance gate. The
      current design fails this; it forces a redesign (coupled to L3/A1).
  P2. Keep power informational but document the limitation (current state).

## Strategic option (recommended for consideration)
The Bonferroni-24/48 design trades almost all power for error control. A cleaner
v4 protocol could pre-register a SMALL candidate set (e.g. G_max=6, per-cell
alpha=0.05/6=0.0083, z~2.39), raising power materially while keeping program
FWER <=0.05, with lifetime error handled by explicit alpha-spending. This is a
new protocol version, not a tweak, and requires full re-pre-registration.

## Decision required
Owner must select one option per inconsistency (or commission v4), after which:
  1. the chosen parameters are written into a re-frozen protocol document;
  2. the null suite is re-run at 5000 reps with the frozen seeds;
  3. ALL checks pass;
  4. the owner explicitly approves activation.
Until then Phase 2 remains DISABLED. No threshold is changed to force a pass.
