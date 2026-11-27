# QUANT-M1 Phase 2 Protocol v4 (3/6 roster — owner-selected)

Status: PROPOSED frozen candidate. Ratified by owner selection of the 3/6
design; activation still requires a green 5000-rep CI run on the ratified
commit plus explicit owner approval. Protocol v3 is retained untouched as
historical evidence. Phase 2 remains DISABLED until the activation gate passes.

## 1. Design (owner-selected 3/6)
A smaller confirmatory roster balances candidate capacity against power.
  G_program = 3            candidates per program
  G_lifetime = 6           maximum lifetime candidates
  alpha_global = 0.05      program family-wise budget
  alpha_lifetime = 0.10    lifetime family-wise budget
  alpha_cell = alpha_lifetime / G_lifetime = 0.10/6 = 0.016667
  z_cell = Phi^-1(1 - alpha_cell) = 2.1280
  N_eval = 399             evaluation observations (unchanged)
  half_life_obs = 60       (unchanged)
  estimator = prewhitened-AR(1) HAC (verified; adopted)

Lifetime alpha reservation is exactly alpha_cell * G_lifetime = 0.10. The 12%
lifetime ACCEPTANCE limit adds a 2-percentage-point Monte Carlo tolerance; it
does NOT increase the nominal 10% lifetime alpha budget. Program FWER <=
3*alpha_cell ~ 0.049; lifetime FWER <= 6*alpha_cell = 0.10 by construction.

## 2. Acceptance bounds (Wilson one-sided 95% upper)
  program3_rho{0.0,0.3,0.5}  <= 0.07
  per_cell_zero_edge         <= 0.026667 (alpha_cell + 0.01)
  lifetime6_iid              <= 0.12     (0.10 + 0.02 tolerance)
  program3_weak_negative     <= 0.07
  program3_cost_domination   <= 0.07
  strong_negative_retirement_by_141 >= 0.95
  positive_control_power     INFORMATIONAL (disclosed weak, ~0.14)

## 3. Calibration (owner local reproduction, 5000 reps, fixed seeds)
  Program IID        6.04%  upper 6.62%  PASS
  Program AR(1) 0.3  5.86%  upper 6.43%  PASS
  Program AR(1) 0.5  5.52%  upper 6.08%  PASS
  Lifetime 6 cells  10.16%  upper 10.88% PASS
  Single cell        1.78%  upper 2.12%  PASS
  Weak-negative      3.40%  upper 3.85%  PASS
  Cost domination    0.78%  upper 1.01%  PASS
  Power             14.06%  informational (weak)
Sandbox cross-check at 4000 reps reproduces PASS on the boundary checks
(program IID upper 0.0646; per-cell upper 0.0234; lifetime upper 0.1104).
At 1500 reps two boundary checks exceed their Wilson upper (small-sample
noise below the frozen 5000); this is expected and is not a design failure.

## 4. Honest power disclosure
Power ~0.14 is weak: a genuine +10 bp/day edge (IR ~1.6) is detected ~14% of
the time. The owner accepts this as informational, not adequate. Higher power
requires a longer evaluation window (more observations) or a larger effect;
that trade-off is documented in the v4 amendment history. No "perfect" claim.

## 5. Activation gate (unchanged)
  1. Owner ratifies this 3/6 design (this document becomes the frozen v4).
  2. scripts/run_null_suite_v4.py --reps 5000 passes in CI (hashed report).
  3. PromotionLedger default roster set to g_max=3, g_lifetime=6 (or constructed
     with those args); Phase 2 tests updated to z_cell=2.1280.
  4. Owner explicitly approves activation.
Until then the SE path stays OFF and the runner reports NO AUTHORITY / NO TRADES.

## 6. Disclosures
Panel back-adjusted, non-PIT; no confirmatory alpha claim. Synthetic calibration
fixtures are not market validation. No profitability or trading edge asserted.
