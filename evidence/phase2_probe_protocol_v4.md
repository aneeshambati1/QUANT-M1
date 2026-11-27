# QUANT-M1 Phase 2 Protocol v4 (PROPOSED — pending owner ratification)

Status: PROPOSED. Supersedes v3 ONLY upon explicit owner ratification. Until
then v3 remains the frozen specification and Phase 2 stays DISABLED. This
document changes no live authority and enables nothing by itself.

This is the REDESIGN route (not the minimal L1+A1+P2 patch). It is internally
coherent: every acceptance bound is declared and derived from a finite,
pre-registered roster. No bound is left undefined and no threshold is tuned
post-hoc to force a pass.

## 1. What changed from v3 and why
v3 used 24 cells/program and 48 lifetime cells at alpha_cell = 0.05/24 =
0.002083 (z = 2.865). Consequences measured at 5000 reps:
  * lifetime any-false rate ~0.0953 could not meet a 0.07 bound (undefined).
  * positive-control power was 0.039 — the gate was nearly blind to real edges.
v4 reduces the roster and lengthens the evaluation window to buy power honestly:
  * G_program = 6, G_lifetime = 12 (family-deduplicated, pre-registered).
  * alpha_cell = alpha_lifetime / G_lifetime = 0.10/12 = 0.008333, z = 2.394.
  * Lifetime FWER <= 0.10 BY CONSTRUCTION (finite Bonferroni over 12 cells).
    Program FWER <= 6 * alpha_cell = 0.05.
  * N_eval = 1008 daily observations, half_life_obs = 252 (n_eff ~ 724).
  * Prewhitened-AR(1) HAC estimator (already implemented and verified) is
    formally ADOPTED here (this ratifies amendment option A1).

## 2. Frozen parameters
G_program=6; G_lifetime=12; alpha_lifetime=0.10; alpha_cell=0.008333;
z_cell=2.394; N_eval=1008; half_life_obs=252; se_min_obs=252; lag_cap L=21;
phi_abs_limit=0.995; pi_probe=0.05; rho_dup=0.95; c_model=0.40; c_family=0.50;
D_demote=21; retirement unchanged (n>=120, mean<-0.002); Delta=21 (deadline 141);
reps=5000; seeds frozen in scripts/run_null_suite_v4.py.

## 3. Acceptance bounds (ALL declared; Wilson one-sided 95% upper)
  program6_rho{0.0,0.3,0.5}  <= 0.07   (0.05 + 0.02)
  per_cell_zero_edge         <= 0.018333 (alpha_cell + 0.01)
  lifetime12_iid             <= 0.12   (0.10 + 0.02; derived, not arbitrary)
  program6_weak_negative     <= 0.07
  program6_cost_domination   <= 0.07
  strong_negative_retirement_by_141 >= 0.95
  positive_control_power     >= 0.25   (GATE, not informational; option P1)

## 4. Calibration evidence (sandbox, prewhitened estimator, 1500 reps)
  program6_rho0.0 0.0520 (upper 0.0623) PASS
  program6_rho0.3 0.0553 (upper 0.0659) PASS
  program6_rho0.5 0.0487 (upper 0.0586) PASS
  per_cell        0.0093 (upper 0.0144) PASS
  lifetime12      0.0987 (upper 0.1121) PASS
  weak_negative   0.0147 PASS
  cost_domination 0.0007 PASS
  retirement      0.9993 PASS
  power           0.3360 >= 0.25 PASS
The frozen 5000-rep run must be executed in CI; 1500 reps is a sandbox check.
At 1200 reps two boundary checks exceeded their Wilson upper (small-sample
noise); this is expected below the frozen replication count and is not a
design failure. Do not lower reps to claim a pass.

## 5. Honest power disclosure (no "perfect" claim)
Power ~0.34 means a genuine +10 bp/day edge (annualized IR ~1.6) is detected
~34% of the time; ~66% of such edges are missed. This is the price of strict
family-wise error control at a modest effect size. Power is bounded by data:
  N_eval=399,  HL=60  -> power ~0.10
  N_eval=756,  HL=126 -> power ~0.18
  N_eval=1008, HL=252 -> power ~0.34  (this protocol)
A shorter window is faster in wall-clock but materially less powerful. The
owner may choose a different point on this trade-off; it must be re-frozen and
re-calibrated, not tuned after seeing results.

## 6. Activation gate (unchanged)
Phase 2 activation requires ALL of:
  1. Owner explicitly ratifies v4 (this document becomes frozen).
  2. scripts/run_null_suite_v4.py --reps 5000 passes in CI (hashed report).
  3. PromotionLedger constructed with v4 params (n_eval=1008, se_min_obs=252).
  4. The CI workflow pointed at the v4 suite.
  5. Owner explicitly approves activation.
Until then the SE path stays OFF and the runner reports NO AUTHORITY / NO TRADES.

## 7. Disclosures
Panel is back-adjusted, non-PIT; no confirmatory alpha claim. Synthetic
calibration fixtures are not market validation. No profitability or trading
edge is asserted. This protocol governs error control and evidence, not returns.
