# QUANT-M1 Phase 2 Protocol v3 (frozen)

Status: APPROVED FOR IMPLEMENTATION by owner directive; ENABLING gated on the
null suite passing. SE path default OFF. Kappa and the original authority rule
unchanged. Panel non-PIT; no alpha claim.

## Frozen parameters
alpha_global=0.05 (one-sided); G_max=24 cells; alpha_cell=0.05/24=0.002083;
z_cell=2.865; alpha_lifetime=0.10; G_lifetime=48; N_eval=399; se_min_obs=120;
lag_cap L=21 (Bartlett taper); pi_probe=0.05 (forecast-blend coefficient);
rho_dup=0.95; c_model=0.40; c_family=0.50; D_demote=21 consecutive sessions;
retirement unchanged (n>=120, mean<-0.002); Delta=21 (deadline 141);
null acceptance: Wilson one-sided 95% upper <= nominal + tolerance
(program-wide 0.05+0.02=0.07; per-cell alpha_cell+0.01=0.0121).

## A. Global budget
One test per candidate-horizon cell at n=399; failed cells close their FAMILY
for the system lifetime (signal-hash / rho>=0.95 family). Epochs consume the
finite roster; no fresh budget. Lifetime caps bound all programs. Ledger:
quant_m1/research/promotion_ledger.py (hash-chained).

## B. Blend algebra
mu_active frozen at epoch start. b_c=(1-pi_c)*mu_active + pi_c*s_c.
pi_c=0.05 iff c is its family representative and that rep is not already
active; else 0. Family cap sum(pi_c)<=0.05. Duplicates: pi_c=0 -> obs==0 ->
fail-closed, not promotable. No combined multi-candidate probe blend exists.
Probe coefficients never become live weights. Greedy = admission ordering
only (descending incremental lcb_SE), updating the active set for the NEXT
epoch.

## C. Three schedules
Certificate: one evaluation at n=399 (ledger). Live weight: continuous
max(0, se_lcb(t)), normalized, water-filling caps, remainder/all-zero -> cash
(consumes no alpha). Demotion: lcb_SE<=0 for 21 consecutive sessions ->
terminal; cell closed. Retirement unchanged. Risk Kernel final veto.

## D/E. Calibration
Wilson-upper acceptance as above; weak-negative rate-based; strong-negative
(-50bp/day, sigma=1%) retirement detection >=95% by deadline 141 (Delta=21
frozen before simulation). Suite: scripts/run_null_suite.py; hashed report;
must pass before enabling.

## Isolation
Probes run in disposable shadows (own ledger/gateway/MockVenue/workdir);
real system digest-guarded; never routed to live endpoints. _ProbeGate is a
counterfactual construct used only inside shadow tracker maps.
