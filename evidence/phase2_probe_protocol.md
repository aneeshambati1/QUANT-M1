# QUANT-M1 Phase 2 Pre-Registration v2: Probation Probe + Standard-Error LCB
Status: PROPOSED — NOT APPROVED. Design-only. kappa and current lcb unchanged.
Scope: research/paper only; daily horizon (h=1) only in this revision.
Panel back-adjusted, non-PIT. No alpha claim. Implementation requires owner
approval AND a passing null suite (Sec 7). SE path is opt-in (flag default OFF).

## 1. Observation journal (prerequisite)
Durable, append-only, horizon-keyed journal of RAW paired after-cost
contributions c_i per (model_id, horizon). Hash-chained (reuse Store/EventLog
integrity); verified on restore. The aggregate tracker mean/var is insufficient
for autocovariances; the journal is the source for gamma_k and finite weights.
Retention >= N_use. No journal => no SE promotion (fail closed).

## 2. Probe / shadow evaluation (research-only)
probe_blend_weight = 0.05: a FORECAST-BLEND coefficient in blend_signals(), NOT
a capital allocation. It guarantees no particular exposure/notional. The
allocator, covariance, risk_aversion, trade band, costs, min-trade threshold and
Risk Kernel determine execution. No scale-invariance is claimed; no notional is
forced and the allocator is never bypassed. If the blend yields no incremental
trade, the resulting zero contribution is a valid tested deployment outcome.

Per evaluation, from the SAME starting state, build isolated shadows (own
MockVenue + temp workdir; real sx NEVER mutated; no live endpoint):
  - Baseline B0: active ensemble at earned weights (cash if none).
  - Candidate Bm: B0 + candidate m's signal at probe_blend_weight.
  - c_m = (NAV_after(Bm) - NAV_after(B0)) / starting_NAV, after cost.
Add-one-in is marginal value CONDITIONAL on the baseline; it is not an additive
decomposition. Simultaneous candidates use GREEDY sequential add-one-in in a
pre-registered deterministic order (sorted model_id): each candidate is measured
against the baseline PLUS previously-added candidates. Duplicate guard: signals
with sample correlation > 0.99 are one family; at most the first is promotable.
Order-dependence is a documented limitation; Shapley is deferred (separate
proposal). Risk Kernel veto applies inside shadows. Horizon isolation, probation,
retirement unchanged.

## 3. _shadow_system repair (prerequisite)
Build shadow with REAL instruments/marks/allowlist/product_classes at
construction; ledger.propagate_config(real_version, real_digest); install
cfg/instruments/scenarios on sx top-level AND kernel AND gw. Own venue/workdir.

## 4. Finite-sample SE estimator (daily; replaces stationary v1)
lambda = 1 - 2^(-1/half_life_obs). For journal c_1..c_n:
  w_i = lambda*(1-lambda)^(n-i), i=1..n;  S_n = sum w_i = 1-(1-lambda)^n.
  Renormalized mean m_hat = (sum w_i c_i)/S_n   [removes zero-prior bias].
  gamma_k from the most recent N_use=min(n,250) obs (demeaned by their sample
  mean); bandwidth L=21; Bartlett taper (1-k/(L+1)).
  Var(m_hat) = (1/S_n^2)*[ gamma_0*sum(w_i^2)
               + 2*sum_{k=1..L}(1-k/(L+1))*gamma_k*sum_i(w_i*w_{i+k}) ].
  SE_HAC = sqrt(Var(m_hat)) ONLY IF Var finite and >0 and gamma_0>0 and
  n>=se_min_obs; otherwise FAIL CLOSED (no authority, diagnostic logged).
  Never clamp a non-positive/non-finite variance to zero.
  n_eff_corr = gamma_0 / Var(m_hat)   [IID limit -> (2-lambda)/lambda].
Cross-check: circular moving-block bootstrap, block=21 daily obs (NOT h=1),
>=ceil(n/21) blocks, B>=2000; require |SE_HAC-SE_boot|/SE_HAC <= 0.15, else
fail closed / investigate. Block length preserves serial dependence.

## 5. Evaluation schedule (controls repeated looks)
PRIMARY: single fixed evaluation EPOCH. Each candidate is tested for promotion
exactly ONCE, at its N_eval-th daily observation (N_eval=399, S_n>=0.99).
se_min_obs=120 (minimum for gamma stability) is distinct from probation_obs=30;
promotion requires n>=N_eval (=>=se_min_obs), so the autocov window always exists.
No per-session re-testing. Candidates not promoted at N_eval are not re-tested
for promotion without a NEW pre-registered epoch (alpha budget tracked in the
trial ledger). Retirement (unchanged rule) is a separate stopping rule, not a
promotion look. New candidates => new epoch => new pre-registration.
ALTERNATIVE (deferred, needs own calibration): sequential alpha-spending
alpha_{h,t}=0.05*pi_h/(t(t+1)), sum_h pi_h=1, divided across candidates; valid
only if per-look tests are calibrated (asymptotic HAC z is not exact sequentially).
Initially pi_1=1, pi_h=0 (h>1): cross-horizon error controlled by not testing them.

## 6. Promotion / weight / demotion / retirement (exact)
M = candidates evaluated in the epoch (<= pre-declared M_max=5).
z_promote = Phi^{-1}(1 - 0.05/M)  [M=1:1.645; M=5:2.326; M=10:2.576].
lcb_SE = m_hat - z_promote*SE_HAC.
PROMOTE iff n>=N_eval AND Var valid AND lcb_SE>0 AND not retired AND after-cost.
SE-eligible weight: raw_weight_SE = max(0, lcb_SE); weights() normalizes over
SE-eligible models. THIS CHANGES BOTH GATE AND WEIGHTING (approval required);
kappa and the old lcb/raw_weight path remain intact behind the flag (OFF).
DEMOTE iff SE-eligible and lcb_SE<=0. RETIRED iff n>=retire_min_obs(120) and
mean<-retire_band(0.002) [UNCHANGED]. Probe promotions are a new ledger family,
counted separately from the 13 market trials; DSR/PBO (O-03) on RETURN SERIES,
never on the HAC contribution statistic.

## 7. Null suite (hard gate before enabling; pre-registered seeds, B>=5000, daily)
Seeds fixed (e.g., default_rng(20261127)+index). Acceptance uses WILSON 95%
UPPER BOUND, not point estimates. Evaluate the WHOLE epoch procedure.
 (a) IID zero-edge: false-promotion Wilson-upper <= 0.065.
 (b) AR(1) rho in {0.3,0.5} zero-edge: Wilson-upper <= 0.065 AND
     SE-HAC upper < SE-IID upper (must reduce inflation).
 (c) Weak-negative mean=-0.001 (> -band): promotion=0; demotion verified; NOT retirement.
 (d) Strong-negative mean=-0.003 (< -band): retirement detection >= 0.95 by
     n=retire_min_obs+Delta; retirement rule unchanged.
 (e) Positive control: mean=+0.001, sigma=0.01, AR(1) rho=0.3, N_eval=399:
     power floor M=1 >= 0.25; M=5 (Bonferroni) >= 0.10. Measured, not assumed.
 (f) Cost-domination (gross edge < cost): promotion=0.
 (g) Duplicate/interaction: two identical signals, greedy order -> second
     conditional |incremental| <= eps; not both independently promoted.
 (h) Epoch family-wise false-promotion <= 0.05 (Wilson CI) across the procedure.
Any failure blocks enabling. Results hashed into the trial ledger.

## 8. Power / limitation disclosure
n_eff is bounded by half_life_obs, not sample length. SE makes the gate VALID
and reachable, not powerful enough to overcome the panel's fundamental limit.
A promotion is not evidence of profitability.

## 9. Implementation plan (gated; NOT executed)
 1. paper_loop._shadow_system repair + tests.
 2. Durable horizon-keyed contribution journal + tests.
 3. quant_m1/meta/probe.py: greedy add-one-in probe driver (isolated shadows).
 4. contribution.py: finite-sample SE helpers + opt-in SE status/weight (flag OFF);
    old lcb/kappa preserved.
 5. Wire probe -> journal -> SE evaluation at epoch (research-only, behind flag).
 6. tests: cold-start probe earns observations; null suite (a)-(h); finite-sample
    variance vs brute force; fail-closed invalid variance; duplicate guard;
    horizon isolation; kernel veto in shadow; real sx unchanged; shadow digest sync.

## 10. Phase 1 reporting correction (separate small patch; not Phase 2)
Runner: no_authority must reflect tracker authority (any eligible), reported
SEPARATELY from no_trades (intents_total==0). An authorized model can legitimately
produce no orders (band/optimizer/risk). Not permission to force trades.

## 11. Approval
Owner must approve Secs 2,4,5,6,7 explicitly. Until then: no rule change, no
probe wiring, kappa unchanged, runner remains NO AUTHORITY / NO TRADES.
