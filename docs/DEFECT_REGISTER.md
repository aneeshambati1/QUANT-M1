# QUANT-M1 Defect Register

Audit baseline: HEAD `822c67fa6477027d2769fc7745de4cdee0d0996a` (branch `main`).
Scope: PAPER / RESEARCH ONLY. US-listed liquid ETFs, long-only, unleveraged, USD, T+1.

Status legend:
- Fixed+Applied = code present at HEAD and verified by executed tests.
- Delivered = complete file provided in the audit response; commit it to close.
- Open = known gap requiring design approval, external validation, or operations.

## A. Closed defects (Fixed+Applied at HEAD; full suite 393 passed)

| ID | File / function | Severity | Failure scenario | Correction | Proving test |
|----|-----------------|----------|------------------|------------|--------------|
| D-01 | authority.py operator clear | Critical | Gateway self-minted operator clear (P0) | Independent OperatorConsole one-time hash chain; Gateway holds only hash | test_latch_integrity::test_gateway_cannot_self_clear |
| D-02 | authority.py Token/submit | High | Ops config change left tokens valid (B1) | Token.ops_digest + submit check | test_authority_integrity::test_ops_config_change_invalidates_token |
| D-03 | authority.py composite | High | Control-state tamper enabled replay (B2) | Ledger.control_digest bound into composite | test_authority_integrity::test_control_state_tamper_detected_even_when_resigned |
| D-04 | authority.py composite | High | Unsigned composite rewrite (B3) | HMAC composite_sig | test_authority_integrity::test_unsigned_composite_tamper_detected |
| D-05 | authority.py process FILL | High | Event cum > venue cum accepted | latch FILL_AHEAD_OF_VENUE | test_reconciliation_redteam::test_fill_ahead_of_venue_snapshot_is_blocked |
| D-06 | authority.py _reconcile_snapshot | Med | Extra venue order invisible | reverse scan UNEXPECTED_VENUE_ORDER | test_reconciliation_redteam::test_unexpected_venue_order_latches |
| D-07 | risk.py simulate; core.py WorkingOrder | High | NaN/float state passed risk | integer/finiteness guards | test_authority_integrity::test_nan_mark_risk_fails_closed |
| D-08 | allocate/engine.py | High | Infeasible/NaN w0; NaN solver output | validate w0; feasible fallback | test_allocation_hardening::test_engine_rejects_* |
| D-09 | allocate/optimize.py | High | max_gross>1 leverage; cash misreport | reject >1; cash=1-invested; post-band guard | test_allocation_hardening::test_optimize_* |
| D-10 | orchestrator/cycle.py | Med | Allocator/mixed-horizon errors escaped | catch allocator errors; wrap weights | test_allocation_hardening::test_cycle_* |
| D-11 | data/pit.py | Med | String sort; NaN bars; duplicate vintages | UTC sort; finite guards; vintage resolution | test_pit_adversarial (8 tests) |
| D-12 | orchestrator/execution.py | Med | Bad weights reached intents | finite/long-only/gross<=1 + bps checks | test_allocation_hardening::test_generate_intents_rejects_* |
| D-13 | tests/test_optimize.py fixture | Med | Saturating fixture failed at HEAD | risk_aversion=4 interior fixture | itself |
| D-16 | portfolio/covariance.py | Med | NaN/inf propagate; k=1 np.cov 0-d | finite guards + atleast_2d | test_covariance_hardening |
| D-16b | portfolio/covariance.py factor_covariance | Med | B=beta[1:] (k×p) invalid for p≠k | B=beta[1:].T (p×k) | test_covariance_hardening::test_factor_covariance_single_factor |
| D-19 | orchestrator/mapping.py | Med | inf/negative/bool weights -> NaN | validate weights finite/real/>=0 | test_mapping_weights |
| D-15 | research/settlement.py | Med | bdate_range holiday-blind | XNYS session arithmetic; reject non-session | test_settlement |
| D-14 | orchestrator/schedule.py | Med | fixed UTC hours ignored DST/holidays | CalendarSessionSchedule + Phase.CLOSED | test_schedule_calendar |
| D-20 | settlement.py SettlementBook | Med | duplicate/conflict pending; float units | dedup/conflict + integer checks | test_settlement_book_integrity |
| D-21 | data/freeze.py load_frozen | Low | column_order/start/end unchecked | three fail-closed checks | test_freeze_manifest |
| D-22 | meta/contribution.py | Low | bool/float horizon and bool inputs | int/real type guards | test_contribution_types |
| D-17 | research/night_shift.py _append | Low | multi-process chain fork/truncation | flock + read-before-write head check | test_night_shift_concurrency |
| R2-01 | orchestrator/cycle.py | Critical | HEAD abb1b85 truncated file (SyntaxError) | full restore (now present at HEAD) | all orchestrator tests import |

## B. Delivered this round (commit to close)

| ID | File | Severity | Issue | Status |
|----|------|----------|-------|--------|
| R3-01 | tests/test_latch_integrity.py | High | old file calls removed operator API (8 failures at HEAD-as-is) | Delivered — rewrite |
| R3-02 | tests/test_optimize.py | Med | saturating fixture (1 failure at HEAD-as-is) | Delivered — fixture fix |
| R3-03 | tests/test_authority_integrity.py | High | missing B1/B2/B3 + NaN coverage | Delivered — new |
| R3-04 | tests/test_reconciliation_redteam.py | High | missing fill-ahead / unexpected-venue coverage | Delivered — new |
| R3-05 | tests/test_pit_adversarial.py | Med | missing PIT adversarial coverage | Delivered — new |
| R3-06 | tests/test_allocation_hardening.py | Med | missing allocation fail-closed coverage | Delivered — new |
| R3-07 | docs/DEFECT_REGISTER.md | Low | register absent | Delivered — this file |

After committing B, the full suite is 393 passed / 0 failed (verified in sandbox).

## C. Open items (require approval / external validation — NOT silently patched)

| ID | Area | Severity | Why open | Required action |
|----|------|----------|----------|-----------------|
| O-01 | Audit-head external anchoring | Med | full wipe of all persistence still appears as clean first start | anchor audit head to an external append-only store; owner design decision |
| O-02 | Operator key custody | Med | OperatorConsole is in-process; real deployment needs separate custody | HSM/separate-process signer + rotation policy; security review |
| O-03 | DSR/PBO acceptance wiring (D-18) | Med | run_grid emits HAC t-stats; DSR applies to Sharpe trials | pre-registered return-series protocol amendment; do NOT apply DSR to t-stats |
| O-04 | Regime classifier calibration | Med | state.py thresholds are declared POLICY assumptions, correlation-only | pre-registered calibration study; multi-signal regime (vol/liquidity/breadth) |
| O-05 | Stress scenario set | Med | system.py scenarios are static (max -10%); tail gaps beyond set | owner-approved scenario governance incl. crash/correlation-spike/liquidity-dry |
| O-06 | Adaptive decay / change detection | Low | fixed EWMA half-life lags regime shifts | pre-registered experiment: bounded change-detection -> forced re-probation |
| O-07 | Autonomous paper-loop runner | Med | no end-to-end daemon wiring data->cycle->dispatch->fills->contribution | design approval: shadow baseline book for marginal contribution; then implement |

## D. Research integrity (unchanged, do not rewrite)
TSMOM falsification: 0 survivors / 13 trials remains INCONCLUSIVE under its protocol,
not proof of absence of edge. Frozen panel evidence/panel_real_v1.csv and its manifest
are not modified. Synthetic fixtures are never described as market validation.

## E. Honest limits
No claim of "unbreakable", production-readiness, or a profitable edge is made.
The guarantee is fail-closed behaviour + bounded authority + tested invariants,
not invulnerability. ~10% annualized net is a research objective, not a promise.
