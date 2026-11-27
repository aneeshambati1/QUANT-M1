# QUANT-M1 Defect Register

Audit baseline: HEAD `c4f69ab982999f993be1c50f27d825372d7b669b` (branch `main`).
Scope: PAPER / RESEARCH ONLY. US-listed liquid ETFs, long-only, unleveraged, USD, T+1.

Status legend:
- Fixed+Applied = code present at HEAD and verified by executed tests.
- Delivered = complete file provided in the audit response; commit it to close.
- Open = known gap requiring design approval, external validation, or operations.

TEST-COUNT POLICY: markdown test counts are NOT authoritative. The authoritative
result is the GitHub Actions `tests` workflow run at the current HEAD. Earlier
figures (293 in RESULTS.md; 393 in this register at baseline 822c67f) were verified
at their own commits and do not establish the result at later HEADs. At baseline
c4f69ab9 with the corrected tests/test_paper_loop.py delivered below, the full
suite was executed in-sandbox: 409 passed / 0 failed. The first CI run at c4f69ab9
was RED solely because of the flawed restart test (see R6-04); committing the
corrected test turns it green.

## A. Closed defects (Fixed+Applied at HEAD)

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

## B. Round-3 test files (now Fixed+Applied at HEAD)

| ID | File | Severity | Issue | Status |
|----|------|----------|-------|--------|
| R3-01 | tests/test_latch_integrity.py | High | old file called removed operator API | Fixed+Applied — rewrite |
| R3-02 | tests/test_optimize.py | Med | saturating fixture | Fixed+Applied — fixture fix |
| R3-03 | tests/test_authority_integrity.py | High | missing B1/B2/B3 + NaN coverage | Fixed+Applied — new |
| R3-04 | tests/test_reconciliation_redteam.py | High | missing fill-ahead / unexpected-venue coverage | Fixed+Applied — new |
| R3-05 | tests/test_pit_adversarial.py | Med | missing PIT adversarial coverage | Fixed+Applied — new |
| R3-06 | tests/test_allocation_hardening.py | Med | missing allocation fail-closed coverage | Fixed+Applied — new |
| R3-07 | docs/DEFECT_REGISTER.md | Low | register absent | Fixed+Applied — this file |

## R6. Paper-loop integration defects (cross-review found; confirmed + corrected)

| ID | File / function | Severity | Failure scenario | Correction | Proving test | Status |
|----|-----------------|----------|------------------|------------|--------------|--------|
| R6-01 | paper_loop.run_paper_step | High | Shared Clock not advanced; kernel rejected every order STALE_MARKS (reproduced: 0 submitted / 3 rejected) | _advance_session catches clock up to mark wall under lock | test_successful_order_reaches_fill_and_reconciles | Fixed+Applied |
| R6-02 | paper_loop / core.advance_session | High | Session persisted to ledger.json but not the signed composite; restart could roll session back | coordinated transition appends SESSION_ADVANCE + gw._persist(); idempotent re-run | test_restart_continuity_keeps_session; test_same_session_rerun_is_idempotent | Fixed+Applied |
| R6-03 | paper_loop fill loop | High | venue.fill is incremental; passing full qty crashes on partial fills; fills counted without outcome check | fill by o.remaining; check gw.process outcome; fail closed | test_partial_fill_completes_on_next_step | Fixed+Applied |
| R6-04 | tests/test_paper_loop.py::test_restart_continuity_keeps_session | Med | Asserted continued trading after a naive restart; mock venue reuses exec_ids so gateway correctly latches EXEC_ID_PAYLOAD_CONFLICT. First CI run RED on this test only | test corrected to assert session durability + fail-closed on exec-id conflict (the guaranteed behaviour) | itself | Delivered — commit to turn CI green |
| R6-05 | CI absent | Med | No executed test result at HEAD | .github/workflows/tests.yml added | workflow run | Fixed+Applied (first run red pending R6-04) |

## C. Open items (require approval / external validation — NOT silently patched)

| ID | Area | Severity | Why open | Required action |
|----|------|----------|----------|-----------------|
| O-01 | Audit-head external anchoring | Med | full wipe of all persistence still appears as clean first start | anchor audit head to an external append-only store; owner design decision |
| O-02 | Operator key custody | Med | OperatorConsole is in-process; real deployment needs separate custody | HSM/separate-process signer + rotation policy; security review |
| O-03 | DSR/PBO acceptance wiring (D-18) | Med | run_grid emits HAC t-stats; DSR applies to Sharpe trials | pre-registered return-series protocol amendment; do NOT apply DSR to t-stats |
| O-04 | Regime classifier calibration | Med | state.py thresholds are declared POLICY assumptions, correlation-only | pre-registered calibration study; multi-signal regime (vol/liquidity/breadth) |
| O-05 | Stress scenario set | Med | system.py scenarios are static (max -10%); tail gaps beyond set | owner-approved scenario governance incl. crash/correlation-spike/liquidity-dry |
| O-06 | Adaptive decay / change detection | Low | fixed EWMA half-life lags regime shifts | pre-registered experiment: bounded change-detection -> forced re-probation |
| O-07 | Autonomous paper-loop runner | Med | single-step function exists; no scheduled data->decision->execution->reconciliation->learning daemon | design approval: shadow baseline + venue reconciliation on restart; then implement |
| O-08 | Restart venue reconciliation | Med | mock venue reuses exec_ids on rebuild; continued trading after restart needs persistent broker exec_ids + gw.recover() | venue adapter with durable exec_id sequence; reconcile before resuming (part of O-07) |
| O-09 | Per-model attribution | Med | paper loop returns one ensemble-vs-hold number; no per-model leave-one-out | pre-registered leave-one-out counterfactual books for per-model earned authority |
| O-10 | AI audit-trail hardening | Low | ai_advisory logs plaintext; no hash-chain/lock/redaction; adversary fields loosely validated | hash-chained + locked + redacted audit log; strict adversary schema |
| O-11 | Symbol-order interface enforcement | Low | allocation vector must match sorted price-symbol order by caller convention | enforce ordering in the interface, not by documentation |

## D. Research integrity (unchanged, do not rewrite)
TSMOM falsification: 0 survivors / 13 trials remains INCONCLUSIVE under its protocol,
not proof of absence of edge. Frozen panel evidence/panel_real_v1.csv and its manifest
are not modified. Synthetic fixtures are never described as market validation.

## E. Honest limits
No claim of "unbreakable", production-readiness, or a profitable edge is made.
The guarantee is fail-closed behaviour + bounded authority + tested invariants,
not invulnerability. ~10% annualized net is a research objective, not a promise.
The paper loop is a single-step component, not yet a complete autonomous system.

## F. Phase 2 verification (HEAD 274b8ad6b475a9a221db968d36e86cf1964beebb, 2026-11-27)

Full-suite Colab run at b23f8fe (2026-11-27): 558 passed / 0 failed. Fast subset at
274b8ad (tests/test_phase2.py + tests/test_runner_integration.py): 24 passed.

| ID | Item | Status | Evidence |
|----|------|--------|----------|
| P2-01 | Runner baselines KeyError 'spy_buy_hold_gross' | RESOLVED | generic benchmark_* + SPY-conditional spy_* keys; 8/8 integration tests |
| P2-02 | no_authority vs no_trades conflated | RESOLVED | no_authority from tracker eligibility; no_trades from intent count; separated |
| P2-03 | Promotion ledger program/lifetime budgets, config freeze, terminal closure, single-writer lock | VERIFIED | 16 phase2 tests pass; thread-lock + thread-local depth + flock; CONFIG record; TERMINAL events |
| P2-04 | Probe journal separation, immutable families, operational guards, blocked-shadow rejection | VERIFIED | phase2 tests pass; no live authority wiring |
| P2-05 | SE weights certificate-gated, fail-closed without ledger | VERIFIED | se_weights requires PromotionLedger.is_promoted; excludes terminal |
| P2-06 | Null suite whole-program + per-cell + frozen reps/seeds | APPLIED | Batch 1 replacement committed; per-cell check added; undeclared LIFETIME_BOUND=0.10 removed |
| P2-07 | CI null-suite gate | APPLIED | .github/workflows/phase2_null_suite.yml (5000 reps, hashed report artifact) |
| P2-08 | Null suite calibration FAILS | OPEN — PROTOCOL | 5000 reps: program24_ar0.5 = 0.0738 > 0.07; lifetime48_iid = 0.0974 > 0.07 |
| P2-09 | Positive-control power | OPEN — PROTOCOL | 0.0444 at z_cell=2.865; no declared numeric floor |
| P2-10 | Protocol amendment | PROPOSED, NOT APPROVED | evidence/phase2_probe_protocol_v3_amendment_PROPOSED.md |

Root causes of P2-08/P2-09 are protocol inconsistencies, NOT code bugs:
(1) per-cell Bonferroni-24 across 48 lifetime cells yields theoretical any-false
rate 1-(1-0.05/24)^48 = 0.0953, incompatible with a 0.07 acceptance bound; the
protocol declares a lifetime budget (0.10) but no lifetime acceptance bound.
(2) Bartlett HAC L=21 under-corrects the EWMA-mean variance at AR(1) rho=0.5
(~2.96x IID inflation), pushing program-wise false promotion above 0.07.
No frozen parameter has been changed and no threshold tuned to force a pass.
Phase 2 remains DISABLED pending owner selection of amendment options
(L1/L2/L3, A1/A2/A3, P1/P2) or a v4 redesign, re-freeze, and a passing 5000-rep suite.
