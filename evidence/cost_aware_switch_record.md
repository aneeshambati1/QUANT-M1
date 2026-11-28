# Cost-Aware Allocator Switch Record

Date: 2026-11-27. Owner approved the runner-default switch after the
pre-registered Batch A ablation (evidence/batchA_cost_aware_protocol.md).

## Verified at switch time
- Full suite: 574 passed / 0 failed (owner Colab run).
- Runner banner: "Allocator: cost_aware_allocator".
- Runner result: NO AUTHORITY / NO TRADES (fail-closed; no model has earned
  authority; Phase 2 SE path remains disabled).
- Ablation: control cost 654bp / turnover 131x vs treatment 86bp / 17x;
  accept_switch_candidate=True; both arms fail-closed; gross <= 1.

## Scope of the change
- scripts/run_simulation.py now passes allocator=cost_aware_allocator.
- tests/test_runner_uses_cost_aware.py is the permanent regression gate.
- Mechanical friction reduction only. NOT alpha. Both ablation arms lost money
  on the placeholder signal; the panel is non-PIT; the 13-trial result is
  inconclusive.

## Unchanged
Risk Kernel veto, ledger chain, authority gates, Phase 2 disabled, long-only,
unleveraged, T+1, paper-only.

## Register cross-reference
docs/DEFECT_REGISTER.md section F should add:
  P2-14 | Runner default switched to cost-aware allocator | APPLIED |
        574 passed; ablation accept=True; mechanical friction only, not alpha.
(Apply that one-line addition when convenient; this record is the authoritative
source for the switch.)
