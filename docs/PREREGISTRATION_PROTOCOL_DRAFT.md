# Preregistration Protocol — DRAFT

Status: DRAFT. Decisions marked TBD are intentionally unresolved and must be
chosen AFTER the Round 9A data audit and BEFORE any confirmatory result is
viewed. This draft establishes structure and anti-adjustment rules, not values.

Global rule: once the panel is frozen and the family registered, no threshold,
horizon, comparator, utility, or block rule may be changed because a result is
unfavorable. Any change requires a NEW registered family.

---

## Decision 1 — Primary utility

- Definition & rationale: per-episode net log growth, u_t = ln(end_nav/start_nav),
  cost already deducted by the ledger. Downside CVaR of the paired differential
  is applied at the SERIES level as Score = LCB(mean Δ) − λ_C·CVaR(−Δ). MDD is a
  separate path-level gate, not inside the per-episode utility.
- Candidate values: λ_C ∈ {0.5, 1.0, 2.0}.
- Evidence needed: sensitivity of the paired differential's downside tail across
  the audited episodes.
- Confirmatory or exploratory: CONFIRMATORY.
- Anti-adjustment rule: λ_C fixed before the first confirmatory episode is scored.

## Decision 2 — Evaluation horizon h

- Definition & rationale: number of sessions per decision episode; must match the
  intended rebalance cadence. Episodes use the PCDL `valuation` phase.
- Candidate values: h ∈ {1, 5, 21} sessions.
- Evidence needed: turnover/cost profile at each cadence from the cost model;
  number of independent h-length episodes available after the data audit.
- Confirmatory or exploratory: CONFIRMATORY.
- Anti-adjustment rule: h registered before episodes are generated.

## Decision 3 — Minimum economically relevant effect δ

- Definition & rationale: smallest incremental utility per episode that justifies
  the adaptive policy's complexity. Controls the power analysis.
- Candidate values: TBD pending the paired-differential variance σ_d from the audit.
- Evidence needed: pilot estimate of σ_d; δ must be set as a multiple of σ_d/√n.
- Confirmatory or exploratory: CONFIRMATORY.
- Anti-adjustment rule: δ fixed before the power calculation is finalized; it may
  NOT be lowered to rescue an underpowered result.

## Decision 4 — Comparator family

- Definition & rationale: frozen set of paired comparators, all starting from the
  identical pre-state. Paired: hold-current, cash-liquidation, frozen-config pooled.
  Any broad-market benchmark is UNPAIRED context and reported separately.
- Candidate values: {hold, cash} minimum; add pooled if a frozen-config policy exists.
- Evidence needed: none beyond the audit (comparators are definitions, not data).
- Confirmatory or exploratory: CONFIRMATORY.
- Anti-adjustment rule: comparator set frozen before scoring; no post-hoc additions.

## Decision 5 — Multiplicity correction & family

- Definition & rationale: one-sided primary test; Holm within a declared family;
  α = 0.05. Family = every registered spec, case, variant, and selection-influencing
  ablation. Family count derived from the immutable trial ledger depth.
- Candidate values: α ∈ {0.05, 0.01}; Holm (primary) vs Bonferroni (conservative).
- Evidence needed: expected number of registered specs from the research program.
- Confirmatory or exploratory: CONFIRMATORY for the primary family; ablations are
  exploratory but still counted in the ledger.
- Anti-adjustment rule: failed experiments are retained in the family; the count
  cannot be reduced retrospectively.

## Decision 6 — Contiguous temporal block rule

- Definition & rationale: resampling blocks must be CONTIGUOUS in time; a shared
  regime label on separated periods is NOT one block. Block length chosen by a
  pre-declared dependence rule (e.g., autocorrelation decay), never by significance.
- Candidate values: block length from the audit's provisional regime count; method
  = paired block bootstrap on the active-minus-hold differential.
- Evidence needed: autocorrelation structure of the paired differential from the
  audited PIT panel (requires real data first).
- Confirmatory or exploratory: CONFIRMATORY.
- Anti-adjustment rule: block rule registered before any bootstrap is run.

---

## Sequencing

1. Approve the Round 9A data spec and run the audit on candidate datasets.
2. Acquire a PIT panel that returns SUITABLE (or accept NOT_SUITABLE and stop).
3. Freeze the panel and its manifest.
4. Choose Decisions 1–6 using the audit's evidence.
5. Register the family in the trial ledger.
6. Only then wire the Round 9 statistical layer.

Until step 2 returns a SUITABLE panel, confirmatory testing is not possible and
the correct output of any experiment is NOT_IDENTIFIABLE.
