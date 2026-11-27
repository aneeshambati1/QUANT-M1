"""ETF universe construction and sanctioned post-build registration (Phase 1).

Verified facts respected here (HEAD 6ad2af9):
  - Instrument is defined in quant_m1.core (no quant_m1.meta.instrument exists).
  - The mandate permits product_class "PLAIN_ETF"; the allowlist is a frozenset
    on a frozen Mandate. The only sanctioned mutation path is
    ConfigAuthority.apply_config (version bump + digest) then
    Ledger.propagate_config. In-place mandate edits are impossible/forbidden.
  - holdings=None maps every ETF into the shared UNK:ALL issuer bucket
    (core.Instrument.lookthrough). With issuer_bps=4000 the entire
    unknown-holdings book is capped at 40% of NAV. Preserved by design;
    do not fabricate holdings to evade it.
  - beta_bps=10_000 is a conservative unit-beta placeholder, NOT an estimate.
    For a long-only book it maximizes measured absolute factor usage.
  - adv is Money = CENTS. 1_000_000_000 cents = $10M placeholder ADV.
  - Stress scenarios must cover every registered instrument. Kernel/Gateway
    hold references to the build-time instruments dict and scenarios tuple;
    registration updates those same objects under sx["lock"]. No Risk Kernel
    limit is read or written here.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from .core import Instrument

ETF_PRODUCT_CLASS = "PLAIN_ETF"
ETF_ADV_CENTS = 1_000_000_000
ETF_UNIT_BETA_BPS = 10_000
SCENARIO_FACTORS = (0.90, 0.95, 1.02, 1.05)


def make_etf_instrument(symbol: str) -> Instrument:
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol must be a non-empty string")
    return Instrument(
        symbol=symbol,
        product_class=ETF_PRODUCT_CLASS,
        adv=ETF_ADV_CENTS,
        beta_bps=ETF_UNIT_BETA_BPS,
        holdings=None,
        halted=False,
    )


def etf_instruments(symbols: Iterable[str]) -> dict[str, Instrument]:
    syms = list(symbols)
    if not syms:
        raise ValueError("symbols must be non-empty")
    if len(set(syms)) != len(syms):
        raise ValueError("duplicate symbols")
    return {s: make_etf_instrument(s) for s in syms}


def build_scenarios(marks: Mapping[str, int]) -> tuple[dict[str, int], ...]:
    if not marks:
        raise ValueError("marks must be non-empty")
    for s, m in marks.items():
        if isinstance(m, bool) or not isinstance(m, int) or m <= 0:
            raise ValueError(f"mark for {s} must be a positive integer (cents)")
    return tuple({s: int(m * f) for s, m in marks.items()} for f in SCENARIO_FACTORS)


def register_etf_universe(sx: dict, symbols: Iterable[str],
                          *, marks: Mapping[str, int] | None = None) -> dict:
    """Register ETFs on an already-built system through sanctioned APIs.

    marks: integer-cent marks used to rebuild stress scenarios; defaults to the
    ledger's current marks. Fails closed if any registered instrument would lack
    scenario coverage. Nothing is mutated on validation/configuration failure.
    Returns a factual report dict; prints nothing.
    """
    syms = list(symbols)
    if not syms:
        raise ValueError("symbols must be non-empty")
    for s in syms:
        if not isinstance(s, str) or not s:
            raise ValueError("symbols must be non-empty strings")
    if len(set(syms)) != len(syms):
        raise ValueError("duplicate symbols")

    kernel, gw, cfg, ledger = sx["kernel"], sx["gw"], sx["cfg"], sx["ledger"]
    with sx["lock"]:
        instruments = sx["instruments"]
        for s in syms:
            existing = instruments.get(s)
            if existing is not None and existing.product_class != ETF_PRODUCT_CLASS:
                raise ValueError(
                    f"{s} already registered as {existing.product_class}; refusing to overwrite")

        resolved_marks = dict(marks) if marks is not None else dict(ledger.state.marks)
        required = set(instruments) | set(syms)
        missing = sorted(required - set(resolved_marks))
        if missing:
            raise ValueError(f"scenario marks missing for: {missing}")
        new_scenarios = build_scenarios(resolved_marks)

        need_allow = set(cfg.mandate.allowlist) | set(syms)
        need_classes = set(cfg.mandate.product_classes) | {ETF_PRODUCT_CLASS}
        config_changed = (need_allow != set(cfg.mandate.allowlist)
                          or need_classes != set(cfg.mandate.product_classes))
        if config_changed:
            cfg.apply_config({"mandate": {
                "allowlist": sorted(need_allow),
                "product_classes": sorted(need_classes),
            }})
            ledger.propagate_config(cfg.mandate.version, cfg.mandate.digest())

        for s in syms:
            if s not in instruments:
                instruments[s] = make_etf_instrument(s)

        sx["scenarios"] = new_scenarios
        kernel.scenarios = new_scenarios
        gw.scenarios = new_scenarios

        return {
            "registered": sorted(syms),
            "instruments_total": len(instruments),
            "mandate_version": cfg.mandate.version,
            "config_digest": cfg.mandate.digest(),
            "config_changed": config_changed,
            "scenario_symbols": sorted(resolved_marks),
            "n_scenarios": len(new_scenarios),
        }
