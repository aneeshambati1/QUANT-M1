"""M1.3 dependency wiring. Offline / mock only. No live transport, no secrets.

Phase 1 extension (backward compatible): build() accepts an optional custom
universe (instruments, marks, scenarios, allowlist, product_classes) constructed
at build time through the real Instrument / ConfigAuthority APIs. When
instruments is None, the legacy A/B/C default universe and behavior are
preserved exactly. Custom-universe rules are fail-closed: marks are required
and must cover every instrument; default stress scenarios are derived from
those marks; the allowlist defaults to the instrument symbols and
product_classes to {"PLAIN_ETF"}; no Risk Kernel limit is changed here.
"""
from __future__ import annotations

import os
import threading

from .core import (Store, EventLog, Ledger, VerifiedState, Clock, ConfigAuthority,
                   ImmutableFloor, Mandate, Operational, Instrument)
from .flags import Flags
from .venue import MockVenue
from .authority import Kernel, Gateway, OperatorConsole
from .universe import build_scenarios


def _validate_instruments(instruments) -> None:
    if not isinstance(instruments, dict) or not instruments:
        raise ValueError("instruments must be a non-empty dict")
    for sym, inst in instruments.items():
        if not isinstance(sym, str) or not sym:
            raise ValueError("instrument symbols must be non-empty strings")
        if not isinstance(inst, Instrument):
            raise ValueError(f"{sym}: not a core.Instrument")
        if inst.symbol != sym:
            raise ValueError(f"{sym}: instrument.symbol mismatch ({inst.symbol})")
        if isinstance(inst.adv, bool) or not isinstance(inst.adv, int) or inst.adv <= 0:
            raise ValueError(f"{sym}: adv must be a positive integer (cents)")
        if isinstance(inst.beta_bps, bool) or not isinstance(inst.beta_bps, int):
            raise ValueError(f"{sym}: beta_bps must be an integer")
        if inst.holdings is not None:
            if not isinstance(inst.holdings, tuple):
                raise ValueError(f"{sym}: holdings must be a tuple or None")
            for item in inst.holdings:
                if (not isinstance(item, tuple) or len(item) != 2
                        or not isinstance(item[0], str)
                        or isinstance(item[1], bool) or not isinstance(item[1], int)
                        or item[1] < 0):
                    raise ValueError(f"{sym}: invalid holdings entry {item!r}")
        if not isinstance(inst.halted, bool):
            raise ValueError(f"{sym}: halted must be bool")


def _validate_marks(marks, required) -> None:
    if not isinstance(marks, dict) or not marks:
        raise ValueError("marks must be a non-empty dict")
    missing = sorted(set(required) - set(marks))
    if missing:
        raise ValueError(f"marks missing for instruments: {missing}")
    for s, m in marks.items():
        if isinstance(m, bool) or not isinstance(m, int) or m <= 0:
            raise ValueError(f"mark for {s} must be a positive integer (cents)")


def _validate_scenarios(scenarios, required) -> None:
    if not isinstance(scenarios, (tuple, list)) or not scenarios:
        raise ValueError("scenarios must be a non-empty tuple of dicts")
    for sc in scenarios:
        if not isinstance(sc, dict):
            raise ValueError("each scenario must be a dict")
        missing = sorted(set(required) - set(sc))
        if missing:
            raise ValueError(f"scenario missing marks for: {missing}")
        for s, m in sc.items():
            if isinstance(m, bool) or not isinstance(m, int) or m <= 0:
                raise ValueError(f"scenario mark for {s} must be a positive integer (cents)")


def build(tmpdir: str, cash: int = 10_000_000, positions=None, marks=None,
          *, settle_lag_days: int = 1, m_vertex: int = 6,
          trade_date: int = 100, mark_wall: int = 1_000_000,
          allow_live: bool = False,
          instruments: dict[str, Instrument] | None = None,
          scenarios: tuple[dict[str, int], ...] | None = None,
          allowlist=None,
          product_classes=None):
    if allow_live:
        raise ValueError("M1 build() is offline/mock only; live transport is out of scope")

    os.makedirs(tmpdir, exist_ok=True)
    custom_universe = instruments is not None

    if not custom_universe:
        instruments = {
            "A": Instrument("A", "PLAIN_ETF", 50_000_000, 10_000,
                            (("ISS1", 6000), ("ISS2", 4000))),
            "B": Instrument("B", "PLAIN_ETF", 50_000_000, 8_000,
                            (("ISS1", 5000), ("ISS3", 5000))),
            "C": Instrument("C", "PLAIN_ETF", 50_000_000, 9_000, None),
        }
        if positions is None:
            positions = {"A": 300, "B": 300}
        marks = marks or {"A": 10_000, "B": 10_000, "C": 10_000}
        default_allowlist = frozenset({"A", "B", "C"})
    else:
        _validate_instruments(instruments)
        if marks is None:
            raise ValueError("marks are required when instruments are provided "
                             "(stress scenarios must cover every instrument)")
        _validate_marks(marks, required=set(instruments))
        if positions is None:
            positions = {}
        unknown = sorted(set(positions) - set(instruments))
        if unknown:
            raise ValueError(f"positions reference unregistered instruments: {unknown}")
        unmarked = sorted(set(positions) - set(marks))
        if unmarked:
            raise ValueError(f"positions reference symbols without marks: {unmarked}")
        default_allowlist = frozenset(instruments)

    if allowlist is None:
        allowlist = default_allowlist
    if product_classes is None:
        product_classes = frozenset({"PLAIN_ETF"})
    allowlist = frozenset(allowlist)
    product_classes = frozenset(product_classes)

    if custom_universe:
        bad = sorted(s for s, i in instruments.items()
                     if i.product_class not in product_classes)
        if bad:
            raise ValueError(f"instruments with product_class outside mandate: {bad}")

    if scenarios is None:
        if custom_universe:
            scenarios = build_scenarios(marks)
        else:
            scenarios = (
                {s: int(m * 0.90) for s, m in marks.items()},
                {s: int(m * 0.95) for s, m in marks.items()},
                {s: int(m * 1.02) for s, m in marks.items()},
                {s: int(m * 1.05) for s, m in marks.items()},
            )
    else:
        scenarios = tuple(scenarios)
        _validate_scenarios(scenarios, required=set(marks))

    cfg = ConfigAuthority(ImmutableFloor(),
                          Mandate(version=1, settle_lag_days=settle_lag_days,
                                  allowlist=allowlist,
                                  product_classes=product_classes),
                          Operational(m_vertex=m_vertex))

    nav = cash + sum(q * marks[s] for s, q in positions.items())
    st = VerifiedState(0, trade_date, cash, (), dict(positions), dict(marks),
                       mark_wall, (), nav, 1, cfg.mandate.digest())

    ledger = Ledger(Store(os.path.join(tmpdir, "ledger.json")), st)
    log = EventLog(Store(os.path.join(tmpdir, "events.json")))
    clock = Clock(seq=0, wall=mark_wall)
    venue = MockVenue(positions=dict(positions), cash=cash)
    flags = Flags()

    # TEST-ONLY KEY. Never use for an external broker connection.
    key = b"m1-test-key"

    wd = os.path.join(tmpdir, "STOP_NEW")
    flags_store = Store(os.path.join(tmpdir, "flags.json"))
    lock = threading.RLock()

    kernel = Kernel(key, cfg, instruments, scenarios, lock=lock)
    operator = OperatorConsole()
    gw = Gateway(key, cfg, ledger, log, venue, clock, instruments, scenarios,
                 flags, wd, lock=lock, flags_store=flags_store)
    gw.restore_flags()
    if gw._operator_hash is None:
        gw.register_operator(operator.current_hash())
    ledger.persist()

    return {"cfg": cfg, "ledger": ledger, "log": log, "clock": clock, "venue": venue,
            "flags": flags, "kernel": kernel, "gw": gw, "instruments": instruments,
            "scenarios": scenarios, "wd": wd, "lock": lock, "key": key,
            "flags_store": flags_store, "operator": operator}
