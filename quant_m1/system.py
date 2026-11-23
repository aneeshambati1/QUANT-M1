from .settlement import SettlementBook, ExchangeCalendar
calendar = calendar or ExchangeCalendar("XNYS")
settlement = SettlementBook(calendar, pending=pending_settlements)
from __future__ import annotations
import os
import threading
from .core import (
    Store, EventLog, Ledger, VerifiedState, Clock, ConfigAuthority,
    ImmutableFloor, Mandate, Operational, Instrument
)
from .flags import Flags
from .venue import MockVenue
from .authority import Kernel, Gateway
def build(
    tmpdir: str,
    cash: int = 10_000_000,          # keep: settled cash for backward compatibility
    positions=None,
    marks=None,
    cash_unsettled: int = 0,         # ADD
    pending_settlements=(),          # ADD
    calendar=None,                   # ADD: exchange calendar
    as_of=None,                      # ADD: current business date
    allow_live: bool = False,        # ADD: must default False
):
    positions = positions or {"A": 300, "B": 300}
    marks = marks or {"A": 10_000, "B": 10_000, "C": 10_000}

    instruments = {
        "A": Instrument(
            "A", "PLAIN_ETF", 50_000_000, 10_000,
            (("ISS1", 6000), ("ISS2", 4000))
        ),
        "B": Instrument(
            "B", "PLAIN_ETF", 50_000_000, 8_000,
            (("ISS1", 5000), ("ISS3", 5000))
        ),
        "C": Instrument("C", "PLAIN_ETF", 50_000_000, 9_000, None),
    }

    scenarios = (
        {s: int(m * 0.90) for s, m in marks.items()},
        {s: int(m * 0.95) for s, m in marks.items()},
        {s: int(m * 1.02) for s, m in marks.items()},
        {s: int(m * 1.05) for s, m in marks.items()},
    )

    cfg = ConfigAuthority(
        ImmutableFloor(),
        Mandate(version=1),
        Operational()
    )

    nav = cash + sum(q * marks[s] for s, q in positions.items())
    st = VerifiedState(
        0, 100, cash, (), dict(positions), dict(marks),
        1_000_000, (), nav, 1, cfg.mandate.digest()
    )

    ledger = Ledger(Store(os.path.join(tmpdir, "ledger.json")), st)
    log = EventLog(Store(os.path.join(tmpdir, "events.json")))
    clock = Clock()
    venue = MockVenue(positions=dict(positions), cash=cash)
    flags = Flags()

    # Test-only secret. Never use this key for an external broker connection.
    key = b"m1-test-key"

    wd = os.path.join(tmpdir, "STOP_NEW")
    lock = threading.RLock()

    kernel = Kernel(key, cfg, instruments, scenarios, lock=lock)
    gw = Gateway(
        key, cfg, ledger, log, venue, clock,
        instruments, scenarios, flags, wd, lock=lock
    )

    ledger.persist()

    return {
        "cfg": cfg,
        "ledger": ledger,
        "log": log,
        "clock": clock,
        "venue": venue,
        "flags": flags,
        "kernel": kernel,
        "gw": gw,
        "instruments": instruments,
        "scenarios": scenarios,
        "wd": wd,
        "lock": lock,
        "key": key,
    }
