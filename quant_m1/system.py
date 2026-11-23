"""M1.3 dependency wiring. Offline / mock only. No live transport, no secrets."""
from __future__ import annotations

import os
import threading

from .core import (Store, EventLog, Ledger, VerifiedState, Clock, ConfigAuthority,
                   ImmutableFloor, Mandate, Operational, Instrument)
from .flags import Flags
from .venue import MockVenue
from .authority import Kernel, Gateway


def build(tmpdir: str, cash: int = 10_000_000, positions=None, marks=None,
          *, settle_lag_days: int = 1, m_vertex: int = 6,
          trade_date: int = 100, mark_wall: int = 1_000_000,
          allow_live: bool = False):
    if allow_live:
        raise ValueError("M1 build() is offline/mock only; live transport is out of scope")

    os.makedirs(tmpdir, exist_ok=True)
    positions = positions if positions is not None else {"A": 300, "B": 300}
    marks = marks or {"A": 10_000, "B": 10_000, "C": 10_000}

    instruments = {
        "A": Instrument("A", "PLAIN_ETF", 50_000_000, 10_000,
                        (("ISS1", 6000), ("ISS2", 4000))),
        "B": Instrument("B", "PLAIN_ETF", 50_000_000, 8_000,
                        (("ISS1", 5000), ("ISS3", 5000))),
        "C": Instrument("C", "PLAIN_ETF", 50_000_000, 9_000, None),
    }

    scenarios = (
        {s: int(m * 0.90) for s, m in marks.items()},
        {s: int(m * 0.95) for s, m in marks.items()},
        {s: int(m * 1.02) for s, m in marks.items()},
        {s: int(m * 1.05) for s, m in marks.items()},
    )

    cfg = ConfigAuthority(ImmutableFloor(),
                          Mandate(version=1, settle_lag_days=settle_lag_days),
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
    gw = Gateway(key, cfg, ledger, log, venue, clock, instruments, scenarios,
                 flags, wd, lock=lock, flags_store=flags_store)
    gw.restore_flags()
    ledger.persist()

    return {"cfg": cfg, "ledger": ledger, "log": log, "clock": clock, "venue": venue,
            "flags": flags, "kernel": kernel, "gw": gw, "instruments": instruments,
            "scenarios": scenarios, "wd": wd, "lock": lock, "key": key,
            "flags_store": flags_store}
