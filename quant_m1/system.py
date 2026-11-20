from __future__ import annotations
import os, threading
from .core import (Store, EventLog, Ledger, VerifiedState, Clock, ConfigAuthority,
                   ImmutableFloor, Mandate, Operational, Instrument)
from .flags import Flags
from .venue import MockVenue
from .authority import Kernel, Gateway


def build(tmpdir: str, cash: int = 10_000_000, positions=None, marks=None):
    os.makedirs(tmpdir, exist_ok=True)
    positions = positions or {"A": 300, "B": 300}
    marks = marks or {"A": 10_000, "B": 10_000, "C": 10_000}
    instruments = {
        "A": Instrument("A", "PLAIN_ETF", 50_000_000, 10_000, (("ISS1", 6000), ("ISS2", 4000))),
        "B": Instrument("B", "PLAIN_ETF", 50_000_000, 8_000, (("ISS1", 5000), ("ISS3", 5000))),
        "C": Instrument("C", "PLAIN_ETF", 50_000_000, 9_000, None),
    }
    scenarios = (
        {s: int(m * 0.90) for s, m in marks.items()},
        {s: int(m * 0.95) for s, m in marks.items()},
        {s: int(m * 1.02) for s, m in marks.items()},
        {s: int(m * 1.05) for s, m in marks.items()},
    )
    cfg = ConfigAuthority(ImmutableFloor(), Mandate(version=1), Operational())
    nav = cash + sum(q * marks[s] for s, q in positions.items())
    st = VerifiedState(0, 100, cash, (), dict(positions), dict(marks), 1_000_000,
                       (), nav, 1, cfg.mandate.digest())
    ledger = Ledger(Store(os.path.join(tmpdir, "ledger.json")), st)
    log = EventLog(Store(os.path.join(tmpdir, "events.json")))
    clock = Clock()
    venue = MockVenue(positions=dict(positions), cash=cash)
    flags = Flags()
    key = b"m1-test-key"
    wd = os.path.join(tmpdir, "STOP_NEW")
    kernel = Kernel(key, cfg, instruments, scenarios)
    gw = Gateway(key, cfg, ledger, log, venue, clock, instruments, scenarios, flags, wd)
    ledger.persist()
    return dict(cfg=cfg, ledger=ledger, log=log, clock=clock, venue=venue,
                flags=flags, kernel=kernel, gw=gw, instruments=instruments,
                scenarios=scenarios, wd=wd, lock=threading.Lock(), key=key)
