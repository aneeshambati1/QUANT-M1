from __future__ import annotations
from dataclasses import dataclass, field

PERMS = ("NEW_RISK", "REDUCE_RISK", "SUBMIT", "RECOVER_AUTO")


@dataclass
class Flags:
    data: str = "OK"          # OK|DEGRADED|STALE|CONFLICTED
    model: str = "OK"         # OK|OOD|INADEQUATE|SUPPORT_EXIT
    exec: str = "OK"          # OK|DEGRADED|DIVERGENT|DISCONNECTED
    envelope: str = "OK"      # OK|CONTRACTED|EMPTY|OVERFLOW
    governance: str = "OK"    # OK|UNATTESTED|ROLLBACK_REQUIRED
    market: str = "NORMAL"    # NORMAL|STRESS|RESTRICTED

    def permissions(self) -> frozenset[str]:
        p = set(PERMS)
        if self.data != "OK":
            p -= {"NEW_RISK"}
        if self.model != "OK":
            p -= {"NEW_RISK", "RECOVER_AUTO"}
        if self.exec != "OK":
            p -= {"NEW_RISK", "SUBMIT", "RECOVER_AUTO"}
        if self.envelope in {"EMPTY", "OVERFLOW"}:
            p -= {"NEW_RISK"}
        if self.governance != "OK":
            p -= {"NEW_RISK", "SUBMIT", "RECOVER_AUTO"}
        if self.market == "RESTRICTED":
            p -= {"NEW_RISK"}
        return frozenset(p)


@dataclass(frozen=True)
class Monitor:
    name: str
    data: str            # INDEPENDENT | SHARED:<x> | UNKNOWN
    compute: str
    config: str
    clock: str
    infra: str
    authority: str

    def qualifies_for_recovery(self) -> bool:
        need = ("data", "compute", "config", "authority")
        return all(getattr(self, d) == "INDEPENDENT" for d in need)


def may_auto_recover(mon: Monitor, flags: Flags) -> bool:
    return mon.qualifies_for_recovery() and "RECOVER_AUTO" in flags.permissions()
