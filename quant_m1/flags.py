"""M1.3 permission flags. killed/frozen are additive and default False."""
from __future__ import annotations

from dataclasses import dataclass

PERMS = ("NEW_RISK", "REDUCE_RISK", "SUBMIT", "RECOVER_AUTO")


@dataclass
class Flags:
    data: str = "OK"
    model: str = "OK"
    exec: str = "OK"
    envelope: str = "OK"
    governance: str = "OK"
    market: str = "NORMAL"
    killed: bool = False
    frozen: bool = False

    def permissions(self) -> frozenset[str]:
        if self.killed:
            return frozenset()
        p = set(PERMS)
        if self.frozen:
            p -= {"NEW_RISK", "SUBMIT"}
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
    data: str
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
