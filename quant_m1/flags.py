"""M1 permission flags and independent recovery monitor."""
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

        permissions = set(PERMS)

        if self.frozen:
            permissions -= {"NEW_RISK", "SUBMIT"}

        if self.data != "OK":
            permissions -= {"NEW_RISK"}

        if self.model != "OK":
            permissions -= {"NEW_RISK", "RECOVER_AUTO"}

        if self.exec != "OK":
            permissions -= {"NEW_RISK", "SUBMIT", "RECOVER_AUTO"}

        if self.envelope in {"EMPTY", "OVERFLOW"}:
            permissions -= {"NEW_RISK"}

        if self.governance != "OK":
            permissions -= {"NEW_RISK", "SUBMIT", "RECOVER_AUTO"}

        if self.market == "RESTRICTED":
            permissions -= {"NEW_RISK"}

        return frozenset(permissions)


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
        """Every required monitor must independently report healthy status."""
        required = (
            "data",
            "compute",
            "config",
            "clock",
            "infra",
            "authority",
        )
        return all(
            getattr(self, component) == "INDEPENDENT"
            for component in required
        )


def may_auto_recover(mon: Monitor, flags: Flags) -> bool:
    """Recovery is allowed only when monitoring and permission checks agree."""
    return (
        mon.qualifies_for_recovery()
        and "RECOVER_AUTO" in flags.permissions()
    )
