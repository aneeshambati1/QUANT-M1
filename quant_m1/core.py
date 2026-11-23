"""Core M1.1: integer units, config authority, durable ledger and state."""
from __future__ import annotations
import hashlib, hmac, json, os, tempfile
from dataclasses import dataclass, field, asdict
from typing import Any

Money = int  # cents
Shares = int
BPS = 10_000


def ceil_bps(amount: int, bps: int) -> int:
    if amount < 0 or bps < 0:
        raise ValueError("amount and bps must be nonnegative")
    return (amount * bps + BPS - 1) // BPS


def floor_bps(amount: int, bps: int) -> int:
    if amount < 0 or bps < 0:
        raise ValueError("amount and bps must be nonnegative")
    return amount * bps // BPS


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


@dataclass(frozen=True)
class ImmutableFloor:
    long_only: bool = True
    no_leverage: bool = True
    require_token: bool = True
    single_use_nonce: bool = True
    fail_closed_on_stale: bool = True
    require_kernel_recompute: bool = True

    def digest(self) -> str:
        return sha(asdict(self))


IMMUTABLE_FIELDS = frozenset(ImmutableFloor().__dict__.keys())


@dataclass(frozen=True)
class Mandate:
    version: int
    gross_bps: int = 10_000
    issuer_bps: int = 4_000
    factor_bps: int = 6_000
    worst_loss_bps: int = 1_500
    cvar_bps: int = 1_200
    cvar_q_bps: int = 2_500
    drawdown_trigger_bps: int = 1_000
    order_notional_cap: Money = 2_000_000
    participation_bps: int = 500
    rate_limit: int = 5
    rate_window: int = 100
    settle_lag_days: int = 1
    tau_data_secs: int = 120
    allowlist: frozenset[str] = frozenset({"A", "B", "C"})
    product_classes: frozenset[str] = frozenset({"PLAIN_ETF"})

    def digest(self) -> str:
        d = asdict(self)
        d["allowlist"] = sorted(self.allowlist)
        d["product_classes"] = sorted(self.product_classes)
        return sha(d)


@dataclass(frozen=True)
class Operational:
    m_vertex: int = 6
    auth_ttl_seqs: int = 50
    version: int = 1

    def digest(self) -> str:
        return sha(asdict(self))


class ConfigAuthorityError(Exception):
    pass


@dataclass
class ConfigAuthority:
    floor: ImmutableFloor
    mandate: Mandate
    ops: Operational
    _floor_digest: str = field(init=False)

    def __post_init__(self):
        self._floor_digest = self.floor.digest()

    def apply_config(self, diff: dict) -> None:
        illegal = IMMUTABLE_FIELDS & set(diff)
        if illegal:
            raise ConfigAuthorityError(f"CONFIG attempted immutable change: {sorted(illegal)}")
        if "mandate" in diff:
            md = dict(diff["mandate"])
            md["allowlist"] = frozenset(md.get("allowlist", self.mandate.allowlist))
            md["product_classes"] = frozenset(md.get("product_classes", self.mandate.product_classes))
            self.mandate = Mandate(**{**asdict(self.mandate), **md,
                                      "version": self.mandate.version + 1})
        if "ops" in diff:
            self.ops = Operational(**{**asdict(self.ops), **diff["ops"],
                                      "version": self.ops.version + 1})

    def attested(self) -> bool:
        return hmac.compare_digest(self.floor.digest(), self._floor_digest)


class Clock:
    def __init__(self, seq: int = 0, wall: int = 1_000_000):
        self._seq, self._wall = seq, wall

    def next(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def wall(self) -> int:
        return self._wall

    def advance_wall(self, dt: int) -> None:
        if dt < 0:
            raise ValueError("wall clock cannot be advanced by a negative duration")
        self._wall += dt

    def skew_wall(self, dt: int) -> None:
        self._wall += dt


@dataclass
class Store:
    path: str

    def save(self, obj: Any) -> None:
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(canonical(obj))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            # Persist directory entry where supported.
            try:
                dfd = os.open(d, os.O_DIRECTORY)
                try: os.fsync(dfd)
                finally: os.close(dfd)
            except (AttributeError, OSError):
                pass
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def load(self) -> Any | None:
        if not os.path.exists(self.path):
            return None
        with open(self.path) as f:
            return json.load(f)


@dataclass
class EventLog:
    store: Store
    chain: list[dict] = field(default_factory=list)

    def append(self, kind: str, payload: dict, seq: int) -> str:
        prev = self.chain[-1]["hash"] if self.chain else "0" * 64
        rec = {"seq": seq, "kind": kind, "payload": payload, "prev": prev}
        rec["hash"] = hashlib.sha256((prev + canonical(rec)).encode()).hexdigest()
        self.chain.append(rec)
        self.store.save(self.chain)
        return rec["hash"]

    def head(self) -> str:
        return self.chain[-1]["hash"] if self.chain else "0" * 64

    def restore(self) -> None:
        data = self.store.load()
        self.chain = data if data else []
        prev = "0" * 64
        for rec in self.chain:
            supplied = rec.get("hash")
            body = {k: v for k, v in rec.items() if k != "hash"}
            expected = hashlib.sha256((prev + canonical(body)).encode()).hexdigest()
            if rec.get("prev") != prev or supplied != expected:
                raise ValueError("event log hash-chain verification failed")
            prev = supplied


@dataclass(frozen=True)
class Instrument:
    symbol: str
    product_class: str
    adv: Money
    beta_bps: int
    holdings: tuple[tuple[str, int], ...] | None
    halted: bool = False

    def lookthrough(self) -> tuple[tuple[str, int], ...]:
        if self.holdings is None:
            return ((f"UNK:{self.symbol}", BPS),)
        tot = sum(w for _, w in self.holdings)
        if tot != BPS or any(w < 0 for _, w in self.holdings):
            return ((f"UNK:{self.symbol}", BPS),)
        return self.holdings


@dataclass
class WorkingOrder:
    order_id: str
    symbol: str
    side: str
    qty: Shares
    cum_filled: Shares
    price: Money
    cost_bps: int
    status: str
    trade_date: int
    nonce: str
    filled_notional: Money = 0
    filled_cost: Money = 0

    @property
    def remaining(self) -> Shares:
        return self.qty - self.cum_filled

    @property
    def terminal(self) -> bool:
        return self.status in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}


@dataclass
class VerifiedState:
    seq: int
    trade_date: int
    cash_settled: Money
cash_unsettled: int
pending_settlements: tuple[PendingSettlement, ...]
    receivables: tuple[tuple[int, Money], ...]
    positions: dict[str, Shares]
    marks: dict[str, Money]
    mark_wall: int
    working: tuple[WorkingOrder, ...]
    peak_nav: Money
    mandate_version: int
    config_digest: str

    def receivable_total(self) -> Money:
        return sum(c for _, c in self.receivables)

    def nav(self) -> Money:
        if any(s not in self.marks for s, q in self.positions.items() if q):
            raise ValueError("position has no current mark")
        return (self.cash_settled + self.receivable_total()
                + sum(q * self.marks[s] for s, q in self.positions.items() if q))

    def digest(self) -> str:
        return sha({
            "seq": self.seq, "date": self.trade_date, "cash": self.cash_settled,
            "recv": sorted(self.receivables), "pos": self.positions,
            "marks": self.marks, "peak": self.peak_nav,
            "working": [asdict(o) for o in self.working],
            "mv": self.mandate_version, "cd": self.config_digest,
        })


@dataclass
class Ledger:
    store: Store
    state: VerifiedState
    spent_nonces: set[str] = field(default_factory=set)
    applied_exec_ids: set[str] = field(default_factory=set)
    submit_seq: list[int] = field(default_factory=list)
    reserved: dict[str, int] = field(default_factory=dict)
    applied_exec_hashes: dict[str, str] = field(default_factory=dict)

    def persist(self) -> None:
        self.store.save({
            "state": {
                "seq": self.state.seq, "trade_date": self.state.trade_date,
                "cash_settled": self.state.cash_settled,
                "receivables": list(self.state.receivables),
                "positions": self.state.positions, "marks": self.state.marks,
                "mark_wall": self.state.mark_wall, "peak_nav": self.state.peak_nav,
                "mandate_version": self.state.mandate_version,
                "config_digest": self.state.config_digest,
                "working": [asdict(o) for o in self.state.working],
            },
            "spent_nonces": sorted(self.spent_nonces),
            "applied_exec_ids": sorted(self.applied_exec_ids),
            "applied_exec_hashes": self.applied_exec_hashes,
            "submit_seq": self.submit_seq,
            "reserved": self.reserved,
        })

    @staticmethod
    def restore(store: Store) -> "Ledger | None":
        d = store.load()
        if d is None:
            return None
        s = d["state"]
        working = []
        for item in s["working"]:
            # Backward compatible with pre-M1.1 ledgers.
            working.append(WorkingOrder(**item))
        st = VerifiedState(
            seq=s["seq"], trade_date=s["trade_date"], cash_settled=s["cash_settled"],
            receivables=tuple(tuple(x) for x in s["receivables"]),
            positions=dict(s["positions"]), marks=dict(s["marks"]),
            mark_wall=s["mark_wall"], peak_nav=s["peak_nav"],
            mandate_version=s["mandate_version"], config_digest=s["config_digest"],
            working=tuple(working),
        )
        return Ledger(store, st, set(d.get("spent_nonces", [])),
                      set(d.get("applied_exec_ids", [])), list(d.get("submit_seq", [])),
                      dict(d.get("reserved", {})), dict(d.get("applied_exec_hashes", {})))

    def settle(self, today: int) -> None:
        due = sum(c for d, c in self.state.receivables if d <= today)
        self.state.receivables = tuple((d, c) for d, c in self.state.receivables if d > today)
        self.state.cash_settled += due

    def mark_peak(self) -> None:
        self.state.peak_nav = max(self.state.peak_nav, self.state.nav())

    def drawdown_bps(self) -> int:
        pk = self.state.peak_nav
        if pk <= 0:
            return BPS
        return (pk - self.state.nav()) * BPS // pk

    def working_active(self) -> tuple[WorkingOrder, ...]:
        return tuple(o for o in self.state.working if not o.terminal and o.remaining > 0)
