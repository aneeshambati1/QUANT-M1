"""Governed night-shift research intake and morning data revalidation.

This is the deterministic boundary between research-agent suggestions and the
QUANT experiment process. It stores falsifiable, long-only ETF hypotheses and
rechecks proposal/evidence freshness plus the supplied price panel before a
candidate may enter a separately pre-registered experiment.

This module deliberately does NOT call LLMs or market-data vendors, estimate
expected returns, create weights/orders, or promote a strategy. The panel's
``pit`` flag is an upstream provenance attestation: this module can verify the
panel's current content hash and timestamps, but cannot independently prove a
vendor's point-in-time publication history. A queue pass is NOT alpha evidence
or authority to enter the portfolio decision/execution path.

Persistence is an append-only SHA-256 hash chain. It detects accidental or
partial changes; it is not an authenticated signature against an adversary
who can rewrite the entire file. On POSIX, an advisory flock plus a
read-before-write head check make concurrent writers fail closed rather than
fork the chain; the single-writer-per-file contract is still recommended.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
try:
    import fcntl as _fcntl
except ImportError:  # Windows: no advisory flock; single-writer contract applies
    _fcntl = None
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ..data.loader import PricePanel, content_hash as panel_content_hash

GENESIS = "0" * 64
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")

# Initial bounded ETF universe. It is an allowlist, not a liquidity guarantee.
# An expanded universe must be explicitly supplied by the caller.
DEFAULT_ETF_UNIVERSE = frozenset({
    "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "USO", "UUP",
})

SIGNAL_FAMILIES = frozenset({
    "time_series_momentum",
    "cross_sectional_momentum",
    "mean_reversion",
    "volatility_filter",
    "trend_filter",
    "macro_filter",
    "multi_signal",
})


class NightShiftError(Exception):
    """Base error for invalid proposals or queue operations."""


class QueueIntegrityError(NightShiftError):
    """Raised when the persistent event chain cannot be verified."""


class CandidateStatus(str, Enum):
    PROPOSED = "PROPOSED"
    SCREENED_FOR_TEST = "SCREENED_FOR_TEST"
    QUARANTINED = "QUARANTINED"


def _utc(value: str, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise NightShiftError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NightShiftError(f"{field} is not valid ISO-8601") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise NightShiftError(f"{field} must include a timezone")
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NightShiftError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise NightShiftError(f"{field} must be finite")
    return out


def _positive_int(value: Any, field: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NightShiftError(f"{field} must be an integer >= {minimum}")


@dataclass(frozen=True)
class EvidenceReference:
    """Minimal source provenance carried with each proposal."""
    source: str
    raw_ref: str
    release_ts: str
    known_at: str
    vintage_id: str

    def __post_init__(self) -> None:
        for field_name in ("source", "raw_ref", "vintage_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise NightShiftError(f"evidence {field_name} is required")
        release = _utc(self.release_ts, "evidence.release_ts")
        known = _utc(self.known_at, "evidence.known_at")
        if known < release:
            raise NightShiftError("evidence known_at cannot precede release_ts")
        object.__setattr__(self, "release_ts", _iso(release))
        object.__setattr__(self, "known_at", _iso(known))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvidenceReference":
        if not isinstance(raw, Mapping):
            raise NightShiftError("each evidence reference must be an object")
        allowed = {"source", "raw_ref", "release_ts", "known_at", "vintage_id"}
        extra, missing = set(raw) - allowed, allowed - set(raw)
        if extra or missing:
            raise NightShiftError(
                f"evidence fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return cls(**{k: raw[k] for k in allowed})


@dataclass(frozen=True)
class ResearchCandidate:
    """A falsifiable research idea, not a prediction or allocation instruction."""
    candidate_id: str
    created_at: str
    title: str
    hypothesis: str
    mechanism: str
    signal_family: str
    universe: tuple[str, ...]
    lookback_sessions: int
    skip_sessions: int
    holding_sessions: int
    rebalance_sessions: int
    volatility_window: int
    estimated_round_trip_cost_bps: float
    falsification_rule: str
    evidence: tuple[EvidenceReference, ...]
    long_only: bool = True
    max_gross: float = 1.0
    cash_allowed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not _CANDIDATE_ID.fullmatch(self.candidate_id):
            raise NightShiftError("candidate_id must be a 3-80 character lowercase slug")
        created = _utc(self.created_at, "created_at")
        object.__setattr__(self, "created_at", _iso(created))

        for name in ("title", "hypothesis", "mechanism", "falsification_rule"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value.strip()) < 12:
                raise NightShiftError(f"{name} must contain at least 12 characters")

        if not isinstance(self.signal_family, str) or self.signal_family not in SIGNAL_FAMILIES:
            raise NightShiftError(f"unsupported signal_family: {self.signal_family!r}")

        if not isinstance(self.universe, tuple) or not 1 <= len(self.universe) <= 50:
            raise NightShiftError("universe must be a tuple containing 1-50 symbols")
        if any(not isinstance(s, str) or not _SYMBOL.fullmatch(s) for s in self.universe):
            raise NightShiftError("universe contains an invalid ticker symbol")
        if len(set(self.universe)) != len(self.universe):
            raise NightShiftError("universe contains duplicate symbols")

        for name, minimum in (
            ("lookback_sessions", 1),
            ("holding_sessions", 1),
            ("rebalance_sessions", 1),
            ("volatility_window", 2),
        ):
            _positive_int(getattr(self, name), name, minimum)
        if (isinstance(self.skip_sessions, bool)
                or not isinstance(self.skip_sessions, int)
                or self.skip_sessions < 0
                or self.skip_sessions >= self.lookback_sessions):
            raise NightShiftError("skip_sessions must be in [0, lookback_sessions)")

        cost = _finite_number(self.estimated_round_trip_cost_bps,
                              "estimated_round_trip_cost_bps")
        if not 0 <= cost <= 10_000:
            raise NightShiftError("estimated round-trip cost must be in [0,10000] bps")
        object.__setattr__(self, "estimated_round_trip_cost_bps", cost)

        if self.long_only is not True:
            raise NightShiftError("QUANT research proposals must be long-only")
        if self.cash_allowed is not True:
            raise NightShiftError("cash must remain an allowed position")
        gross = _finite_number(self.max_gross, "max_gross")
        if not 0 < gross <= 1.0:
            raise NightShiftError("max_gross must be in (0,1]; leverage is prohibited")
        object.__setattr__(self, "max_gross", gross)

        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise NightShiftError("at least one evidence reference is required")
        for ref in self.evidence:
            if not isinstance(ref, EvidenceReference):
                raise NightShiftError("evidence entries must be EvidenceReference objects")
            if _utc(ref.known_at, "evidence.known_at") > created:
                raise NightShiftError("proposal cannot cite evidence not yet known at creation")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ResearchCandidate":
        """Parse untrusted agent output; unknown fields are rejected."""
        if not isinstance(raw, Mapping):
            raise NightShiftError("candidate payload must be an object")
        allowed = {
            "candidate_id", "created_at", "title", "hypothesis", "mechanism",
            "signal_family", "universe", "lookback_sessions", "skip_sessions",
            "holding_sessions", "rebalance_sessions", "volatility_window",
            "estimated_round_trip_cost_bps", "falsification_rule", "evidence",
            "long_only", "max_gross", "cash_allowed",
        }
        extra, missing = set(raw) - allowed, allowed - set(raw)
        if extra or missing:
            raise NightShiftError(
                f"candidate fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        evidence_raw, universe_raw = raw["evidence"], raw["universe"]
        if not isinstance(evidence_raw, (list, tuple)):
            raise NightShiftError("evidence must be a list")
        if not isinstance(universe_raw, (list, tuple)):
            raise NightShiftError("universe must be a list")
        evidence = tuple(EvidenceReference.from_mapping(x) for x in evidence_raw)
        return cls(
            candidate_id=raw["candidate_id"], created_at=raw["created_at"],
            title=raw["title"], hypothesis=raw["hypothesis"],
            mechanism=raw["mechanism"], signal_family=raw["signal_family"],
            universe=tuple(universe_raw),
            lookback_sessions=raw["lookback_sessions"],
            skip_sessions=raw["skip_sessions"],
            holding_sessions=raw["holding_sessions"],
            rebalance_sessions=raw["rebalance_sessions"],
            volatility_window=raw["volatility_window"],
            estimated_round_trip_cost_bps=raw["estimated_round_trip_cost_bps"],
            falsification_rule=raw["falsification_rule"], evidence=evidence,
            long_only=raw["long_only"], max_gross=raw["max_gross"],
            cash_allowed=raw["cash_allowed"],
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResearchCandidate":
        return cls.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["universe"] = list(self.universe)
        result["evidence"] = [asdict(x) for x in self.evidence]
        return result

    @property
    def proposal_hash(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class MorningScreenResult:
    candidate_id: str
    status: CandidateStatus
    reasons: tuple[str, ...]
    screened_at: str
    commit_sha: str
    proposal_hash: str
    panel_source: str
    panel_vintage: str
    panel_declared_hash: str
    panel_content_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "screened_at": self.screened_at,
            "commit_sha": self.commit_sha,
            "proposal_hash": self.proposal_hash,
            "panel_source": self.panel_source,
            "panel_vintage": self.panel_vintage,
            "panel_declared_hash": self.panel_declared_hash,
            "panel_content_hash": self.panel_content_hash,
        }


class ResearchQueue:
    """Single-writer append-only candidate queue with event-chain verification."""

    def __init__(
        self,
        path: str,
        *,
        allowed_symbols: Sequence[str] = tuple(DEFAULT_ETF_UNIVERSE),
    ) -> None:
        if isinstance(allowed_symbols, (str, bytes)):
            raise NightShiftError("allowed_symbols must be a sequence of ticker symbols")
        try:
            symbols = tuple(allowed_symbols)
        except TypeError as exc:
            raise NightShiftError("allowed_symbols must be a sequence of ticker symbols") from exc
        if not symbols or any(not isinstance(x, str) or not _SYMBOL.fullmatch(x) for x in symbols):
            raise NightShiftError("allowed_symbols must be a non-empty ticker allowlist")
        if len(set(symbols)) != len(symbols):
            raise NightShiftError("allowed_symbols contains duplicates")

        self.path = Path(path)
        self.allowed_symbols = frozenset(symbols)
        self._lock = threading.RLock()
        self._events: list[dict[str, Any]] = []
        self._candidates: dict[str, ResearchCandidate] = {}
        self._statuses: dict[str, CandidateStatus] = {}
        self._screens: dict[str, dict[str, Any]] = {}
        self._load_and_replay()

    def _validate_universe(self, candidate: ResearchCandidate) -> None:
        unknown = set(candidate.universe) - self.allowed_symbols
        if unknown:
            raise NightShiftError(f"unapproved ETF symbols: {sorted(unknown)}")

    def _validate_screen_payload(
        self, payload: Any, candidate: ResearchCandidate, event_timestamp: str
    ) -> tuple[CandidateStatus, dict[str, Any]]:
        required = {
            "candidate_id", "status", "reasons", "screened_at", "commit_sha",
            "proposal_hash", "panel_source", "panel_vintage",
            "panel_declared_hash", "panel_content_hash",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise QueueIntegrityError("morning-screen payload fields invalid")
        if payload["candidate_id"] != candidate.candidate_id:
            raise QueueIntegrityError("morning-screen candidate id mismatch")
        try:
            status = CandidateStatus(payload["status"])
        except (TypeError, ValueError) as exc:
            raise QueueIntegrityError("morning-screen status invalid") from exc
        if status not in {CandidateStatus.SCREENED_FOR_TEST, CandidateStatus.QUARANTINED}:
            raise QueueIntegrityError("invalid morning-screen transition")
        if not isinstance(payload["reasons"], list) or any(not isinstance(x, str) for x in payload["reasons"]):
            raise QueueIntegrityError("morning-screen reasons invalid")
        if (status is CandidateStatus.SCREENED_FOR_TEST) != (len(payload["reasons"]) == 0):
            raise QueueIntegrityError("morning-screen status/reasons conflict")
        if not isinstance(payload["commit_sha"], str) or not _GIT_SHA.fullmatch(payload["commit_sha"]):
            raise QueueIntegrityError("morning-screen commit SHA invalid")
        if payload["proposal_hash"] != candidate.proposal_hash:
            raise QueueIntegrityError("morning-screen proposal hash mismatch")
        if not isinstance(payload["panel_source"], str) or not isinstance(payload["panel_vintage"], str):
            raise QueueIntegrityError("morning-screen panel provenance invalid")
        for field_name in ("panel_declared_hash", "panel_content_hash"):
            value = payload[field_name]
            if not isinstance(value, str):
                raise QueueIntegrityError("morning-screen panel hash invalid")
            if value and not _SHA256.fullmatch(value):
                raise QueueIntegrityError("morning-screen panel hash malformed")
        screened = _utc(payload["screened_at"], "screened_at")
        event_ts = _utc(event_timestamp, "event.timestamp")
        if screened != event_ts:
            raise QueueIntegrityError("screen timestamp/event timestamp mismatch")
        clean = dict(payload)
        clean["status"] = status.value
        return status, clean

    def _load_and_replay(self) -> None:
        if not self.path.exists():
            return
        previous = GENESIS
        events: list[dict[str, Any]] = []
        candidates: dict[str, ResearchCandidate] = {}
        statuses: dict[str, CandidateStatus] = {}
        screens: dict[str, dict[str, Any]] = {}

        for seq, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                raise QueueIntegrityError(f"blank event at sequence {seq}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QueueIntegrityError(f"invalid JSON at sequence {seq}") from exc
            required = {"seq", "kind", "timestamp", "payload", "prev_hash", "hash"}
            if not isinstance(record, dict) or set(record) != required:
                raise QueueIntegrityError(f"event fields invalid at sequence {seq}")
            if (isinstance(record["seq"], bool) or not isinstance(record["seq"], int)
                    or record["seq"] != seq or record["prev_hash"] != previous):
                raise QueueIntegrityError(f"event chain sequence/link error at {seq}")
            body = {k: record[k] for k in ("seq", "kind", "timestamp", "payload", "prev_hash")}
            try:
                expected = _digest(body)
                _utc(record["timestamp"], "event.timestamp")
            except (TypeError, ValueError, NightShiftError) as exc:
                raise QueueIntegrityError(f"invalid event at sequence {seq}") from exc
            if record["hash"] != expected:
                raise QueueIntegrityError(f"event hash mismatch at sequence {seq}")

            payload = record["payload"]
            if not isinstance(payload, dict):
                raise QueueIntegrityError(f"event payload invalid at {seq}")
            if record["kind"] == "PROPOSAL":
                if set(payload) != {"candidate"}:
                    raise QueueIntegrityError(f"proposal payload fields invalid at {seq}")
                try:
                    candidate = ResearchCandidate.from_dict(payload["candidate"])
                    cid = candidate.candidate_id
                    if cid in candidates:
                        raise QueueIntegrityError(f"duplicate proposal event for {cid}")
                    self._validate_universe(candidate)
                    if _utc(candidate.created_at, "candidate.created_at") > _utc(record["timestamp"], "event.timestamp"):
                        raise QueueIntegrityError("proposal event predates candidate creation")
                except (KeyError, TypeError, ValueError, NightShiftError) as exc:
                    if isinstance(exc, QueueIntegrityError):
                        raise
                    raise QueueIntegrityError(f"invalid proposal at sequence {seq}") from exc
                candidates[cid] = candidate
                statuses[cid] = CandidateStatus.PROPOSED
            elif record["kind"] == "MORNING_SCREEN":
                cid = payload.get("candidate_id")
                if cid not in candidates or statuses.get(cid) is not CandidateStatus.PROPOSED:
                    raise QueueIntegrityError(f"invalid screen transition for {cid}")
                status, clean = self._validate_screen_payload(payload, candidates[cid], record["timestamp"])
                screens[cid] = clean
                statuses[cid] = status
            else:
                raise QueueIntegrityError(f"unknown event kind at sequence {seq}")

            events.append(record)
            previous = record["hash"]

        self._events, self._candidates = events, candidates
        self._statuses, self._screens = statuses, screens

    def _disk_chain_state(self) -> tuple[int, str]:
        """Count and head hash of the on-disk chain (read under the write lock)."""
        if not self.path.exists():
            return 0, GENESIS
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return 0, GENESIS
        try:
            last = json.loads(lines[-1])
            head = last["hash"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise QueueIntegrityError("on-disk queue tail is unreadable") from exc
        return len(lines), head

    def _append(self, kind: str, payload: dict[str, Any], now: datetime) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        previous = self._events[-1]["hash"] if self._events else GENESIS
        body = {
            "seq": len(self._events), "kind": kind, "timestamp": _iso(now),
            "payload": payload, "prev_hash": previous,
        }
        record = {**body, "hash": _digest(body)}
        encoded = (_canonical(record) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
            # Fail closed if another writer advanced (or truncated) the chain
            # since this instance loaded it. Prevents multi-process chain forks.
            n_disk, head_disk = self._disk_chain_state()
            if n_disk != len(self._events) or head_disk != previous:
                raise QueueIntegrityError(
                    "stale queue writer: on-disk chain diverged; reload before appending"
                )
            offset = 0
            while offset < len(encoded):
                written = os.write(fd, encoded[offset:])
                if written <= 0:
                    raise OSError("short write while appending research event")
                offset += written
            os.fsync(fd)
        finally:
            if _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            os.close(fd)
        self._events.append(record)

    def add_proposal(
        self,
        proposal: ResearchCandidate | Mapping[str, Any],
        *,
        now_utc: str | None = None,
    ) -> bool:
        """Append one proposal. Identical duplicate IDs are idempotent."""
        candidate = proposal if isinstance(proposal, ResearchCandidate) else ResearchCandidate.from_mapping(proposal)
        self._validate_universe(candidate)
        now = _utc(now_utc, "now_utc") if now_utc is not None else datetime.now(timezone.utc)
        if _utc(candidate.created_at, "created_at") > now:
            raise NightShiftError("proposal creation time is in the future")

        with self._lock:
            existing = self._candidates.get(candidate.candidate_id)
            if existing is not None:
                if existing.proposal_hash == candidate.proposal_hash:
                    return False
                raise NightShiftError("candidate_id already exists with different content")
            self._append("PROPOSAL", {"candidate": candidate.to_dict()}, now)
            self._candidates[candidate.candidate_id] = candidate
            self._statuses[candidate.candidate_id] = CandidateStatus.PROPOSED
            return True

    def screen_morning(
        self,
        candidate_id: str,
        panel: PricePanel,
        *,
        commit_sha: str,
        now_utc: str,
        max_candidate_age_hours: float = 24.0,
        max_evidence_age_days: float = 90.0,
        max_panel_age_days: float = 7.0,
        min_extra_sessions: int = 40,
    ) -> MorningScreenResult:
        """Revalidate a proposal and panel before separately preregistered tests.

        A pass means only SCREENED_FOR_TEST. ``panel.pit`` is trusted as an
        upstream attestation; the actual vendor's PIT provenance is not proven
        by this function. The panel content hash is recomputed and compared.
        """
        now = _utc(now_utc, "now_utc")
        if not isinstance(commit_sha, str) or not _GIT_SHA.fullmatch(commit_sha):
            raise NightShiftError("commit_sha must be a 7-64 character hexadecimal Git SHA")
        for value, field_name in (
            (max_candidate_age_hours, "max_candidate_age_hours"),
            (max_evidence_age_days, "max_evidence_age_days"),
            (max_panel_age_days, "max_panel_age_days"),
        ):
            number = _finite_number(value, field_name)
            if number <= 0:
                raise NightShiftError(f"{field_name} must be positive and finite")
        _positive_int(min_extra_sessions, "min_extra_sessions", 0)
        if not isinstance(candidate_id, str) or not candidate_id:
            raise NightShiftError("candidate_id must be a non-empty string")

        with self._lock:
            if candidate_id not in self._candidates:
                raise NightShiftError(f"unknown candidate_id: {candidate_id}")
            if self._statuses[candidate_id] is not CandidateStatus.PROPOSED:
                raise NightShiftError("candidate has already received a morning screen")
            candidate = self._candidates[candidate_id]
            reasons: list[str] = []

            created = _utc(candidate.created_at, "candidate.created_at")
            age = now - created
            if age < timedelta(0):
                reasons.append("PROPOSAL_FROM_FUTURE")
            elif age > timedelta(hours=float(max_candidate_age_hours)):
                reasons.append("PROPOSAL_STALE")

            for evidence in candidate.evidence:
                known = _utc(evidence.known_at, "evidence.known_at")
                if known > now:
                    reasons.append("EVIDENCE_FROM_FUTURE")
                    break
                if now - known > timedelta(days=float(max_evidence_age_days)):
                    reasons.append("EVIDENCE_STALE")
                    break

            panel_source = panel_vintage = ""
            panel_declared_hash = panel_hash = ""
            if not isinstance(panel, PricePanel):
                reasons.append("PANEL_TYPE_INVALID")
            else:
                panel_source = panel.source if isinstance(panel.source, str) else ""
                panel_vintage = panel.vintage if isinstance(panel.vintage, str) else ""
                panel_declared_hash = panel.content_hash if isinstance(panel.content_hash, str) else ""

                if panel.pit is not True:
                    reasons.append("PANEL_NOT_POINT_IN_TIME")
                if not panel_source or not panel_vintage:
                    reasons.append("PANEL_PROVENANCE_MISSING")

                close = panel.close
                if not isinstance(close, pd.DataFrame) or close.empty:
                    reasons.append("PANEL_CLOSE_INVALID")
                else:
                    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
                        reasons.append("PANEL_INDEX_INVALID")
                    if not isinstance(close.index, pd.DatetimeIndex) or close.index.tz is None:
                        reasons.append("PANEL_INDEX_MUST_BE_TIMEZONE_AWARE")
                    else:
                        panel_end = close.index[-1].to_pydatetime().astimezone(timezone.utc)
                        if panel_end > now:
                            reasons.append("PANEL_CONTAINS_FUTURE_ROWS")
                        elif now - panel_end > timedelta(days=float(max_panel_age_days)):
                            reasons.append("PANEL_STALE")

                    columns = [str(c) for c in close.columns]
                    if len(set(columns)) != len(columns):
                        reasons.append("PANEL_SYMBOL_COLUMNS_DUPLICATED")
                    missing = set(candidate.universe) - set(columns)
                    if missing:
                        reasons.append("UNIVERSE_SYMBOLS_MISSING:" + ",".join(sorted(missing)))

                    required_rows = (
                        max(candidate.lookback_sessions + candidate.skip_sessions,
                            candidate.volatility_window)
                        + candidate.holding_sessions + min_extra_sessions
                    )
                    if len(close) < required_rows:
                        reasons.append("INSUFFICIENT_PANEL_HISTORY")

                    try:
                        panel_hash = panel_content_hash(close)
                        if not _SHA256.fullmatch(panel_declared_hash):
                            reasons.append("PANEL_CONTENT_HASH_MISSING_OR_INVALID")
                        elif panel_hash != panel_declared_hash:
                            reasons.append("PANEL_CONTENT_HASH_MISMATCH")
                    except Exception:
                        panel_hash = ""
                        reasons.append("PANEL_CONTENT_HASH_COMPUTE_FAILED")

            status = CandidateStatus.SCREENED_FOR_TEST if not reasons else CandidateStatus.QUARANTINED
            screened_at = _iso(now)
            payload = {
                "candidate_id": candidate_id,
                "status": status.value,
                "reasons": reasons,
                "screened_at": screened_at,
                "commit_sha": commit_sha.lower(),
                "proposal_hash": candidate.proposal_hash,
                "panel_source": panel_source,
                "panel_vintage": panel_vintage,
                "panel_declared_hash": panel_declared_hash if _SHA256.fullmatch(panel_declared_hash) else "",
                "panel_content_hash": panel_hash if _SHA256.fullmatch(panel_hash) else "",
            }
            self._append("MORNING_SCREEN", payload, now)
            self._statuses[candidate_id] = status
            self._screens[candidate_id] = payload
            return MorningScreenResult(
                candidate_id=candidate_id, status=status, reasons=tuple(reasons),
                screened_at=screened_at, commit_sha=commit_sha.lower(),
                proposal_hash=candidate.proposal_hash, panel_source=panel_source,
                panel_vintage=panel_vintage,
                panel_declared_hash=payload["panel_declared_hash"],
                panel_content_hash=payload["panel_content_hash"],
            )

    def get_status(self, candidate_id: str) -> CandidateStatus:
        try:
            return self._statuses[candidate_id]
        except KeyError as exc:
            raise NightShiftError(f"unknown candidate_id: {candidate_id}") from exc

    def get_candidate(self, candidate_id: str) -> ResearchCandidate:
        try:
            return self._candidates[candidate_id]
        except KeyError as exc:
            raise NightShiftError(f"unknown candidate_id: {candidate_id}") from exc

    def get_screen(self, candidate_id: str) -> dict[str, Any] | None:
        item = self._screens.get(candidate_id)
        return None if item is None else json.loads(_canonical(item))

    def list_candidates(self, status: CandidateStatus | None = None) -> list[ResearchCandidate]:
        return [
            candidate for cid, candidate in self._candidates.items()
            if status is None or self._statuses[cid] is status
        ]

    def verify(self) -> bool:
        """Re-read and replay the persisted chain without mutating this object."""
        other = ResearchQueue(str(self.path), allowed_symbols=tuple(self.allowed_symbols))
        return other._events == self._events

    @property
    def event_count(self) -> int:
        return len(self._events)
