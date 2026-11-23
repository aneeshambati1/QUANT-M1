"""
Tamper-evident append-only audit log.

No secret values may be logged. Secret-like keys are redacted.
Hash chain includes previous entry hash.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

GENESIS = "0" * 64
SECRET_KEYS = {
    "key", "secret", "token", "password", "api_key", "apikey",
    "private_key", "secret_key", "access_token", "refresh_token",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in SECRET_KEYS:
                out[k] = "[REDACTED]"
            else:
                out[k] = _sanitize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    return obj


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(_sanitize(payload), sort_keys=True, separators=(",", ":"), default=str)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    ts: str
    event_type: str
    payload_hash: str
    prev_hash: str
    entry_hash: str


def verify_entries(entries: Tuple[AuditEntry, ...]) -> bool:
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.seq != i:
            return False
        if e.prev_hash != prev:
            return False
        expected = _sha(f"{e.seq}|{e.ts}|{e.event_type}|{e.payload_hash}|{e.prev_hash}")
        if e.entry_hash != expected:
            return False
        prev = e.entry_hash
    return True


class AuditLog:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._lock = threading.RLock()

    def append(self, event_type: str, payload: Dict[str, Any]) -> AuditEntry:
        with self._lock:
            seq = len(self._entries)
            ts = _utc_now()
            payload_hash = _sha(_canonical(payload))
            prev = self._entries[-1].entry_hash if self._entries else GENESIS
            entry_hash = _sha(f"{seq}|{ts}|{event_type}|{payload_hash}|{prev}")
            e = AuditEntry(seq, ts, event_type, payload_hash, prev, entry_hash)
            self._entries.append(e)
            return e

    def entries(self) -> Tuple[AuditEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def verify_chain(self) -> bool:
        return verify_entries(self.entries())

    def __len__(self) -> int:
        return len(self._entries)
