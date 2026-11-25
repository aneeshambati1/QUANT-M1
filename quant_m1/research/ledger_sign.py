"""HMAC signing for trial ledger entries.

Each entry is signed with HMAC-SHA256 using a secret key. The signature is
stored alongside the entry and verified on load. This prevents tampering
even if an attacker can write to the ledger file.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def get_key(key_path: str = ".ledger_key") -> bytes:
    """Load or generate the signing key."""
    path = Path(key_path)
    if path.exists():
        return path.read_bytes()
    # Generate a new key
    key = os.urandom(32)
    path.write_bytes(key)
    path.chmod(0o600)  # Owner read/write only
    return key

def sign_entry(entry: dict, key: bytes) -> str:
    """Sign a ledger entry and return the HMAC hex digest."""
    body = {k: entry[k] for k in ("seq", "kind", "prev_hash", "body")}
    msg = _canonical(body).encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

def verify_entry(entry: dict, key: bytes, expected_sig: str) -> bool:
    """Verify the signature on a ledger entry."""
    actual = sign_entry(entry, key)
    return hmac.compare_digest(actual, expected_sig)