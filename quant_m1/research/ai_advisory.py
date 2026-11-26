"""Injected-client AI proposer / adversary / methodologist. ADVISORY ONLY.

This module orchestrates LLM calls OUTSIDE the safety kernel. The client is an
injected callable (system, user) -> str returning JSON text; there is NO network
dependency here, so it is offline-testable and the kernel stays deterministic.

Hard rules (the constitution):
  - AI output is untrusted input. It never sets weights or grants execution.
  - Proposals become ResearchCandidate objects validated by the existing
    night_shift schema, then enter the ResearchQueue (which re-validates and
    hash-chains them). Promotion still requires pre-registered after-cost trials.
  - Every prompt/response/model is logged to an append-only audit file if a path
    is given. Evidence must come from data, not the model's claims.
  - No eval/exec of model text. Unknown fields are rejected by the schema.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from .night_shift import NightShiftError, ResearchCandidate, ResearchQueue

# An injected client: (system_prompt, user_prompt) -> JSON text.
Client = Callable[[str, str], str]


@dataclass
class AdvisoryResult:
    candidates: list
    errors: list
    raw_responses: list

    def as_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "errors": list(self.errors),
            "n_raw": len(self.raw_responses),
        }


def _audit(path: Optional[str], record: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def propose(
    client: Client,
    context: str,
    *,
    model: str = "unspecified",
    max_candidates: int = 5,
    audit_path: Optional[str] = None,
) -> AdvisoryResult:
    """Ask the model for falsifiable ETF hypotheses; return validated candidates.

    The model MUST return a JSON array of ResearchCandidate objects. Invalid
    entries are rejected (collected in errors), never silently accepted. This
    function does NOT add to the queue; call submit() for that.
    """
    system = (
        "You are a quant research proposer. Return ONLY a JSON array of "
        "falsifiable, long-only, unleveraged ETF hypothesis objects matching the "
        "ResearchCandidate schema. No prose, no markdown, no weights, no orders. "
        "Each must include evidence with release_ts and known_at."
    )
    user = f"Context:\n{context}\n\nReturn at most {max_candidates} candidates as a JSON array."
    raw = client(system, user)
    _audit(audit_path, {"ts": _now(), "role": "proposer", "model": model,
                        "system": system, "user": user, "response": raw})

    candidates, errors = [], []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return AdvisoryResult([], [f"INVALID_JSON:{type(exc).__name__}"], [raw])
    if not isinstance(parsed, list):
        return AdvisoryResult([], ["NOT_A_JSON_ARRAY"], [raw])

    for i, item in enumerate(parsed[:max_candidates]):
        try:
            candidates.append(ResearchCandidate.from_mapping(item))
        except (NightShiftError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"candidate[{i}]:{type(exc).__name__}:{exc}")
    return AdvisoryResult(candidates, errors, [raw])


def submit(queue: ResearchQueue, result: AdvisoryResult, *, now_utc: str) -> dict:
    """Add validated candidates to the governed queue (which re-validates)."""
    added, dup, rejected = 0, 0, []
    for cand in result.candidates:
        try:
            if queue.add_proposal(cand, now_utc=now_utc):
                added += 1
            else:
                dup += 1
        except NightShiftError as exc:
            rejected.append(f"{cand.candidate_id}:{exc}")
    return {"added": added, "duplicate": dup, "rejected": rejected}


def adversarial_review(
    client: Client,
    candidate: ResearchCandidate,
    *,
    model: str = "unspecified",
    audit_path: Optional[str] = None,
) -> dict:
    """Ask the model to ATTACK the candidate. Advisory critique only.

    Returns a dict of flags. It NEVER grants authority or modifies the queue.
    A malformed response yields a conservative 'review_unavailable' flag.
    """
    system = (
        "You are a quant research adversary. Attack the hypothesis for lookahead "
        "bias, survivorship, cost realism, regime dependence, and multiple "
        "testing. Return ONLY a JSON object with boolean flags and a 'notes' "
        "string. Do not approve or promote anything."
    )
    user = json.dumps(candidate.to_dict(), sort_keys=True)
    raw = client(system, user)
    _audit(audit_path, {"ts": _now(), "role": "adversary", "model": model,
                        "system": system, "user": user, "response": raw})
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {"review_available": True, **parsed}
    except (json.JSONDecodeError, TypeError):
        pass
    return {"review_available": False, "notes": "review_unavailable"}


def methodologist_suggestions(
    client: Client,
    protocol_summary: str,
    *,
    model: str = "unspecified",
    audit_path: Optional[str] = None,
) -> dict:
    """Suggest ablations/baselines/protocol improvements. Advisory only.

    Suggestions require owner approval and pre-registration before any use.
    """
    system = (
        "You are a quant methodologist. Suggest ablations, baselines, and "
        "protocol improvements as a JSON object with a 'suggestions' array of "
        "strings. Do not change any governance rule yourself."
    )
    raw = client(system, protocol_summary)
    _audit(audit_path, {"ts": _now(), "role": "methodologist", "model": model,
                        "system": system, "user": protocol_summary, "response": raw})
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("suggestions"), list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {"suggestions": []}
