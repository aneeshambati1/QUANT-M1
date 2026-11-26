"""AI advisory: injected client, schema validation, no authority, offline."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from quant_m1.research.ai_advisory import propose, submit, adversarial_review
from quant_m1.research.night_shift import ResearchQueue

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


def _candidate_dict(cid="mom-a"):
    created = (NOW - timedelta(hours=2)).isoformat()
    ev = (NOW - timedelta(hours=3)).isoformat()
    return {
        "candidate_id": cid, "created_at": created,
        "title": "ETF trend persistence hypothesis",
        "hypothesis": "Positive trailing ETF trends persist over the next five sessions.",
        "mechanism": "Slow investor underreaction may create gradual price adjustment.",
        "signal_family": "time_series_momentum", "universe": ["SPY", "QQQ", "TLT"],
        "lookback_sessions": 21, "skip_sessions": 1, "holding_sessions": 5,
        "rebalance_sessions": 5, "volatility_window": 21,
        "estimated_round_trip_cost_bps": 8.0,
        "falsification_rule": "Reject when after-cost OOS returns fail the preregistered baseline test.",
        "evidence": [{"source": "internal_research_note", "raw_ref": "sha256:research-note-001",
                      "release_ts": ev, "known_at": ev, "vintage_id": "note-v1"}],
        "long_only": True, "max_gross": 1.0, "cash_allowed": True,
    }


def _client(payload):
    def c(system, user):
        return json.dumps(payload)
    return c


def test_valid_proposal_parses():
    res = propose(_client([_candidate_dict()]), "context")
    assert len(res.candidates) == 1 and not res.errors


def test_invalid_json_rejected():
    res = propose(lambda s, u: "not json", "context")
    assert not res.candidates and res.errors


def test_non_array_rejected():
    res = propose(_client({"candidate_id": "x"}), "context")
    assert not res.candidates and res.errors


def test_schema_violation_rejected():
    bad = _candidate_dict(); bad["max_gross"] = 2.0  # leverage prohibited
    res = propose(_client([bad]), "context")
    assert not res.candidates and res.errors


def test_submit_to_queue_and_adversary_is_advisory(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    res = propose(_client([_candidate_dict()]), "context")
    out = submit(q, res, now_utc=NOW.isoformat())
    assert out["added"] == 1
    critique = adversarial_review(_client({"lookahead_risk": True, "notes": "check PIT"}),
                                  res.candidates[0])
    assert critique["review_available"] is True
    # Advisory critique must NOT change queue status.
    assert q.get_status("mom-a").value == "PROPOSED"


def test_malformed_adversary_conservative():
    res = propose(_client([_candidate_dict()]), "context")
    critique = adversarial_review(lambda s, u: "garbage", res.candidates[0])
    assert critique["review_available"] is False
