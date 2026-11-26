"""Multi-writer queue safety: stale writer and truncation fail closed."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from quant_m1.research.night_shift import QueueIntegrityError, ResearchQueue

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


def _proposal(cid):
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


def test_stale_second_writer_is_rejected(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    q1 = ResearchQueue(path)
    q2 = ResearchQueue(path)
    assert q1.add_proposal(_proposal("mom-a"), now_utc=NOW.isoformat()) is True
    with pytest.raises(QueueIntegrityError, match="stale"):
        q2.add_proposal(_proposal("mom-b"), now_utc=NOW.isoformat())
    q3 = ResearchQueue(path)
    assert q3.add_proposal(_proposal("mom-b"), now_utc=NOW.isoformat()) is True


def test_truncated_chain_blocks_append(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    q = ResearchQueue(path)
    q.add_proposal(_proposal("mom-a"), now_utc=NOW.isoformat())
    q.add_proposal(_proposal("mom-b"), now_utc=NOW.isoformat())
    lines = open(path).read().splitlines()
    open(path, "w").write(lines[0] + "\n")
    with pytest.raises(QueueIntegrityError):
        q.add_proposal(_proposal("mom-c"), now_utc=NOW.isoformat())
