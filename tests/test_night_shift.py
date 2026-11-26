from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quant_m1.data.loader import panel_from_frame
from quant_m1.research.night_shift import (
    CandidateStatus,
    NightShiftError,
    QueueIntegrityError,
    ResearchCandidate,
    ResearchQueue,
)

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


def _proposal(candidate_id="mom-21-5", created_at=None):
    created_at = created_at or (NOW - timedelta(hours=2)).isoformat()
    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    evidence_dt = created_dt - timedelta(hours=1)
    return {
        "candidate_id": candidate_id,
        "created_at": created_at,
        "title": "ETF trend persistence hypothesis",
        "hypothesis": "Positive trailing ETF trends persist over the next five sessions.",
        "mechanism": "Slow investor underreaction may create gradual price adjustment.",
        "signal_family": "time_series_momentum",
        "universe": ["SPY", "QQQ", "TLT"],
        "lookback_sessions": 21,
        "skip_sessions": 1,
        "holding_sessions": 5,
        "rebalance_sessions": 5,
        "volatility_window": 21,
        "estimated_round_trip_cost_bps": 8.0,
        "falsification_rule": "Reject when after-cost OOS returns fail the preregistered baseline test.",
        "evidence": [{
            "source": "internal_research_note",
            "raw_ref": "sha256:research-note-001",
            "release_ts": evidence_dt.isoformat(),
            "known_at": evidence_dt.isoformat(),
            "vintage_id": "note-v1",
        }],
        "long_only": True,
        "max_gross": 1.0,
        "cash_allowed": True,
    }


def _panel(pit=True, n=160, end=None):
    rng = np.random.default_rng(4)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 3)), axis=0))
    idx = pd.bdate_range(end=end or "2026-11-25", periods=n, tz="UTC")
    frame = pd.DataFrame(px, index=idx, columns=["SPY", "QQQ", "TLT"])
    return panel_from_frame(
        frame,
        source="fixture_feed",
        vintage="fixture-v1",
        pit=pit,
        label="TEST FIXTURE - no performance claim",
    )


def test_candidate_schema_and_hash_are_deterministic():
    a = ResearchCandidate.from_mapping(_proposal())
    b = ResearchCandidate.from_mapping(_proposal())
    assert a.proposal_hash == b.proposal_hash
    assert a.universe == ("SPY", "QQQ", "TLT")


def test_unknown_candidate_fields_are_rejected():
    raw = _proposal()
    raw["target_weights"] = {"SPY": 1.0}
    with pytest.raises(NightShiftError, match="extra"):
        ResearchCandidate.from_mapping(raw)


def test_non_long_only_candidate_is_rejected():
    raw = _proposal()
    raw["long_only"] = False
    with pytest.raises(NightShiftError, match="long-only"):
        ResearchCandidate.from_mapping(raw)


def test_leveraged_candidate_is_rejected():
    raw = _proposal()
    raw["max_gross"] = 1.1
    with pytest.raises(NightShiftError, match="leverage"):
        ResearchCandidate.from_mapping(raw)


def test_future_evidence_at_proposal_creation_is_rejected():
    raw = _proposal()
    raw["evidence"][0]["known_at"] = (NOW + timedelta(minutes=1)).isoformat()
    with pytest.raises(NightShiftError, match="not yet known"):
        ResearchCandidate.from_mapping(raw)


def test_naive_timestamps_are_rejected():
    raw = _proposal(created_at="2026-11-26T08:00:00")
    with pytest.raises(NightShiftError, match="timezone"):
        ResearchCandidate.from_mapping(raw)


def test_unapproved_symbol_rejected(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    raw = _proposal()
    raw["universe"] = ["SPY", "NVDA"]
    with pytest.raises(NightShiftError, match="unapproved"):
        q.add_proposal(raw, now_utc=NOW.isoformat())


def test_proposal_append_is_persistent_and_idempotent(tmp_path):
    path = tmp_path / "queue.jsonl"
    q = ResearchQueue(str(path))
    raw = _proposal()
    assert q.add_proposal(raw, now_utc=NOW.isoformat()) is True
    assert q.add_proposal(raw, now_utc=NOW.isoformat()) is False
    q2 = ResearchQueue(str(path))
    assert q2.event_count == 1
    assert q2.get_status("mom-21-5") is CandidateStatus.PROPOSED
    assert q2.verify()


def test_same_id_different_content_rejected(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    altered = _proposal()
    altered["lookback_sessions"] = 63
    with pytest.raises(NightShiftError, match="different content"):
        q.add_proposal(altered, now_utc=NOW.isoformat())


def test_valid_pit_content_verified_panel_passes_screen_but_not_model_validation(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    result = q.screen_morning(
        "mom-21-5", _panel(pit=True),
        commit_sha="a" * 40,
        now_utc=NOW.isoformat(),
    )
    assert result.status is CandidateStatus.SCREENED_FOR_TEST
    assert result.reasons == ()
    assert q.get_status("mom-21-5") is CandidateStatus.SCREENED_FOR_TEST
    # This queue's pass only permits the separate pre-registration stage.
    assert "validated" not in result.status.value.lower()


def test_non_pit_panel_is_quarantined(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    result = q.screen_morning(
        "mom-21-5", _panel(pit=False),
        commit_sha="a" * 40,
        now_utc=NOW.isoformat(),
    )
    assert result.status is CandidateStatus.QUARANTINED
    assert "PANEL_NOT_POINT_IN_TIME" in result.reasons


def test_stale_candidate_is_quarantined(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    old = _proposal(created_at=(NOW - timedelta(days=2)).isoformat())
    q.add_proposal(old, now_utc=NOW.isoformat())
    result = q.screen_morning(
        "mom-21-5", _panel(),
        commit_sha="a" * 40,
        now_utc=NOW.isoformat(),
        max_candidate_age_hours=24,
    )
    assert result.status is CandidateStatus.QUARANTINED
    assert "PROPOSAL_STALE" in result.reasons


def test_missing_universe_symbol_quarantines_candidate(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    p = _panel()
    df = p.close.drop(columns=["QQQ"])
    p2 = panel_from_frame(
        df, source=p.source, vintage=p.vintage,
        pit=True, label=p.label,
    )
    result = q.screen_morning(
        "mom-21-5", p2, commit_sha="a" * 40,
        now_utc=NOW.isoformat(),
    )
    assert result.status is CandidateStatus.QUARANTINED
    assert any(x.startswith("UNIVERSE_SYMBOLS_MISSING") for x in result.reasons)


def test_bad_commit_sha_rejected(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    with pytest.raises(NightShiftError, match="commit_sha"):
        q.screen_morning(
            "mom-21-5", _panel(), commit_sha="not-a-sha",
            now_utc=NOW.isoformat(),
        )


def test_future_panel_rows_quarantined(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    future = _panel(end="2026-12-20")
    result = q.screen_morning(
        "mom-21-5", future, commit_sha="a" * 40,
        now_utc=NOW.isoformat(),
    )
    assert result.status is CandidateStatus.QUARANTINED
    assert "PANEL_CONTAINS_FUTURE_ROWS" in result.reasons


def test_screen_cannot_be_repeated_or_overwritten(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    q.screen_morning(
        "mom-21-5", _panel(), commit_sha="a" * 40,
        now_utc=NOW.isoformat(),
    )
    with pytest.raises(NightShiftError, match="already received"):
        q.screen_morning(
            "mom-21-5", _panel(), commit_sha="b" * 40,
            now_utc=NOW.isoformat(),
        )


def test_event_tampering_is_detected(tmp_path):
    path = tmp_path / "queue.jsonl"
    q = ResearchQueue(str(path))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["payload"]["candidate"]["title"] = "tampered title"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(QueueIntegrityError, match="hash mismatch"):
        ResearchQueue(str(path))


def test_candidate_creation_cannot_be_future_dated(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    future = _proposal(created_at=(NOW + timedelta(hours=1)).isoformat())
    with pytest.raises(NightShiftError, match="future"):
        q.add_proposal(future, now_utc=NOW.isoformat())


def test_queue_rejects_non_json_nan_during_candidate_parse():
    raw = _proposal()
    raw["estimated_round_trip_cost_bps"] = float("nan")
    with pytest.raises(NightShiftError, match="finite"):
        ResearchCandidate.from_mapping(raw)


def test_mutated_panel_after_hashing_is_quarantined(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    p = _panel()
    p.close.iloc[-1, 0] *= 1.01
    result = q.screen_morning(
        "mom-21-5", p, commit_sha="a" * 40,
        now_utc=NOW.isoformat(),
    )
    assert result.status is CandidateStatus.QUARANTINED
    assert "PANEL_CONTENT_HASH_MISMATCH" in result.reasons


def test_stale_panel_is_quarantined(tmp_path):
    q = ResearchQueue(str(tmp_path / "queue.jsonl"))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    old_panel = _panel(end="2026-11-10")
    result = q.screen_morning(
        "mom-21-5", old_panel, commit_sha="a" * 40,
        now_utc=NOW.isoformat(),
        max_panel_age_days=5,
    )
    assert result.status is CandidateStatus.QUARANTINED
    assert "PANEL_STALE" in result.reasons


def test_unhashable_signal_family_is_rejected_cleanly():
    raw = _proposal()
    raw["signal_family"] = ["time_series_momentum"]
    with pytest.raises(NightShiftError, match="signal_family"):
        ResearchCandidate.from_mapping(raw)


def test_allowed_symbols_requires_valid_sequence(tmp_path):
    with pytest.raises(NightShiftError, match="sequence"):
        ResearchQueue(str(tmp_path / "q.jsonl"), allowed_symbols="SPY")


def test_screen_event_replay_verifies_proposal_hash(tmp_path):
    path = tmp_path / "queue.jsonl"
    q = ResearchQueue(str(path))
    q.add_proposal(_proposal(), now_utc=NOW.isoformat())
    q.screen_morning(
        "mom-21-5", _panel(), commit_sha="a" * 40,
        now_utc=NOW.isoformat(),
    )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[1]["payload"]["proposal_hash"] = "0" * 64
    from hashlib import sha256
    body = {
        k: events[1][k]
        for k in ("seq", "kind", "timestamp", "payload", "prev_hash")
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    events[1]["hash"] = sha256(canonical.encode("utf-8")).hexdigest()
    path.write_text(
        "\n".join(json.dumps(e, sort_keys=True, separators=(",", ":")) for e in events)
        + "\n"
    )
    with pytest.raises(QueueIntegrityError, match="proposal hash mismatch"):
        ResearchQueue(str(path))