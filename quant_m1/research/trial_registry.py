"""
Durable append-only trial registry.

Every evaluated configuration must be logged, including failures.
DSR uses registry count, not survivor count.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List


def canonical_config_hash(config: dict) -> str:
    s = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrialRecord:
    config_hash: str
    family: str
    seed: int
    data_vintage: str
    registered_at: str
    metric_name: str
    metric_value: float
    horizon: str
    gross_net: str

    def __post_init__(self) -> None:
        if not self.config_hash or not self.family:
            raise ValueError("config_hash/family required")
        if self.gross_net not in {"gross", "net"}:
            raise ValueError("gross_net must be gross or net")
        if self.metric_name.lower() in {"return", "sharpe", "cagr"} and not self.horizon:
            raise ValueError("horizon required for return-like metric")


class TrialRegistry:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            open(path, "w").close()

    def log(self, record: TrialRecord) -> None:
        line = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> List[TrialRecord]:
        out: List[TrialRecord] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(TrialRecord(**json.loads(line)))
        return out

    def n_trials(self) -> int:
        return len(self.read_all())

    def by_family(self, family: str) -> List[TrialRecord]:
        return [r for r in self.read_all() if r.family == family]

    def sigma_sr(self) -> float:
        vals = [r.metric_value for r in self.read_all() if r.metric_name.lower() == "sharpe"]
        if len(vals) < 2:
            return 1.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        sd = var ** 0.5
        return sd if sd > 0 else 1.0

    def effective_trials(self) -> int:
        """Unique config hashes. Correlated configs require return-correlation analysis."""
        return len({r.config_hash for r in self.read_all()})
