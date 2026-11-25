"""Freeze a price panel with a verifiable, order-invariant content hash.

Writes the canonical close matrix plus a manifest carrying the aggregate hash
and per-symbol hashes, then RELOADS and verifies bit-exactness before
returning. Per-symbol hashes let a later revision be attributed to specific
symbols instead of only flagged in aggregate.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import (PricePanel, panel_from_frame, content_hash,
                     column_hashes, DataError)


def freeze_panel(panel: PricePanel, out_dir: str, name: str) -> dict:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    csv_path = d / f"{name}.csv"
    man_path = d / f"{name}.manifest.json"

    panel.close.to_csv(csv_path, float_format="%.17g")
    manifest = {
        "name": name,
        "content_hash": panel.content_hash or content_hash(panel.close),
        "column_hashes": column_hashes(panel.close),
        "column_order": [str(c) for c in panel.close.columns],
        "source": panel.source, "vintage": panel.vintage,
        "pit": panel.pit, "label": panel.label,
        "n_rows": int(panel.close.shape[0]),
        "n_symbols": int(panel.close.shape[1]),
        "symbols": sorted(str(c) for c in panel.close.columns),
        "start": str(panel.close.index[0]), "end": str(panel.close.index[-1]),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "csv_float_format": "%.17g",
        "hash_scheme": "sha256(index_iso + sorted_cols + float64_le); columns sorted",
    }
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    back = load_frozen(out_dir, name)
    if back.content_hash != manifest["content_hash"]:
        raise DataError("freeze round-trip changed content hash")
    return manifest


def load_frozen(out_dir: str, name: str) -> PricePanel:
    d = Path(out_dir)
    man_path = d / f"{name}.manifest.json"
    csv_path = d / f"{name}.csv"
    if not man_path.exists() or not csv_path.exists():
        raise DataError(f"frozen artifact {name} not found in {out_dir}")
    man = json.loads(man_path.read_text())
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True,
                     float_precision="round_trip")
    panel = panel_from_frame(df, source=man["source"], vintage=man["vintage"],
                             pit=bool(man["pit"]), label=man["label"])
    if panel.content_hash != man["content_hash"]:
        raise DataError("frozen artifact hash mismatch - file was altered")
    return panel
