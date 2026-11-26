"""Freeze and reload a price panel with strict manifest validation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import (
    PricePanel,
    panel_from_frame,
    content_hash,
    column_hashes,
    DataError,
)


def freeze_panel(panel: PricePanel, out_dir: str, name: str) -> dict:
    if not isinstance(panel, PricePanel):
        raise DataError("panel must be a PricePanel")
    if not isinstance(panel.pit, bool):
        raise DataError("panel.pit must be a boolean")
    if not name or Path(name).name != name:
        raise DataError("name must be a simple filename stem")

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    csv_path = d / f"{name}.csv"
    man_path = d / f"{name}.manifest.json"

    panel.close.to_csv(csv_path, float_format="%.17g")

    actual_hash = content_hash(panel.close)
    manifest = {
        "name": name,
        "content_hash": actual_hash,
        "column_hashes": column_hashes(panel.close),
        "column_order": [str(c) for c in panel.close.columns],
        "source": panel.source,
        "vintage": panel.vintage,
        "pit": panel.pit,
        "label": panel.label,
        "n_rows": int(panel.close.shape[0]),
        "n_symbols": int(panel.close.shape[1]),
        "symbols": sorted(str(c) for c in panel.close.columns),
        "start": str(panel.close.index[0]),
        "end": str(panel.close.index[-1]),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "csv_float_format": "%.17g",
        "hash_scheme": "sha256(index_iso + sorted_cols + float64_le); columns sorted",
    }
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    back = load_frozen(out_dir, name)
    if back.content_hash != actual_hash:
        raise DataError("freeze round-trip changed content hash")
    return manifest


def load_frozen(out_dir: str, name: str) -> PricePanel:
    if not name or Path(name).name != name:
        raise DataError("name must be a simple filename stem")

    d = Path(out_dir)
    man_path = d / f"{name}.manifest.json"
    csv_path = d / f"{name}.csv"

    if not man_path.exists() or not csv_path.exists():
        raise DataError(f"frozen artifact {name} not found in {out_dir}")

    try:
        man = json.loads(man_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError("invalid frozen manifest") from exc

    if not isinstance(man, dict):
        raise DataError("manifest must be a JSON object")

    required = {
        "name", "content_hash", "column_hashes", "column_order",
        "source", "vintage", "pit", "label", "n_rows", "n_symbols",
        "symbols", "start", "end",
    }
    missing = required - set(man)
    if missing:
        raise DataError(f"manifest missing fields: {sorted(missing)}")

    if man["name"] != name:
        raise DataError("manifest name mismatch")
    if not isinstance(man["pit"], bool):
        raise DataError("manifest pit must be a JSON boolean")
    if not isinstance(man["content_hash"], str) or len(man["content_hash"]) != 64:
        raise DataError("manifest content_hash invalid")

    try:
        df = pd.read_csv(
            csv_path, index_col=0, parse_dates=True,
            float_precision="round_trip",
        )
    except Exception as exc:
        raise DataError("could not read frozen price CSV") from exc

    panel = panel_from_frame(
        df,
        source=man["source"],
        vintage=man["vintage"],
        pit=man["pit"],
        label=man["label"],
    )

    if panel.content_hash != man["content_hash"]:
        raise DataError("frozen artifact hash mismatch - file was altered")

    if int(man["n_rows"]) != panel.close.shape[0]:
        raise DataError("manifest row count mismatch")
    if int(man["n_symbols"]) != panel.close.shape[1]:
        raise DataError("manifest symbol count mismatch")
    if sorted(man["symbols"]) != sorted(map(str, panel.close.columns)):
        raise DataError("manifest symbols mismatch")
    if man["column_hashes"] != column_hashes(panel.close):
        raise DataError("manifest per-symbol hash mismatch")

    return panel
