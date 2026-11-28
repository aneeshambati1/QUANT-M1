"""Carry signal family. HONEST STUB.

The frozen panel evidence/panel_real_v1.csv is PRICE-ONLY: it contains no bond
yields, dividend yields, or funding rates. True cross-asset carry therefore
cannot be computed from it and is NOT fabricated here. This module reports the
family as unavailable on a price-only panel and provides an adapter that works
only when real yield data is supplied.

This is a data limitation, not a code shortcut. Feeding real carry requires the
free/paid data work (yields per asset), which is a separate step.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np


def carry_signal(yields: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Return a carry signal if real yields are provided, else NOT_AVAILABLE."""
    if yields is None:
        return {
            "available": False,
            "reason": "price-only panel lacks yield data; carry requires real yields",
            "signal": None,
        }
    y = np.asarray(yields, dtype=float)
    if not np.all(np.isfinite(y)):
        return {"available": False, "reason": "non-finite yields", "signal": None}
    # Long high yield, underweight low yield (rank, placeholder magnitude).
    order = y.argsort()
    n = len(y)
    sig = np.zeros(n)
    if n >= 2:
        sig[order[-max(1, n // 5):]] = 0.001
        sig[order[:max(1, n // 5)]] = -0.001
    return {"available": True, "reason": "real yields supplied", "signal": sig}
