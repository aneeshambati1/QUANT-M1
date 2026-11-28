"""Price-based signal families: momentum and reversal (rank, no lookahead).

RESEARCH ONLY. Non-PIT panel; no confirmatory alpha claim.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def rank_signal(close: pd.DataFrame, i: int, lookback: int, contrarian: bool) -> np.ndarray:
    """Cross-sectional rank signal using returns over [i-lookback, i].
    
    Uses absolute rank instead of pct_rank to handle small universes robustly.
    Long the top-k winners (momentum) or top-k losers (reversal), where k = max(1, n//5).
    """
    n = len(close.columns)
    if i < lookback + 1:
        return np.zeros(n)
    ret = close.iloc[i] / close.iloc[i - lookback] - 1.0
    ranks = ret.rank()  # absolute rank, not pct
    k = max(1, n // 5)  # top-k by absolute rank
    
    if contrarian:
        # Reversal: long the losers (bottom-k ranks), short the winners (top-k ranks)
        return np.where(ranks <= k, 0.001, np.where(ranks > n - k, -0.001, 0.0))
    else:
        # Momentum: long the winners (top-k ranks), short the losers (bottom-k ranks)
        return np.where(ranks > n - k, 0.001, np.where(ranks <= k, -0.001, 0.0))


def momentum_signal(close: pd.DataFrame, i: int, lookback: int = 21) -> np.ndarray:
    """21-day momentum: long recent winners, short recent losers."""
    return rank_signal(close, i, lookback, contrarian=False)


def reversal_signal(close: pd.DataFrame, i: int, lookback: int = 5) -> np.ndarray:
    """5-day reversal: long recent losers, short recent winners."""
    return rank_signal(close, i, lookback, contrarian=True)
