"""
Covariance estimation.

Ledoit-Wolf identity target implemented explicitly.
Constant-correlation target preserves individual variance structure better.
Factor covariance for ETF factor models.
"""
from __future__ import annotations

import numpy as np


def ledoit_wolf_identity(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004) shrinkage toward mu * I.
    X: (n_samples, n_features).
    Returns (Sigma_shrunk, shrinkage_intensity).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError("need n>=2, p>=1")
    X = X - X.mean(axis=0)
    S = (X.T @ X) / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = float(np.sum((S - F) ** 2) / p)

    b2 = 0.0
    for t in range(n):
        xt = X[t][:, None]
        b2 += float(np.sum((xt @ xt.T - S) ** 2))
    b2 /= p * n * n
    b2 = min(b2, d2)
    delta = b2 / d2 if d2 > 0 else 0.0
    return delta * F + (1.0 - delta) * S, float(delta)


def ledoit_wolf_constant_correlation(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Shrink correlation toward average off-diagonal correlation.
    Preserves individual variances.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n < 2 or p < 2:
        raise ValueError("need n>=2, p>=2")
    X = X - X.mean(axis=0)
    S = (X.T @ X) / n
    var = np.diag(S).copy()
    if np.any(var <= 0):
        raise ValueError("zero variance feature")
    std = np.sqrt(var)
    corr = S / np.outer(std, std)
    off = corr[~np.eye(p, dtype=bool)]
    rho = float(np.mean(off))
    F = np.outer(std, std) * rho
    np.fill_diagonal(F, var)

    d2 = float(np.sum((S - F) ** 2) / p)
    b2 = 0.0
    for t in range(n):
        xt = X[t][:, None]
        b2 += float(np.sum((xt @ xt.T - S) ** 2))
    b2 /= p * n * n
    b2 = min(b2, d2)
    delta = b2 / d2 if d2 > 0 else 0.0
    return delta * F + (1.0 - delta) * S, float(delta)


def factor_covariance(
    returns: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    """
    OLS factor covariance: B Cov_f B' + diagonal residual variance.
    returns: T x p
    factors: T x k
    """
    R = np.asarray(returns, dtype=float)
    F = np.asarray(factors, dtype=float)
    if R.ndim != 2 or F.ndim != 2 or len(R) != len(F):
        raise ValueError("shape mismatch")
    T, p = R.shape
    k = F.shape[1]
    if T <= k + 1:
        raise ValueError("insufficient observations")

    F1 = np.column_stack([np.ones(T), F])
    beta = np.linalg.lstsq(F1, R, rcond=None)[0]
    resid = R - F1 @ beta
    cov_f = np.cov(F.T, bias=True)
    B = beta[1:]
    D = np.diag(np.var(resid, axis=0, ddof=k + 1))
    return B @ cov_f @ B.T + D
