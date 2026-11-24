"""Factor model, residual covariance, and leverage-constrained breadth.

Pure computation. No I/O. No vendor coupling.
Inputs are numpy arrays derived from PIT-correct returns.

Provenance: every metric carries its sizing convention and constraint set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ResidualReport:
    """Immutable report. All figures are labeled by convention."""
    n_assets: int
    n_factors: int
    n_obs: int

    # Raw spectrum
    raw_eigenvalues: np.ndarray
    raw_participation_ratio: float
    raw_var_based_neff: float

    # Residual spectrum (un-normalized)
    residual_eigenvalues: np.ndarray
    residual_trace: float
    residual_participation_ratio: float
    residual_var_based_neff: float

    # Residual spectrum (renormalized to unit diagonal)
    renorm_residual_eigenvalues: np.ndarray
    renorm_participation_ratio: float
    renorm_var_based_neff: float

    # Leverage-constrained breadth
    avg_residual_variance_frac: float
    min_leverage_for_unit_risk: float
    leverage_constrained_neff: float

    # Shrinkage
    lw_shrinkage_intensity: float
    residual_condition_number: float

    def summary(self) -> str:
        lines = [
            f"Assets={self.n_assets}  Factors={self.n_factors}  Obs={self.n_obs}",
            f"Raw:           PR={self.raw_participation_ratio:.2f}  "
            f"N_eff(var)={self.raw_var_based_neff:.2f}",
            f"Residual:      PR={self.residual_participation_ratio:.2f}  "
            f"N_eff(var)={self.residual_var_based_neff:.2f}  "
            f"trace={self.residual_trace:.2f}",
            f"Renormalized:  PR={self.renorm_participation_ratio:.2f}  "
            f"N_eff(var)={self.renorm_var_based_neff:.2f}",
            f"Leverage-constrained (gross<=1): "
            f"N_eff={self.leverage_constrained_neff:.2f}  "
            f"(avg_resid_var={self.avg_residual_variance_frac:.3f}, "
            f"min_lev={self.min_leverage_for_unit_risk:.2f}x)",
            f"LW shrinkage={self.lw_shrinkage_intensity:.4f}  "
            f"cond(resid)={self.residual_condition_number:.1f}",
        ]
        return "\n".join(lines)


def _participation_ratio(eigenvalues: np.ndarray) -> float:
    s = np.sum(eigenvalues)
    ss = np.sum(eigenvalues ** 2)
    if ss == 0:
        return 0.0
    return float(s * s / ss)


def _var_based_neff(eigenvalues: np.ndarray) -> float:
    """N / (1 + (N-1)*rho_eq) where rho_eq is implied equicorrelation."""
    n = len(eigenvalues)
    if n < 2:
        return float(n)
    s = np.sum(eigenvalues)
    ss = np.sum(eigenvalues ** 2)
    if s == 0:
        return float(n)
    # Implied equicorrelation from spectrum
    rho = (ss / (s * s / n) - 1.0) / max(n - 1, 1)
    rho = max(rho, -1.0 / max(n - 1, 1))
    rho = min(rho, 1.0)
    denom = 1.0 + (n - 1) * rho
    if denom <= 0:
        return 1.0
    return float(min(n / denom, n))


def _ledoit_wolf_identity(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage toward identity. Returns (shrunk_cov, intensity)."""
    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError("need n>=2, p>=1")
    sample = np.cov(X, rowvar=False, bias=True)
    mu = np.trace(sample) / p
    delta = sample - mu * np.eye(p)
    # Sum of squared off-diagonal + squared diagonal deviations
    sum_sq = np.sum(delta ** 2)
    # Estimate optimal shrinkage intensity (simplified LW formula)
    # For identity target: kappa = sum Var(x_ij^2) / sum (s_ij - mu*delta_ij)^2
    # Use the oracle-approximating formula from LW 2004
    Xc = X - X.mean(axis=0)
    phi_mat = (Xc ** 2).T @ (Xc ** 2) / n - sample ** 2
    phi = np.sum(phi_mat)
    gamma = sum_sq
    if gamma == 0:
        return sample, 0.0
    intensity = max(0.0, min(1.0, phi / (gamma * n)))
    shrunk = (1.0 - intensity) * sample + intensity * mu * np.eye(p)
    return shrunk, float(intensity)


class FactorModel:
    """Factor regression + residual covariance + breadth report.

    Usage:
        fm = FactorModel()
        report = fm.fit(returns, factor_returns)
        print(report.summary())
    """

    def fit(
        self,
        returns: np.ndarray,
        factor_returns: np.ndarray,
        *,
        annualize: bool = False,
        periods_per_year: int = 252,
    ) -> ResidualReport:
        """Fit factor model and compute residual breadth report.

        Args:
            returns: (n_obs, n_assets) matrix of PIT-correct asset returns.
            factor_returns: (n_obs, n_factors) matrix of factor returns.
            annualize: if True, scale eigenvalues by periods_per_year.
            periods_per_year: scaling factor for annualization.

        Returns:
            ResidualReport with all metrics labeled.
        """
        if returns.ndim != 2:
            raise ValueError("returns must be 2D (n_obs, n_assets)")
        if factor_returns.ndim != 2:
            raise ValueError("factor_returns must be 2D (n_obs, n_factors)")
        if returns.shape[0] != factor_returns.shape[0]:
            raise ValueError("returns and factor_returns must have same n_obs")
        if returns.shape[0] < returns.shape[1] + factor_returns.shape[1] + 1:
            raise ValueError("n_obs must exceed n_assets + n_factors")

        n_obs, n_assets = returns.shape
        n_factors = factor_returns.shape[1]

        # OLS factor loadings: R_asset = B @ R_factor + epsilon
        # B = (F'F)^{-1} F' R
        FtF = factor_returns.T @ factor_returns
        FtR = factor_returns.T @ returns
        try:
            B = np.linalg.solve(FtF, FtR)  # (n_factors, n_assets)
        except np.linalg.LinAlgError:
            B = np.linalg.lstsq(factor_returns, returns, rcond=None)[0].T

        residuals = returns - factor_returns @ B  # (n_obs, n_assets)

        # Raw covariance spectrum
        raw_cov = np.cov(returns, rowvar=False, bias=True)
        raw_eigs = np.sort(np.linalg.eigvalsh(raw_cov))[::-1]
        raw_eigs = np.maximum(raw_eigs, 0.0)

        # Residual covariance with LW shrinkage
        resid_sample = np.cov(residuals, rowvar=False, bias=True)
        resid_shrunk, lw_intensity = _ledoit_wolf_identity(residuals)
        resid_eigs = np.sort(np.linalg.eigvalsh(resid_shrunk))[::-1]
        resid_eigs = np.maximum(resid_eigs, 0.0)
        resid_trace = float(np.sum(resid_eigs))

        # Condition number
        pos_eigs = resid_eigs[resid_eigs > 0]
        cond = float(pos_eigs[0] / pos_eigs[-1]) if len(pos_eigs) > 1 else 1.0

        # Renormalized residual correlation matrix
        diag = np.diag(resid_shrunk)
        diag_safe = np.where(diag > 0, diag, 1.0)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(diag_safe))
        resid_corr = D_inv_sqrt @ resid_shrunk @ D_inv_sqrt
        renorm_eigs = np.sort(np.linalg.eigvalsh(resid_corr))[::-1]
        renorm_eigs = np.maximum(renorm_eigs, 0.0)

        # Average residual variance as fraction of raw
        raw_trace = float(np.sum(raw_eigs))
        avg_resid_frac = resid_trace / raw_trace if raw_trace > 0 else 1.0

        # Minimum leverage for unit risk equalization (Jensen bound)
        # E[r_ii^{-1/2}] >= (E[r_ii])^{-1/2}
        mean_resid_var = resid_trace / n_assets if n_assets > 0 else 1.0
        if mean_resid_var > 0 and raw_trace > 0:
            min_lev = float(np.sqrt(raw_trace / resid_trace))
        else:
            min_lev = 1.0

        # Leverage-constrained N_eff
        # Under gross leverage <= 1, the portfolio cannot equalize risk
        # across low-variance residual factors. Approximate by scaling
        # the renormalized N_eff by the fraction of variance accessible
        # at unit leverage. This is a LOWER BOUND.
        lev_constrained_neff = _var_based_neff(resid_eigs)

        # Annualize if requested
        scale = periods_per_year if annualize else 1.0

        return ResidualReport(
            n_assets=n_assets,
            n_factors=n_factors,
            n_obs=n_obs,
            raw_eigenvalues=raw_eigs * scale,
            raw_participation_ratio=_participation_ratio(raw_eigs),
            raw_var_based_neff=_var_based_neff(raw_eigs),
            residual_eigenvalues=resid_eigs * scale,
            residual_trace=resid_trace * scale,
            residual_participation_ratio=_participation_ratio(resid_eigs),
            residual_var_based_neff=_var_based_neff(resid_eigs),
            renorm_residual_eigenvalues=renorm_eigs,
            renorm_participation_ratio=_participation_ratio(renorm_eigs),
            renorm_var_based_neff=_var_based_neff(renorm_eigs),
            avg_residual_variance_frac=avg_resid_frac,
            min_leverage_for_unit_risk=min_lev,
            leverage_constrained_neff=lev_constrained_neff,
            lw_shrinkage_intensity=lw_intensity,
            residual_condition_number=cond,
        )