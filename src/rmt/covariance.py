"""
Covariance estimators: sample MLE, Ledoit-Wolf shrinkage, and RMT-denoised.

All three share the same interface: fit(returns) -> self, covariance_ after fit.

Ref: Ledoit & Wolf (2004), J. Multivariate Analysis 88(2), 365-411.
"""

import numpy as np


class SampleCovariance:
    """
    MLE sample covariance: S = (1/T) X^T X.

    Divides by T, not T-1, to match the convention used throughout RMT
    literature where the Marchenko-Pastur law is stated for (1/n) X^T X.
    """

    def fit(self, returns: np.ndarray) -> "SampleCovariance":
        returns = np.asarray(returns, dtype=float)
        T, p = returns.shape
        if T < p:
            raise ValueError("T=%d < p=%d: insufficient observations." % (T, p))
        X = returns - returns.mean(axis=0)
        self.covariance_ = X.T @ X / T
        self.n_observations_ = T
        self.n_assets_ = p
        return self


class LedoitWolfCovariance:
    r"""
    Ledoit-Wolf analytical shrinkage toward a scaled identity target.

    Estimates the optimal shrinkage intensity alpha* analytically (no CV)
    from the Oracle Approximating Shrinkage formula in Ledoit & Wolf (2004):

    .. math::

        \hat{\Sigma} = (1 - \alpha^*) S + \alpha^* \mu I_p,
        \quad \mu = \frac{\operatorname{tr}(S)}{p}

    where

    .. math::

        \alpha^* = \frac{\beta^2}{\delta^2}, \quad
        \delta^2 = \frac{\|S - \mu I\|_F^2}{p}, \quad
        \beta^2 = \min\!\left(\delta^2,\;
            \frac{1}{T^2 p} \sum_k \bigl[\|x_k\|^4
            - 2\,x_k^\top S\,x_k + \operatorname{tr}(S^2)\bigr]
        \right)

    The min-cap on beta^2 ensures alpha* stays in [0, 1].
    Sets ``self.alpha_`` after fit.
    """

    def fit(self, returns: np.ndarray) -> "LedoitWolfCovariance":
        returns = np.asarray(returns, dtype=float)
        T, p = returns.shape
        if T < p:
            raise ValueError("T=%d < p=%d: insufficient observations." % (T, p))

        X = returns - returns.mean(axis=0)
        S = X.T @ X / T
        mu = np.trace(S) / p
        I = np.eye(p)

        delta_sq = np.linalg.norm(S - mu * I, "fro") ** 2 / p

        # Vectorised O(T*p) computation of sum_k ||x_k x_k^T - S||_F^2
        # using ||A - B||_F^2 = tr(A^T A) - 2 tr(A^T B) + tr(B^T B)
        norms_sq = np.einsum("ni,ni->n", X, X)
        XSX = np.einsum("ni,ij,nj->n", X, S, X)
        tr_S2 = np.einsum("ij,ji->", S, S)
        beta_bar_sq = np.sum(norms_sq**2 - 2.0 * XSX + tr_S2) / (T**2 * p)

        beta_sq = min(delta_sq, beta_bar_sq)

        if delta_sq < 1e-15:
            self.alpha_ = 0.0
        else:
            self.alpha_ = float(np.clip(beta_sq / delta_sq, 0.0, 1.0))

        self.covariance_ = (1.0 - self.alpha_) * S + self.alpha_ * mu * I
        self.n_observations_ = T
        self.n_assets_ = p
        return self


class RMTCovariance:
    """
    RMT-denoised covariance estimator.

    Converts sample covariance to correlation, applies eigenvalue clipping
    via the Marchenko-Pastur law, then converts back.  The correlation/
    covariance split is important: the MP distribution is parameterised for
    unit-variance variables, so denoising must happen on the correlation matrix.

    Parameters
    ----------
    method : str
        Passed to ``eigenvalue_clip``. Default 'clip' (trace-preserving).
    """

    def __init__(self, method: str = "clip") -> None:
        self.method = method

    def fit(self, returns: np.ndarray) -> "RMTCovariance":
        returns = np.asarray(returns, dtype=float)
        T, p = returns.shape
        if T < p:
            raise ValueError("T=%d < p=%d: insufficient observations." % (T, p))

        from rmt.denoising import eigenvalue_clip

        S = SampleCovariance().fit(returns).covariance_
        vols = np.sqrt(np.diag(S))
        corr = S / np.outer(vols, vols)

        result = eigenvalue_clip(corr, p / T, method=self.method)
        self.covariance_ = result.denoised_corr * np.outer(vols, vols)
        self.denoising_result_ = result
        self.n_observations_ = T
        self.n_assets_ = p
        return self
