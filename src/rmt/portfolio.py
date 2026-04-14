"""Min-variance portfolio via closed-form solution."""

import numpy as np
import pandas as pd


class MinVariancePortfolio:
    r"""
    Global minimum-variance portfolio.

    Solves :math:`\min_w w^\top \Sigma w` s.t. :math:`\mathbf{1}^\top w = 1`
    via the closed-form:

    .. math::

        w^* = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}

    Uses ``np.linalg.solve`` rather than explicit inversion.
    """

    def fit(self, covariance_matrix: np.ndarray) -> "MinVariancePortfolio":
        Sigma = np.asarray(covariance_matrix, dtype=float)
        ones = np.ones(Sigma.shape[0])
        v = np.linalg.solve(Sigma, ones)
        self.weights_ = v / v.sum()
        self.covariance_ = Sigma
        return self

    def portfolio_variance(self) -> float:
        return float(self.weights_ @ self.covariance_ @ self.weights_)

    @staticmethod
    def compare(
        estimators: dict,
        returns_test: np.ndarray,
        risk_free: float = 0.0,
    ) -> pd.DataFrame:
        """
        OOS comparison of min-variance portfolios across covariance estimators.

        Parameters
        ----------
        estimators : dict
            {name: covariance_matrix} — covariances must already be fit on train data.
        returns_test : ndarray (T_test, p)
            Held-out return matrix.
        risk_free : float
            Daily risk-free rate. Default 0.

        Returns a DataFrame with annualised OOS variance and Sharpe ratio.
        """
        R = np.asarray(returns_test, dtype=float)
        rows = []
        for name, cov in estimators.items():
            w = MinVariancePortfolio().fit(cov).weights_
            r_p = R @ w
            exc = r_p - risk_free
            rows.append({
                "Estimator": name,
                "OOS Variance (ann.)": float(r_p.var() * 252),
                "Sharpe Ratio": float(exc.mean() / (exc.std() + 1e-12) * np.sqrt(252)),
            })
        return pd.DataFrame(rows).set_index("Estimator")
