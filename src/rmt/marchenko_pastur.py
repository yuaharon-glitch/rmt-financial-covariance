"""
Marchenko-Pastur distribution.

Ref: Marchenko & Pastur (1967), Sbornik: Mathematics 1(4), 457-483.
"""

import numpy as np
from scipy import integrate, optimize, stats


class MarchenkoPastur:
    r"""
    Marchenko-Pastur spectral distribution for large Wishart matrices.

    Describes the limiting eigenvalue density of S = (1/T) X^T X where X
    has i.i.d. mean-zero entries with variance sigma^2, as p/T -> q.

    After ``fit()``, ``sigma_`` holds the MLE noise level.
    """

    def lambda_plus(self, sigma: float, q: float) -> float:
        # lambda_+ = sigma^2 * (1 + sqrt(q))^2
        return float(sigma**2 * (1.0 + np.sqrt(q)) ** 2)

    def lambda_minus(self, sigma: float, q: float) -> float:
        # lambda_- = sigma^2 * (1 - sqrt(q))^2
        return float(sigma**2 * (1.0 - np.sqrt(q)) ** 2)

    def pdf(self, x: np.ndarray, sigma: float, q: float) -> np.ndarray:
        r"""
        Marchenko-Pastur density.

        .. math::

            \rho(x) = \frac{\sqrt{(\lambda_+ - x)(x - \lambda_-)}}{2\pi\,\sigma^2\,q\,x}

        Zero outside :math:`[\lambda_-, \lambda_+]`.  Returns a scalar when x
        is scalar, array otherwise.
        """
        scalar = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        lp = self.lambda_plus(sigma, q)
        lm = self.lambda_minus(sigma, q)
        out = np.zeros_like(x_arr)
        mask = (x_arr > lm) & (x_arr < lp)
        xm = x_arr[mask]
        out[mask] = np.sqrt((lp - xm) * (xm - lm)) / (2.0 * np.pi * sigma**2 * q * xm)
        return float(out[0]) if scalar else out

    def _pdf_scalar(self, x: float, sigma: float, q: float) -> float:
        lp = self.lambda_plus(sigma, q)
        lm = self.lambda_minus(sigma, q)
        if x <= lm or x >= lp:
            return 0.0
        return float(np.sqrt((lp - x) * (x - lm)) / (2.0 * np.pi * sigma**2 * q * x))

    def _mp_cdf_scalar(self, x: float, sigma: float, q: float) -> float:
        lm = self.lambda_minus(sigma, q)
        lp = self.lambda_plus(sigma, q)
        if x <= lm:
            return 0.0
        result, _ = integrate.quad(
            self._pdf_scalar, lm, min(x, lp), args=(sigma, q), limit=200
        )
        return float(np.clip(result, 0.0, 1.0))

    def _mp_cdf_vec(self, x_arr: np.ndarray, sigma: float, q: float) -> np.ndarray:
        # kstest passes the full array at once, so we need this wrapper
        return np.array([self._mp_cdf_scalar(float(xi), sigma, q) for xi in x_arr])

    def fit(self, eigenvalues: np.ndarray, q: float) -> "MarchenkoPastur":
        r"""
        MLE fit of sigma^2 to an empirical eigenvalue spectrum.

        Optimises :math:`\ell(\sigma^2) = \sum_i \log \rho(\lambda_i; \sigma^2, q)`
        in log-space to keep sigma positive without explicit bounds.
        Sets ``self.sigma_`` and returns self.
        """
        eigs = np.asarray(eigenvalues, dtype=float)

        def _neg_ll(log_s2: float) -> float:
            s = float(np.exp(0.5 * log_s2))
            lp = self.lambda_plus(s, q)
            lm = self.lambda_minus(s, q)
            ll = 0.0
            for x in eigs:
                if lm < x < lp:
                    v = np.sqrt((lp - x) * (x - lm)) / (2.0 * np.pi * s**2 * q * x)
                else:
                    v = 1e-300
                ll += np.log(max(float(v), 1e-300))
            return -ll

        res = optimize.minimize_scalar(_neg_ll, bounds=(-5.0, 5.0), method="bounded")
        self.sigma_ = float(np.exp(0.5 * res.x))
        return self

    def ks_test(
        self, eigenvalues: np.ndarray, sigma: float, q: float
    ) -> tuple[float, float]:
        r"""
        KS goodness-of-fit between the bulk eigenvalue spectrum and the MP law.

        Only tests eigenvalues inside :math:`[\lambda_-, \lambda_+]`; signal
        eigenvalues above the upper edge are excluded before the test.

        Returns ``(statistic, pvalue)``.
        """
        eigs = np.asarray(eigenvalues, dtype=float)
        lm = self.lambda_minus(sigma, q)
        lp = self.lambda_plus(sigma, q)
        bulk = eigs[(eigs >= lm) & (eigs <= lp)]
        if len(bulk) == 0:
            return (1.0, 0.0)
        result = stats.kstest(bulk, lambda x: self._mp_cdf_vec(x, sigma, q))
        return (float(result.statistic), float(result.pvalue))
