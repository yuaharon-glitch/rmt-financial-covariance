import numpy as np
import pytest
from scipy import integrate

from rmt.marchenko_pastur import MarchenkoPastur

RANDOM_SEED = 42


def test_T1_pdf_integrates_to_one():
    mp = MarchenkoPastur()
    for sigma, q in [(1.0, 0.1), (1.0, 0.5), (1.0, 0.9), (2.0, 0.3)]:
        lm = mp.lambda_minus(sigma, q)
        lp = mp.lambda_plus(sigma, q)
        val, _ = integrate.quad(lambda x: mp.pdf(x, sigma, q), lm, lp, limit=500)
        assert abs(val - 1.0) < 1e-4, "sigma=%s q=%s integral=%.6f" % (sigma, q, val)


def test_T2_wishart_sigma_fit():
    rng = np.random.default_rng(RANDOM_SEED)
    p, T = 50, 500
    X = rng.standard_normal((T, p))
    S = X.T @ X / T
    C = S / np.outer(np.sqrt(np.diag(S)), np.sqrt(np.diag(S)))
    eigs = np.linalg.eigvalsh(C)

    mp = MarchenkoPastur()
    mp.fit(eigs, q=p / T)
    assert abs(mp.sigma_**2 - 1.0) < 0.05, "fitted sigma^2=%.4f" % mp.sigma_**2
