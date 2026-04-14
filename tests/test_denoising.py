import numpy as np
import pytest

from rmt.denoising import eigenvalue_clip
from rmt.portfolio import MinVariancePortfolio

RANDOM_SEED = 42


@pytest.fixture
def corr_matrix():
    """Factor-model correlation matrix with clear signal (5 factors, p=100, T=500)."""
    rng = np.random.default_rng(RANDOM_SEED)
    p, T = 100, 500
    F = rng.standard_normal((T, 5)) * 3.0
    B = rng.standard_normal((p, 5))
    X = F @ B.T + rng.standard_normal((T, p))
    X = (X - X.mean(0)) / X.std(0)
    S = X.T @ X / T
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def test_T3_denoised_is_psd(corr_matrix):
    result = eigenvalue_clip(corr_matrix, q=100/500, method="clip")
    min_eig = np.linalg.eigvalsh(result.denoised_corr).min()
    assert min_eig >= -1e-10, "min eigenvalue %.2e" % min_eig


def test_T4_trace_preserved(corr_matrix):
    result = eigenvalue_clip(corr_matrix, q=100/500, method="clip")
    diff = abs(np.trace(result.denoised_corr) - np.trace(corr_matrix))
    assert diff < 1e-6, "trace diff %.2e" % diff


def test_T6_weights_sum_to_one():
    rng = np.random.default_rng(RANDOM_SEED)
    A = rng.standard_normal((20, 20))
    Sigma = A @ A.T + np.eye(20) * 0.5
    w = MinVariancePortfolio().fit(Sigma).weights_
    assert abs(w.sum() - 1.0) < 1e-8
