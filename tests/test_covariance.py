import numpy as np
import pytest

from rmt.covariance import LedoitWolfCovariance, RMTCovariance, SampleCovariance

RANDOM_SEED = 42


def test_T5_lw_alpha_in_unit_interval():
    rng = np.random.default_rng(RANDOM_SEED)
    for p, T in [(20, 100), (50, 200), (10, 500), (80, 250)]:
        lw = LedoitWolfCovariance().fit(rng.standard_normal((T, p)))
        assert 0.0 <= lw.alpha_ <= 1.0, "alpha=%.4f out of [0,1] for p=%d T=%d" % (lw.alpha_, p, T)


def test_T7_raises_when_p_gt_T():
    rng = np.random.default_rng(RANDOM_SEED)
    bad = rng.standard_normal((50, 100))
    for cls in [SampleCovariance, LedoitWolfCovariance, RMTCovariance]:
        with pytest.raises(ValueError, match="T=50 < p=100"):
            cls().fit(bad)
