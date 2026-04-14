"""
Eigenvalue clipping for correlation matrix denoising via the MP law.

Three methods:
  'clip'   - noise eigs -> mean(noise eigs)  [trace-preserving, default]
  'shrink' - noise eigs -> sigma^2
  'zero'   - project onto signal subspace only (rank-deficient)
"""

from dataclasses import dataclass

import numpy as np

from rmt.marchenko_pastur import MarchenkoPastur


@dataclass
class DenoisingResult:
    denoised_corr: np.ndarray
    n_signal: int
    n_noise: int
    lambda_plus: float
    fitted_sigma: float


def eigenvalue_clip(C: np.ndarray, q: float, method: str = "clip") -> DenoisingResult:
    r"""
    Denoise a correlation matrix by replacing noise eigenvalues.

    Fits the Marchenko-Pastur law to the full spectrum of C, classifies
    eigenvalues above :math:`\lambda_+ = \hat{\sigma}^2(1+\sqrt{q})^2` as
    signal, and replaces the rest according to ``method``.

    The 'clip' method is algebraically trace-preserving:
    sum(signal eigs) + n_noise * mean(noise eigs) = tr(C).
    The 'shrink' and 'zero' methods are not.
    """
    C = np.asarray(C, dtype=float)
    eigs, vecs = np.linalg.eigh(C)  # ascending order, real, orthonormal

    mp = MarchenkoPastur()
    mp.fit(eigs, q)
    lp = mp.lambda_plus(mp.sigma_, q)

    signal = eigs > lp
    new_eigs = eigs.copy()

    if method == "clip":
        if (~signal).any():
            new_eigs[~signal] = eigs[~signal].mean()
    elif method == "shrink":
        new_eigs[~signal] = mp.sigma_**2
    elif method == "zero":
        new_eigs[~signal] = 0.0
    else:
        raise ValueError("method must be 'clip', 'shrink', or 'zero', got '%s'" % method)

    C_clean = vecs @ np.diag(new_eigs) @ vecs.T
    C_clean = (C_clean + C_clean.T) * 0.5  # symmetrise away floating-point drift

    return DenoisingResult(
        denoised_corr=C_clean,
        n_signal=int(signal.sum()),
        n_noise=int((~signal).sum()),
        lambda_plus=float(lp),
        fitted_sigma=float(mp.sigma_),
    )
