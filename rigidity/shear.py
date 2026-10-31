"""Simple shear and stress differentiation without bridging failed samples."""

import numpy as np


def affine(positions, gamma):
    result = np.asarray(positions, dtype=np.float64).copy()
    result[..., 0] += gamma * result[..., 1]
    return result


def virial_stress(positions, network, gamma, kappa=0., backend="numpy"):
    from .energy import evaluate
    return float(evaluate(positions, network, gamma, kappa, backend)["stress"])


def differential_modulus(gamma, stress, converged):
    """Three-point derivative on the actual nonuniform gamma grid.

    Interior values require both immediate neighbors; endpoints require three
    consecutive successful samples. Missing or failed neighbors yield NaN.
    """
    gamma, stress = np.asarray(gamma, float), np.asarray(stress, float)
    good = np.asarray(converged, bool) & np.isfinite(stress)
    if gamma.ndim != 1 or stress.shape != gamma.shape or good.shape != gamma.shape:
        raise ValueError("gamma, stress and convergence must be matching vectors")
    if not np.all(np.isfinite(gamma)) or np.any(np.diff(gamma) <= 0):
        raise ValueError("gamma must be finite and strictly increasing")
    result = np.full(gamma.shape, np.nan)
    if len(gamma) < 3:
        return result
    for i in range(len(gamma)):
        start = max(0, min(i - 1, len(gamma) - 3))
        inds = np.arange(start, start + 3)
        if good[inds].all():
            # Shift before solving to reduce conditioning errors at small strain.
            t = gamma[inds] - gamma[i]
            weights = np.linalg.solve(np.stack([np.ones(3), t, t*t]), [0., 1., 0.])
            result[i] = weights @ stress[inds]
    return result
