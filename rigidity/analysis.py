"""Operational finite-grid onset estimates; no fitted universal exponents."""

import numpy as np

from .shear import affine


def nonaffinity(positions, reference, gamma, lc):
    """Total displacement from zero-strain affine state, minus translation."""
    if gamma <= 0 or lc <= 0:
        raise ValueError("nonaffinity requires positive strain and segment length")
    displacement = np.asarray(positions) - affine(reference, gamma)
    displacement -= displacement.mean(axis=0)
    return float(np.mean(np.sum(displacement**2, axis=1)) / (lc * lc * gamma * gamma))


def critical_strains(gamma, modulus, nonaff, converged):
    """Return both onset estimates and log10 ratio; unresolved = None.

    K onset is first interior point at least twice the median first-three-point
    plateau, with positive log slope and an equally elevated next point. The
    nonaffinity onset is its highest strict interior local maximum. All required
    neighbors must be successful. No endpoint is declared a resolved transition.
    """
    g, k, a = map(lambda x: np.asarray(x, float), (gamma, modulus, nonaff))
    valid = np.asarray(converged, bool)
    if g.ndim != 1 or any(v.shape != g.shape for v in (k, a, valid)):
        raise ValueError("critical-strain inputs must be matching vectors")
    if not np.all(np.isfinite(g)) or np.any(g <= 0) or np.any(np.diff(g) <= 0):
        raise ValueError("strains must be positive and increasing")
    result = dict(gamma_c_K=None, gamma_c_nonaffinity=None, agreement_log10=None)
    if len(g) < 5:
        return result
    if np.all(valid[:3] & np.isfinite(k[:3])):
        baseline = np.median(k[:3])
        if baseline > 0:
            for i in range(3, len(g)-1):
                if (valid[i-1:i+2].all() and np.isfinite(k[i-1:i+2]).all()
                        and k[i] >= 2 * baseline and k[i+1] >= 2 * baseline
                        and k[i] > k[i-1]):
                    result["gamma_c_K"] = float(g[i])
                    break
    candidates = [i for i in range(1, len(g)-1)
                  if valid[i-1:i+2].all() and np.isfinite(a[i-1:i+2]).all()
                  and a[i] > a[i-1] and a[i] > a[i+1]]
    if candidates:
        i = max(candidates, key=lambda j: a[j])
        result["gamma_c_nonaffinity"] = float(g[i])
    if result["gamma_c_K"] and result["gamma_c_nonaffinity"]:
        result["agreement_log10"] = float(np.log10(
            result["gamma_c_K"] / result["gamma_c_nonaffinity"]))
    return result
