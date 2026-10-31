"""Finite-grid onsets and conditional power-law fits, with unresolved results."""

import numpy as np
from scipy.stats import linregress, t

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
    nonaffinity onset is its highest observed value, only if a strict interior
    maximum. All required
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
    observed = valid & np.isfinite(a)
    if observed.any():
        i = int(np.argmax(np.where(observed, a, -np.inf)))
        if (0 < i < len(g)-1 and observed[i-1:i+2].all()
                and a[i] > a[i-1] and a[i] > a[i+1]):
            result["gamma_c_nonaffinity"] = float(g[i])
    if result["gamma_c_K"] and result["gamma_c_nonaffinity"]:
        result["agreement_log10"] = float(np.log10(
            result["gamma_c_K"] / result["gamma_c_nonaffinity"]))
    return result


def fit_critical_exponent(gamma, modulus, converged, gamma_c, *, window=None, min_points=5):
    """Fit K' = A * (gamma - gamma_c)**f on the rigid side at fixed gamma_c.

    ``window`` is an inclusive (min, max) distance ABOVE gamma_c, default
    (0.05*gamma_c, 0.5*gamma_c). It is specified before fitting, never optimized
    against the literature exponent. Require >=5 usable points spanning a factor
    of three in distance. Missing/nonpositive modulus and failed states are
    excluded; callers must also mask any failed stress-difference stencils.

    Unweighted log-log least squares gives f, its standard error and a Student-t
    95% interval, conditional on this onset/window. These do not include onset,
    finite-size, seed or bending-crossover uncertainty. No joint onset fit or
    bending-background subtraction is attempted. Unresolved fits carry null f.
    """
    g, k = np.asarray(gamma, float), np.asarray(modulus, float)
    good = np.asarray(converged, bool)
    if g.ndim != 1 or any(v.shape != g.shape for v in (k, good)):
        raise ValueError("exponent inputs must be matching vectors")
    if not np.all(np.isfinite(g)) or np.any(g <= 0) or np.any(np.diff(g) <= 0):
        raise ValueError("strains must be positive and increasing")
    if not isinstance(min_points, (int, np.integer)) or min_points < 5:
        raise ValueError("at least five points are required")
    if gamma_c is not None and (not np.isfinite(gamma_c) or gamma_c <= 0):
        raise ValueError("gamma_c must be positive and finite, or None when unresolved")
    if window is not None:
        window = np.asarray(window, float)
        if window.shape != (2,) or not np.isfinite(window).all() or not 0 < window[0] < window[1]:
            raise ValueError("window must be two increasing positive distances above gamma_c")
    gamma_c = None if gamma_c is None else float(gamma_c)
    result = dict(status="not determined", reason="transition strain unresolved", gamma_c=gamma_c,
                  f=None, f_stderr=None, f_ci95=None, r_squared=None,
                  delta_window=None, fit_window=None, n_points=0, point_indices=[],
                  literature_f=.7, delta_f_literature=None)
    if gamma_c is None:
        return result
    lo, hi = (.05 * gamma_c, .5 * gamma_c) if window is None else window
    delta = g - gamma_c
    selected = good & np.isfinite(k) & (k > 0) & (delta >= lo) & (delta <= hi)
    indices = np.flatnonzero(selected)
    n = len(indices)
    result.update(delta_window=[float(lo), float(hi)], n_points=n, point_indices=indices.tolist())
    if n:
        result["fit_window"] = [float(g[indices[0]]), float(g[indices[-1]])]
    if n < min_points:
        result["reason"] = f"only {n} usable near-transition points; need {min_points}"
        return result
    if delta[indices[-1]] / delta[indices[0]] < 3:
        result["reason"] = "usable distances span less than a factor of three"
        return result
    regression = linregress(np.log(delta[selected]), np.log(k[selected]))
    slope, stderr = float(regression.slope), float(regression.stderr)
    if not np.isfinite([slope, stderr, regression.rvalue]).all() or slope <= 0:
        result["reason"] = "no finite positive stiffening exponent"
        return result
    margin = float(t.ppf(.975, n-2) * stderr)
    result.update(status="determined", reason=None, f=slope, f_stderr=stderr,
                  f_ci95=[slope-margin, slope+margin], r_squared=float(regression.rvalue**2),
                  delta_f_literature=slope - .7)
    return result


def critical_strain_trends(curves):
    """Describe resolved onsets versus measured sub-isostatic z at each kappa.

    This is a finite-grid direction check, never an extrapolation to z=4.
    Missing onsets remain absent and the count of usable connectivities is given.
    """
    trends = []
    for kappa in sorted({curve["kappa"] for curve in curves}):
        group = [curve for curve in curves if curve["kappa"] == kappa]
        for method in ("gamma_c_K", "gamma_c_nonaffinity"):
            points = sorted((float(c["z_actual"]), float(c[method])) for c in group
                            if np.isfinite(c["z_actual"]) and 0 < c["z_actual"] < 4
                            and c.get(method) is not None and np.isfinite(c[method]) and c[method] > 0)
            status = "not determined"
            if len(points) >= 2 and np.all(np.diff([p[0] for p in points]) > 0):
                differences = np.diff([p[1] for p in points])
                status = ("decreasing" if np.all(differences < 0) else
                          "increasing" if np.all(differences > 0) else
                          "flat" if np.all(differences == 0) else "mixed")
            trends.append(dict(kappa=float(kappa), method=method, status=status,
                               n_points=len(points), total_curves=len(group),
                               points=[dict(z_actual=z, gamma_c=gc) for z, gc in points],
                               limit_at_z4="not determined"))
    return trends
