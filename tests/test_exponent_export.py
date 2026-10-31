import json

import numpy as np
import pytest

from rigidity.io import viewer_curves


def values_for_fit():
    gamma = np.r_[[.01, .02, .04, .2], .2 + np.geomspace(.011, .099, 12)]
    shape = (1, 1, 2, len(gamma))
    values = dict(z_target=np.array([3.5]), z_actual=np.array([[3.45, 3.55]]),
                  kappa=np.array([.001]), gamma=gamma, converged=np.ones(shape, bool))
    values["K"] = np.broadcast_to(np.r_[[.001]*4, 2 * (gamma[4:] - .2)**.93], shape).copy()
    nonaff = np.r_[[1, 2, 3, 5], np.linspace(4, 1, 12)]
    for key in ("nonaffinity", "stretch_energy", "bend_energy"):
        values[key] = np.broadcast_to(nonaff, shape).copy()
    return values


def test_export_fits_complete_seed_stencils_and_keeps_both_onset_choices():
    values = values_for_fit()
    # One failed seed cannot contaminate a fit of the mean, even if another is good.
    values["converged"][0, 0, 0, 7] = False
    values["K"][0, 0, :, 7] = 1e12
    values["K"][0, 0, 0, 9] = np.nan  # Successful state, failed derivative stencil.
    values["K"][0, 0, 1, 9] = 1e12
    curve = viewer_curves(values)[0]
    assert set(curve["critical_exponents"]) == {"gamma_c_K", "gamma_c_nonaffinity"}
    fit = curve["critical_exponents"]["gamma_c_nonaffinity"]
    assert fit["gamma_c"] == .2
    assert fit["f"] == pytest.approx(.93, abs=1e-12)
    assert fit["n_points"] == 10
    assert 7 not in fit["point_indices"] and 9 not in fit["point_indices"]
    json.dumps(curve, allow_nan=False)


def test_export_not_determined_when_convergence_leaves_too_few_fit_points():
    values = values_for_fit()
    values["converged"][0, 0, 0, 8:] = False
    curve = viewer_curves(values)[0]
    fit = curve["critical_exponents"]["gamma_c_nonaffinity"]
    assert fit["gamma_c"] == .2
    assert fit["status"] == "not determined"
    assert fit["f"] is None
    assert fit["n_points"] == 4
    assert "need 5" in fit["reason"]
    json.dumps(curve, allow_nan=False)
