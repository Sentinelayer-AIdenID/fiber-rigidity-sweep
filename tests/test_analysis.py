import numpy as np
import pytest

from rigidity.analysis import (critical_strains, critical_strain_trends,
                               fit_critical_exponent, nonaffinity)
from rigidity.shear import affine, differential_modulus


def test_modulus_nonuniform_quadratic_and_failure_gap():
    gamma = np.geomspace(.001, 1., 8)
    stress = 2*gamma**2 + 3*gamma
    good = np.ones(8, bool)
    np.testing.assert_allclose(differential_modulus(gamma, stress, good), 4*gamma+3, rtol=1e-12)
    good[3] = False
    k = differential_modulus(gamma, stress, good)
    assert np.isnan(k[2:5]).all()
    assert np.isfinite(k[[0, 1, 5, 6, 7]]).all()


def test_nonaffinity_translation_and_normalization():
    reference = np.array([[0., 0.], [1., 1.]])
    gamma, lc = .2, 2.
    positions = affine(reference, gamma) + [3., -8.]
    assert nonaffinity(positions, reference, gamma, lc) == pytest.approx(0., abs=1e-25)
    positions += np.array([[.1, 0.], [-.1, 0.]])
    assert nonaffinity(positions, reference, gamma, lc) == pytest.approx(.01 / (.04*4))


def test_critical_strains_known_upturn_peak_and_fail_closed():
    g = np.geomspace(.001, 1., 7)
    k = [1., 1., 1., 3., 6., 9., 12.]
    a = [1., 2., 3., 5., 3., 2., 1.]
    result = critical_strains(g, k, a, [True]*7)
    assert result == dict(gamma_c_K=g[3], gamma_c_nonaffinity=g[3], agreement_log10=0.)
    failed = critical_strains(g, k, a, [True, True, True, False, True, True, True])
    assert failed["gamma_c_nonaffinity"] is None
    # Later plateau samples may qualify as an upturn, but the failed sample cannot.
    assert failed["gamma_c_K"] != g[3]
    unresolved = critical_strains(g, np.ones(7), np.arange(7), [True]*7)
    assert all(v is None for v in unresolved.values())


def test_critical_strains_reject_nonfinite_input():
    with pytest.raises(ValueError, match="strains"):
        critical_strains([.001, .01, np.nan, .1, 1.], [1.]*5, [1.]*5, [True]*5)


def test_endpoint_global_peak_is_unresolved_even_with_interior_bump():
    result = critical_strains([.001, .01, .1, .3, 1.], [1.]*5, [1., 3., 2., 4., 5.], [True]*5)
    assert result["gamma_c_nonaffinity"] is None


def test_critical_exponent_recovers_known_power_and_uncertainty():
    # Deliberately different from 0.7: a literature constant cannot pass this.
    gc, expected = .2, 1.17
    delta = np.geomspace(.01, .1, 31)
    gamma = gc + delta
    noise = np.random.default_rng(23).normal(0., .015, len(delta))
    modulus = 2.4 * delta**expected * np.exp(noise)
    fit = fit_critical_exponent(gamma, modulus, np.ones(len(delta), bool), gc,
                                window=(.009, .101))
    assert fit["status"] == "determined"
    assert fit["f"] == pytest.approx(expected, rel=.05)
    assert 0 < fit["f_stderr"] < .02
    assert fit["f_ci95"][0] < expected < fit["f_ci95"][1]
    assert fit["fit_window"] == pytest.approx([gamma[0], gamma[-1]])
    assert fit["delta_window"] == [.009, .101]
    assert fit["n_points"] == 31
    assert fit["r_squared"] > .99
    assert fit["delta_f_literature"] == pytest.approx(fit["f"] - .7)


def test_exponent_excludes_failed_nonpositive_and_out_of_window_points():
    delta = np.geomspace(.001, .3, 41)
    gamma = .2 + delta
    modulus = 3 * delta**.91
    good = np.ones(41, bool)
    good[19:22] = False
    modulus[~good] = 1e20
    modulus[23], modulus[24], modulus[25] = 0., -1., np.nan
    # Outside the declared near-transition window, this curve has another slope.
    modulus[(delta < .01) | (delta > .1)] = 100
    fit = fit_critical_exponent(gamma, modulus, good, .2)
    assert fit["status"] == "determined"
    assert fit["f"] == pytest.approx(.91, abs=1e-12)
    expected = np.flatnonzero(good & np.isfinite(modulus) & (modulus > 0)
                              & (delta >= .01) & (delta <= .1))
    assert fit["point_indices"] == expected.tolist()


@pytest.mark.parametrize("case", ["missing_onset", "too_few", "narrow_window", "flat"])
def test_exponent_not_determined_without_supported_window(case):
    gamma = .2 + np.geomspace(.011, .099, 10)
    good = np.ones(10, bool)
    gc = .2
    if case == "missing_onset":
        gc = None
    elif case == "too_few":
        good[4:] = False
    elif case == "narrow_window":
        gamma = np.linspace(.22, .23, 10)
    modulus = np.ones(10) if case == "flat" else (gamma - .2)**.83
    fit = fit_critical_exponent(gamma, modulus, good, gc)
    assert fit["status"] == "not determined"
    assert fit["reason"]
    assert all(fit[key] is None for key in ("f", "f_stderr", "f_ci95", "delta_f_literature"))


@pytest.mark.parametrize("kwargs", [dict(gamma=[.3, .2]), dict(modulus=[1.]),
                                     dict(window=(0, .1)), dict(window=(.1, .01)),
                                     dict(gamma_c=float("nan")), dict(min_points=2)])
def test_exponent_rejects_invalid_inputs(kwargs):
    arguments = dict(gamma=[.21, .22], modulus=[1., 2.], converged=[True, True], gamma_c=.2)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        fit_critical_exponent(**arguments)


def test_transition_trend_uses_measured_z_and_keeps_kappa_and_methods_separate():
    curves = [dict(z_target=z, z_actual=actual, kappa=kappa,
                   gamma_c_nonaffinity=gc, gamma_c_K=None)
              for kappa, z, actual, gc in [(1e-3, 3.8, 3.2, .2), (1e-3, 3.2, 3.8, .1),
                                          (1e-2, 3.2, 3.2, .1), (1e-2, 3.8, 3.8, .2)]]
    trends = critical_strain_trends(curves)
    assert [t["status"] for t in trends if t["method"] == "gamma_c_nonaffinity"] == ["decreasing", "increasing"]
    assert all(t["status"] == "not determined" for t in trends if t["method"] == "gamma_c_K")
    assert all(t["limit_at_z4"] == "not determined" for t in trends)
    curves[1]["gamma_c_nonaffinity"] = None
    assert critical_strain_trends(curves)[1]["status"] == "not determined"


@pytest.mark.parametrize("onsets,status", [([.3, .2, .1], "decreasing"),
                                           ([.1, .2, .3], "increasing"),
                                           ([.2, .2, .2], "flat"), ([.2, .3, .1], "mixed")])
def test_transition_direction_is_a_finite_subisostatic_grid_check(onsets, status):
    curves = [dict(z_actual=z, kappa=.001, gamma_c_K=gc, gamma_c_nonaffinity=None)
              for z, gc in zip([3.2, 3.5, 3.8, 4.0], onsets + [100.])]
    trend = critical_strain_trends(curves)[0]
    assert trend["status"] == status
    assert trend["n_points"] == 3
    assert trend["limit_at_z4"] == "not determined"
