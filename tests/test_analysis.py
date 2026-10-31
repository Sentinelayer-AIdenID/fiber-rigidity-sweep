import numpy as np
import pytest

from rigidity.analysis import critical_strains, nonaffinity
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
