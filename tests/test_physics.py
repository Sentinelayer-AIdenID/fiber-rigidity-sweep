from types import SimpleNamespace

import numpy as np
import pytest

from rigidity.energy import evaluate
from rigidity.network import triangular_lattice
from rigidity.shear import affine, virial_stress


def motif(kind):
    points = np.array([[0., 0.], [1., 0.], [2., 0.], [2.4, .7]])
    return SimpleNamespace(
        positions=points, box=np.diag([8., 8.]),
        edges=np.array([[0, 1], [1, 2], [2, 3]], dtype=int) if kind == "stretch" else np.empty((0, 2), int),
        images=np.zeros((3 if kind == "stretch" else 0, 2), int),
        rest_lengths=np.ones(3 if kind == "stretch" else 0),
        triplets=np.array([[0, 1, 2], [1, 2, 3]], int) if kind == "bend" else np.empty((0, 3), int),
        triplet_images=np.zeros((2 if kind == "bend" else 0, 2, 2), int),
        bend_lengths=np.ones(2 if kind == "bend" else 0),
    )


@pytest.mark.parametrize("kind", ["stretch", "bend"])
def test_analytic_gradient_central_difference(kind):
    net = motif(kind)
    x = net.positions + np.random.default_rng(91).normal(0., .09, net.positions.shape)
    result = evaluate(x, net, .13, .7)
    numeric = np.zeros_like(x)
    h = 1e-6
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += h
        minus[index] -= h
        numeric[index] = (evaluate(plus, net, .13, .7)["energy"]
                          - evaluate(minus, net, .13, .7)["energy"]) / (2*h)
    relative_error = np.linalg.norm(numeric + result["forces"]) / np.linalg.norm(numeric)
    assert relative_error < 1e-6


def test_triangular_shear_modulus_energy_and_virial():
    net = triangular_lattice(6, 6)
    h = 1e-4
    volume = abs(np.linalg.det(net.box))
    e0 = evaluate(net.positions, net)["energy"]
    ep = evaluate(affine(net.positions, h), net, h)["energy"]
    em = evaluate(affine(net.positions, -h), net, -h)["energy"]
    curvature = (ep + em - 2*e0) / (volume*h*h)
    virial = (virial_stress(affine(net.positions, h), net, h)
              - virial_stress(affine(net.positions, -h), net, -h)) / (2*h)
    expected = np.sqrt(3) / 4
    assert curvature == pytest.approx(expected, rel=.01)
    assert virial == pytest.approx(expected, rel=.01)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_lees_edwards_affine_equilibrium(backend):
    net = triangular_lattice(6, 6)
    result = evaluate(affine(net.positions, .27), net, .27, .03, backend)
    assert np.max(np.abs(result["forces"])) < 1e-9


def test_bend_straight_zero_and_kink_positive():
    net = motif("bend")
    net.triplets = net.triplets[:1]
    net.triplet_images = net.triplet_images[:1]
    net.bend_lengths = net.bend_lengths[:1]
    for backend in ("numpy", "jax"):
        assert evaluate(net.positions, net, 0., .1, backend)["bend_energy"] == 0.
        kinked = net.positions.copy()
        kinked[1, 1] = .3
        assert evaluate(kinked, net, 0., .1, backend)["bend_energy"] > .001


@pytest.mark.parametrize("kind", ["stretch", "bend"])
def test_numpy_jax_agreement_and_virial_work(kind):
    net = motif(kind)
    x = net.positions + np.random.default_rng(6).normal(0., .08, net.positions.shape)
    result = evaluate(x, net, .2, .17)
    compiled = evaluate(x, net, .2, .17, "jax")
    for key in result:
        np.testing.assert_allclose(compiled[key], result[key], rtol=1e-12, atol=1e-13)
    h = 1e-6
    # Virtual homogeneous shear at the current nonaffine state.
    ep = evaluate(affine(x, h), net, .2+h, .17)["energy"]
    em = evaluate(affine(x, -h), net, .2-h, .17)["energy"]
    assert result["stress"] == pytest.approx((ep-em)/(2*h*abs(np.linalg.det(net.box))), rel=1e-7)
