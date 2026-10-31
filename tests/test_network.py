import hashlib

import numpy as np
import pytest

from rigidity.network import Network, cleanup, generate_network, triangular_lattice


def digest(network):
    h = hashlib.sha256()
    for name in ("positions", "box", "edges", "images", "triplets", "triplet_images",
                 "rest_lengths", "bend_lengths"):
        h.update(getattr(network, name).tobytes())
    return h.hexdigest()


def test_triangular_periodicity_and_straight_fibers():
    net = triangular_lattice(8, 6)
    vectors = net.positions[net.edges[:, 1]] - net.positions[net.edges[:, 0]] + net.images @ net.box
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-14)
    assert net.z_actual == 6
    a = net.positions[net.triplets[:, 1]] - net.positions[net.triplets[:, 0]] + net.triplet_images[:, 0] @ net.box
    b = net.positions[net.triplets[:, 2]] - net.positions[net.triplets[:, 1]] + net.triplet_images[:, 1] @ net.box
    assert np.allclose(a, b, atol=1e-14)
    assert np.any(net.images)


@pytest.mark.parametrize("z", [3.2, 3.5, 3.8])
def test_phantom_cleanup_coordination_and_lengths(z):
    net = generate_network(20, 20, z, 19)
    degree = np.bincount(net.edges.ravel(), minlength=len(net.positions))
    assert degree.min() >= 2
    assert degree.max() <= 4
    assert abs(net.z_actual - z) < 0.06
    assert np.any(net.rest_lengths > 1)
    vectors = net.positions[net.edges[:, 1]] - net.positions[net.edges[:, 0]] + net.images @ net.box
    assert np.allclose(np.linalg.norm(vectors, axis=1), net.rest_lengths)
    a = net.positions[net.triplets[:, 1]] - net.positions[net.triplets[:, 0]] + net.triplet_images[:, 0] @ net.box
    b = net.positions[net.triplets[:, 2]] - net.positions[net.triplets[:, 1]] + net.triplet_images[:, 1] @ net.box
    assert np.allclose(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], 0, atol=1e-12)
    assert np.all(np.sum(a * b, axis=1) > 0)


def test_network_seed_determinism():
    first = generate_network(12, 12, 3.5, 9)
    assert digest(first) == digest(generate_network(12, 12, 3.5, 9))
    assert digest(first) != digest(generate_network(12, 12, 3.5, 10))


def test_cleanup_rejects_finite_seam_crossing_and_dangling_bonds():
    # A winding triangle, a dangling edge, and a finite triangle that crosses
    # the periodic seam twice with opposite signs. Merely testing images != 0
    # would incorrectly retain the latter.
    edges = np.array([[0, 1], [1, 2], [2, 0], [1, 3], [4, 5], [5, 6], [6, 4]])
    images = np.array([[0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [-1, 0], [0, 0]])
    net = Network(np.zeros((7, 2)), np.eye(2), edges, images,
                  np.empty((0, 3), dtype=int), np.empty((0, 2, 2), dtype=int),
                  np.ones(7), np.empty(0))
    result = cleanup(net)
    assert len(result.positions) == 3
    assert len(result.edges) == 3
    assert result.images[:, 0].sum() == 1


@pytest.mark.parametrize("arguments", [(2, 4, 3.5, 0), (4, 5, 3.5, 0),
                                       (4, 4, 4.1, 0), (4, 4, 3.5, -1)])
def test_network_rejects_invalid_parameters(arguments):
    with pytest.raises(ValueError):
        generate_network(*arguments)
