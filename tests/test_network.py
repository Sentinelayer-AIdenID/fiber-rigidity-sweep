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


def test_small_network_retunes_after_removing_finite_decorations():
    net = generate_network(4, 4, 3.2, 56)
    # Before accounting for the removed decoration the chosen retention gives
    # z=3.285714; full-cleanup retuning selects the closer value 3.142857.
    assert abs(net.z_actual - 3.2) < 0.06
    assert np.bincount(net.edges.ravel(), minlength=len(net.positions)).min() >= 2


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


def graph_network(edges, images):
    edges = np.asarray(edges, dtype=int)
    return Network(np.zeros((edges.max() + 1, 2)), np.eye(2), edges,
                   np.asarray(images, dtype=int), np.empty((0, 3), dtype=int),
                   np.empty((0, 2, 2), dtype=int), np.ones(len(edges)), np.empty(0))


def test_cleanup_removes_cycles_attached_at_articulation_or_bridge():
    # Triangle 0-1-2 wraps. Cycle 2-3-4 touches it only at articulation 2.
    # Cycle 5-6-7 attaches by bridge 1-5. Both are finite rattlers despite
    # surviving the graph 2-core (every vertex initially has degree >= 2).
    edges = [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 2],
             [1, 5], [5, 6], [6, 7], [7, 5]]
    images = np.zeros((len(edges), 2), dtype=int)
    images[2, 0] = 1
    images[3, 0], images[4, 0] = 1, -1  # Finite seam-crossing decoration.
    result = cleanup(graph_network(edges, images))
    assert len(result.positions) == 3
    np.testing.assert_array_equal(result.edges, edges[:3])


def test_cleanup_preserves_path_joining_two_winding_blocks():
    edges = [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4],
             [4, 5], [5, 6], [6, 4], [3, 7], [7, 8], [8, 3]]
    images = np.zeros((len(edges), 2), dtype=int)
    images[2, 0], images[7, 1] = 1, 1
    result = cleanup(graph_network(edges, images))
    assert len(result.positions) == 7
    # The degree-two path 2-3-4 transmits between wrapping blocks; the finite
    # decoration 3-7-8 is removed without discarding the connecting path.
    np.testing.assert_array_equal(result.edges, edges[:8])


@pytest.mark.parametrize("loop", [False, True])
def test_cleanup_preserves_parallel_edge_and_self_loop_winding(loop):
    if loop:
        edges = [[0, 0], [0, 1], [1, 2], [2, 0]]
        images = [[1, 0], [0, 0], [0, 0], [0, 0]]
        expected = 1
    else:
        edges = [[0, 1], [0, 1], [1, 2], [2, 3], [3, 1]]
        images = [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]]
        expected = 2
    result = cleanup(graph_network(edges, images))
    assert len(result.edges) == expected
    np.testing.assert_array_equal(result.images, images[:expected])


@pytest.mark.parametrize("arguments", [(2, 4, 3.5, 0), (4, 5, 3.5, 0),
                                       (4, 4, 4.1, 0), (4, 4, 3.5, -1)])
def test_network_rejects_invalid_parameters(arguments):
    with pytest.raises(ValueError):
        generate_network(*arguments)
