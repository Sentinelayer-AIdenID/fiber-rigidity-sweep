"""FIRE physics, padding, independent batch state and failure semantics."""
import numpy as np
import pytest

from rigidity.fire import minimize_batch, pad_networks
from rigidity.network import generate_network, triangular_lattice


def test_fire_lowers_energy_and_converges():
    """A force-free or sign-reversed FIRE step cannot pass this check."""
    network = generate_network(6, 6, 3.8, seed=3)
    result = minimize_batch([network], [.15], [.03], max_steps=3000, force_tol=1e-8)
    assert result["converged"][0]
    assert not result["failed"][0]
    assert result["energy"][0] < .9 * result["initial_energy"][0]
    assert result["max_force"][0] <= 1e-8
    assert 0 < result["steps"][0] < 3000


def test_numpy_jax_padding_and_independent_members_agree():
    # The larger equilibrium member must freeze at zero steps while the other
    # member relaxes; the latter also exercises padded nodes/edges/triplets.
    networks = [triangular_lattice(6, 6), generate_network(4, 4, 3.8, seed=5)]
    gammas, kappas = [.08, .15], [.03, .03]
    settings = dict(max_steps=80, force_tol=1e-10, chunk_steps=25)
    reference = minimize_batch(networks, gammas, kappas, backend="numpy", **settings)
    actual = minimize_batch(networks, gammas, kappas, backend="jax", **settings)
    for key in ("positions", "energy", "initial_energy", "max_force"):
        np.testing.assert_allclose(actual[key], reference[key], atol=2e-12, rtol=2e-11)
    np.testing.assert_array_equal(actual["steps"], [0, 80])
    np.testing.assert_array_equal(actual["converged"], [True, False])
    assert np.all(actual["positions"][1, len(networks[1].positions):] == 0)
    single = minimize_batch(networks[1:], gammas[1:], kappas[1:], backend="numpy", **settings)
    np.testing.assert_allclose(actual["positions"][1, :len(networks[1].positions)],
                               single["positions"][0], atol=2e-12, rtol=2e-11)
    padded = pad_networks(networks)
    assert padded["node_mask"].sum(axis=1).tolist() == [36, 16]
    assert padded["edge_mask"].sum(axis=1).tolist() == [len(n.edges) for n in networks]
    assert padded["triplet_mask"].sum(axis=1).tolist() == [len(n.triplets) for n in networks]


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_fire_max_steps_fails_closed_and_callback_can_interrupt(backend):
    network = generate_network(4, 4, 3.8, seed=5)
    snapshots = []
    result = minimize_batch([network], [.15], [.03], backend=backend,
                            max_steps=7, chunk_steps=3, force_tol=1e-12,
                            callback=snapshots.append)
    assert not result["converged"][0]
    assert result["steps"][0] == 7
    assert [s["steps"][0] for s in snapshots] == [3, 6, 7]
    assert all(not s["converged"][0] for s in snapshots)

    def interrupt(snapshot):
        snapshots.append(snapshot)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        minimize_batch([network], [.15], [.03], backend=backend,
                       max_steps=100, chunk_steps=2, callback=interrupt)
    assert snapshots[-1]["steps"][0] == 2


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_nonfinite_member_is_failed_and_never_converged(backend):
    network = triangular_lattice(4, 4)
    bad = network.positions.copy()
    bad[0, 0] = np.nan
    with np.errstate(invalid="ignore"):
        result = minimize_batch([network], [.1], [.01], backend=backend,
                                max_steps=10, initial_positions=[bad])
    assert result["failed"][0]
    assert not result["converged"][0]
    assert result["steps"][0] == 0
