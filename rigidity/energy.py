"""Stretch and bending mechanics in a sheared periodic cell, in float64.

Images are fixed material connections, not nearest-image recomputations. Positions
are unwrapped Cartesian coordinates; both them and the cell undergo affine shear.
The independent NumPy reference uses analytic gradients. JAX uses the same
physical law with scatter assembly, avoiding a dense Hessian or neighbor matrix.
"""

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def network_data(net):
    """Convert a Network to the unbatched array contract (including masks)."""
    data = {name: np.asarray(getattr(net, name)) for name in (
        "box", "edges", "images", "rest_lengths", "triplets",
        "triplet_images", "bend_lengths",
    )}
    data.update(node_mask=np.ones(len(net.positions), dtype=bool),
                edge_mask=np.ones(len(net.edges), dtype=bool),
                triplet_mask=np.ones(len(net.triplets), dtype=bool))
    return data


def numpy_evaluate(x, data, gamma, kappa):
    """Return energy, force=-gradient and total virial shear stress (mu=1)."""
    x = np.asarray(x, dtype=np.float64)
    box = np.asarray(data["box"], dtype=np.float64).copy()
    box[:, 0] += gamma * box[:, 1]
    edges, trips = data["edges"], data["triplets"]
    em, tm = data["edge_mask"], data["triplet_mask"]
    d = x[edges[:, 1]] - x[edges[:, 0]] + data["images"] @ box
    d = np.where(em[:, None], d, np.array([1., 0.]))
    length = np.maximum(np.linalg.norm(d, axis=-1), 1e-30)
    extension = length - data["rest_lengths"]
    edge_stretch = .5 * extension**2 / data["rest_lengths"] * em
    gd = (extension / (data["rest_lengths"] * length) * em)[:, None] * d

    a = x[trips[:, 1]] - x[trips[:, 0]] + data["triplet_images"][:, 0] @ box
    b = x[trips[:, 2]] - x[trips[:, 1]] + data["triplet_images"][:, 1] @ box
    a = np.where(tm[:, None], a, np.array([1., 0.]))
    b = np.where(tm[:, None], b, np.array([1., 0.]))
    la = np.maximum(np.linalg.norm(a, axis=-1), 1e-30)
    lb = np.maximum(np.linalg.norm(b, axis=-1), 1e-30)
    u, v = a / la[:, None], b / lb[:, None]
    diff = u - v
    coefficient = kappa / data["bend_lengths"] * tm
    triplet_bend = .5 * coefficient * np.sum(diff**2, axis=-1)
    ga = (coefficient / la)[:, None] * (diff - u * np.sum(u * diff, axis=-1)[:, None])
    gb = -(coefficient / lb)[:, None] * (diff - v * np.sum(v * diff, axis=-1)[:, None])
    forces = np.zeros_like(x)
    np.add.at(forces, edges[:, 0], gd)
    np.add.at(forces, edges[:, 1], -gd)
    np.add.at(forces, trips[:, 0], ga)
    np.add.at(forces, trips[:, 1], gb - ga)
    np.add.at(forces, trips[:, 2], -gb)
    forces *= data["node_mask"][:, None]
    stress = (np.sum(gd[:, 0] * d[:, 1])
              + np.sum(ga[:, 0] * a[:, 1] + gb[:, 0] * b[:, 1])) / abs(np.linalg.det(box))
    return dict(energy=edge_stretch.sum() + triplet_bend.sum(),
                stretch_energy=edge_stretch.sum(), bend_energy=triplet_bend.sum(),
                forces=forces, stress=stress, edge_stretch=edge_stretch,
                triplet_bend=triplet_bend)


@jax.jit
def jax_evaluate(x, data, gamma, kappa):
    """Device-resident equivalent of :func:`numpy_evaluate`; accepts JAX arrays."""
    box = jnp.asarray(data["box"]).at[:, 0].add(gamma * data["box"][:, 1])
    edges, trips = data["edges"], data["triplets"]
    em, tm = data["edge_mask"], data["triplet_mask"]
    d = x[edges[:, 1]] - x[edges[:, 0]] + data["images"] @ box
    d = jnp.where(em[:, None], d, jnp.array([1., 0.]))
    length = jnp.maximum(jnp.linalg.norm(d, axis=-1), 1e-30)
    extension = length - data["rest_lengths"]
    edge_stretch = .5 * extension**2 / data["rest_lengths"] * em
    gd = (extension / (data["rest_lengths"] * length) * em)[:, None] * d
    a = x[trips[:, 1]] - x[trips[:, 0]] + data["triplet_images"][:, 0] @ box
    b = x[trips[:, 2]] - x[trips[:, 1]] + data["triplet_images"][:, 1] @ box
    a = jnp.where(tm[:, None], a, jnp.array([1., 0.]))
    b = jnp.where(tm[:, None], b, jnp.array([1., 0.]))
    la = jnp.maximum(jnp.linalg.norm(a, axis=-1), 1e-30)
    lb = jnp.maximum(jnp.linalg.norm(b, axis=-1), 1e-30)
    u, v = a / la[:, None], b / lb[:, None]
    diff = u - v
    coefficient = kappa / data["bend_lengths"] * tm
    triplet_bend = .5 * coefficient * jnp.sum(diff**2, axis=-1)
    ga = (coefficient / la)[:, None] * (diff - u * jnp.sum(u * diff, axis=-1)[:, None])
    gb = -(coefficient / lb)[:, None] * (diff - v * jnp.sum(v * diff, axis=-1)[:, None])
    forces = jnp.zeros_like(x).at[edges[:, 0]].add(gd).at[edges[:, 1]].add(-gd)
    forces = forces.at[trips[:, 0]].add(ga).at[trips[:, 1]].add(gb - ga)
    forces = forces.at[trips[:, 2]].add(-gb) * data["node_mask"][:, None]
    stress = (jnp.sum(gd[:, 0] * d[:, 1])
              + jnp.sum(ga[:, 0] * a[:, 1] + gb[:, 0] * b[:, 1])) / jnp.abs(jnp.linalg.det(box))
    return dict(energy=edge_stretch.sum() + triplet_bend.sum(),
                stretch_energy=edge_stretch.sum(), bend_energy=triplet_bend.sum(),
                forces=forces, stress=stress, edge_stretch=edge_stretch,
                triplet_bend=triplet_bend)


def evaluate(positions, network, gamma=0., kappa=0., backend="numpy"):
    """Convenience API common to both backends; returns host NumPy values."""
    if backend not in ("numpy", "jax"):
        raise ValueError("backend must be numpy or jax")
    if not np.isfinite(gamma) or not np.isfinite(kappa) or kappa < 0:
        raise ValueError("finite shear and nonnegative bending stiffness required")
    fun = numpy_evaluate if backend == "numpy" else jax_evaluate
    return {k: np.asarray(v) for k, v in fun(
        np.asarray(positions, dtype=np.float64), network_data(network), gamma, kappa).items()}
