"""Atomic, non-pickle checkpoints and compact static-viewer exports."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .analysis import critical_strains
from .energy import evaluate
from .shear import affine

SCHEMA_VERSION = 1


def array_hash(arrays):
    """Hash names, dtypes, shapes and array bytes, not ZIP timestamps."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        if value.dtype.hasobject:
            raise ValueError("object arrays are not allowed")
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finite_list(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def viewer_curves(results):
    curves = []
    for iz, z in enumerate(results["z_target"]):
        for ik, kappa in enumerate(results["kappa"]):
            good = results["converged"][iz, ik]
            mean = {}
            for key in ("K", "nonaffinity", "stretch_energy", "bend_energy"):
                data = results[key][iz, ik]
                valid = good & np.isfinite(data)
                count = valid.sum(axis=0)
                mean[key] = np.divide(np.where(valid, data, 0.).sum(axis=0), count,
                                      out=np.full(len(count), np.nan), where=count > 0)
            estimates = critical_strains(results["gamma"], mean["K"], mean["nonaffinity"],
                                         good.all(axis=0))
            curves.append(dict(z_target=float(z), z_actual=float(np.mean(results["z_actual"][iz])),
                               kappa=float(kappa), gamma=results["gamma"].tolist(),
                               **{key: finite_list(value) for key, value in mean.items()},
                               converged_count=good.sum(axis=0).tolist(),
                               modulus_count=np.sum(good & np.isfinite(results["K"][iz, ik]), axis=0).tolist(),
                               **estimates))
    return curves


def frame_projection(net, max_edges=1500):
    """Build the sampled-edge/hinge map once per network, not once per frame."""
    chosen = np.linspace(0, len(net.edges)-1, min(max_edges, len(net.edges)), dtype=int)
    nodes = np.unique(net.edges[chosen])
    remap = np.full(len(net.positions), -1, int)
    remap[nodes] = np.arange(len(nodes))
    dtype = np.dtype([(key, np.int64) for key in ("i", "j", "ix", "iy")])
    def keys(pair, images):
        return np.ascontiguousarray(np.column_stack([pair, images]), dtype=np.int64).view(dtype).ravel()
    selected_keys = keys(net.edges[chosen], net.images[chosen])
    ordering = np.argsort(selected_keys)
    sorted_keys = selected_keys[ordering]
    ti, ei = [], []
    for side in range(2):
        hinge_keys = keys(net.triplets[:, side:side+2], net.triplet_images[:, side])
        found = np.searchsorted(sorted_keys, hinge_keys)
        clipped = np.minimum(found, len(chosen)-1)
        present = (found < len(chosen)) & (sorted_keys[clipped] == hinge_keys)
        ti.append(np.flatnonzero(present))
        ei.append(ordering[found[present]])
    return dict(chosen=chosen, nodes=nodes, edges=remap[net.edges[chosen]],
                triplet_indices=np.concatenate(ti), edge_indices=np.concatenate(ei))


def network_frame(net, positions, gamma, kappa, converged, max_edges=1500, projection=None):
    """Sample edges for display only; full coordinates remain in checkpoints."""
    result = evaluate(positions, net, gamma, kappa)
    projection = frame_projection(net, max_edges) if projection is None else projection
    chosen, nodes = projection["chosen"], projection["nodes"]
    bend = np.zeros(len(chosen))
    # Half of each hinge energy is attributed to each neighboring segment.
    np.add.at(bend, projection["edge_indices"], .5 * result["triplet_bend"][projection["triplet_indices"]])
    box = affine(net.box, gamma)
    fractions = positions[nodes] @ np.linalg.inv(box)
    wrapped = (fractions - np.floor(fractions)) @ box
    vectors = positions[net.edges[chosen, 1]] - positions[net.edges[chosen, 0]] + net.images[chosen] @ box
    return dict(z_target=net.z_target, kappa=float(kappa), seed=net.seed, gamma=float(gamma),
                converged=bool(converged), positions=wrapped.tolist(),
                edges=projection["edges"].tolist(), edge_vectors=vectors.tolist(),
                stretch_energy=result["edge_stretch"][chosen].tolist(), bend_energy=bend.tolist(),
                box=box.tolist(), total_edges=len(net.edges), shown_edges=len(chosen))
