#!/usr/bin/env python3
"""Run independent affine-start shear minimizations and atomically checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from rigidity.analysis import critical_strains, nonaffinity
from rigidity.energy import evaluate
from rigidity.fire import minimize_batch
from rigidity.io import (SCHEMA_VERSION, array_hash, atomic_json, atomic_npz, load_npz,
                         frame_projection, network_frame, viewer_curves)
from rigidity.network import generate_network
from rigidity.shear import affine, differential_modulus


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    required = {"nx", "ny", "z", "kappa", "gamma", "seeds", "backend", "fire", "output"}
    if not required <= config.keys():
        raise ValueError(f"missing config fields: {sorted(required-config.keys())}")
    for key in ("z", "kappa", "gamma"):
        vals = np.asarray(config[key], dtype=float)
        if vals.ndim != 1 or not len(vals) or not np.all(np.isfinite(vals)):
            raise ValueError(f"{key} must be a nonempty finite vector")
        if np.any(np.diff(vals) <= 0):
            raise ValueError(f"{key} must be strictly increasing")
        config[key] = vals.tolist()
    if len(config["gamma"]) < 3 or min(config["gamma"]) <= 0 or min(config["kappa"]) < 0:
        raise ValueError("need at least three positive strains and nonnegative kappa")
    seeds = config["seeds"]
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("seeds must be distinct nonnegative integers")
    if config["backend"] not in ("numpy", "jax"):
        raise ValueError("backend must be numpy or jax")
    config.setdefault("batch_size", len(seeds)*len(config["gamma"]))
    config.setdefault("viewer_max_edges", 1500)
    for key in ("batch_size", "viewer_max_edges"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    return config


def fingerprint(config):
    # Changing solver source or physical config invalidates old checkpoints.
    clean = {k: v for k, v in config.items() if k not in ("output", "notes")}
    digest = hashlib.sha256(json.dumps(clean, sort_keys=True).encode())
    for path in sorted((Path(__file__).parent / "rigidity").glob("*.py")) + [Path(__file__)]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _empty_checkpoint(networks, stamp):
    size, nodes = len(networks), max(len(n.positions) for n in networks)
    data = {key: np.full(size, np.nan) for key in
            ("stress", "stretch_energy", "bend_energy", "nonaffinity", "max_force", "initial_energy")}
    data.update(positions=np.zeros((size, nodes, 2)), converged=np.zeros(size, bool),
                available=np.zeros(size, bool), steps=np.zeros(size, np.int64),
                fingerprint=np.asarray(stamp), schema_version=np.asarray(SCHEMA_VERSION),
                node_count=np.asarray([len(n.positions) for n in networks]))
    return data


def run(config, *, resume=False, output=None, stop_after_checkpoints=None):
    config = dict(config)
    root = Path(output or config["output"])
    stamp = fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    if manifest.exists():
        if not resume:
            raise ValueError("output already contains a run; use --resume or another --output")
        recorded = json.loads(manifest.read_text())
        if recorded["fingerprint"] != stamp:
            raise ValueError("resume refused: config or solver source differs")
    elif any(root.iterdir()):
        raise ValueError("output is not empty and has no run manifest")
    else:
        atomic_json(manifest, dict(schema_version=SCHEMA_VERSION, fingerprint=stamp, config=config))
    shape = (len(config["z"]), len(config["kappa"]), len(config["seeds"]), len(config["gamma"]))
    results = {key: np.full(shape, np.nan) for key in
               ("stress", "stretch_energy", "bend_energy", "nonaffinity", "K", "max_force", "initial_energy")}
    results.update(converged=np.zeros(shape, bool), steps=np.zeros(shape, np.int64),
                   z_target=np.asarray(config["z"]), kappa=np.asarray(config["kappa"]),
                   gamma=np.asarray(config["gamma"]), seeds=np.asarray(config["seeds"], np.int64),
                   z_actual=np.zeros((shape[0], shape[2])), node_count=np.zeros((shape[0], shape[2]), np.int64),
                   node_density=np.zeros((shape[0], shape[2])), junction_density=np.zeros((shape[0], shape[2])),
                   l_c=np.zeros((shape[0], shape[2])), schema_version=np.asarray(SCHEMA_VERSION),
                   fingerprint=np.asarray(stamp))
    for key in ("gamma_c_K", "gamma_c_nonaffinity", "agreement_log10"):
        results[key] = np.full(shape[:3], np.nan)
    frames, writes = [], 0
    start_time = time.perf_counter()
    for iz, z in enumerate(config["z"]):
        networks = [generate_network(config["nx"], config["ny"], z, seed) for seed in config["seeds"]]
        projection = frame_projection(networks[0], config["viewer_max_edges"])
        for si, net in enumerate(networks):
            results["z_actual"][iz, si] = net.z_actual
            results["node_count"][iz, si] = len(net.positions)
            results["l_c"][iz, si] = net.l_c
            area = abs(np.linalg.det(net.box))
            results["node_density"][iz, si] = len(net.positions) / area
            results["junction_density"][iz, si] = np.count_nonzero(
                np.bincount(net.edges.ravel(), minlength=len(net.positions)) >= 3) / area
            atomic_npz(root / f"network_z{iz:02d}_s{si:02d}.npz",
                       **{k: np.asarray(v) for k, v in vars(net).items()})
        for ik, kappa in enumerate(config["kappa"]):
            paths = [root / "checkpoints" / f"z{iz:02d}_k{ik:02d}_g{gi:03d}.npz"
                     for gi in range(shape[3])]
            checkpoints = []
            for path in paths:
                cp = load_npz(path) if path.exists() else _empty_checkpoint(networks, stamp)
                if str(cp["fingerprint"]) != stamp:
                    raise ValueError(f"checkpoint fingerprint mismatch: {path.name}")
                checkpoints.append(cp)
            pairs = [(si, gi) for si in range(shape[2]) for gi in range(shape[3])]
            for offset in range(0, len(pairs), config["batch_size"]):
                group = pairs[offset:offset+config["batch_size"]]
                if all(checkpoints[gi]["available"][si] for si, gi in group):
                    continue
                group_networks = [networks[si] for si, gi in group]
                initial = [checkpoints[gi]["positions"][si, :len(networks[si].positions)]
                           if checkpoints[gi]["available"][si] else affine(networks[si].positions, config["gamma"][gi])
                           for si, gi in group]

                def save_finished(snapshot):
                    nonlocal writes
                    dirty = set()
                    for bi, (si, gi) in enumerate(group):
                        cp = checkpoints[gi]
                        terminal = (snapshot["converged"][bi] or snapshot["failed"][bi]
                                    or snapshot["steps"][bi] >= config["fire"].get("max_steps", 10000))
                        if cp["available"][si] or not terminal:
                            continue
                        net, gamma = networks[si], config["gamma"][gi]
                        x = snapshot["positions"][bi, :len(net.positions)]
                        physical = evaluate(x, net, gamma, kappa)
                        for key in ("stress", "stretch_energy", "bend_energy"):
                            cp[key][si] = physical[key]
                        cp["nonaffinity"][si] = nonaffinity(x, net.positions, gamma, net.l_c)
                        cp["positions"][si, :len(x)] = x
                        for key in ("converged", "steps", "max_force", "initial_energy"):
                            cp[key][si] = snapshot[key][bi]
                        cp["available"][si] = True
                        dirty.add(gi)
                    for gi in sorted(dirty):
                        atomic_npz(paths[gi], **checkpoints[gi])
                        if checkpoints[gi]["available"].all():
                            writes += 1
                            if stop_after_checkpoints is not None and writes >= stop_after_checkpoints:
                                raise KeyboardInterrupt("requested checkpoint interruption")

                snapshot = minimize_batch(group_networks,
                    [config["gamma"][gi] for si, gi in group], [kappa]*len(group),
                    backend=config["backend"], initial_positions=initial, callback=save_finished,
                    skip_mask=[checkpoints[gi]["available"][si] for si, gi in group],
                    **config["fire"])
                save_finished(snapshot)
            for gi, cp in enumerate(checkpoints):
                if not cp["available"].all():
                    raise RuntimeError("incomplete checkpoint after minimization")
                for key in ("stress", "stretch_energy", "bend_energy", "nonaffinity", "max_force",
                            "initial_energy", "steps", "converged"):
                    results[key][iz, ik, :, gi] = cp[key]
                # First seed only in browser; all seeds' full positions on disk.
                net = networks[0]
                frames.append(network_frame(net, cp["positions"][0, :len(net.positions)],
                                            config["gamma"][gi], kappa, cp["converged"][0],
                                            config["viewer_max_edges"], projection))
            for si in range(shape[2]):
                results["K"][iz, ik, si] = differential_modulus(config["gamma"],
                    results["stress"][iz, ik, si], results["converged"][iz, ik, si])
                estimated = critical_strains(config["gamma"], results["K"][iz, ik, si],
                    results["nonaffinity"][iz, ik, si], results["converged"][iz, ik, si])
                for key, value in estimated.items():
                    results[key][iz, ik, si] = np.nan if value is None else value
            count = int(results["converged"][iz, ik].sum())
            print(f"z={z:.2f} kappa={kappa:g}: {count}/{shape[2]*shape[3]} converged", flush=True)
    curves = viewer_curves(results)
    atomic_npz(root / "results.npz", **results)
    document = dict(schema_version=SCHEMA_VERSION,
                    metadata=dict(description="Small-network CPU demonstration" if config["nx"]*config["ny"] < 10000
                                  else "Fiber-network shear sweep", backend=config["backend"],
                                  simulation="independent affine-start minimizations", fingerprint=stamp,
                                  seeds=config["seeds"], sample_only=False,
                                  nonconverged_excluded=True, viewer_first_seed_only=True),
                    curves=curves, network_frames=frames)
    atomic_json(root / "viewer.json", document)
    elapsed = time.perf_counter() - start_time
    summary = dict(elapsed_seconds=elapsed, converged=int(results["converged"].sum()), total=int(np.prod(shape)),
                   arrays_sha256=array_hash(results), backend=config["backend"],
                   total_fire_steps=int(results["steps"].sum()),
                   node_iterations=float(np.sum(results["steps"] * results["node_count"][:, None, :, None])))
    atomic_json(root / "timing.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", help="override the output directory")
    parser.add_argument("--stop-after-checkpoints", type=int, help="controlled interruption for resume verification")
    args = parser.parse_args()
    try:
        run(load_config(args.config), resume=args.resume, output=args.output,
            stop_after_checkpoints=args.stop_after_checkpoints)
    except KeyboardInterrupt:
        print("Interrupted. Atomic completed checkpoints remain; rerun with --resume.", file=sys.stderr)
        return 130
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
