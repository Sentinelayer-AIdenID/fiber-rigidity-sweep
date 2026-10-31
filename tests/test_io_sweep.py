import json
from pathlib import Path

import numpy as np
import pytest

from rigidity.io import array_hash, load_npz, viewer_curves
from run_sweep import load_config, run


def tiny_config():
    return dict(nx=6, ny=6, z=[3.8], kappa=[.03], gamma=[.001, .01, .03, .1, .3],
                seeds=[3, 7], backend="jax", fire=dict(max_steps=5000, force_tol=1e-8,
                dt0=.02, chunk_steps=500), output="unused", batch_size=10, viewer_max_edges=50)


def test_npz_schema_determinism_and_resume(tmp_path):
    config = tiny_config()
    first = run(config, output=tmp_path / "first")
    second = run(config, output=tmp_path / "second")
    assert first["converged"].all()
    assert array_hash(first) == array_hash(second)
    assert array_hash(first) == array_hash(load_npz(tmp_path / "first/results.npz"))
    with pytest.raises(KeyboardInterrupt):
        run(config, output=tmp_path / "interrupted", stop_after_checkpoints=1)
    partial = list((tmp_path / "interrupted/checkpoints").glob("*.npz"))
    assert partial
    before = {p.name: p.read_bytes() for p in partial}
    resumed = run(config, output=tmp_path / "interrupted", resume=True)
    assert array_hash(resumed) == array_hash(first)
    # A completed checkpoint is reused byte-for-byte.
    for name, raw in before.items():
        assert (tmp_path / "interrupted/checkpoints" / name).read_bytes() == raw
    viewer = json.loads((tmp_path / "first/viewer.json").read_text())
    assert viewer["schema_version"] == 1
    assert len(viewer["curves"]) == 1
    assert len(viewer["network_frames"]) == len(config["gamma"])
    for frame in viewer["network_frames"]:
        assert len(frame["edges"]) == len(frame["edge_vectors"]) == len(frame["stretch_energy"])
        assert max(max(pair) for pair in frame["edges"]) < len(frame["positions"])
        assert frame["converged"] is True
    assert first["stress"].shape == (1, 1, 2, 5)
    for key in ("stress", "K", "nonaffinity", "stretch_energy", "bend_energy"):
        assert first[key].dtype == np.float64
        assert np.isfinite(first[key]).all()
    with pytest.raises(ValueError, match="already contains"):
        run(config, output=tmp_path / "first")
    changed = {**config, "seeds": [3, 8]}
    with pytest.raises(ValueError, match="differs"):
        run(changed, output=tmp_path / "first", resume=True)


def test_exhausted_steps_persist_and_never_enter_fits(tmp_path):
    config = tiny_config()
    config["fire"]["max_steps"] = 0
    result = run(config, output=tmp_path)
    assert not result["converged"].any()
    assert np.isnan(result["K"]).all()
    viewer = json.loads((tmp_path / "viewer.json").read_text())
    curve = viewer["curves"][0]
    assert curve["gamma_c_K"] is None
    assert curve["gamma_c_nonaffinity"] is None
    assert curve["K"] == [None]*5
    assert curve["converged_count"] == [0]*5
    assert all(not frame["converged"] for frame in viewer["network_frames"])
    for path in (tmp_path / "checkpoints").glob("*.npz"):
        cp = load_npz(path)
        assert cp["available"].all()
        assert not cp["converged"].any()


def test_config_grid_contract_and_invalid_input(tmp_path):
    full = load_config(Path(__file__).parents[1] / "configs/h100.yaml")
    assert len(full["z"]) == 7
    assert len(full["kappa"]) == 4
    assert len(full["seeds"]) == 10
    np.testing.assert_allclose(full["gamma"], np.geomspace(.001, 1., 20))
    assert full["batch_size"] == 200
    path = tmp_path / "bad.yaml"
    import yaml
    cfg = tiny_config()
    cfg["gamma"] = [.1, float("nan"), .2]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="finite"):
        load_config(path)


def test_mean_modulus_onset_requires_complete_seed_stencils():
    shape = (1, 1, 2, 7)
    values = dict(z_target=np.array([3.5]), kappa=np.array([.01]), z_actual=np.array([[3.5, 3.5]]),
                  gamma=np.geomspace(.001, 1., 7), converged=np.ones(shape, bool))
    values["K"] = np.broadcast_to([1., 1., 1., 1., 3., 4., 5.], shape).copy()
    values["K"][0, 0, 0, 5] = np.nan  # A failed derivative stencil in one realization.
    for key in ("nonaffinity", "stretch_energy", "bend_energy"):
        values[key] = np.broadcast_to([1., 2., 3., 5., 3., 2., 1.], shape).copy()
    curve = viewer_curves(values)[0]
    assert curve["gamma_c_K"] is None
    assert curve["gamma_c_nonaffinity"] == values["gamma"][3]
    assert curve["modulus_count"][5] == 1
