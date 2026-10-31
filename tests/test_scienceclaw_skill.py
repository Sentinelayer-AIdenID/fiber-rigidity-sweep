import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from rigidity.io import load_npz, viewer_curves


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "scienceclaw/skills/fiber-network-rigidity"
SCRIPT = SKILL / "scripts/fiber_network_rigidity.py"


def test_skill_frontmatter():
    frontmatter = yaml.safe_load((SKILL / "SKILL.md").read_text().split("---", 2)[1])
    assert frontmatter["name"] == SKILL.name
    assert isinstance(frontmatter["description"], str) and frontmatter["description"].strip()
    assert (SKILL / "requirements.txt").read_text() == (ROOT / "requirements.txt").read_text()


def test_skill_json_contract_and_copied_install(tmp_path):
    config = dict(nx=6, ny=6, z=[3.8], kappa=[.03], gamma=[.001, .01, .1],
                  seeds=[3], backend="numpy", fire=dict(max_steps=1000, force_tol=1e-7),
                  output="unused", viewer_max_edges=20)
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config))
    # Exercise the actual copied-folder installation, with no cwd imports.
    copied = tmp_path / SKILL.name
    shutil.copytree(SKILL, copied)
    output = tmp_path / "results"
    command = [sys.executable, str(copied / "scripts/fiber_network_rigidity.py"),
               "--config", str(config_path), "--out", str(output),
               "--backend", "numpy", "--format", "json"]
    environment = {**os.environ, "RIGIDITY_SWEEP_ROOT": str(ROOT)}
    process = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert {"summary", "results", "artifacts", "parameters"} <= result.keys()
    assert isinstance(result["summary"], str) and "states converged" in result["summary"]
    assert len(result["results"]) == 1
    row = result["results"][0]
    assert {"z", "kappa", "measured_z", "gamma_c_modulus", "gamma_c_nonaffinity",
            "agreement_log10", "critical_exponents", "converged_count", "converged", "total"} <= row.keys()
    arrays = load_npz(output / "results.npz")
    curve = viewer_curves(arrays)[0]
    assert row["measured_z"] == float(arrays["z_actual"].mean())
    assert row["converged"] == int(np.count_nonzero(arrays["converged"]))
    assert row["total"] == 3
    assert row["converged_count"] == curve["converged_count"]
    for name, original in [("gamma_c_modulus", "gamma_c_K"),
                           ("gamma_c_nonaffinity", "gamma_c_nonaffinity"),
                           ("agreement_log10", "agreement_log10")]:
        expected = curve[original] if curve[original] is not None else "not determined"
        assert row[name] == expected
    for method in ["gamma_c_K", "gamma_c_nonaffinity"]:
        fit = row["critical_exponents"][method]
        assert fit["status"] == "not determined"  # Three strains cannot fit an exponent.
        assert fit["f"] == fit["f_stderr"] == fit["f_ci95"] == "not determined"
    assert all(Path(path).is_file() for path in result["artifacts"].values())
    assert result["parameters"]["backend"] == "numpy"
    assert result["parameters"]["resume"] is False
    checkpoint = next((output / "checkpoints").glob("*.npz"))
    before = checkpoint.read_bytes()
    resumed = subprocess.run(command + ["--resume"], cwd=tmp_path, env=environment,
                             capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["results"] == result["results"]
    assert checkpoint.read_bytes() == before


def test_skill_bad_config_fails_without_json(tmp_path):
    process = subprocess.run([sys.executable, str(SCRIPT), "--config", str(tmp_path / "missing"),
                              "--format", "json"], capture_output=True, text=True)
    assert process.returncode == 1
    assert process.stdout == ""
    assert "Sweep failed" in process.stderr
