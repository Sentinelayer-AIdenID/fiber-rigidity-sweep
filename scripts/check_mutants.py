#!/usr/bin/env python3
"""Run deliberately broken source copies; never modify the working source."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = [
    ("stretch-gradient-sign", "rigidity/energy.py", "gd = (extension", "gd = -(extension",
     "tests/test_physics.py::test_analytic_gradient_central_difference[stretch]"),
    ("bend-gradient-sign", "rigidity/energy.py", "ga = (coefficient", "ga = -(coefficient",
     "tests/test_physics.py::test_analytic_gradient_central_difference[bend]"),
    ("wrong-stretch-normalization", "rigidity/energy.py", "edge_stretch = .5 *", "edge_stretch = .6 *",
     "tests/test_physics.py::test_triangular_shear_modulus_energy_and_virial"),
    ("wrong-virial-sign", "rigidity/energy.py", "stress = (np.sum", "stress = -(np.sum",
     "tests/test_physics.py::test_triangular_shear_modulus_energy_and_virial"),
    ("omit-lees-edwards-offset", "rigidity/energy.py", "box[:, 0] += gamma * box[:, 1]",
     "box[:, 0] += 0. * box[:, 1]", "tests/test_physics.py::test_lees_edwards_affine_equilibrium[numpy]"),
    ("disable-bend-energy", "rigidity/energy.py", "triplet_bend = .5 * coefficient",
     "triplet_bend = 0. * coefficient", "tests/test_physics.py::test_bend_straight_zero_and_kink_positive"),
    ("fire-uphill-step", "rigidity/fire.py", 'candidate = state["x"] + dt[:, None, None] * velocity',
     'candidate = state["x"] - dt[:, None, None] * velocity',
     "tests/test_fire.py::test_fire_lowers_energy_and_converges"),
    ("retain-attached-rattlers", "rigidity/network.py",
     "retained = _load_bearing_edges(len(network.positions), network.edges[mask], network.images[mask])",
     "retained = np.ones(np.count_nonzero(mask), dtype=bool)",
     "tests/test_network.py::test_cleanup_removes_cycles_attached_at_articulation_or_bridge"),
    ("unseeded-randomness", "rigidity/network.py", "rng = np.random.default_rng(int(seed))",
     "rng = np.random.default_rng()", "tests/test_network.py::test_network_seed_determinism"),
    ("include-failed-stress", "rigidity/shear.py", "good = np.asarray(converged, bool) &",
     "good = np.ones_like(converged, dtype=bool) &",
     "tests/test_analysis.py::test_modulus_nonuniform_quadratic_and_failure_gap"),
    ("wrong-nonaffinity-normalization", "rigidity/analysis.py", "/ (lc * lc * gamma * gamma)",
     "/ (lc * lc)", "tests/test_analysis.py::test_nonaffinity_translation_and_normalization"),
    ("accept-incompatible-resume", "run_sweep.py", 'if k not in ("output", "notes")',
     'if k not in ("output", "notes", "seeds")', "tests/test_io_sweep.py::test_npz_schema_determinism_and_resume"),
    ("accept-smaller-interior-peak", "rigidity/analysis.py",
     'i = int(np.argmax(np.where(observed, a, -np.inf)))',
     'i = next((j for j in range(1, len(g)-1) if observed[j-1:j+2].all() and a[j] > a[j-1] and a[j] > a[j+1]), 0)',
     "tests/test_analysis.py::test_endpoint_global_peak_is_unresolved_even_with_interior_bump"),
    ("accept-incomplete-mean-stencil", "rigidity/io.py",
     'complete_modulus = good.all(axis=0) & np.isfinite(results["K"][iz, ik]).all(axis=0)',
     'complete_modulus = good.all(axis=0)',
     "tests/test_io_sweep.py::test_mean_modulus_onset_requires_complete_seed_stencils"),
    ("fit-unshifted-strain", "rigidity/analysis.py", "np.log(delta[selected])", "np.log(g[selected])",
     "tests/test_analysis.py::test_critical_exponent_recovers_known_power_and_uncertainty"),
    ("fit-failed-points", "rigidity/analysis.py", "selected = good &", "selected = np.ones_like(good) &",
     "tests/test_analysis.py::test_exponent_excludes_failed_nonpositive_and_out_of_window_points"),
    ("fit-incomplete-seed-stencils", "rigidity/io.py", 'mean["K"], complete_modulus, estimates[key]',
     'mean["K"], good.any(axis=0), estimates[key]',
     "tests/test_exponent_export.py::test_export_fits_complete_seed_stencils_and_keeps_both_onset_choices"),
    ("fit-too-few-points", "rigidity/analysis.py", "if n < min_points:", "if n < 3:",
     "tests/test_exponent_export.py::test_export_not_determined_when_convergence_leaves_too_few_fit_points"),
    ("trend-target-instead-of-measured-z", "rigidity/analysis.py", 'float(c["z_actual"]), float(c[method])',
     'float(c["z_target"]), float(c[method])',
     "tests/test_analysis.py::test_transition_trend_uses_measured_z_and_keeps_kappa_and_methods_separate"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="evidence/mutations.json")
    parser.add_argument("--only", choices=[row[0] for row in MUTANTS])
    args = parser.parse_args()
    records = []
    for name, filename, old, new, target in MUTANTS:
        if args.only and args.only != name:
            continue
        with tempfile.TemporaryDirectory(prefix=".mutation-", dir=ROOT) as directory:
            copy = Path(directory)
            for folder in ("rigidity", "tests", "configs"):
                shutil.copytree(ROOT / folder, copy / folder, ignore=shutil.ignore_patterns("__pycache__"))
            for file in ("run_sweep.py", "pytest.ini"):
                shutil.copy2(ROOT / file, copy / file)
            path = copy / filename
            source = path.read_text()
            if old not in source:
                raise RuntimeError(f"mutation anchor absent: {name}")
            # First occurrence intentionally mutates only NumPy for energy checks.
            path.write_text(source.replace(old, new, 1))
            completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", target],
                                       cwd=copy, text=True, capture_output=True, timeout=90)
            output = completed.stdout + completed.stderr
            killed = completed.returncode == 1 and ("AssertionError" in output or "Failed:" in output)
            record = dict(mutant=name, source=filename, replacement=dict(before=old, after=new),
                          test=target, returncode=completed.returncode, detected=killed,
                          tail=output.splitlines()[-7:])
            records.append(record)
            print(f"{name}: {'DETECTED' if killed else 'NOT DETECTED'} (exit {completed.returncode})", flush=True)
    report = ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(dict(method="Isolated source copies; working source unchanged", results=records), indent=2)+"\n")
    return 0 if records and all(row["detected"] for row in records) else 1


if __name__ == "__main__":
    sys.exit(main())
