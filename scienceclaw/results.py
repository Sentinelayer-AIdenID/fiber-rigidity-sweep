"""Project completed sweep arrays through the existing analysis contract."""

import json
from pathlib import Path

import numpy as np

from rigidity.io import load_npz, viewer_curves


LIMITATIONS = (
    "Small CPU samples are numerical demonstrations, not reproductions of published "
    "critical curves. Finite size, sparse strain grids and limited seeds can leave "
    "critical strains and exponents not determined. Failed equilibria are excluded. "
    "Exponent uncertainty is conditional on the onset and fit window; it excludes "
    "onset, seed, finite-size and crossover uncertainty. No agreement with literature, "
    "H100 performance or thermodynamic-limit behavior is established by this report."
)


def determined(value):
    return "not determined" if value is None else value


def load_report(output, repo_root):
    """Read final artifacts, reject mismatches, and reuse the lane's analysis."""
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    viewer = json.loads((output / "viewer.json").read_text(encoding="utf-8"))
    arrays = load_npz(output / "results.npz")
    if (str(arrays["fingerprint"]) != manifest["fingerprint"]
            or viewer["metadata"]["fingerprint"] != manifest["fingerprint"]):
        raise ValueError("sweep artifact fingerprints differ")
    if int(arrays["schema_version"]) != 1 or viewer["schema_version"] != 1:
        raise ValueError("unsupported sweep schema")
    curves = viewer_curves(arrays)
    rows = []
    for curve in curves:
        fits = {}
        for method, fit in curve["critical_exponents"].items():
            fits[method] = {**fit, **{key: determined(fit[key]) for key in
                                   ("f", "f_stderr", "f_ci95")}}
        counts = curve["converged_count"]
        rows.append(dict(z=curve["z_target"], kappa=curve["kappa"],
                         measured_z=curve["z_actual"],
                         gamma_c_modulus=determined(curve["gamma_c_K"]),
                         gamma_c_nonaffinity=determined(curve["gamma_c_nonaffinity"]),
                         agreement_log10=determined(curve["agreement_log10"]),
                         critical_exponents=fits, converged_count=counts,
                         converged=sum(counts),
                         total=len(arrays["seeds"])*len(arrays["gamma"])))
    count = int(np.count_nonzero(arrays["converged"]))
    total = int(arrays["converged"].size)
    return dict(summary=f"{len(rows)} (z, kappa) curves; {count}/{total} states converged.",
                results=rows, artifacts=dict(npz=str(output / "results.npz"),
                    viewer_json=str(output / "viewer.json"),
                    viewer_html=str(Path(repo_root).resolve() / "viewer/index.html")),
                parameters=manifest["config"], limitations=LIMITATIONS)
