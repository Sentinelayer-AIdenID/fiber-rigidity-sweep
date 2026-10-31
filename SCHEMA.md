# Output schema, version 1

All numeric mechanics use float64; indices use integers and masks use booleans.
NPZ files contain ordinary arrays, never pickled Python objects. Read with
`np.load(path, allow_pickle=False)`. JSON uses `null` for unavailable scalar
estimates; NPZ uses NaN. Units: original lattice spacing and stretching modulus
are both one. Stress and moduli are energy per reference cell area.

`manifest.json` records the physical/numerical config and a fingerprint of it,
solver source, Python major/minor, numerical package versions, and JAX platform.
Output path and human notes do not change the fingerprint. Resume rejects a
mismatch. `timing.json` records wall seconds, convergence counts, FIRE steps,
node-iterations, backend and the results-array hash. Timing is not part of the
deterministic results hash.

## Topology

`network_zZZ_sSS.npz` stores one cleaned topology per connectivity/seed:

| Array | Shape | Meaning |
|---|---|---|
| positions | N × 2 | Unsheared reference node coordinates |
| box | 2 × 2 | **Row** periodic cell vectors |
| edges | M × 2 | Oriented node indices i,j |
| images | M × 2 | Integer image, so d = x[j] − x[i] + images @ sheared_box |
| rest_lengths | M | Reference segment lengths |
| triplets | T × 3 | Consecutive i,j,k on the same original fiber |
| triplet_images | T × 2 × 2 | Images for i→j and j→k, in that order |
| bend_lengths | T | Mean of the two adjacent reference segment lengths |
| seed, z_target | scalar | Realization identifier and requested coordination |

## Checkpoints

`checkpoints/zZZ_kKK_gGGG.npz` is one (z, kappa, strain-index) checkpoint.
Arrays `available`, `converged`, `steps`, `max_force`, `initial_energy`, `stress`,
`stretch_energy`, `bend_energy`, `nonaffinity` have shape S (seeds).
`positions` is S × Nmax × 2, padded with zero; `node_count[S]` identifies valid
coordinates. Positions are **unwrapped, already sheared** Cartesian coordinates.
`fingerprint` and `schema_version` are scalars.

`available=true` means this seed finished, either converged or exhausted/failed.
It never implies convergence. An interrupted in-progress seed has
`available=false` and is recomputed from its deterministic affine start. Writes
use a flushed temporary file and atomic rename; completed records are reused.
Nonconverged rows retain diagnostic energies/positions and `converged=false`.

## Results

`results.npz` axes are `(Z, KAPPA, S, G)`:

- `stress`, `stretch_energy`, `bend_energy`, `nonaffinity`, `K`, `max_force`,
  `initial_energy`, `steps`, `converged`: full four-dimensional arrays.
- `z_target[Z]`, `kappa[KAPPA]`, `seeds[S]`, `gamma[G]`: axis values.
- `z_actual[Z,S] = 2M/N`, `node_count[Z,S]`, `l_c[Z,S]`,
  `node_density[Z,S] = N/area`, `junction_density[Z,S]`: topology summaries.
  Junction density counts retained nodes of degree at least three per area;
  it is distinct from all-node density and coordination.
- `gamma_c_K`, `gamma_c_nonaffinity`, `agreement_log10`: shape `(Z,KAPPA,S)`,
  separate per-seed onset estimates; unresolved entries are NaN.
- `schema_version`, `fingerprint`: scalars.

Failed samples are excluded when computing K and onsets. K also requires
successful immediate neighboring strains (three-point nonuniform derivative).
Finite energy in a failed row is diagnostic, not a usable equilibrium.

## Viewer JSON

`viewer.json` contains `schema_version: 1`, `metadata`, `curves`, and
`network_frames`. Each curve represents one `(z_target,kappa)` with scalar
`z_actual` (mean across seeds), strain-vector `gamma`, and aligned vectors
`K`, `nonaffinity`, `stretch_energy`, `bend_energy`, `converged_count`,
`modulus_count`. Means include only converged, finite contributions.
`gamma_c_K`, `gamma_c_nonaffinity`, `agreement_log10` are **onsets of the mean
curves**, not means of the per-seed onsets. Onset neighborhoods require all
seeds converged; unresolved estimates are null.

Each curve also contains `critical_exponents`, keyed by `gamma_c_K` and
`gamma_c_nonaffinity`. Each entry fits the **mean** K′ above that onset at fixed
gamma_c using complete-seed derivative stencils. It contains:

- `status`: `determined` or `not determined`; `reason` is null or explanatory text.
- `gamma_c`, `f`, `f_stderr`, `f_ci95` (two-sided Student-t 95% interval),
  `r_squared` (log-space), `literature_f: 0.7`, `delta_f_literature: f - 0.7`.
  Unresolved exponent values and uncertainty are null, never sentinel numbers.
- `delta_window`: requested positive [min,max] distance above gamma_c, default
  [0.05 gamma_c, 0.5 gamma_c], or null if gamma_c is unresolved.
- `fit_window`: actual first/last usable **absolute strains**, or null if none;
  `n_points`, `point_indices`: selected count and indices into the curve's gamma.
  These diagnostics remain available when too few points prevent fitting.

At least five points spanning a factor of three in positive strain distance
are required. Uncertainty conditions on the onset and window, excluding onset,
seed, finite-size and crossover uncertainty. These are JSON analysis fields;
the NPZ schema and checkpoint format remain unchanged.

The root `critical_strain_trends` array has one entry per kappa and onset method:
`kappa`, `method`, `status` (`decreasing`, `increasing`, `flat`, `mixed`, or
`not determined`), `n_points`, `total_curves`, and `points` (resolved
`{z_actual,gamma_c}` pairs sorted by measured z with 0 < z < 4).
At least two distinct resolved connectivities are required for a direction.
`limit_at_z4` is always `not determined`: direction is not extrapolation.
The viewer accepts older version-1 exports lacking these additive fields and
shows fits/trends as not determined.

Frames contain `z_target`, `kappa`, `seed`, `gamma`, `converged`, `box`,
`positions`, `edges`, `edge_vectors`, `stretch_energy`, `bend_energy`,
`total_edges`, `shown_edges`. The first seed is exported for every z/kappa/
strain. At most `viewer_max_edges` evenly selected edges are shown, remapped
to the included nodes; this sampling never changes the simulation or NPZ data.
Starting positions are wrapped into the sheared cell, but `edge_vectors` are
true material-image displacements. Draw a line from `positions[i]` to
`positions[i] + edge_vectors[e]`; do not join the two independently wrapped
endpoints. Half of each hinge's bend energy is assigned to each neighboring
edge. Summed bend colors equal total bend energy only for an unsampled frame.

`viewer/sample.json` is a committed CPU output. `viewer/sample-data.js` embeds
the identical object as `window.RIGIDITY_SAMPLE`, allowing `file://` use without
fetch or a server. Load a new JSON through the viewer's file picker.
