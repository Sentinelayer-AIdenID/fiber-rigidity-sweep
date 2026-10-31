# Rigidity Sweep

An MIT-licensed, athermal 2D fiber-network simulation and static viewer for
exploring strain stiffening. NumPy provides an analytic reference; JAX runs
independent FIRE minimizers as a padded batch in float64. CPU physics tests,
reproducibility, interruption/resume and deliberate broken-code checks accompany
the code. The included small sample is a numerical demonstration, **not a
reproduction of published critical curves**. No accelerator run has been made.

## Run on CPU

Python 3.12 and Node.js (only for viewer checks) are used for the proofs.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# requirements.lock records the complete CPU environment used for the evidence.
pytest -q
python run_sweep.py --config configs/cpu_small.yaml
python run_sweep.py --config configs/cpu_small.yaml --resume
node --check viewer/app.js
node --test tests/test_viewer.mjs
python scripts/check_mutants.py
```

The small config has two connectivities, two bending stiffnesses, nine strains,
two seeds, and 64 precursor sites: 72 independent states. The output directory
is relative to the current working directory; run commands from this repository.
Use `--output another-directory` for a separate experiment. Existing data require
`--resume`. Ctrl-C is safe at any time: atomic completed seed records survive;
unfinished seed/strain states restart from their original affine positions.
For a controlled interruption, add `--stop-after-checkpoints 1` (exit 130), then
rerun with `--resume`. A checkpoint exists for each z/kappa/strain index.

Same config/seed/backend and pinned environment give identical CPU arrays in
the tests. Hashes include names, shapes, dtypes and array bytes, excluding ZIP
timestamps and wall timing. Resume rejects changed config, solver source,
numerical versions, Python major/minor or JAX platform. GPU bitwise
reproducibility and behavior across hardware/compiler versions are untested.

## Run on one H100

The following commands are provided for the participant's machine; **they were
not executed on an accelerator**. Install a compatible CUDA-capable driver first.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install "jax[cuda12]==0.12.0.1"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c 'import jax; print(jax.devices()); assert jax.devices()[0].platform == "gpu"'
python run_sweep.py --config configs/h100.yaml
python run_sweep.py --config configs/h100.yaml --resume
```

The pinned command is the version-specific form of `pip install "jax[cuda12]"`.
GPU dependencies beyond the CPU lock have not been validated here. Record
`pip freeze` on the target machine alongside the run. `JAX_PLATFORMS=cuda`
prevents an unintended CPU fallback for this expensive grid.

The full config specifies z = 3.2…3.8 in 0.1 increments, kappa = 10⁻⁵…10⁻²
in decades, 20 log strains 10⁻³…1, and ten seeds: 5,600 states. For each
z/kappa, all 200 seed/strain states form one JAX batch. Kappa groups run
sequentially. `batch_size` can split that batch without changing the physical
protocol; very large networks require this. Padding masks cover nodes, edges
and bending hinges, with independent time step and convergence for every state.
Completed rows in a partially resumed batch are skipped without changing shape.

The default 400×250 grid has 100,000 **precursor** sites; cleanup removes some,
so use `node_count`, not the precursor count, for actual system size. To study
approximately one million sites, copy the config and set nx=1000, ny=1000;
choose a smaller batch and measure actual node count and memory before the
full sweep. ny must be even. Larger precursor dimensions may be needed to meet
an exact retained-node target. These sizes have not been mechanically solved here.

## Conventions and assumptions

- **Phantom construction:** start from a periodic triangular lattice of unit
  spacing. At every site, randomly choose one of the three fiber directions to
  bypass the crosslink; the other two share a freely rotating crossing. Contract
  bypass sites into straight, longer segments between crosslinks. This gives
  coordination four before dilution without counting uncoupled crossings as
  extra crosslinks. It is our coarse-grained variant, not an asserted exact
  implementation of any cited paper. Internal curvature within a contracted
  segment is not discretized.
- **Dilution:** a seeded random permutation of contracted segments defines
  nested uniform bond deletions. Choose retention to approach requested z after
  cleanup. Longer and shorter contracted segments have the same deletion
  probability; this differs from independent deletion of elementary unit bonds.
  After degree<2 pruning, an iterative block decomposition removes finite
  dangling components and cycles attached by one articulation or bridge. Keep
  winding blocks and paths between them. Degree-two bending hinges and internal
  floppy mechanisms of the load-bearing network remain. Report **actual**
  z=2M/N after cleanup; small systems cannot attain every target exactly.
- **Stretch:** E_s = sum [mu/(2 l0)] (l−l0)², with mu=1 and l0 the
  reference segment length. Thus every bond in the full triangular reference
  has spring constant one. Compression is harmonic; no tension-only cutoff,
  thermal forces, excluded volume or prestress is introduced.
- **Bend:** for consecutive directed segments a,b of the same original fiber,
  E_b = sum [kappa/(2 l_b)] |a/|a|−b/|b||²
  = sum [kappa/l_b] (1−cos theta), where l_b is the mean of the two
  reference lengths. Straight fibers have zero energy. This agrees with
  kappa theta²/(2 l_b) at small angles, and stays smooth at a straight state.
  Different fibers are not coupled by a bending spring at their crossing.
- **Lees–Edwards shear:** x→x+gamma*y at fixed area; row box vectors transform
  by the same shear. Bond images are fixed integer material connections.
  Forces use unwrapped coordinates plus images times the **sheared** box;
  images are never recomputed by a nearest-image rule. Translation is removed
  as a null mode. Singular collapsed active segments fail convergence.
- **Loading:** each strain starts independently from its affine image of the
  zero-strain reference, permitting seeds and strains in one batch. This is
  not quasistatic continuation through neighboring strains. Metastability,
  buckling and different basin selection may create nonmonotone stress or
  nonaffinity, especially in tiny periodic systems.
- **FIRE:** semi-implicit Euler update, unit fictitious node mass, alpha0=0.1,
  f_inc=1.1, f_dec=0.5, f_alpha=0.99, N_min=5, dt_max=10 dt0.
  Default dt0=.02; each state has its own velocity, dt and alpha. Convergence
  is maximum active-node force norm ≤ force_tol, not a small energy change.
  Reaching max_steps or nonfinite mechanics stores converged=false; diagnostic
  energies remain, but such rows never enter K or onset analysis. FIRE finds
  a force-balanced local state, not a certified global minimum.

## Stress, nonaffinity and transition estimates

The total shear virial is sigma = (dE/dgamma)_affine / area, including both
stretch and bend contributions. With g_d=dE_s/dd and g_a,g_b the hinge
gradients, it is sum(g_d,x d_y + g_a,x a_y + g_b,x b_y)/area. The bending
term must not be omitted. We measure K′ = d sigma/d gamma with the derivative
of the three-point quadratic interpolant on the actual nonuniform strain grid.
Interior points use immediate neighbors; boundaries use one-sided three-point
stencils. A failed point invalidates every stencil containing it. No stencil
bridges a failure gap. Negative K can occur and is saved; the logarithmic viewer
heatmap masks nonpositive K rather than plotting its absolute value.

Nonaffinity is mean |u−u_affine|² / (l_c² gamma²), with displacement measured
from the zero-strain reference, translation removed, and l_c the mean retained
reference **segment** length. It is the total-displacement definition requested
here, not the incremental nonaffinity often used in criticality analyses.

Two **operational grid estimates** of the critical strain:

1. K′ onset: first interior point after the first three samples that reaches
   twice their median K′, rises from the preceding point, and stays above that
   threshold at the next point. The baseline must be positive and all required
   values usable. The assumed low-strain plateau may be absent in tiny samples;
   this threshold is then not a reliable physical transition.
2. Nonaffinity onset: highest observed value, accepted only if it is a strict
   interior peak with successful immediate neighbors. An endpoint global maximum
   is unresolved, even if a smaller interior bump exists. Flat or monotone curves
   produce no peak estimate.

Both return null/NaN when unresolved. `agreement_log10` is
log10(gamma_c,K / gamma_c,NA): zero means agreement, ±1 a factor of ten.
Individual seeds' estimates are in NPZ; the viewer overlays onsets of mean
curves, using converged finite contributions and complete-seed onset neighborhoods.
The sample's resolved pair differs by **1.875 decades** (factor about 75);
several K′ estimates are unresolved. This disagreement is displayed, not hidden.
Coarse strain spacing, finite size, periodic surviving fibers and the difference
between total and incremental nonaffinity prevent interpreting these points as
published critical strains. Refine the strain grid around candidate transitions,
increase network size and compare seed distributions before drawing conclusions.

To study crosslink density, compare gamma_c against **measured** z and retained
junction density (degree≥3 nodes/area) from the output. Coordination and density
are different controls; this dilution protocol changes both. The small sample
does not establish a monotone density-to-transition law.

### Conditional critical-exponent fits

`rigidity.analysis.fit_critical_exponent(gamma, modulus, converged, gamma_c)`
fits K′ = A (gamma − gamma_c)^f above a **fixed** transition estimate by
unweighted least squares in log K′ versus log(gamma − gamma_c). The slope is
free; it is never constrained to the approximately **0.7** reported for the
sub-isostatic strain-controlled class by Sharma et al. (2016). The viewer reports
f − 0.7 for comparison. This is an effective finite-window fit, not evidence of
a universal exponent or a reproduction of the paper.

The default distance window is **0.05 gamma_c ≤ gamma − gamma_c ≤ 0.5 gamma_c**;
an explicit `window=(minimum_distance, maximum_distance)` overrides it. Choose
the window before fitting and check sensitivity to it. At least five converged,
finite, positive modulus points spanning a factor of three in distance are
required. The export fits the mean K′ only where **every seed has converged and
has a finite stress-difference stencil**, separately for each onset method and
kappa. Failed points, missing derivatives and points outside the window cannot
enter the fit. An unresolved onset, insufficient data or nonpositive fitted
slope gives **"not determined"**, with null f and a reason; no fallback number
is substituted. The function reports selected indices, actual strain bounds,
requested distance bounds, point count and log-space R².

Uncertainty is the regression standard error and a two-sided Student-t 95%
interval (n−2 degrees of freedom). Both are **conditional on the chosen onset
and window**. They exclude uncertainty in gamma_c, seed variability, correlated
finite-difference errors, finite size and bending crossover. No bending plateau
is subtracted. Finite kappa rounds the transition and may prevent a power law;
inspect fit quality and refine the grid, kappa and system size before physical
interpretation. Neither window nor gamma_c is optimized to obtain 0.7.

The bundled CPU grid supports **no exponent fit** for either onset method: it
has fewer than five usable points in every prescribed window. Its measured
connectivities are approximately 3.1905 and 3.8125. At kappa=0.001, the
nonaffinity onset **increases** from 0.0056234 to 0.0133352; at kappa=0.01 it is
**flat** at 0.0749894. Modulus-onset trends are not determined (one resolved
connectivity at kappa=0.001, none at kappa=0.01). Thus these data **do not show**
the expected falling gamma_c as z approaches the 2D central-force isostatic
point z=4. A finite-grid direction check cannot establish a zero limit at z=4;
that limit is reported as not determined. The trend calculation keeps kappa and
onset methods separate and sorts by measured, cleaned z, not the dilution target.

## Viewer and files

Open `viewer/index.html` directly. The committed CPU sample is embedded locally,
so no server is required; an internet connection loads pinned Plotly 2.35.2 and
regl 2.1.1 scripts. Without those CDNs, the WebGL network has a Canvas fallback,
but interactive Plotly charts require Plotly. Publish the `viewer` directory as
a static site or use a static-pages host. Choose a new `viewer.json` with the
file picker; no upload or backend is involved.

Choose kappa and z; the heatmap shows log10 K′ with both resolved gamma_c lines.
Gray/missing cells represent absent or nonpositive modulus. The response plot
compares K′ and nonaffinity. The strain slider selects network frames; colors
show stretch or bend energy, normalized to a fixed scale across saved strains
for the chosen network. Failed frames are explicitly marked. Periodic edge
vectors may extend outside the cell; they are not long bonds joining wrapped
endpoints. Large exports sample at most 1,200 edges and the first seed for
display; all states' full coordinates remain in NPZ checkpoints.

The transition table lists measured z, both gamma_c estimates, f ± its standard
error, the used strain window and point count, and f − 0.7 for the selected
kappa. Hover over a fitted exponent for the conditional 95% interval. Unresolved
fits show their reason. The text below the table reports the observed direction
of gamma_c(z) for each estimator and the unresolved z=4 limit. Older JSON
exports without fits still load and show "not determined".

See [SCHEMA.md](SCHEMA.md) for axes, units, masks and file layout. No wall-clock
timing is included in results-array hashes. `evidence/mutations.json` records
which deliberately incorrect mechanics were rejected, with exact replacement
strings and test targets. `scripts/check_mutants.py` tests disposable copies and
never modifies the working source.

## Performance evidence and limits

Measured CPU demonstration: 72/72 states converged in 5.34 seconds on a
16-CPU host. The warmed CPU benchmark in `evidence/cpu_benchmark.json` measures
8 states × 1,019 retained nodes × 1,000 steps, approximately 6.48 million
node-iterations/s; topology generation, compilation warmup and export are excluded.
CPU throughput is workload- and machine-specific; rerun `python scripts/benchmark.py`.

**ESTIMATED H100 runtime:** assume the full 100,000-site grid takes 10,000–100,000
steps per state and achieves an unmeasured 10×–100× the above CPU throughput.
The formula is 5,600 × 100,000 × steps / (6.48e6 × speedup), giving approximately
2.4 hours–10 days for minimization. This wide range is a planning assumption,
not a GPU benchmark or a convergence prediction. Low kappa and near-critical
states may hit max_steps. Generation, JSON/NPZ compression, transfers and JIT
compilation add time. **ESTIMATED** million-site runtime is roughly ten times
larger under a linear-scaling assumption that is itself untested.

**ESTIMATED H100 memory:** a CPU topology measurement at z≈3.8 uses about
196 bytes/node/state; positions, velocity and force add 48 bytes/node/state.
At 100,000 sites × 200 states that is about 4.9 GB for these arrays alone.
Allowing an **ESTIMATED** 8×–16× multiplier for intermediates, compilation and
buffers gives about 39–78 GB. That is close to an 80 GB device's capacity;
the 200-state full batch may still exhaust memory. Reduce `batch_size` and
measure on the target. **ESTIMATED** million-site/200-state storage is
390–780 GB with the same multiplier; batch_size=10 instead gives roughly
20–39 GB. These are decimal GB and are not measured peaks. Checkpoint files
also contain all full positions, so provision disk space (uncompressed positions
alone are **ESTIMATED** about 9 GB for the default full grid).

The Python/NumPy network generator runs on the host; JAX minimizes on the chosen
device. Topology stays resident across FIRE chunks. Checkpoint callbacks transfer
positions to the host once per chunk (1,000 steps in the full config), which
can become a bottleneck. Plot sampling does not reduce solver memory. No full-grid
mechanical sweep, H100 execution, million-node solve, browser/WebGL execution,
3D variant, or literature-curve reproduction has been claimed. The optional 3D
extension is deferred; this deliverable is 2D.

## ScienceClaw

The skill in `scienceclaw/skills/fiber-network-rigidity/` imports this project's
existing sweep and analysis. It follows ScienceClaw's skill folder and YAML
frontmatter contract; no ScienceClaw code is vendored. From this checkout and
the Python environment above:

```bash
python scienceclaw/skills/fiber-network-rigidity/scripts/fiber_network_rigidity.py --out outputs/scienceclaw-cpu --format json
python scienceclaw/skills/fiber-network-rigidity/scripts/fiber_network_rigidity.py --out outputs/scienceclaw-cpu --resume --format json
```

JSON stdout contains `summary`, per-curve `results` (measured z, both γc estimates,
agreement, conditional exponent fits and convergence counts), `artifacts`,
`parameters` and `limitations`. Sweep progress goes to stderr. Missing estimates
are `"not determined"`. Open the returned viewer HTML and choose its JSON file.
The full H100 configuration is supported with `--config configs/h100.yaml
--backend jax` after the CUDA setup above; its runtime remains **ESTIMATED**.

To install into the participant's ScienceClaw checkout, substitute its path:

```bash
export RIGIDITY_SWEEP_ROOT="$PWD"
cp -R scienceclaw/skills/fiber-network-rigidity /path/to/scienceclaw/skills/
cd /path/to/scienceclaw
python3 skill_catalog.py --stats
python3 skill_catalog.py --search rigidity
python3 skills/fiber-network-rigidity/scripts/fiber_network_rigidity.py --out outputs/rigidity --format json
```

Add `"fiber-network-rigidity"` to `preferred_tools` in
`~/.scienceclaw/agent_profile.json`. Keep the solver checkout accessible, and
make its Python environment and `RIGIDITY_SWEEP_ROOT` available to the agent
process. [SKILL.md](scienceclaw/skills/fiber-network-rigidity/SKILL.md) has the
parameters, CPU/H100 examples, output contract and suggested upstream artifact
coordination entries. ScienceClaw discovery itself has not been run here;
offline tests exercise a copied skill directory.

The separate poster reads completed sweep artifacts and builds an Infinite
summary with a γc(z) table, fits and scientific limitations. **Its default is a
dry run:** it prints the exact JSON payload, makes no network calls and reads
no credentials. These example URLs are placeholders; replace them with your
public repo and viewer URLs or omit both flags for an empty `dataSources` list.
From the rigidity-sweep checkout:

```bash
python scienceclaw/post_results.py --out outputs/scienceclaw-cpu --community materials --repo-url https://example.org/your-repo --viewer-url https://example.org/your-viewer --draft-out outputs/scienceclaw-cpu/infinite-draft.json
```

To publish deliberately, use **your own** Infinite account, configured by
ScienceClaw's setup. The config supports `api_key` or `apiKey`. Credentials and
session tokens are kept in memory and never written by the poster or logged.
No posting has been performed as part of this integration.

```bash
python scienceclaw/post_results.py --out outputs/scienceclaw-cpu --community materials --repo-url https://example.org/your-repo --viewer-url https://example.org/your-viewer --post --config ~/.scienceclaw/infinite_config.json
# Alternative credential source: --post --api-key-env INFINITE_API_KEY
# The named variable must already contain your own key; do not put keys in commands.
```

`--community` is required for both modes. `--config` here means the credentials
JSON, not the sweep YAML. `--api-base` or `INFINITE_API_BASE` changes the API base;
the default is `https://infinite-lamm.vercel.app/api`, matching the supplied
Infinite client (its README also names `https://lamm.mit.edu/infinite/api`).
HTTPS is required except for loopback HTTP tests; redirects are refused.
The poster logs in at `/agents/login`, then sends the payload to `/posts` and
prints only the returned post id. Missing credentials, login failures, HTTP
errors and karma refusal exit 1; posting requires at least 10 karma. If a request
fails after sending, check your account before retrying to avoid a duplicate.
ScienceClaw's own alternative is
`scienceclaw-post --agent YOUR_AGENT --topic "Rigidity sweep" --community materials --dry-run`.

Offline verification (including a local fake login/post server):

```bash
pytest -q
python scripts/check_scienceclaw_mutants.py
```

The three mutants rename the required `results` key, enable posting by default,
and print the API key. Each must fail its targeted test; the runner uses isolated
source copies and records only the exit status and mutation, never leaked output.
See [evidence/SCIENCECLAW.md](evidence/SCIENCECLAW.md) for the observed proof.

## References

- Sharma et al. (2016), *Nature Physics*, strain-controlled criticality in
  fiber networks: [doi:10.1038/nphys3628](https://doi.org/10.1038/nphys3628).
- Jansen et al. (2018), *Biophysical Journal*:
  [doi:10.1016/j.bpj.2018.04.043](https://doi.org/10.1016/j.bpj.2018.04.043).
- Bitzek et al. (2006), *Physical Review Letters* **97**, 170201, FIRE:
  [doi:10.1103/PhysRevLett.97.170201](https://doi.org/10.1103/PhysRevLett.97.170201).

All numerical choices above are this implementation's assumptions; no unstated
parameters or agreement with these publications are inferred.
