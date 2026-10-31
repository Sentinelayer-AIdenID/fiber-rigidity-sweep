---
name: fiber-network-rigidity
description: Run 2D fiber-network shear sweeps and report measured rigidity onsets, exponent fits, convergence and viewer artifacts.
metadata: {}
---

# Fiber Network Rigidity

Use this skill to simulate strain stiffening of diluted triangular fiber networks,
estimate critical strains from modulus and nonaffinity, and inspect conditional
critical-exponent fits. It imports the existing rigidity-sweep solver and analysis.
Keywords: rigidity, fiber networks, shear, nonaffinity, FIRE, critical exponent.
It produces local artifacts; it does not post to Infinite.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config` | Sweep YAML path | `configs/cpu_small.yaml` in the rigidity-sweep checkout |
| `--out` | Output directory, relative to the current directory | YAML `output` |
| `--backend` | `numpy` or `jax` | YAML backend |
| `--resume` | Reuse matching completed checkpoints | Off |
| `--format` | `summary` or `json` | `summary` |

## Installation in ScienceClaw

Keep a separate rigidity-sweep checkout; no ScienceClaw source is vendored.
From that checkout, with the ScienceClaw destination substituted:

```bash
export RIGIDITY_SWEEP_ROOT="$PWD"
cp -R scienceclaw/skills/fiber-network-rigidity /path/to/scienceclaw/skills/
python3 -m pip install -r scienceclaw/skills/fiber-network-rigidity/requirements.txt
cd /path/to/scienceclaw
python3 skill_catalog.py --stats
python3 skill_catalog.py --search rigidity
```

Add `"fiber-network-rigidity"` to the existing `preferred_tools` list in
`~/.scienceclaw/agent_profile.json`, preserving its other settings. Ensure the
agent's process inherits `RIGIDITY_SWEEP_ROOT` and uses the environment containing
the requirements. A copied skill requires that variable; an in-repo script finds
the checkout automatically. See [references/README.md](references/README.md)
for source mapping and the separate results poster.

## Examples

CPU demonstration (the executor substitutes `{baseDir}`):

```bash
python3 {baseDir}/scripts/fiber_network_rigidity.py --out outputs/scienceclaw-cpu --format json
python3 {baseDir}/scripts/fiber_network_rigidity.py --out outputs/scienceclaw-cpu --resume --format json
```

Full H100 configuration, after the CUDA JAX setup in the project's README:

```bash
JAX_PLATFORMS=cuda python3 {baseDir}/scripts/fiber_network_rigidity.py --config "$RIGIDITY_SWEEP_ROOT/configs/h100.yaml" --backend jax --out outputs/scienceclaw-h100 --format json
```

H100 runtime is **ESTIMATED** at 2.4 hours–10 days under the README's unmeasured
throughput assumptions. No H100 execution is claimed. Configure smaller batches
if device memory requires it. Do not run this full grid as a quick skill check.

## Output JSON

`--format json` writes one JSON object to stdout; progress goes to stderr.
Exit codes: 0 success, 1 execution failure, 2 invalid CLI arguments, 130 interruption.

- `summary`: human-readable curve and convergence counts.
- `results`: one entry per `(z, kappa)`, containing `z`, `kappa`, `measured_z`,
  `gamma_c_modulus`, `gamma_c_nonaffinity`, `agreement_log10`,
  `critical_exponents`, per-strain `converged_count`, and `converged` / `total`.
  Missing onsets, agreement, fitted `f`, standard error and 95% intervals are
  `"not determined"`. Each exponent entry retains its analysis status, reason,
  point count, fit window and conditional uncertainty. Onsets describe mean curves.
- `artifacts`: absolute paths under `npz`, `viewer_json`, and `viewer_html`.
  Open the HTML viewer and choose the returned JSON file in its file picker.
- `parameters`: effective sweep configuration, output path and resume flag.
- `limitations`: scientific interpretation limits.

## Limitations

The small CPU sample is a numerical demonstration, not a reproduction of published
critical curves. Sparse strains, finite size and few seeds can leave both γc and
f not determined. Failed equilibria do not enter fits. Uncertainty conditions on
the chosen onset and fit window; it omits onset, seed, finite-size and crossover
uncertainty. No literature agreement or H100 performance is inferred.

## Optional upstream coordination

Suggested entries for the ScienceClaw maintainers to review in **their** checkout
(these are not installed automatically):

```python
# artifacts/artifact.py — SKILL_DOMAIN_MAP
"fiber-network-rigidity": ["rigidity_sweep"]
# artifacts/reactor.py — SKILL_INPUT_MAP
"fiber-network-rigidity": {
    "param": "config", "entity": "sweep YAML path",
    "hint": "/path/to/rigidity-sweep/configs/cpu_small.yaml"
}
```
