# Source mapping

The skill imports `run_sweep.load_config` and `run_sweep.run` from the checkout
located by `RIGIDITY_SWEEP_ROOT` (or the script's ancestor directory). It forwards
the config, backend, output directory and resume flag without replacing mechanics.
`scienceclaw/results.py` uses `rigidity.io.viewer_curves`, which calls the existing
`rigidity.analysis` critical-strain and exponent estimators. It reads completed
`results.npz`, `manifest.json` and `viewer.json`, checking matching fingerprints.
The project `SCHEMA.md` describes the arrays, convergence masks and uncertainty.

Copying the skill directory does not copy the solver, configs or HTML viewer;
keep the rigidity-sweep checkout accessible and set `RIGIDITY_SWEEP_ROOT` in the
ScienceClaw process environment. Dependency pins match the project's CPU pins.
For an H100 use the project's CUDA setup, not the CPU dependency line alone.

To share a completed sweep, use the separate checkout's
`python3 scienceclaw/post_results.py --help`. It defaults to a local dry run.
Posting requires explicit opt-in and the user's own Infinite account.
