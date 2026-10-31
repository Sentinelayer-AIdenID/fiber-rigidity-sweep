# CPU validation

Executed with Python 3.12.3 and the pinned CPU environment in requirements.lock.
No accelerator or real-browser execution was performed.

| Command | Observed result |
|---|---|
| `python -m pytest -q` | **42 passed in 16.27s** |
| `python run_sweep.py --config configs/cpu_small.yaml` | **72/72 converged, 6.0656s** |
| `node --check viewer/app.js` | Exit 0 |
| `node --test tests/test_viewer.mjs` | **6 passed**, including the real sample and every saved selection |
| `python scripts/check_mutants.py` | **14/14 detected**, each target exited 1 with an assertion failure |
| `pip check` | No broken requirements found |
| `git diff --check` | Exit 0 |

The JavaScript tests use a DOM/WebGL-command stub and inspect generated geometry,
chart traces, controls and JSON validation. They are not browser rendering proof.

The Python tests cover separate stretch/bend central-difference gradients
(relative error < 1e-6), G=sqrt(3)/4 via both energy curvature and virial stress
(within 1%), full-lattice affine equilibrium (force components < 1e-9), exactly
zero straight-fiber bending and positive kink energy, FIRE energy decrease and
convergence, NumPy/JAX parity, cleanup, masks, failure exclusion, array-hash
determinism, resume, output schema and both onset estimators.

The mutation runner copies source and tests into disposable directories and
applies the exact replacements recorded in [mutations.json](mutations.json).
Working source is never changed. Detected defects:

1. Reversed stretch gradient.
2. Reversed bend gradient.
3. Wrong spring-energy normalization.
4. Reversed virial stress.
5. Removed Lees–Edwards cell-image shear offset.
6. Disabled bend energy.
7. FIRE displacement uphill.
8. Retained finite attached rattlers.
9. Unseeded network randomness.
10. Included failed stress in differentiation.
11. Removed strain normalization from nonaffinity.
12. Allowed a different seed list to reuse checkpoints.
13. Accepted a smaller interior bump despite a higher endpoint peak.
14. Accepted a mean-modulus onset with an incomplete seed stencil.

[resume.json](resume.json) records a real SIGINT during the entry-point sweep,
exit 130, completed checkpoints and a subsequent successful `--resume`. Its
result-array hash equals the uninterrupted sweep. The pytest resume test also
checks completed checkpoint bytes remain unchanged and incompatible seeds are
rejected.

[cpu_small_timing.json](cpu_small_timing.json) records the committed sample's
timing and hash; [cpu_benchmark.json](cpu_benchmark.json) records warmed JAX CPU
throughput. The sample contains unresolved modulus onsets and substantial
disagreement with nonaffinity onset (1.875 decades for its resolved pair).
This is disclosed numerical behavior, not agreement with published curves.

H100 execution, full-grid mechanical solves, million-node mechanics,
real-browser/WebGL rendering, 3D and literature-curve reproduction remain
unexecuted. README accelerator timing/memory figures are explicitly ESTIMATED.
