# CPU validation

Executed with Python 3.12.3 and the pinned CPU environment in requirements.lock.
No accelerator or real-browser execution was performed.

| Command | Observed result |
|---|---|
| `python -m pytest -q` | **61 passed in 11.83s** |
| `python run_sweep.py --config configs/cpu_small.yaml --output outputs/addendum_final` | **72/72 converged, 5.3383s** |
| Same command with `--resume` | **Exit 0**, identical results-array hash |
| `node --check viewer/app.js` | Exit 0 |
| `node --test tests/test_viewer.mjs` | **8 passed**, including the real sample, table and saved selections |
| `python scripts/check_mutants.py` | **19/19 detected**, each target exited 1 with an assertion failure |
| `pip check` | No broken requirements found |
| `python -m compileall -q rigidity run_sweep.py scripts tests` | Exit 0 |
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
15. Fit log K′ against log gamma instead of log(gamma − gamma_c).
16. Included unconverged points in the exponent fit.
17. Fit mean modulus with incomplete seed derivative stencils.
18. Allowed an exponent fit with fewer than five usable points.
19. Used target connectivity instead of measured connectivity in the trend.

## Exponent addendum

The synthetic fixture has fixed gamma_c=0.2, amplitude 2.4, true f=1.17,
31 log-spaced distances from 0.01 to 0.1 and seeded log noise with standard
deviation 0.015. The declared distance window is [0.009,0.101], using absolute
strains [0.21,0.30]. The recovered **f=1.17140385**, standard error **0.00402752**,
and conditional 95% interval **[1.16316664,1.17964106]** recover the known exponent
within 5%. See [exponent_synthetic.json](exponent_synthetic.json).

The requested mutant replaces `np.log(delta[selected])` with
`np.log(g[selected])` in `fit_critical_exponent`: the synthetic test exits 1
at `assert fit["f"] == pytest.approx(expected, rel=.05)`.
[exponent-mutant.json](exponent-mutant.json) records that isolated run;
the full mutation report also includes it.

[addendum_regressions.json](addendum_regressions.json) records new tests run
against unchanged base c8daeb1b2f7a22c897894605e4a90844cda4973d in a detached
worktree. Analysis tests cannot import the missing functions (exit 2); both
export regressions fail on missing `critical_exponents` (exit 1); both new
viewer checks fail without the table (exit 1). Fixed tests pass in the suites
above. These are local checks, not independent review.

The current bundled sample was regenerated from the final analysis source;
[addendum_timing.json](addendum_timing.json) records its timing and array hash.
Every exponent is **not determined**: no usable samples lie in the prescribed
near-onset windows, and some onsets themselves are unresolved. Measured z rises
from 3.190524 to 3.8125. At kappa=0.001, the nonaffinity onset rises from
0.00562341 to 0.0133352; at kappa=0.01 it stays at 0.0749894. The sample does
**not** show the expected decreasing onset toward z=4. Modulus-onset trends and
the z=4 limit are not determined. No fit was forced to the literature value.

[resume.json](resume.json) records the original pre-addendum SIGINT during the entry-point sweep,
exit 130, completed checkpoints and a subsequent successful `--resume`. Its
result-array hash equals the uninterrupted sweep. The pytest resume test also
checks completed checkpoint bytes remain unchanged and incompatible seeds are
rejected.

[cpu_small_timing.json](cpu_small_timing.json) records the original pre-addendum sample's
timing and hash; [cpu_benchmark.json](cpu_benchmark.json) records warmed JAX CPU
throughput. The sample contains unresolved modulus onsets and substantial
disagreement with nonaffinity onset (1.875 decades for its resolved pair).
This is disclosed numerical behavior, not agreement with published curves.

H100 execution, full-grid mechanical solves, million-node mechanics,
real-browser/WebGL rendering, 3D and literature-curve reproduction remain
unexecuted. README accelerator timing/memory figures are explicitly ESTIMATED.
