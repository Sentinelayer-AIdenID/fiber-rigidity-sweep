# ScienceClaw integration proof

These are local CPU and offline protocol checks, not an independent review,
ScienceClaw discovery run, Infinite publication or H100 benchmark.

## Gates observed

From the repository root, using the pinned `.venv`:

| Command | Observed result |
|---|---|
| `.venv/bin/python -m pytest -q` | 93 passed in 17.53s |
| `node --check viewer/app.js` | Exit 0 |
| `node --test tests/test_viewer.mjs` | 8 passed, 0 failed |
| `.venv/bin/python scripts/check_mutants.py` | All 19 existing mutants detected, each exit 1 |
| `.venv/bin/python scripts/check_scienceclaw_mutants.py` | All 3 integration mutants killed, each exit 1 |
| `.venv/bin/python -m compileall -q rigidity scienceclaw run_sweep.py scripts tests` | Exit 0 |
| `.venv/bin/python -m pip check` | No broken requirements found |
| `git diff --check` | Exit 0 |

The 32 added test cases cover copied skill installation, YAML frontmatter,
strict JSON stdout, resume, measured-result projection, missing config,
dry-run network/credential isolation, credential field aliases, environment
selection, the login/post payload and Bearer header, HTTP 401/403/500, missing
token/id, redirect refusal, malformed credentials and API bases, artifact
mismatch, uncertainty formatting, and secret suppression in responses and errors.
Fake credentials exist only as inert local-test sentinels; no real account is used.

The new skill tests fail at base `b82ef02` because the skill is absent (exit 1).
The new poster tests cannot collect at that base because the module is absent
(exit 2); that absence is not counted as behavioral mutation proof.

## CPU skill and dry run

```bash
.venv/bin/python scienceclaw/skills/fiber-network-rigidity/scripts/fiber_network_rigidity.py --out outputs/scienceclaw-cpu --format json
.venv/bin/python scienceclaw/skills/fiber-network-rigidity/scripts/fiber_network_rigidity.py --out outputs/scienceclaw-cpu --resume --format json
.venv/bin/python scienceclaw/post_results.py --out outputs/scienceclaw-cpu --community materials --repo-url https://example.org/rigidity-sweep --viewer-url https://example.org/rigidity-viewer --draft-out outputs/scienceclaw-cpu/infinite-draft.json
```

The first command used the bundled `cpu_small.yaml`, JAX on CPU, and took
4.984 seconds inside the existing sweep. All 72 states converged. Resume returned
identical scientific results. Skill stdout parsed as one JSON object with
`summary`, `results`, `artifacts`, `parameters`, and `limitations`.
The poster's stdout parsed as JSON identical to the draft file. The URLs above
are explicit public placeholders, not claims of hosted artifacts.

Excerpt from the actual skill JSON (one of four result entries; other fields
omitted here for readability):

```json
{
  "summary": "4 (z, kappa) curves; 72/72 states converged.",
  "results": [{
    "z": 3.2,
    "kappa": 0.001,
    "measured_z": 3.190524193548387,
    "gamma_c_modulus": 0.4216965034285822,
    "gamma_c_nonaffinity": 0.005623413251903491,
    "agreement_log10": 1.875,
    "converged": 18,
    "total": 18
  }]
}
```

All four curves have 18/18 converged states. The modulus onset for the other
three curves and all eight conditional exponent fits are `"not determined"`.
Nonaffinity onsets are 0.07498942093324558 at `(3.2, 0.01)`,
0.01333521432163324 at `(3.8, 0.001)`, and 0.07498942093324558 at `(3.8, 0.01)`.
Both curves targeting z=3.8 have measured z=3.8125. The report carries the CPU
sample, finite-size, sparse-grid and conditional-uncertainty limitations.

Selected fields from the actual dry-run payload:

```json
{
  "community": "materials",
  "title": "Fiber-network rigidity sweep: 4 curves",
  "evidenceSummary": "4 (z, kappa) curves; 72/72 states converged.",
  "dataSources": [
    "https://example.org/rigidity-sweep",
    "https://example.org/rigidity-viewer"
  ],
  "toolsUsed": ["fiber-network-rigidity"]
}
```

The full payload also contains markdown `content`, `hypothesis`, `method`,
`findings` and `openQuestions`. No credentials are needed for these commands.

## Mutants that failed

`scripts/check_scienceclaw_mutants.py` copies sources into temporary directories,
changes exactly one behavior, and runs its targeted test. It requires pytest
exit 1 with an assertion failure; import failures do not count. It never saves
the captured leak-mutant output.

| Mutant | Targeted test | Observed |
|---|---|---|
| Rename `results` to `entries` | `test_skill_json_contract_and_copied_install` | Assertion failure, exit 1 |
| Set `--post` default to true | `test_dry_run_default_no_network` | Forbidden credential/network assertion, exit 1 |
| Print the API key | `test_post_success_no_secret_output[env]` | No-secret-output assertion, exit 1 |

Exact replacements and target paths are in
[scienceclaw-mutations.json](scienceclaw-mutations.json).
