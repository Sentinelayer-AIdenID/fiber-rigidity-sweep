#!/usr/bin/env python3
"""Prove the ScienceClaw contract, dry-run and secret-output tests reject mutants."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = [
    ("rename-results-key", "scienceclaw/results.py", "results=rows, artifacts=", "entries=rows, artifacts=",
     "tests/test_scienceclaw_skill.py::test_skill_json_contract_and_copied_install"),
    ("post-by-default", "scienceclaw/post_results.py",
     'parser.add_argument("--post", action="store_true",',
     'parser.add_argument("--post", action="store_true", default=True,',
     "tests/test_scienceclaw_post.py::test_dry_run_default_no_network"),
    ("print-api-key", "scienceclaw/post_results.py",
     'key = load_api_key(args.config, args.api_key_env) if args.post else None',
     'key = load_api_key(args.config, args.api_key_env) if args.post else None\n        if key:\n            print(key)',
     "tests/test_scienceclaw_post.py::test_post_success_no_secret_output[env]"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="evidence/scienceclaw-mutations.json")
    args = parser.parse_args()
    records = []
    for name, filename, old, new, target in MUTANTS:
        with tempfile.TemporaryDirectory(prefix=".mutation-scienceclaw-", dir=ROOT) as directory:
            copied = Path(directory)
            for folder in ("rigidity", "scienceclaw", "tests", "configs", "viewer"):
                shutil.copytree(ROOT / folder, copied / folder, ignore=shutil.ignore_patterns("__pycache__"))
            for filename_root in ("run_sweep.py", "pytest.ini", "requirements.txt"):
                shutil.copy2(ROOT / filename_root, copied / filename_root)
            path = copied / filename
            source = path.read_text()
            if source.count(old) != 1:
                raise RuntimeError(f"expected exactly one mutation anchor: {name}")
            path.write_text(source.replace(old, new, 1))
            process = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", target],
                                     cwd=copied, text=True, capture_output=True, timeout=90)
            output = process.stdout + process.stderr
            killed = process.returncode == 1 and ("AssertionError" in output or "Failed:" in output)
            # Do not save captured output: the leak mutant intentionally prints an inert sentinel.
            records.append(dict(mutant=name, source=filename, target=target,
                                replacement=dict(before=old, after=new),
                                exit_code=process.returncode, killed=killed))
            print(f"{name}: {'killed' if killed else 'SURVIVED or inconclusive'} (exit {process.returncode})")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, indent=2) + "\n")
    return 0 if all(record["killed"] for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
