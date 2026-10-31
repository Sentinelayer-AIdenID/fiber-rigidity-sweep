#!/usr/bin/env python3
"""Run the existing rigidity sweep and emit a ScienceClaw result."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys


def repository_root():
    explicit = os.environ.get("RIGIDITY_SWEEP_ROOT")
    candidates = [Path(explicit).expanduser()] if explicit else Path(__file__).resolve().parents
    for candidate in candidates:
        if (candidate / "run_sweep.py").is_file() and (candidate / "rigidity").is_dir():
            return candidate.resolve()
    raise ValueError("Set RIGIDITY_SWEEP_ROOT to the rigidity-sweep checkout")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML path; defaults to the checkout's configs/cpu_small.yaml")
    parser.add_argument("--out", help="Output directory; defaults to the YAML output")
    parser.add_argument("--backend", choices=["numpy", "jax"], help="Override the YAML backend")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--format", default="summary", choices=["summary", "json"])
    args = parser.parse_args(argv)
    try:
        root = repository_root()
        sys.path.insert(0, str(root))
        from run_sweep import load_config, run
        from scienceclaw.results import load_report

        config = load_config(args.config or root / "configs/cpu_small.yaml")
        if args.backend:
            config["backend"] = args.backend
        output = Path(args.out or config["output"]).resolve()
        # The executor parses stdout as one JSON document; sweep progress is stderr.
        with redirect_stdout(sys.stderr):
            run(config, resume=args.resume, output=output)
        report = load_report(output, root)
        report["parameters"] = {**config, "output": str(output), "resume": args.resume}
        print(json.dumps(report, allow_nan=False) if args.format == "json" else report["summary"])
    except KeyboardInterrupt:
        print("Interrupted; completed checkpoints can be reused with --resume.", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Sweep failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
