#!/usr/bin/env python3
"""Measure CPU throughput; does not predict accelerator performance by itself."""

import json
import os
from pathlib import Path
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rigidity.fire import minimize_batch
from rigidity.network import generate_network


def main():
    n = generate_network(32, 32, 3.5, 0)
    batch, steps = 8, 1000
    parameters = dict(networks=[n]*batch, gammas=[.1]*batch, kappas=[.001]*batch,
                      max_steps=steps, force_tol=1e-30, chunk_steps=1000)
    minimize_batch(**parameters)  # Compile and warm this shape.
    start = time.perf_counter()
    result = minimize_batch(**parameters)
    elapsed = time.perf_counter()-start
    iterations = int(result["steps"].sum()) * len(n.positions)
    report = dict(platform="cpu", batch=batch, nodes=len(n.positions), steps_per_state=steps,
                  elapsed_seconds=elapsed, node_iterations=iterations,
                  node_iterations_per_second=iterations/elapsed,
                  method="One warmup followed by one timed float64 JAX CPU batch; excludes export and topology generation")
    target = Path(__file__).resolve().parents[1] / "evidence/cpu_benchmark.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
