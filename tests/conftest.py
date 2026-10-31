import os

# All proofs run on CPU, even on hosts that happen to have an accelerator.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
