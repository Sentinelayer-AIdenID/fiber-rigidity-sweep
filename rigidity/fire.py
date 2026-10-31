"""Independent FIRE minimizers batched with padding and resumable host callbacks.

The JAX backend evaluates a whole batch inside a compiled ``lax.while_loop``.
Each member owns its time step, mixing coefficient, and convergence state.
Callbacks receive host snapshots after each chunk; raising KeyboardInterrupt
propagates after the preceding snapshot has been delivered.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .energy import network_data, numpy_evaluate


def pad_networks(networks):
    """Return stacked topology arrays, using masks for every padded element."""
    if not networks:
        raise ValueError("at least one network is required")
    originals = [network_data(net) for net in networks]
    n = max(len(net.positions) for net in networks)
    if any(len(net.positions) == 0 for net in networks):
        raise ValueError("networks must contain at least one node")
    m = max(1, max(len(d["edges"]) for d in originals))
    t = max(1, max(len(d["triplets"]) for d in originals))
    sizes = {"edges": m, "images": m, "rest_lengths": m, "edge_mask": m,
             "triplets": t, "triplet_images": t, "bend_lengths": t,
             "triplet_mask": t, "node_mask": n}
    batch = {}
    for key in originals[0]:
        rows = []
        for data in originals:
            value = np.asarray(data[key])
            if key in sizes:
                shape = (sizes[key],) + value.shape[1:]
                # Positive dummy rest lengths keep masked divisions finite.
                row = np.ones(shape, dtype=value.dtype) if key in (
                    "rest_lengths", "bend_lengths") else np.zeros(shape, dtype=value.dtype)
                row[:len(value)] = value
                value = row
            rows.append(value)
        batch[key] = np.stack(rows)
    return batch


def _evaluate_numpy(x, data, gammas, kappas):
    results = [numpy_evaluate(x[i], {k: v[i] for k, v in data.items()},
                             gammas[i], kappas[i]) for i in range(len(x))]
    return (np.asarray([r["energy"] for r in results]),
            np.stack([r["forces"] for r in results]))


def _force_max(force, node_mask, xp):
    return xp.max(xp.sqrt(xp.sum(force * force, axis=2)) * node_mask, axis=1)


def _step(state, data, gammas, kappas, force_tol, max_steps, dt_max, evaluate, xp):
    """One semi-implicit Euler FIRE update (Bitzek et al., 2006)."""
    mask = data["node_mask"][:, :, None]
    active = (~state["converged"] & ~state["failed"] & ~state["skipped"]
              & (state["steps"] < max_steps))
    force = state["force"]
    velocity = state["v"] + state["dt"][:, None, None] * force
    power = xp.sum(velocity * force, axis=(1, 2))
    positive = power > 0
    n_positive = xp.where(positive, state["positive"] + 1, 0)
    accelerate = positive & (n_positive > 5)
    dt = xp.where(positive, xp.where(accelerate, xp.minimum(state["dt"] * 1.1, dt_max),
                                    state["dt"]), state["dt"] * 0.5)
    alpha = xp.where(positive, xp.where(accelerate, state["alpha"] * 0.99,
                                      state["alpha"]), 0.1)
    vnorm = xp.sqrt(xp.sum(velocity * velocity, axis=(1, 2)))
    fnorm = xp.sqrt(xp.sum(force * force, axis=(1, 2)))
    ratio = vnorm / xp.maximum(fnorm, 1e-300)
    mixed = ((1 - state["alpha"][:, None, None]) * velocity
             + state["alpha"][:, None, None] * ratio[:, None, None] * force)
    velocity = xp.where(positive[:, None, None], mixed, 0.0) * mask
    candidate = state["x"] + dt[:, None, None] * velocity
    # Translation is a null mode. Preserve the initial center of mass.
    count = xp.sum(mask, axis=1)
    center = xp.sum(candidate * mask, axis=1) / count
    candidate = (candidate - (center - state["center"])[:, None, :]) * mask
    candidate = xp.where(active[:, None, None], candidate, state["x"])
    energy, new_force = evaluate(candidate, data, gammas, kappas)
    new_force = new_force * mask
    maximum = _force_max(new_force, data["node_mask"], xp)
    finite = (xp.isfinite(energy) & xp.all(xp.isfinite(new_force), axis=(1, 2))
              & xp.all(xp.isfinite(candidate), axis=(1, 2)))
    accepted = active & finite
    failed = state["failed"] | (active & ~finite)
    # Failed members retain their last finite positions and cannot enter fits.
    update = lambda new, old: xp.where(accepted.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)
    result = dict(state)
    result.update(x=update(candidate, state["x"]), v=update(velocity, state["v"]),
                  force=update(new_force, state["force"]), energy=update(energy, state["energy"]),
                  max_force=update(maximum, state["max_force"]),
                  steps=state["steps"] + active.astype(state["steps"].dtype),
                  converged=state["converged"] | (accepted & (maximum <= force_tol)),
                  failed=failed, dt=xp.where(active, dt, state["dt"]),
                  alpha=xp.where(active, alpha, state["alpha"]),
                  positive=xp.where(active, n_positive, state["positive"]))
    return result


@lru_cache(maxsize=1)
def _jax_functions():
    import jax
    import jax.numpy as jnp
    from .energy import jax_evaluate
    jax.config.update("jax_enable_x64", True)

    def evaluate(x, data, gammas, kappas):
        def one(position, topology, gamma, kappa):
            result = jax_evaluate(position, topology, gamma, kappa)
            return result["energy"], result["forces"]
        return jax.vmap(one)(x, data, gammas, kappas)

    @jax.jit
    def chunk(state, data, gammas, kappas, force_tol, max_steps, dt_max, chunk_steps):
        def condition(carry):
            tick, current = carry
            active = (~current["converged"] & ~current["failed"] & ~current["skipped"]
                      & (current["steps"] < max_steps))
            return (tick < chunk_steps) & jnp.any(active)

        def body(carry):
            tick, current = carry
            return tick + 1, _step(current, data, gammas, kappas, force_tol,
                                   max_steps, dt_max, evaluate, jnp)
        return jax.lax.while_loop(condition, body, (jnp.asarray(0), state))[1]

    return jax.jit(evaluate), chunk


def _snapshot(state):
    return {"positions": np.asarray(state["x"]).copy(),
            **{key: np.asarray(state[key]).copy() for key in
               ("converged", "steps", "max_force", "energy", "initial_energy", "failed", "skipped")}}


def minimize_batch(networks, gammas, kappas, *, backend="jax", max_steps=10000,
                   force_tol=1e-8, dt0=0.02, initial_positions=None, callback=None,
                   chunk_steps=100, skip_mask=None):
    """Minimize each (network, gamma, kappa) independently in float64.

    ``positions`` in the returned dictionary has shape ``(B, Nmax, 2)``.
    A false ``converged`` is never a usable equilibrium, including exhaustion
    of ``max_steps``. ``failed`` marks nonfinite calculations specifically.
    ``initial_positions`` may be padded or a list of unpadded arrays; otherwise
    every strain starts from its own affine image of the reference network.
    ``skip_mask`` freezes already checkpointed members while retaining batch
    shape; skipped members have ``skipped=True, converged=False``.
    """
    if backend not in ("numpy", "jax"):
        raise ValueError("backend must be 'numpy' or 'jax'")
    if (not isinstance(max_steps, (int, np.integer)) or max_steps < 0
            or not isinstance(chunk_steps, (int, np.integer)) or chunk_steps < 1):
        raise ValueError("max_steps must be nonnegative and chunk_steps positive integers")
    if not (np.isfinite(dt0) and dt0 > 0 and np.isfinite(force_tol) and force_tol > 0):
        raise ValueError("dt0 and force_tol must be positive and finite")
    batch = len(networks)
    skipped = np.zeros(batch, dtype=bool) if skip_mask is None else np.asarray(skip_mask, dtype=bool)
    if skipped.shape != (batch,):
        raise ValueError("skip_mask must be a vector of batch length")
    gammas, kappas = np.asarray(gammas, dtype=np.float64), np.asarray(kappas, dtype=np.float64)
    if (gammas.shape != (batch,) or kappas.shape != (batch,)
            or not np.all(np.isfinite(gammas)) or not np.all(np.isfinite(kappas))
            or np.any(kappas < 0)):
        raise ValueError("gammas and nonnegative kappas must be finite vectors of batch length")
    data = pad_networks(networks)
    mask = data["node_mask"][:, :, None]
    x = np.zeros((batch, data["node_mask"].shape[1], 2), dtype=np.float64)
    for i, net in enumerate(networks):
        count = len(net.positions)
        if initial_positions is None:
            position = np.asarray(net.positions, dtype=np.float64).copy()
            position[:, 0] += gammas[i] * position[:, 1]
        else:
            position = np.asarray(initial_positions[i], dtype=np.float64)
            if position.ndim != 2 or position.shape[1] != 2 or position.shape[0] < count:
                raise ValueError("initial positions must supply every node with two coordinates")
            position = position[:count]
        x[i, :count] = position
    evaluate, chunk = (_evaluate_numpy, None) if backend == "numpy" else _jax_functions()
    if backend == "jax":
        import jax
        # Keep topology and sweep parameters on the device across all chunks.
        data = jax.device_put(data)
        gammas, kappas = jax.device_put(gammas), jax.device_put(kappas)
    energy, force = evaluate(x, data, gammas, kappas)
    energy, force = np.asarray(energy), np.asarray(force) * mask
    maximum = _force_max(force, data["node_mask"], np)
    finite = np.isfinite(energy) & np.all(np.isfinite(force), axis=(1, 2)) & np.all(np.isfinite(x), axis=(1, 2))
    state = {"x": x, "v": np.zeros_like(x), "force": force, "energy": energy,
             "initial_energy": energy.copy(), "max_force": maximum,
             "converged": finite & (maximum <= force_tol) & ~skipped, "failed": ~finite,
             "skipped": skipped,
             "steps": np.zeros(batch, dtype=np.int64), "dt": np.full(batch, dt0),
             "alpha": np.full(batch, 0.1), "positive": np.zeros(batch, dtype=np.int64),
             "center": np.sum(x * mask, axis=1) / np.sum(mask, axis=1)}
    while np.any(~np.asarray(state["converged"]) & ~np.asarray(state["failed"])
                 & ~np.asarray(state["skipped"]) & (np.asarray(state["steps"]) < max_steps)):
        if backend == "jax":
            state = chunk(state, data, gammas, kappas, force_tol, max_steps, 10 * dt0, chunk_steps)
        else:
            for _ in range(chunk_steps):
                active = (~state["converged"] & ~state["failed"] & ~state["skipped"]
                          & (state["steps"] < max_steps))
                if not np.any(active):
                    break
                state = _step(state, data, gammas, kappas, force_tol, max_steps,
                              10 * dt0, evaluate, np)
        if callback is not None:
            callback(_snapshot(state))
    return _snapshot(state)
