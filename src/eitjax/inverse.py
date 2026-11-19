"""Sensitivity, linearised difference imaging and nonlinear absolute imaging."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Array, Float

from .forward import ForwardModel
from .mesh import Mesh


def jacobian(model: ForwardModel, sigma: Float[Array, " n_elem"]) -> Float[Array, "n_meas n_elem"]:
    """d(voltages)/d(conductivity) by reverse-mode autodiff."""
    return jax.jacrev(model)(sigma)


def neighbour_differences(mesh: Mesh) -> Float[Array, "n_edges n_elem"]:
    """First-difference operator between elements that share an edge.

    Used as a smoothness penalty: it is zero for a uniform image.
    """
    elements = np.asarray(mesh.elements)
    edges: dict[tuple[int, int], list[int]] = {}
    for e, tri in enumerate(elements):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges.setdefault(tuple(sorted((int(tri[a]), int(tri[b])))), []).append(e)
    pairs = [v for v in edges.values() if len(v) == 2]
    d = np.zeros((len(pairs), len(elements)))
    for r, (i, j) in enumerate(pairs):
        d[r, i], d[r, j] = 1.0, -1.0
    return jnp.asarray(d)


def reconstruct_difference(
    model: ForwardModel,
    v1: Float[Array, " n_meas"],
    v0: Float[Array, " n_meas"],
    sigma0: Float[Array, " n_elem"] | None = None,
    lam: float = 1e-2,
) -> Float[Array, " n_elem"]:
    """One regularised Gauss-Newton step from a reference frame.

    Returns the change in conductivity that explains v1 - v0, with a
    smoothness penalty of relative weight `lam`.
    """
    if sigma0 is None:
        sigma0 = jnp.ones(model.mesh.n_elem)
    j = jacobian(model, sigma0)
    d = neighbour_differences(model.mesh)
    jtj = j.T @ j
    reg = d.T @ d
    scale = jnp.trace(jtj) / jnp.trace(reg)
    return jnp.linalg.solve(jtj + lam * scale * reg, j.T @ (v1 - v0))


def reconstruct_absolute(
    model: ForwardModel,
    v: Float[Array, " n_meas"],
    sigma0: Float[Array, " n_elem"] | None = None,
    lam: float = 1e-3,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 50,
    log_bound: float = 6.0,
) -> tuple[Float[Array, " n_elem"], optx.Solution]:
    """Nonlinear least squares for conductivity with Levenberg-Marquardt.

    The unknown is log-conductivity, which keeps conductivity positive. The
    residual is the relative data misfit stacked with a smoothness penalty on
    neighbouring elements, weighted by `lam`. Log-conductivity is confined to +/- `log_bound`.
    """
    if sigma0 is None:
        sigma0 = jnp.ones(model.mesh.n_elem)
    d = neighbour_differences(model.mesh)
    v_scale = jnp.sqrt(jnp.mean(v**2))

    def residual(log_sigma, _):
        # keep trial steps inside a physically sensible range (contrast of e^6)
        data = (model(jnp.exp(jnp.clip(log_sigma, -log_bound, log_bound))) - v) / v_scale
        return data, jnp.sqrt(lam) * (d @ log_sigma)

    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    sol = optx.least_squares(
        residual, solver, jnp.log(sigma0), max_steps=max_steps, throw=False
    )
    return jnp.exp(jnp.clip(sol.value, -log_bound, log_bound)), sol


def fisher_information(
    j: Float[Array, "n_meas n_param"], noise_sd: float | Float[Array, " n_meas"]
) -> Float[Array, "n_param n_param"]:
    """Fisher information for independent Gaussian noise of s.d. `noise_sd`.

    Its eigenvalues say how many parameter combinations the measurement can
    resolve at that noise level: an eigenvalue well below 1 means that
    direction is not identifiable.
    """
    w = j / jnp.asarray(noise_sd)[..., None] if jnp.ndim(noise_sd) else j / noise_sd
    return w.T @ w
