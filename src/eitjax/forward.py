"""Finite-element forward model: conductivity -> boundary voltages.

Linear (P1) triangles, piecewise-constant conductivity per element, point
electrodes. Everything is written in JAX, so the map from conductivity to
voltages can be differentiated, vmapped and jitted.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from .mesh import Mesh
from .protocol import Protocol


def _local_stiffness(mesh: Mesh) -> Float[Array, "n_elem 3 3"]:
    """Element stiffness matrices for unit conductivity."""
    p = mesh.nodes[mesh.elements]  # (n_elem, 3, 2)
    # edge opposite each vertex
    e = jnp.stack([p[:, 2] - p[:, 1], p[:, 0] - p[:, 2], p[:, 1] - p[:, 0]], axis=1)
    area = mesh.areas
    return jnp.einsum("eik,ejk->eij", e, e) / (4.0 * area)[:, None, None]


def stiffness(mesh: Mesh, sigma: Float[Array, " n_elem"]) -> Float[Array, "n_nodes n_nodes"]:
    """Assemble the global stiffness matrix, with the reference node pinned to zero."""
    ke = _local_stiffness(mesh) * sigma[:, None, None]
    rows = jnp.repeat(mesh.elements[:, :, None], 3, axis=2)
    cols = jnp.repeat(mesh.elements[:, None, :], 3, axis=1)
    k = jnp.zeros((mesh.n_nodes, mesh.n_nodes), dtype=ke.dtype).at[rows, cols].add(ke)
    ref = mesh.ref_node
    k = k.at[ref, :].set(0.0).at[:, ref].set(0.0).at[ref, ref].set(1.0)
    return k


class ForwardModel(eqx.Module):
    """Boundary voltages for a mesh and scan protocol.

    Calling the model with a per-element conductivity returns the n_meas
    voltage differences for a unit drive current.
    """

    mesh: Mesh
    protocol: Protocol

    def potentials(self, sigma: Float[Array, " n_elem"]) -> Float[Array, "n_drive n_nodes"]:
        """Node potentials for each distinct drive pair.

        One Cholesky factorisation of the stiffness matrix serves all drive
        pairs. (A Lineax solve here is equivalent, but nesting it inside
        Optimistix's Levenberg-Marquardt gave non-finite steps with lineax
        0.1.1 / optimistix 0.1.0, so the factorisation uses jax.scipy.)
        """
        mesh, proto = self.mesh, self.protocol
        n_drive = proto.drives.shape[0]
        rows = jnp.arange(n_drive)
        b = jnp.zeros((n_drive, mesh.n_nodes), dtype=sigma.dtype)
        b = b.at[rows, mesh.electrodes[proto.drives[:, 0]]].add(1.0)
        b = b.at[rows, mesh.electrodes[proto.drives[:, 1]]].add(-1.0)
        b = b.at[:, mesh.ref_node].set(0.0)
        factor = jsl.cho_factor(stiffness(mesh, sigma))
        return jsl.cho_solve(factor, b.T).T

    def __call__(self, sigma: Float[Array, " n_elem"]) -> Float[Array, " n_meas"]:
        proto = self.protocol
        u = self.potentials(sigma)[proto.drive_index]  # (n_meas, n_nodes)
        el = self.mesh.electrodes
        idx = jnp.arange(proto.n_meas)
        return u[idx, el[proto.measure[:, 0]]] - u[idx, el[proto.measure[:, 1]]]
