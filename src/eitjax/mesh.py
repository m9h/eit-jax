"""Triangular meshes of the unit disc with point electrodes on the boundary."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from scipy.spatial import Delaunay


class Mesh(eqx.Module):
    """A 2D triangular mesh.

    nodes:      (n_nodes, 2) coordinates
    elements:   (n_elem, 3) node indices, counter-clockwise
    electrodes: (n_el,) node index of each electrode
    ref_node:   node held at zero potential
    """

    nodes: Float[Array, "n_nodes 2"]
    elements: Int[Array, "n_elem 3"]
    electrodes: Int[Array, " n_el"]
    ref_node: int = 0

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elem(self) -> int:
        return self.elements.shape[0]

    @property
    def n_el(self) -> int:
        return self.electrodes.shape[0]

    @property
    def centroids(self) -> Float[Array, "n_elem 2"]:
        return self.nodes[self.elements].mean(axis=1)

    @property
    def areas(self) -> Float[Array, " n_elem"]:
        p = self.nodes[self.elements]
        d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        return 0.5 * jnp.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])

    @classmethod
    def from_arrays(cls, nodes, elements, electrodes, ref_node: int = 0) -> "Mesh":
        """Build from NumPy arrays, fixing element orientation to counter-clockwise."""
        nodes = np.asarray(nodes, dtype=float)[:, :2]
        elements = np.asarray(elements, dtype=int).copy()
        p = nodes[elements]
        d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        clockwise = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) < 0
        elements[clockwise] = elements[clockwise][:, [0, 2, 1]]
        return cls(
            nodes=jnp.asarray(nodes),
            elements=jnp.asarray(elements),
            electrodes=jnp.asarray(np.asarray(electrodes, dtype=int)),
            ref_node=int(ref_node),
        )


def disc_mesh(n_el: int = 16, n_rings: int = 8, radius: float = 1.0) -> Mesh:
    """Mesh the disc with concentric rings of nodes.

    Ring k (1..n_rings) carries k * n_el nodes, so the boundary ring has a node
    at every electrode angle. Electrode 0 sits at angle 0 and numbering runs
    counter-clockwise. The centre node is the potential reference.
    """
    pts = [np.zeros((1, 2))]
    for k in range(1, n_rings + 1):
        # stagger alternate rings a little for better-shaped triangles,
        # but never the boundary ring, which must hit the electrode angles
        offset = 0.0 if k == n_rings else 0.5 * (k % 2) * 2 * np.pi / (k * n_el)
        theta = offset + 2 * np.pi * np.arange(k * n_el) / (k * n_el)
        r = radius * k / n_rings
        pts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    nodes = np.vstack(pts)
    elements = Delaunay(nodes).simplices
    first_boundary = len(nodes) - n_rings * n_el
    electrodes = first_boundary + n_rings * np.arange(n_el)
    return Mesh.from_arrays(nodes, elements, electrodes, ref_node=0)


def disc_inclusion(
    mesh: Mesh, center, radius: float, value: float, background: float = 1.0
) -> Float[Array, " n_elem"]:
    """Per-element conductivity: `value` inside a disc, `background` elsewhere."""
    d = jnp.linalg.norm(mesh.centroids - jnp.asarray(center, dtype=float), axis=1)
    return jnp.where(d < radius, value, background)
