"""Differentiable electrical impedance tomography in JAX."""

import jax

# EIT is ill-conditioned; single precision is not enough for the inverse problem.
jax.config.update("jax_enable_x64", True)

from .forward import ForwardModel, stiffness  # noqa: E402
from .inverse import (  # noqa: E402
    fisher_information,
    jacobian,
    reconstruct_absolute,
    reconstruct_difference,
)
from .mesh import Mesh, disc_mesh, disc_inclusion  # noqa: E402
from .protocol import Protocol, opposition, scan  # noqa: E402

__all__ = [
    "ForwardModel",
    "Mesh",
    "Protocol",
    "disc_inclusion",
    "disc_mesh",
    "fisher_information",
    "jacobian",
    "opposition",
    "reconstruct_absolute",
    "reconstruct_difference",
    "scan",
    "stiffness",
]
