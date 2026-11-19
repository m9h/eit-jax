"""Four-electrode scan patterns: which pair drives and which pairs are measured."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int


class Protocol(eqx.Module):
    """drive:   (n_meas, 2) electrode indices [source A, sink B] for each measurement
    measure: (n_meas, 2) electrode indices [N, M]; the measurement is v[N] - v[M]
    drives:  (n_drive, 2) the distinct drive pairs, in scan order
    drive_index: (n_meas,) row of `drives` used by each measurement
    """

    drive: Int[Array, "n_meas 2"]
    measure: Int[Array, "n_meas 2"]
    drives: Int[Array, "n_drive 2"]
    drive_index: Int[Array, " n_meas"]

    @property
    def n_meas(self) -> int:
        return self.measure.shape[0]


def scan(n_el: int = 16, drive_dist: int = 1, meas_step: int = 1) -> Protocol:
    """Rotating scan. Drive pair (i, i + drive_dist); measured pairs
    (m + meas_step, m) for m = 0..n_el-1, skipping any pair that touches a
    driven electrode. This is the ordering pyEIT calls the 'std' parser and the
    ordering the OpenEIT Spectra firmware transmits.
    """
    drives, drive, measure, index = [], [], [], []
    for i in range(n_el):
        a, b = i, (i + drive_dist) % n_el
        drives.append((a, b))
        for m in range(n_el):
            n = (m + meas_step) % n_el
            if m in (a, b) or n in (a, b):
                continue
            drive.append((a, b))
            measure.append((n, m))
            index.append(i)
    return Protocol(
        drive=jnp.asarray(np.array(drive)),
        measure=jnp.asarray(np.array(measure)),
        drives=jnp.asarray(np.array(drives)),
        drive_index=jnp.asarray(np.array(index)),
    )


def opposition(n_el: int = 16) -> Protocol:
    """Opposite-electrode drive with adjacent measurement (Spectra default):
    32, 192 and 896 measurements for 8, 16 and 32 electrodes."""
    return scan(n_el, drive_dist=n_el // 2, meas_step=1)
