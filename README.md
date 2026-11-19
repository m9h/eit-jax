# eit-jax

Differentiable electrical impedance tomography in [JAX](https://github.com/jax-ml/jax).

A finite-element forward model whose map from conductivity to boundary voltages can be differentiated, jitted and vmapped, plus reconstruction built on it: sensitivity by autodiff, one-step difference imaging, and nonlinear absolute imaging with [Optimistix](https://github.com/patrick-kidger/optimistix). Models are [Equinox](https://github.com/patrick-kidger/equinox) modules.

It was written alongside work on the [OpenEIT](https://github.com/OpenEIT) Spectra board and uses the same scan ordering as its firmware, but it does not depend on OpenEIT and works for any point-electrode 2D setup.

**Status: early.** Validated against pyEIT and on simulated data only. Nothing here has been checked against tank or phantom measurements yet.

## Install

    git clone https://github.com/m9h/eit-jax.git
    cd eit-jax
    uv sync
    uv run pytest

## Example

```python
import jax.numpy as jnp
import eitjax as ej

# 16 electrodes, opposite drive, adjacent measurement: 192 values per frame
mesh = ej.disc_mesh(n_el=16, n_rings=6)
model = ej.ForwardModel(mesh, ej.opposition(16))

sigma_true = ej.disc_inclusion(mesh, center=(0.4, 0.3), radius=0.25, value=3.0)
v = model(sigma_true)                      # simulated frame

J = ej.jacobian(model, jnp.ones(mesh.n_elem))            # sensitivity, by autodiff
delta = ej.reconstruct_difference(model, v, model(jnp.ones(mesh.n_elem)))
sigma, solution = ej.reconstruct_absolute(model, v)      # Levenberg-Marquardt
```

## What is in it

| Module | Contents |
|---|---|
| `mesh` | `Mesh`, a ring-structured `disc_mesh`, `Mesh.from_arrays` for meshes made elsewhere |
| `protocol` | rotating four-electrode scans; `opposition(n_el)` is the Spectra default |
| `forward` | `ForwardModel`: P1 triangles, per-element conductivity, point electrodes |
| `inverse` | `jacobian`, `reconstruct_difference`, `reconstruct_absolute`, `fisher_information` |

## What has been checked

- **Against pyEIT 1.2.4**, on pyEIT's own meshes, for 8, 16 and 32 electrodes: identical drive and measurement ordering, and boundary voltages equal to a relative tolerance of 1e-8, for a uniform disc and for one with an inclusion.
- **Jacobian** against central finite differences.
- **Reconstruction** from data simulated on a finer mesh than the one used to reconstruct. For a disc of three times background conductivity with 0.3 % multiplicative noise (the frame-to-frame figure measured on a Spectra board), the absolute reconstruction returns about 2.6 inside the target (peak 2.9) and 1.01 in the background.

## Limits

- 2D only, point electrodes (no complete electrode model, so no contact impedance yet).
- Real conductivity only; no complex admittivity.
- Dense matrices. Fine for a few thousand nodes, not beyond.
- Double precision is switched on at import (`jax_enable_x64`).

## Planned

- Complete electrode model with contact impedances as fitted parameters.
- Calibration fits (per-channel gain, electrode position) from homogeneous-tank and resistor-phantom data.
- Multi-frequency models across the Spectra's 15 spectroscopy frequencies.
- Validation on measured data.

## Licence

MIT. See `LICENSE`.
