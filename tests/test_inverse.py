"""Reconstruction from data simulated on a different, finer mesh."""
import jax.numpy as jnp
import numpy as np

import eitjax as ej

CENTER, RADIUS = (0.4, 0.3), 0.25


def _simulate(value, noise=0.0, seed=0):
    fine = ej.disc_mesh(n_el=16, n_rings=10)
    model = ej.ForwardModel(fine, ej.opposition(16))
    v0 = model(jnp.ones(fine.n_elem))
    v1 = model(ej.disc_inclusion(fine, CENTER, RADIUS, value))
    if noise:
        rng = np.random.default_rng(seed)
        v1 = v1 * (1 + noise * jnp.asarray(rng.normal(size=v1.shape)))
    return v0, v1


def _peak(mesh, image):
    return np.asarray(mesh.centroids[int(jnp.argmax(jnp.abs(image)))])


def test_difference_image_locates_target():
    v0, v1 = _simulate(3.0, noise=0.003)  # 0.3 % noise, as measured on the Spectra
    mesh = ej.disc_mesh(n_el=16, n_rings=6)
    model = ej.ForwardModel(mesh, ej.opposition(16))
    delta = ej.reconstruct_difference(model, v1, v0, lam=1e-2)
    assert np.linalg.norm(_peak(mesh, delta) - CENTER) < 0.2
    assert float(delta[jnp.argmax(jnp.abs(delta))]) > 0  # more conductive


def test_absolute_reconstruction_recovers_contrast():
    _, v1 = _simulate(3.0)
    mesh = ej.disc_mesh(n_el=16, n_rings=6)
    model = ej.ForwardModel(mesh, ej.opposition(16))
    sigma, sol = ej.reconstruct_absolute(model, v1, lam=1e-3)
    inside = np.linalg.norm(np.asarray(mesh.centroids) - CENTER, axis=1) < RADIUS * 0.6
    far = np.linalg.norm(np.asarray(mesh.centroids) - CENTER, axis=1) > RADIUS * 2
    assert np.linalg.norm(_peak(mesh, jnp.log(sigma)) - CENTER) < 0.2
    assert float(jnp.median(sigma[inside])) > 1.5      # raised towards the true 3.0
    assert abs(float(jnp.median(sigma[far])) - 1.0) < 0.1  # background stays near 1
    # the fit explains the data much better than a uniform tank does
    misfit = float(jnp.linalg.norm(model(sigma) - v1))
    uniform = float(jnp.linalg.norm(model(jnp.ones(mesh.n_elem)) - v1))
    assert misfit < 0.2 * uniform


def test_fisher_information_counts_resolvable_directions():
    mesh = ej.disc_mesh(n_el=16, n_rings=4)
    model = ej.ForwardModel(mesh, ej.opposition(16))
    j = ej.jacobian(model, jnp.ones(mesh.n_elem))
    v = model(jnp.ones(mesh.n_elem))
    fim = ej.fisher_information(j, 0.003 * jnp.abs(v))
    eig = np.linalg.eigvalsh(np.asarray(fim))
    # 192 measurements with reciprocity cannot resolve all elements
    assert (eig > 1).sum() < min(192, mesh.n_elem)
    assert (eig > 1).sum() > 10
