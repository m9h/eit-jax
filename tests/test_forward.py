"""The forward model must agree with pyEIT on the same mesh and scan."""
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import eitjax as ej

pyeit_mesh = pytest.importorskip("pyeit.mesh")
from pyeit.eit import protocol as pyeit_protocol  # noqa: E402
from pyeit.eit.fem import EITForward  # noqa: E402
from pyeit.mesh.wrapper import PyEITAnomaly_Circle  # noqa: E402


@pytest.mark.parametrize("n_el", [8, 16, 32])
def test_protocol_matches_pyeit(n_el):
    ours = ej.opposition(n_el)
    theirs = pyeit_protocol.create(n_el, dist_exc=n_el // 2, step_meas=1, parser_meas="std")
    assert ours.n_meas == {8: 32, 16: 192, 32: 896}[n_el]
    np.testing.assert_array_equal(np.asarray(ours.drives), theirs.ex_mat)
    # pyEIT stores [N, M] pairs per drive, shape (n_drive, n_meas_per_drive, 2)
    np.testing.assert_array_equal(np.asarray(ours.measure), theirs.meas_mat.reshape(-1, 2))


@pytest.mark.parametrize("n_el", [8, 16, 32])
def test_voltages_match_pyeit(n_el):
    pm = pyeit_mesh.create(n_el, h0=0.1)
    pp = pyeit_protocol.create(n_el, dist_exc=n_el // 2, step_meas=1, parser_meas="std")
    target = pyeit_mesh.set_perm(
        pm, anomaly=PyEITAnomaly_Circle(center=[0.4, 0.3], r=0.2, perm=10.0), background=1.0
    )
    fwd = EITForward(pm, pp)

    mesh = ej.Mesh.from_arrays(pm.node, pm.element, pm.el_pos, ref_node=pm.ref_node)
    model = ej.ForwardModel(mesh, ej.opposition(n_el))

    for perm in (pm.perm_array, target.perm_array):
        expected = fwd.solve_eit(perm=perm)
        got = np.asarray(model(jnp.asarray(np.real(perm))))
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)


def test_jacobian_matches_finite_differences():
    mesh = ej.disc_mesh(n_el=16, n_rings=4)
    model = ej.ForwardModel(mesh, ej.opposition(16))
    sigma = ej.disc_inclusion(mesh, (0.3, -0.2), 0.3, 2.0)
    j = ej.jacobian(model, sigma)
    assert j.shape == (192, mesh.n_elem)
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.normal(size=mesh.n_elem))
    eps = 1e-6
    fd = (model(sigma + eps * direction) - model(sigma - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(j @ direction), np.asarray(fd), rtol=1e-6, atol=1e-9)


def test_reciprocity_and_scaling():
    mesh = ej.disc_mesh(n_el=16, n_rings=5)
    model = eqx.filter_jit(ej.ForwardModel(mesh, ej.opposition(16)))
    v = model(jnp.ones(mesh.n_elem))
    # doubling conductivity halves every voltage
    np.testing.assert_allclose(np.asarray(model(2.0 * jnp.ones(mesh.n_elem))), np.asarray(v) / 2, rtol=1e-10)
    # a rotationally symmetric tank gives the same 12 values for every drive pair
    # (in a different order, because measurement always starts at electrode 0)
    np.testing.assert_allclose(np.sort(np.asarray(v).reshape(16, 12), axis=1).std(axis=0), 0, atol=1e-9)
