"""Tests for new scientific applications: conservation laws, stability checks."""

import shutil
import numpy as np
import pytest
from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.registry import clear
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.reference.stencil_ref import apply_stencil, time_march, _ArrayProxy




@pytest.fixture(autouse=True)
def _clear():
    clear()
    yield
    clear()


class TestAdvectionUpwind:
    def test_upwind_stability(self):
        """Upwind advection with CFL=0.5 should remain bounded."""
        CFL = 0.5

        @stencil(ndim=1, order=2)
        def advect(u, i):
            return u[i] - CFL * (u[i] - u[i - 1])

        spec = advect._stencil_spec
        accesses = extract_footprint(spec)
        spec.halo_widths = compute_halo(accesses, 1)

        N = 100
        h = spec.halo_widths[0]
        u0 = np.zeros(N + 2 * h)
        u0[h + 20:h + 40] = 1.0

        history = time_march(u0, spec, steps=50)
        assert np.max(np.abs(history[-1])) <= 1.0 + 1e-10

    def test_upwind_asymmetric_footprint(self):
        """Upwind stencil has asymmetric offsets."""
        CFL = 0.5

        @stencil(ndim=1, order=2)
        def advect(u, i):
            return u[i] - CFL * (u[i] - u[i - 1])

        spec = advect._stencil_spec
        accesses = extract_footprint(spec)
        offsets = {a.offsets for a in accesses}
        assert (-1,) in offsets
        assert (0,) in offsets
        # No +1 offset (left-biased)
        assert (1,) not in offsets


class TestGrayScott:
    def test_gray_scott_bounded(self):
        """Gray-Scott fields u and v remain in [0, 1] range."""
        Du = 0.16
        Dv = 0.08
        F = 0.035
        k = 0.065
        dt = 1.0

        @stencil(ndim=2, order=2)
        def gs_u(u, v, i, j):
            lap = u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
            return u[i, j] + dt * (Du * lap - u[i, j] * v[i, j] * v[i, j] + F * (1 - u[i, j]))

        @stencil(ndim=2, order=2)
        def gs_v(u, v, i, j):
            lap = v[i - 1, j] + v[i + 1, j] + v[i, j - 1] + v[i, j + 1] - 4 * v[i, j]
            return v[i, j] + dt * (Dv * lap + u[i, j] * v[i, j] * v[i, j] - (F + k) * v[i, j])

        spec_u = gs_u._stencil_spec
        spec_v = gs_v._stencil_spec
        for sp in [spec_u, spec_v]:
            acc = extract_footprint(sp)
            sp.halo_widths = compute_halo(acc, 2)

        N = 20
        hx, hy = spec_u.halo_widths
        u = np.ones((N + 2 * hx, N + 2 * hy))
        v = np.zeros_like(u)
        cx, cy = (N + 2 * hx) // 2, (N + 2 * hy) // 2
        u[cx - 2:cx + 2, cy - 2:cy + 2] = 0.50
        v[cx - 2:cx + 2, cy - 2:cy + 2] = 0.25

        for _ in range(20):
            proxy_u = {'u': _ArrayProxy(u, spec_u.halo_widths), 'v': _ArrayProxy(v, spec_u.halo_widths)}
            result_u = spec_u.update_fn(proxy_u['u'], proxy_u['v'], 0, 0)
            u_new = u.copy()
            interior = tuple(slice(h, s - h) for h, s in zip(spec_u.halo_widths, u.shape))
            u_new[interior] = result_u

            proxy_v = {'u': _ArrayProxy(u, spec_v.halo_widths), 'v': _ArrayProxy(v, spec_v.halo_widths)}
            result_v = spec_v.update_fn(proxy_v['u'], proxy_v['v'], 0, 0)
            v_new = v.copy()
            v_new[interior] = result_v

            u, v = u_new, v_new

        assert u.min() >= -0.1
        assert u.max() <= 1.1
        assert v.min() >= -0.1
        assert v.max() <= 1.1

    def test_multi_input_footprint(self):
        """Gray-Scott stencils have accesses to both u and v."""
        @stencil(ndim=2, order=2)
        def gs(u, v, i, j):
            return u[i, j] + v[i, j]

        spec = gs._stencil_spec
        accesses = extract_footprint(spec)
        arr_names = {a.array_name for a in accesses}
        assert "u" in arr_names
        assert "v" in arr_names


class TestFDTDMaxwell:
    def test_fdtd_stable(self):
        """FDTD simulation remains stable with CFL < 1."""
        dx = 0.01
        dt = 0.005
        eps = 1.0
        mu = 1.0

        @stencil(ndim=1, order=2)
        def upd_E(E, H, i):
            return E[i] + (dt / (eps * dx)) * (H[i] - H[i - 1])

        @stencil(ndim=1, order=2)
        def upd_H(E, H, i):
            return H[i] + (dt / (mu * dx)) * (E[i + 1] - E[i])

        spec_E = upd_E._stencil_spec
        spec_H = upd_H._stencil_spec
        for sp in [spec_E, spec_H]:
            acc = extract_footprint(sp)
            sp.halo_widths = compute_halo(acc, 1)

        N = 100
        h = 1
        E = np.zeros(N + 2 * h)
        H = np.zeros(N + 2 * h)
        E[N // 2 + h] = 1.0  # Initial pulse

        for _ in range(50):
            proxy_h = {'E': _ArrayProxy(E, spec_H.halo_widths), 'H': _ArrayProxy(H, spec_H.halo_widths)}
            result_H = spec_H.update_fn(proxy_h['E'], proxy_h['H'], 0)
            H_new = H.copy()
            H_new[h:-h] = result_H
            H = H_new

            proxy_e = {'E': _ArrayProxy(E, spec_E.halo_widths), 'H': _ArrayProxy(H, spec_E.halo_widths)}
            result_E = spec_E.update_fn(proxy_e['E'], proxy_e['H'], 0)
            E_new = E.copy()
            E_new[h:-h] = result_E
            E = E_new

        # FDTD should be stable
        assert np.max(np.abs(E)) < 10.0
        assert np.max(np.abs(H)) < 10.0

    def test_two_field_coupling(self):
        """E and H stencils access both arrays."""
        @stencil(ndim=1, order=2)
        def upd_E(E, H, i):
            return E[i] + H[i] - H[i - 1]

        spec = upd_E._stencil_spec
        accesses = extract_footprint(spec)
        arr_names = {a.array_name for a in accesses}
        assert "E" in arr_names
        assert "H" in arr_names


class TestShallowWater:
    def test_shallow_water_stable(self):
        """Linearized shallow water remains stable for small dt."""
        g = 9.81
        H0 = 1.0
        dt = 0.001  # Small for stability
        dx = 0.1

        @stencil(ndim=2, order=2)
        def sw_h(h, hu, hv, i, j):
            dhu = (hu[i + 1, j] - hu[i - 1, j]) / (2 * dx)
            dhv = (hv[i, j + 1] - hv[i, j - 1]) / (2 * dx)
            return h[i, j] - dt * H0 * (dhu + dhv)

        spec = sw_h._stencil_spec
        accesses = extract_footprint(spec)
        spec.halo_widths = compute_halo(accesses, 2)

        # Verify 3-input stencil
        assert spec.inputs == ("h", "hu", "hv")

    def test_three_field_inputs(self):
        """Shallow water h-equation accesses all three fields."""
        @stencil(ndim=2, order=2)
        def sw(h, hu, hv, i, j):
            return h[i, j] + hu[i + 1, j] + hv[i, j + 1]

        spec = sw._stencil_spec
        accesses = extract_footprint(spec)
        arr_names = {a.array_name for a in accesses}
        assert "h" in arr_names
        assert "hu" in arr_names
        assert "hv" in arr_names


class TestExampleSmoke:
    def test_advection_upwind_example(self):
        from examples.advection_upwind import main
        main()

    def test_gray_scott_example(self):
        from examples.gray_scott import main
        main()

    def test_shallow_water_example(self):
        from examples.shallow_water import main
        main()

    def test_fdtd_maxwell_example(self):
        from examples.fdtd_maxwell_1d import main
        main()
