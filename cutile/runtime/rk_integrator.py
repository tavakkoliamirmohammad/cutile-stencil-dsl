"""Runge-Kutta time integrator for cuTile stencil operators.

Wraps a compiled stencil kernel and generates Python source code
that performs RK time stepping (Euler, RK2, RK4, or SSPRK3).
"""

from __future__ import annotations

import importlib.util
import tempfile
from dataclasses import dataclass
from typing import Optional

from cutile.runtime.launcher import compile as stencil_compile, CompileResult


# ---------------------------------------------------------------------- #
# RKCompileResult
# ---------------------------------------------------------------------- #


@dataclass
class RKCompileResult:
    """Result of compiling an RK integrator."""

    stencil_result: CompileResult  # the underlying stencil compilation
    method: str                    # "euler", "rk2", "rk4", "ssprk3"
    num_stages: int
    code: str                      # generated Python source with RK stepping

    def emit_to_file(self, path: str) -> None:
        """Write the generated source to *path*."""
        with open(path, "w") as f:
            f.write(self.code)

    def load_module(self):
        """Load the generated code as a Python module."""
        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w"
        ) as f:
            f.write(self.code)
            tmp_path = f.name
        spec = importlib.util.spec_from_file_location("_rk", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


# ---------------------------------------------------------------------- #
# RKIntegrator
# ---------------------------------------------------------------------- #


class RKIntegrator:
    """Runge-Kutta time integrator for stencil operators.

    Usage::

        @stencil(ndim=2, order=2)
        def heat(u, i, j):
            return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

        integrator = RKIntegrator(heat, method="RK4")
        result = integrator.compile(domain=(256, 256))
    """

    _STAGES = {
        "euler": 1,
        "rk2": 2,
        "rk4": 4,
        "ssprk3": 3,
    }

    def __init__(self, stencil_fn, method: str = "rk4"):
        self.stencil_fn = stencil_fn
        self.method = method.lower()
        if self.method not in self._STAGES:
            raise ValueError(
                f"Unknown RK method {method!r}. "
                f"Supported: {', '.join(self._STAGES)}"
            )

    def compile(
        self,
        domain: tuple[int, ...] | None = None,
        hw=None,
    ) -> RKCompileResult:
        """Compile the stencil and wrap it in an RK time stepping loop.

        Parameters
        ----------
        domain
            Optional domain shape for the underlying stencil compilation.
        hw
            Optional hardware specification.

        Returns
        -------
        RKCompileResult
        """
        # 1. Compile the underlying stencil (no temporal blocking)
        stencil_result = stencil_compile(
            self.stencil_fn,
            domain=domain,
            hw=hw,
            temporal_blocking=False,
        )

        # 2. Generate RK stepping code that calls the stencil kernel
        rk_code = self._generate_rk_code(stencil_result)

        return RKCompileResult(
            stencil_result=stencil_result,
            method=self.method,
            num_stages=self._STAGES[self.method],
            code=rk_code,
        )

    # ------------------------------------------------------------------ #
    # Code generation
    # ------------------------------------------------------------------ #

    def _generate_rk_code(self, result: CompileResult) -> str:
        """Generate Python code with RK time stepping."""
        from cutile.lowering.emitter import CodeEmitter

        e = CodeEmitter()

        # Include the stencil kernel code
        e.line("# --- Stencil kernel (auto-generated) ---")
        for line in result.code.splitlines():
            e.line(line)
        e.blank()

        name = result.name

        # Generate the RK stepping function
        e.line(
            f"def rk_{self.method}_step(u, dt, "
            f"launch_fn=launch_{name}):"
        )
        with e.indent():
            e.line('"""Perform one RK time step."""')
            e.line("import cupy as cp")
            e.blank()

            if self.method == "euler":
                self._emit_euler(e)
            elif self.method == "rk2":
                self._emit_rk2(e)
            elif self.method == "rk4":
                self._emit_rk4(e)
            elif self.method == "ssprk3":
                self._emit_ssprk3(e)

            e.line("return u")

        e.blank()

        # Generate the solve (multi-step) function
        e.line(
            f"def rk_{self.method}_solve(u0, dt, num_steps, "
            f"launch_fn=launch_{name}):"
        )
        with e.indent():
            e.line(
                f'"""Run {self.method.upper()} for num_steps time steps."""'
            )
            e.line("import cupy as cp")
            e.line("u = u0.copy()")
            e.line("for step in range(num_steps):")
            with e.indent():
                e.line(f"rk_{self.method}_step(u, dt, launch_fn)")
            e.line("return u")

        return e.render()

    # ------------------------------------------------------------------ #
    # Per-method emitters
    # ------------------------------------------------------------------ #

    @staticmethod
    def _emit_euler(e) -> None:
        """Emit forward Euler: u_new = u + dt * L(u)."""
        e.line("Lu = cp.zeros_like(u)")
        e.line("launch_fn(u, Lu)")
        e.line("u[:] = u + dt * Lu")

    @staticmethod
    def _emit_rk2(e) -> None:
        """Emit midpoint RK2."""
        e.line("k1 = cp.zeros_like(u)")
        e.line("launch_fn(u, k1)")
        e.line("u1 = u + 0.5 * dt * k1")
        e.line("k2 = cp.zeros_like(u)")
        e.line("launch_fn(u1, k2)")
        e.line("u[:] = u + dt * k2")

    @staticmethod
    def _emit_rk4(e) -> None:
        """Emit classic four-stage RK4."""
        e.line("k1 = cp.zeros_like(u)")
        e.line("launch_fn(u, k1)")
        e.blank()
        e.line("u1 = u + 0.5 * dt * k1")
        e.line("k2 = cp.zeros_like(u)")
        e.line("launch_fn(u1, k2)")
        e.blank()
        e.line("u2 = u + 0.5 * dt * k2")
        e.line("k3 = cp.zeros_like(u)")
        e.line("launch_fn(u2, k3)")
        e.blank()
        e.line("u3 = u + dt * k3")
        e.line("k4 = cp.zeros_like(u)")
        e.line("launch_fn(u3, k4)")
        e.blank()
        e.line("u[:] = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)")

    @staticmethod
    def _emit_ssprk3(e) -> None:
        """Emit strong stability preserving RK3."""
        e.line("k1 = cp.zeros_like(u)")
        e.line("launch_fn(u, k1)")
        e.line("u1 = u + dt * k1")
        e.blank()
        e.line("k2 = cp.zeros_like(u)")
        e.line("launch_fn(u1, k2)")
        e.line("u2 = 0.75 * u + 0.25 * (u1 + dt * k2)")
        e.blank()
        e.line("k3 = cp.zeros_like(u)")
        e.line("launch_fn(u2, k3)")
        e.line("u[:] = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * k3)")
