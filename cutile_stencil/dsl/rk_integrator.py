"""Multi-stage Runge-Kutta time integrators for stencil operators.

Generates Python code that orchestrates multiple stencil kernel launches
per timestep according to standard Butcher tableaux.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cutile_stencil.codegen.emitter import CodeEmitter


# Butcher tableaux for standard RK methods
# Format: (stages, a_matrix, b_weights, c_nodes)
_RK_METHODS: Dict[str, Tuple[int, List[List[float]], List[float], List[float]]] = {
    "RK2": (
        2,
        [[0.0], [1.0]],
        [0.5, 0.5],
        [0.0, 1.0],
    ),
    "RK4": (
        4,
        [[0.0], [0.5], [0.0, 0.5], [0.0, 0.0, 1.0]],
        [1/6, 1/3, 1/3, 1/6],
        [0.0, 0.5, 0.5, 1.0],
    ),
    "SSPRK3": (
        3,
        [[0.0], [1.0], [0.25, 0.25]],
        [1/6, 1/6, 2/3],
        [0.0, 1.0, 0.5],
    ),
}


@dataclass
class RKCompileResult:
    """Result of compiling an RK integrator."""
    method: str
    stages: int
    stencil_code: str       # The base stencil kernel code
    integrator_code: str    # The RK stepping code
    stencil_name: str

    def full_code(self) -> str:
        """Return combined stencil kernel + integrator code."""
        return self.stencil_code + "\n\n" + self.integrator_code


class RKIntegrator:
    """Multi-stage Runge-Kutta time integrator for stencil operators.

    Parameters
    ----------
    stencil_fn : decorated stencil function
        The spatial operator (must be a @stencil-decorated function).
    method : str
        RK method name: 'RK2', 'RK4', or 'SSPRK3'.
    """

    def __init__(self, stencil_fn, method: str = "RK4"):
        if method not in _RK_METHODS:
            raise ValueError(
                f"Unknown RK method '{method}'. "
                f"Choose from: {list(_RK_METHODS.keys())}"
            )
        if not hasattr(stencil_fn, '_stencil_spec'):
            raise TypeError(
                f"Expected a @stencil-decorated function, "
                f"got {type(stencil_fn).__name__}"
            )
        self.stencil_fn = stencil_fn
        self.spec = stencil_fn._stencil_spec
        self.method = method
        self.stages, self.a, self.b, self.c = _RK_METHODS[method]

    def compile(
        self,
        domain: Tuple[int, ...],
        hw=None,
    ) -> RKCompileResult:
        """Compile the stencil and generate RK integration code.

        Parameters
        ----------
        domain : tuple of int
            Domain size per dimension.
        hw : HardwareSpec, optional
            GPU hardware parameters.

        Returns
        -------
        RKCompileResult
            Contains the stencil kernel code and RK stepping code.
        """
        from cutile_stencil.dsl.pipeline import compile as stencil_compile

        compiled = stencil_compile(self.stencil_fn, domain, hw)
        integrator_code = self._emit_integrator(compiled.spec.name)

        return RKCompileResult(
            method=self.method,
            stages=self.stages,
            stencil_code=compiled.code,
            integrator_code=integrator_code,
            stencil_name=compiled.spec.name,
        )

    def _emit_integrator(self, stencil_name: str) -> str:
        """Generate the RK time-stepping function."""
        e = CodeEmitter()
        e.line(f"# {self.method} integrator for {stencil_name}")
        e.line(f"# Stages: {self.stages}")
        e.blank()

        launch = f"launch_{stencil_name}"

        e.line(f"def rk_step_{stencil_name}(u, dt, stream=None):")
        with e.indent():
            e.line(f'"""Advance one {self.method} timestep of dt."""')
            # Allocate stage arrays
            for s in range(self.stages):
                e.line(f"k{s} = cp.zeros_like(u)")
            e.blank()

            # Compute each stage
            for s in range(self.stages):
                e.line(f"# Stage {s + 1}")
                if s == 0:
                    # k0 = L(u)
                    e.line(f"{launch}(u, k{s}, stream=stream)")
                else:
                    # u_stage = u + dt * sum(a[s][j] * k[j])
                    e.line(f"u_stage = u.copy()")
                    for j in range(s):
                        a_sj = self.a[s][j] if j < len(self.a[s]) else 0.0
                        if a_sj != 0.0:
                            e.line(f"u_stage += {a_sj} * dt * k{j}")
                    e.line(f"{launch}(u_stage, k{s}, stream=stream)")
                e.blank()

            # Final combination: u_new = u + dt * sum(b[s] * k[s])
            e.line("# Combine stages")
            e.line("u_new = u.copy()")
            for s in range(self.stages):
                b_s = self.b[s]
                if b_s != 0.0:
                    e.line(f"u_new += {b_s} * dt * k{s}")
            e.line("return u_new")

        return e.render()
