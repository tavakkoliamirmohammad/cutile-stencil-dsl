"""Implicit stencil decorator for backward-Euler and similar implicit time-stepping.

An @implicit_stencil captures the same StencilSpec as @stencil, but additionally
provides compile_solver() which wires the stencil into a CG solver for implicit
time-stepping problems of the form:
    (I - dt * L) u^{n+1} = u^n
where L is the stencil operator.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

from cutile_stencil.dsl.decorator import stencil as stencil_decorator
from cutile_stencil.dsl.types import StencilSpec, HardwareSpec


@dataclass
class ImplicitCompileResult:
    """Result of compiling an implicit stencil into a CG solver."""
    spec: StencilSpec
    stencil_cg_result: object  # StencilCGCompileResult from stencil_cg module
    domain: Tuple[int, ...]

    def print_summary(self):
        """Print solver compilation summary."""
        print(f"Implicit stencil solver: {self.spec.name}")
        print(f"  Domain: {self.domain}")
        print(f"  ndim={self.spec.ndim}, order={self.spec.order}")
        if hasattr(self.stencil_cg_result, 'print_summary'):
            self.stencil_cg_result.print_summary()


def implicit_stencil(
    fn=None,
    *,
    ndim: int = None,
    order: int = None,
    dtype: str = "float64",
):
    """Decorator for implicit stencil definitions.

    Works exactly like @stencil but adds a compile_solver() method to the
    wrapper function for convenient CG solver compilation.

    Usage::

        @implicit_stencil(ndim=1, order=2)
        def backward_euler_heat(u, i):
            return u[i-1] - 2*u[i] + u[i+1]

        result = backward_euler_heat.compile_solver(domain=(1024,))
    """
    def decorator(func):
        # Apply the regular @stencil decorator
        wrapped = stencil_decorator(
            func, ndim=ndim, order=order, dtype=dtype
        )
        spec = wrapped._stencil_spec

        def compile_solver(
            domain: Tuple[int, ...],
            hw: HardwareSpec = None,
            dtype: str = "float64",
            tile_size: int = 256,
        ) -> ImplicitCompileResult:
            """Compile this implicit stencil into a CG solver.

            Parameters
            ----------
            domain : tuple of int
                Domain size per dimension (interior points).
            hw : HardwareSpec, optional
                GPU hardware parameters.
            dtype : str
                Solver precision.
            tile_size : int
                Tile size for solver kernels.

            Returns
            -------
            ImplicitCompileResult
            """
            from cutile_stencil.solvers.stencil_cg import compile_stencil_cg
            from cutile_stencil.config import SolverConfig

            solver_config = SolverConfig(dtype=dtype, tile_size=tile_size)
            cg_result = compile_stencil_cg(
                spec, domain=domain, solver_config=solver_config, hw=hw,
            )
            return ImplicitCompileResult(
                spec=spec,
                stencil_cg_result=cg_result,
                domain=domain,
            )

        wrapped.compile_solver = compile_solver
        return wrapped

    if fn is not None:
        return decorator(fn)
    return decorator
