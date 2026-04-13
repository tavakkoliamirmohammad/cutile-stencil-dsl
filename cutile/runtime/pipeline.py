"""Composable pass pipeline for stencil compilation.

Wraps a sequence of xDSL ``ModulePass`` instances into a single callable
unit, providing fluent ``add()`` chaining and factory methods for common
configurations (e.g. single-GPU with all standard passes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass


@dataclass
class Pipeline:
    """Composable pass pipeline for stencil compilation."""

    passes: list[ModulePass] = field(default_factory=list)

    def add(self, pass_: ModulePass) -> Pipeline:
        """Append a pass and return *self* for fluent chaining."""
        self.passes.append(pass_)
        return self

    def run(self, module: ModuleOp) -> ModuleOp:
        """Run all passes in order on *module* and return it."""
        ctx = Context()
        for p in self.passes:
            p.apply(ctx, module)
        return module

    @classmethod
    def single_gpu(
        cls,
        shared_mem_bytes: int = 49152,
        dtype_bytes: int = 8,
        max_temporal_steps: int = 8,
    ) -> Pipeline:
        """Default single-GPU pipeline.

        Pass order: normalize -> analysis -> roofline -> tiling -> boundary
        -> temporal blocking.
        """
        from cutile.lowering.normalize import NormalizePass
        from cutile.passes.analysis.footprint import AnalysisPass
        from cutile.passes.analysis.roofline import RooflinePass
        from cutile.passes.boundary import BoundaryPass
        from cutile.passes.tiling import TilingPass
        from cutile.passes.temporal import TemporalBlockingPass

        p = cls()
        p.add(NormalizePass())
        p.add(AnalysisPass())
        p.add(RooflinePass())
        p.add(TilingPass(shared_mem_bytes=shared_mem_bytes, dtype_bytes=dtype_bytes))
        p.add(BoundaryPass())
        p.add(TemporalBlockingPass(
            max_steps=max_temporal_steps,
            shared_mem_bytes=shared_mem_bytes,
            dtype_bytes=dtype_bytes,
        ))
        return p
