"""Composable pass pipeline for stencil compilation.

A :class:`Pipeline` is an ordered list of xDSL ``ModulePass`` instances
plus a tiny ``run`` driver. The set of passes for a target/optimisation
level is described by a :class:`PipelineSpec` (a thin declarative
struct), which :meth:`Pipeline.from_spec` turns into a Pipeline.

This split lets users dial optimisation, swap passes, or add custom
passes without touching the launcher::

    pipeline = Pipeline.from_spec(PipelineSpec(
        opt_level="O2",
        shared_mem_bytes=99 * 1024,   # Hopper smem
        max_temporal_steps=4,
        extra_passes=[MyCustomPass()],
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass


# Optimisation levels — what each enables, declaratively.
#
# O0: just analysis (no tiling, no boundary, no temporal blocking)
# O1: + tiling   (the minimum needed for cutile_target lowering)
# O2: + boundary + temporal blocking (the previous "single_gpu" default)
_OPT_LEVELS = {
    "O0": {"tiling": False, "boundary": False, "temporal": False},
    "O1": {"tiling": True,  "boundary": False, "temporal": False},
    "O2": {"tiling": True,  "boundary": True,  "temporal": True},
}


@dataclass
class PipelineSpec:
    """Declarative description of which passes to run, in what order.

    The hardware budgets (``shared_mem_bytes``, ``dtype_bytes``) and
    ``max_temporal_steps`` flow into every pass that needs them.
    Set ``opt_level`` for a stock combination, or override individual
    flags for fine control. ``extra_passes`` are appended at the end
    of the standard pipeline — useful for experimental passes without
    touching the standard build.
    """

    opt_level: str = "O2"
    shared_mem_bytes: int = 49152
    dtype_bytes: int = 8
    max_temporal_steps: int = 8

    enable_normalize: bool = True
    enable_analysis: bool = True
    enable_roofline: bool = True
    enable_tiling: bool | None = None     # None => use opt_level default
    enable_boundary: bool | None = None
    enable_temporal: bool | None = None

    extra_passes: list[ModulePass] = field(default_factory=list)

    def resolve_flag(self, name: str) -> bool:
        explicit = getattr(self, f"enable_{name}")
        if explicit is not None:
            return explicit
        if self.opt_level not in _OPT_LEVELS:
            raise ValueError(f"unknown opt_level: {self.opt_level!r}; "
                             f"choose from {sorted(_OPT_LEVELS)}")
        return _OPT_LEVELS[self.opt_level][name]


@dataclass
class Pipeline:
    """Ordered sequence of passes + a ``run`` driver."""

    passes: list[ModulePass] = field(default_factory=list)

    def add(self, pass_: ModulePass) -> Pipeline:
        """Append a pass and return ``self`` for fluent chaining."""
        self.passes.append(pass_)
        return self

    def run(self, module: ModuleOp) -> ModuleOp:
        """Run all passes in order on ``module`` and return it."""
        ctx = Context()
        for p in self.passes:
            p.apply(ctx, module)
        return module

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: PipelineSpec) -> Pipeline:
        """Build a pipeline from a declarative spec."""
        from cutile.lowering.normalize import NormalizePass
        from cutile.passes.analysis.footprint import AnalysisPass
        from cutile.passes.analysis.roofline import RooflinePass
        from cutile.passes.boundary import BoundaryPass
        from cutile.passes.tiling import TilingPass
        from cutile.passes.temporal import TemporalBlockingPass

        p = cls()
        if spec.enable_normalize:
            p.add(NormalizePass())
        if spec.enable_analysis:
            p.add(AnalysisPass())
        if spec.enable_roofline:
            p.add(RooflinePass())
        if spec.resolve_flag("tiling"):
            p.add(TilingPass(
                shared_mem_bytes=spec.shared_mem_bytes,
                dtype_bytes=spec.dtype_bytes,
            ))
        if spec.resolve_flag("boundary"):
            p.add(BoundaryPass())
        if spec.resolve_flag("temporal"):
            p.add(TemporalBlockingPass(
                max_steps=spec.max_temporal_steps,
                shared_mem_bytes=spec.shared_mem_bytes,
                dtype_bytes=spec.dtype_bytes,
            ))
        for extra in spec.extra_passes:
            p.add(extra)
        return p

    @classmethod
    def single_gpu(
        cls,
        shared_mem_bytes: int = 49152,
        dtype_bytes: int = 8,
        max_temporal_steps: int = 8,
    ) -> Pipeline:
        """Backward-compatible factory matching the pre-spec default."""
        return cls.from_spec(PipelineSpec(
            opt_level="O2",
            shared_mem_bytes=shared_mem_bytes,
            dtype_bytes=dtype_bytes,
            max_temporal_steps=max_temporal_steps,
        ))
