"""cuTile Stencil DSL — High-order stencil compilation and tile-based solvers."""

from cutile_stencil.dsl.decorator import stencil
from cutile_stencil.dsl.types import StencilSpec, HardwareSpec, TileConfig, RooflineResult, BrickLayout
from cutile_stencil.dsl.registry import register, lookup, all_stencils
from cutile_stencil.dsl.boundary import BoundaryType, BoundaryCondition, BoundarySpec
from cutile_stencil.dsl.pipeline import analyze, AnalysisResult, compile, CompileResult
from cutile_stencil.config import GPU_PRESETS
