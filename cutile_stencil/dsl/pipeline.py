"""One-call analysis and compilation pipeline for stencil specifications.

Provides ``analyze()`` for analysis-only and ``compile()`` for the full
analyze + codegen pipeline in a single call.
"""

from __future__ import annotations

import ast
import importlib.util
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from cutile_stencil.dsl.types import (
    StencilSpec, HardwareSpec, TileConfig, TemporalConfig, RooflineResult,
    BrickLayout,
)
from cutile_stencil.analysis.footprint import extract_footprint, compute_halo
from cutile_stencil.analysis.tiling import compute_tile_config
from cutile_stencil.analysis.temporal import compute_temporal_config
from cutile_stencil.analysis.roofline import roofline_analysis


@dataclass
class AnalysisResult:
    """Complete analysis result for a stencil."""
    spec: StencilSpec
    tile_config: TileConfig
    temporal_config: TemporalConfig
    roofline: RooflineResult
    warnings: list


@dataclass
class CompileResult:
    """Result of compile(): analysis + generated kernel code."""
    spec: StencilSpec
    analysis: AnalysisResult
    code: str
    layout: Optional[BrickLayout] = field(default=None, repr=False)
    _emitted_path: Optional[Path] = field(default=None, repr=False)

    def print_summary(self) -> None:
        """Print analysis summary: footprint, tiling, temporal, roofline."""
        spec = self.spec
        ar = self.analysis
        print(f"Stencil: {spec.name}, ndim={spec.ndim}, order={spec.order}")
        print(f"Footprint: {len(spec.accesses)} accesses, halo={spec.halo_widths}")
        print(f"  Inputs: {spec.inputs}")
        tc = ar.tile_config
        print(f"Tile config: sizes={tc.tile_sizes}, "
              f"halo={tc.halo_widths}, overhead={tc.overhead_fraction:.4f}")
        tmp = ar.temporal_config
        print(f"Temporal blocking: {tmp.steps} steps, "
              f"expanded_halo={tmp.expanded_halo}, "
              f"BW reduction={tmp.bandwidth_reduction_factor:.1f}x")
        rf = ar.roofline
        print(f"Roofline: {rf.flops_per_point} FLOPs/pt, "
              f"{rf.bytes_per_point} B/pt, AI={rf.arithmetic_intensity:.3f}")
        print(f"  Bound: {rf.bound}, peak ~ {rf.peak_gpoints_s:.2f} Gpts/s")
        if ar.warnings:
            for w in ar.warnings:
                print(f"  WARNING: {w}")

    def emit_to_file(self, path: str | Path) -> Path:
        """Write generated kernel code to a file. Returns the Path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.code)
        self._emitted_path = path
        return path

    def validate(self, *inputs, atol: float = 1e-10) -> None:
        """Validate GPU kernel against NumPy reference.

        Parameters
        ----------
        *inputs : numpy.ndarray
            Input arrays (flat, including halo). For single-input stencils
            pass one array; for multi-input pass one per field.  For bricked
            kernels the inputs should still be *flat* — they are converted to
            bricked layout automatically.
        atol : float
            Absolute tolerance for np.allclose.

        Raises
        ------
        AssertionError
            If GPU result differs from CPU reference beyond tolerance.
        """
        import cupy as cp

        spec = self.spec

        # Emit to temp file if not already emitted
        if self._emitted_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
            tmp.write(self.code)
            tmp.close()
            self._emitted_path = Path(tmp.name)

        # Load generated module
        mod_spec = importlib.util.spec_from_file_location(
            f"kernel_{spec.name}", str(self._emitted_path)
        )
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        launcher = getattr(mod, f"launch_{spec.name}")

        # Determine input arrays
        if len(inputs) == 0:
            raise ValueError("At least one input array is required")

        multi_input = len(spec.inputs) > 1
        halo = spec.halo_widths

        if self.layout is not None:
            self._validate_bricked(
                inputs, launcher, spec, halo, multi_input, atol
            )
        else:
            self._validate_flat(
                inputs, launcher, spec, halo, multi_input, atol
            )

    def _validate_flat(self, inputs, launcher, spec, halo, multi_input, atol):
        import cupy as cp

        gpu_inputs = [cp.asarray(arr, dtype=cp.float64) for arr in inputs]
        out_gpu = cp.zeros_like(gpu_inputs[0])

        if multi_input:
            launcher(*gpu_inputs, out_gpu)
        else:
            launcher(gpu_inputs[0], out_gpu)
        cp.cuda.Device().synchronize()
        gpu_result = cp.asnumpy(out_gpu)

        # CPU reference
        from cutile_stencil.reference.stencil_ref import _ArrayProxy
        if multi_input:
            proxies = [_ArrayProxy(arr, halo) for arr in inputs]
            idx_args = [0] * spec.ndim
            cpu_interior = spec.update_fn(*proxies, *idx_args)
            cpu_ref = inputs[0].copy()
            interior = tuple(slice(h, s - h) for h, s in zip(halo, cpu_ref.shape))
            cpu_ref[interior] = cpu_interior
        else:
            from cutile_stencil.reference.stencil_ref import apply_stencil
            cpu_ref = apply_stencil(inputs[0], spec)
            interior = tuple(slice(h, s - h) for h, s in zip(halo, inputs[0].shape))

        assert np.allclose(
            gpu_result[interior], cpu_ref[interior], atol=atol
        ), f"GPU kernel mismatch for {spec.name}!"
        print(f"  ✓ GPU kernel ({spec.name}) matches NumPy reference")

    def _validate_bricked(self, inputs, launcher, spec, halo, multi_input, atol):
        import cupy as cp
        from cutile_stencil.layout.bricks import to_bricks, from_bricks

        brick_sizes = self.layout.brick_sizes

        # Convert flat inputs to bricked layout
        bricked_inputs_np = [to_bricks(arr, brick_sizes, halo) for arr in inputs]
        gpu_inputs = [cp.asarray(b, dtype=cp.float64) for b in bricked_inputs_np]
        out_gpu = cp.zeros_like(gpu_inputs[0])

        if multi_input:
            launcher(*gpu_inputs, out_gpu)
        else:
            launcher(gpu_inputs[0], out_gpu)
        cp.cuda.Device().synchronize()

        # Convert bricked GPU output back to flat
        gpu_bricked = cp.asnumpy(out_gpu)
        gpu_flat = from_bricks(gpu_bricked, brick_sizes, halo)

        # CPU reference on flat data
        from cutile_stencil.reference.stencil_ref import _ArrayProxy
        if multi_input:
            proxies = [_ArrayProxy(arr, halo) for arr in inputs]
            idx_args = [0] * spec.ndim
            cpu_interior = spec.update_fn(*proxies, *idx_args)
            cpu_ref = inputs[0].copy()
            interior = tuple(slice(h, s - h) for h, s in zip(halo, cpu_ref.shape))
            cpu_ref[interior] = cpu_interior
        else:
            from cutile_stencil.reference.stencil_ref import apply_stencil
            cpu_ref = apply_stencil(inputs[0], spec)
            interior = tuple(slice(h, s - h) for h, s in zip(halo, inputs[0].shape))

        assert np.allclose(
            gpu_flat[interior], cpu_ref[interior], atol=atol
        ), f"GPU bricked kernel mismatch for {spec.name}!"
        print(f"  ✓ GPU bricked kernel ({spec.name}) matches NumPy reference")


def analyze(
    spec: StencilSpec,
    domain: Tuple[int, ...],
    hw: HardwareSpec | None = None,
) -> AnalysisResult:
    """One-call full analysis: footprint -> halo -> tiling -> temporal -> roofline.

    Parameters
    ----------
    spec : StencilSpec
        The stencil specification (from @stencil decorator).
    domain : tuple of int
        Domain size per dimension (interior points).
    hw : HardwareSpec, optional
        GPU hardware parameters. Defaults to HardwareSpec().

    Returns
    -------
    AnalysisResult
        Complete analysis including tile config, temporal blocking, and roofline.
    """
    if hw is None:
        hw = HardwareSpec()

    # Step 1: Extract footprint
    accesses = extract_footprint(spec)

    # Step 2: Compute halo widths
    halo = compute_halo(accesses, spec.ndim)
    spec.halo_widths = halo

    # Step 3: Tile configuration
    tile_cfg = compute_tile_config(spec, domain, hw)

    # Step 4: Temporal blocking
    temp_cfg = compute_temporal_config(spec, tile_cfg, hw)
    spec.temporal_steps = temp_cfg.steps

    # Step 5: Roofline analysis
    roof = roofline_analysis(spec, hw)

    # Step 6: Validation warnings
    warnings = spec.validate()

    return AnalysisResult(
        spec=spec,
        tile_config=tile_cfg,
        temporal_config=temp_cfg,
        roofline=roof,
        warnings=warnings,
    )


def compile(
    stencil_fn,
    domain: Tuple[int, ...],
    hw: HardwareSpec | None = None,
    layout: BrickLayout | None = None,
) -> CompileResult:
    """One-call analyze + codegen. Returns a CompileResult with generated code.

    Parameters
    ----------
    stencil_fn : decorated stencil function or StencilSpec
        The stencil to compile. Accepts either a @stencil-decorated function
        (which has ._stencil_spec) or a StencilSpec directly.
    domain : tuple of int
        Domain size per dimension (interior points).
    hw : HardwareSpec, optional
        GPU hardware parameters. Defaults to HardwareSpec().
    layout : BrickLayout, optional
        If provided, generate a bricked memory layout kernel instead of a
        flat kernel. The domain must be divisible by the brick sizes.

    Returns
    -------
    CompileResult
        Contains analysis results and generated kernel code.
    """
    # Extract spec
    if isinstance(stencil_fn, StencilSpec):
        spec = stencil_fn
    elif hasattr(stencil_fn, '_stencil_spec'):
        spec = stencil_fn._stencil_spec
    else:
        raise TypeError(
            f"Expected a @stencil-decorated function or StencilSpec, "
            f"got {type(stencil_fn).__name__}"
        )

    # Run analysis
    ar = analyze(spec, domain, hw)

    # Validate brick layout if provided
    if layout is not None:
        layout.validate(spec.ndim, spec.halo_widths)
        layout.num_bricks(domain)  # validates divisibility

    # Generate code
    from cutile_stencil.codegen.stencil_codegen import StencilCodeGenerator
    gen = StencilCodeGenerator(
        spec, ar.tile_config, ar.temporal_config, layout=layout
    )
    code = gen.emit()

    # Validate syntax
    ast.parse(code)

    return CompileResult(spec=spec, analysis=ar, code=code, layout=layout)
