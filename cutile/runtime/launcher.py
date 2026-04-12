"""Compilation and execution API for the cuTile stencil DSL.

Provides :func:`compile` which takes a ``@stencil``-decorated function
through the analysis pipeline, generates GPU kernel source, and returns
a :class:`CompileResult` with generated code, metadata, and helper
methods for validation and benchmarking.
"""

from __future__ import annotations

import importlib.util
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

from xdsl.dialects.stencil import ApplyOp

from cutile.runtime.pipeline import Pipeline


# ---------------------------------------------------------------------- #
# CompileResult
# ---------------------------------------------------------------------- #


@dataclass
class CompileResult:
    """Result of compiling a stencil through the pipeline."""

    name: str
    code: str
    ndim: int
    halo_widths: tuple[int, ...]
    tile_sizes: tuple[int, ...]
    temporal_steps: int
    analysis: dict = field(default_factory=dict)

    # -- original function for CPU reference --
    _ref_fn: Any = field(default=None, repr=False)

    def emit_to_file(self, path: str) -> None:
        """Write the generated source to *path*."""
        with open(path, "w") as f:
            f.write(self.code)

    def load_module(self):
        """Load the generated code as an importable Python module."""
        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w"
        ) as f:
            f.write(self.code)
            tmp_path = f.name
        spec = importlib.util.spec_from_file_location(
            f"_gen_{self.name}", tmp_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def validate(self, u_input=None, atol: float = 1e-10) -> bool:
        """Run the GPU kernel and compare against the CPU reference.

        Parameters
        ----------
        u_input : array-like, optional
            Input array.  If ``None`` a random array is created.
        atol : float
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            ``True`` if GPU and CPU results agree within *atol*.
        """
        import cupy as cp
        import numpy as np

        from cutile.reference.stencil_ref import apply_stencil

        if self._ref_fn is None:
            raise RuntimeError(
                "No reference function available -- pass a @stencil-decorated "
                "function to compile() to enable validation."
            )

        hw = self.halo_widths
        ndim = self.ndim
        T = self.temporal_steps

        if u_input is None:
            base = 128 if ndim == 1 else 64
            shape = tuple(base + 2 * T * h for h in hw)
            u_np = np.random.rand(*shape)
        else:
            u_np = np.asarray(u_input)
        ref_out = u_np.copy()
        for _ in range(T):
            ref_out = apply_stencil(ref_out, self._ref_fn, ndim, hw)

        # GPU kernel
        mod = self.load_module()
        launcher = getattr(mod, f"launch_{self.name}")
        u_gpu = cp.asarray(u_np)
        out_gpu = cp.zeros_like(u_gpu)
        launcher(u_gpu, out_gpu)
        cp.cuda.Device().synchronize()
        gpu_out = cp.asnumpy(out_gpu)

        margins = tuple(T * h for h in hw)
        interior = tuple(slice(m, s - m) for m, s in zip(margins, u_np.shape))
        max_diff = float(np.max(np.abs(ref_out[interior] - gpu_out[interior])))
        ok = np.allclose(ref_out[interior], gpu_out[interior], atol=atol)
        print(f"Validation: max_diff={max_diff:.2e}, pass={ok}")
        return ok

    def benchmark(
        self,
        u_input=None,
        warmup: int = 5,
        iters: int = 20,
    ) -> dict:
        """Benchmark the generated kernel.

        Returns
        -------
        dict
            ``time_ms``, ``gpoints_per_s``, ``gbytes_per_s``.
        """
        import cupy as cp
        import numpy as np

        hw = self.halo_widths
        ndim = self.ndim

        if u_input is None:
            shape = tuple(128 + 2 * h for h in hw) if ndim == 1 else tuple(
                64 + 2 * h for h in hw
            )
            u_np = np.random.rand(*shape)
        else:
            u_np = np.asarray(u_input)

        mod = self.load_module()
        launcher = getattr(mod, f"launch_{self.name}")
        u_gpu = cp.asarray(u_np)
        out_gpu = cp.zeros_like(u_gpu)

        # Warmup
        for _ in range(warmup):
            launcher(u_gpu, out_gpu)
        cp.cuda.Device().synchronize()

        # Timed runs
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(iters):
            launcher(u_gpu, out_gpu)
        end.record()
        end.synchronize()
        elapsed_ms = float(cp.cuda.get_elapsed_time(start, end)) / iters

        # Metrics
        interior_size = 1
        for h, s in zip(hw, u_np.shape):
            interior_size *= s - 2 * h
        gpoints = interior_size / elapsed_ms * 1e-6  # GPoints/s
        dtype_bytes = 8  # float64
        # Memory traffic: reads + writes per point
        gbytes = interior_size * 2 * dtype_bytes / elapsed_ms * 1e-6

        return {
            "time_ms": elapsed_ms,
            "gpoints_per_s": gpoints,
            "gbytes_per_s": gbytes,
        }


# ---------------------------------------------------------------------- #
# compile()
# ---------------------------------------------------------------------- #


def compile(
    stencil_fn,
    domain: tuple[int, ...] | None = None,
    hw=None,
    pipeline: Pipeline | None = None,
    temporal_blocking: bool = True,
    autotune: bool = False,
) -> CompileResult:
    """Compile a ``@stencil``-decorated function through the pipeline.

    Parameters
    ----------
    stencil_fn
        Decorated stencil function (must have a ``_ir`` attribute).
    domain
        Optional domain shape for autotuning / validation.
    hw
        Optional :class:`~cutile.config.HardwareSpec`.
    pipeline
        Custom :class:`Pipeline`.  Defaults to ``Pipeline.single_gpu()``.
    temporal_blocking
        Enable temporal blocking (default ``True``).
    autotune
        Enable autotuning (default ``False``).

    Returns
    -------
    CompileResult
    """
    ir = stencil_fn._ir
    name = stencil_fn._fn.__name__

    # -------------------------------------------------------------- #
    # 1. Run the analysis pipeline on a clone
    # -------------------------------------------------------------- #
    analysis_clone = ir.clone()

    if pipeline is None:
        shared_mem = 49152
        dtype_bytes = 4 if getattr(stencil_fn, "_dtype", "float64") in (
            "float32", "fp32"
        ) else 8
        if hw is not None:
            shared_mem = hw.shared_mem_bytes
            dtype_bytes = hw.dtype_bytes
        max_ts = 8 if temporal_blocking else 1
        pipeline = Pipeline.single_gpu(
            shared_mem_bytes=shared_mem,
            dtype_bytes=dtype_bytes,
            max_temporal_steps=max_ts,
        )

    pipeline.run(analysis_clone)

    # -------------------------------------------------------------- #
    # 2. Extract analysis results from annotated ApplyOp
    # -------------------------------------------------------------- #
    halo_widths: tuple[int, ...] = (1,)
    tile_sizes: tuple[int, ...] = (256,)
    temporal_steps: int = 1
    analysis: dict = {}

    for op in analysis_clone.walk():
        if isinstance(op, ApplyOp):
            if "halo_widths" in op.attributes:
                halo_widths = tuple(
                    a.data for a in op.attributes["halo_widths"]
                )
            if "tile_sizes" in op.attributes:
                tile_sizes = tuple(
                    a.data for a in op.attributes["tile_sizes"]
                )
            if "temporal_steps" in op.attributes:
                temporal_steps = op.attributes["temporal_steps"].data

            # Gather roofline analysis
            for key in (
                "flops", "unique_loads", "arithmetic_intensity", "bound",
            ):
                attr = op.attributes.get(key)
                if attr is not None:
                    # IntAttr / StringAttr -> .data
                    # FloatAttr -> .value.data
                    if hasattr(attr, "value") and hasattr(attr.value, "data"):
                        analysis[key] = attr.value.data
                    elif hasattr(attr, "data"):
                        analysis[key] = attr.data
                    else:
                        analysis[key] = str(attr)
            break  # first ApplyOp is enough

    if not temporal_blocking:
        temporal_steps = 1

    # -------------------------------------------------------------- #
    # 3. Optionally autotune
    # -------------------------------------------------------------- #
    if autotune and domain is not None:
        from cutile.runtime.autotune import autotune as _autotune
        result = _autotune(stencil_fn, domain, hw=hw)
        tile_sizes = result.tile_sizes
        temporal_steps = result.temporal_steps

    # -------------------------------------------------------------- #
    # 4. Lower to Python source using the *original* Dialect 1 IR
    # -------------------------------------------------------------- #
    from cutile.lowering.stencil_to_cutile import lower_stencil_to_python

    boundary_spec = None
    if hasattr(stencil_fn, "_boundary") and stencil_fn._boundary is not None:
        boundary_spec = stencil_fn._boundary

    code = lower_stencil_to_python(
        ir.clone(),
        domain=domain,
        tile_sizes=tile_sizes,
        halo_widths=halo_widths,
        temporal_steps=temporal_steps,
        boundary_spec=boundary_spec,
    )

    return CompileResult(
        name=name,
        code=code,
        ndim=stencil_fn._ndim or len(halo_widths),
        halo_widths=halo_widths,
        tile_sizes=tile_sizes,
        temporal_steps=temporal_steps,
        analysis=analysis,
        _ref_fn=stencil_fn._fn,
    )
