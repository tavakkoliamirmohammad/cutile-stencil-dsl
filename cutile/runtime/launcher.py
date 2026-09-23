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

from xdsl.dialects.stencil import AccessOp as StencilAccessOp
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
    # Footprint of the stencil itself; ``halo_widths`` may be wider because
    # the innermost halo is padded so interior rows start 128B-aligned.
    stencil_halo: tuple[int, ...] = field(default=())
    # ``@ct.kernel`` keyword arguments the generated kernel was emitted with.
    kernel_hints: dict = field(default_factory=dict)
    # Register/stack usage measured for the chosen configuration (wide and
    # very wide stencils; ``None`` when the probe did not run).
    resources: Any = field(default=None)

    # Fused kernels: global input names and the member stencils' names, in
    # launcher argument order (``launch_<name>(*inputs, *outputs)``).
    inputs: tuple = ()
    outputs: tuple = ()

    # -- original function(s) for CPU reference --
    _ref_fn: Any = field(default=None, repr=False)
    _ref_fns: Any = field(default=None, repr=False)
    _ref_inputs: Any = field(default=None, repr=False)

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

        if self._ref_fns and (len(self._ref_fns) > 1 or len(self.inputs) > 1):
            return self._validate_multi(atol)
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

    def _validate_multi(self, atol: float) -> bool:
        """Run a multi-input or multi-output kernel once and compare every
        output with its stencil evaluated by the NumPy reference on the same
        inputs.  The launcher takes only the fields a stencil reads (in
        ``self.inputs`` order); the Python reference takes the function's
        full parameter list, so unread parameters get a zero array."""
        import inspect

        import cupy as cp
        import numpy as np

        from cutile.reference.elementwise import array_aware
        from cutile.reference.stencil_ref import _ArrayProxy

        hw = self.halo_widths
        base = 128 if self.ndim == 1 else 64
        shape = tuple(base + 2 * h for h in hw)
        inputs = {name: np.random.rand(*shape) + 0.5 for name in self.inputs}
        mod = self.load_module()
        launcher = getattr(mod, f"launch_{self.name}")
        gpu_in = [cp.asarray(inputs[n]) for n in self.inputs]
        gpu_out = [cp.zeros(shape) for _ in self._ref_fns]
        launcher(*gpu_in, *gpu_out)
        cp.cuda.Device().synchronize()
        interior = tuple(slice(h, s - h) for h, s in zip(hw, shape))
        ok = True
        for fn, out in zip(self._ref_fns, gpu_out):
            params = list(inspect.signature(fn).parameters)[: -self.ndim]
            args = [_ArrayProxy(inputs.get(n, np.zeros(shape)), hw) for n in params]
            ref = array_aware(fn)(*args, *([0] * self.ndim))
            got = cp.asnumpy(out)[interior]
            max_diff = float(np.max(np.abs(ref - got)))
            good = bool(np.allclose(ref, got, atol=atol))
            print(f"Validation[{fn.__name__}]: max_diff={max_diff:.2e}, pass={good}")
            ok = ok and good
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


def _input_names(ir) -> tuple:
    """Array parameter names of the stencil, in launcher order."""
    from cutile.lowering.stencil_to_target import _arg_names, _func_and_block

    func_op, block = _func_and_block(ir)
    return tuple(_arg_names(func_op, block))


def _num_inputs(ir) -> int:
    """Number of input arrays of the stencil (block arguments of its FuncOp)."""
    from cutile.dialects.cutile_stencil.dialect import FuncOp

    for op in ir.body.ops:
        if isinstance(op, FuncOp):
            return len(list(op.body.blocks)[0].args)
    return 1


def pad_inner_halo(
    halo_widths: tuple[int, ...], dtype_bytes: int, align_bytes: int = 128
) -> tuple[int, ...]:
    """Round the innermost halo up so the interior starts on an aligned row.

    With an unpadded halo of 1 element every interior row begins 8 bytes
    into a cache line, which costs a few percent of memory bandwidth on the
    row-shaped tiles the compiler emits.  Padding the innermost halo to a
    multiple of ``align_bytes`` (16 float64 or 32 float32 elements) fixes
    that; the outer halos are left alone because they do not affect
    alignment.
    """
    if not halo_widths:
        return halo_widths
    align = max(1, align_bytes // dtype_bytes)
    inner = -(-halo_widths[-1] // align) * align  # ceil to multiple
    return tuple(halo_widths[:-1]) + (inner,)


def _budgeted_configs(tiling, candidates):
    """Wrap the tiling pass's ``(tile, hints)`` candidates as
    ``regcheck.Config`` with the register budget of each one's occupancy
    hint (unhinted candidates get the pass's occupancy target)."""
    from cutile.runtime import regcheck

    return [
        regcheck.Config(
            tuple(tile), dict(hints),
            max_registers=regcheck.register_budget(hints.get("occupancy", tiling.very_wide_occupancy)),
        )
        for tile, hints in candidates
    ]


_SELECT_MODES = ("probe", "measure")


def _selection_mode(select, tiling, num_loads) -> str:
    """``"probe"`` keeps the first candidate whose cubin fits its register
    budget; ``"measure"`` compiles every candidate and times it on a slab.
    By default very wide stencils are measured (their best configuration is
    not predictable from registers alone: a 112-load product kernel runs
    15.0 ms as a single 128-row at occupancy 3 and 20.3 ms at the first
    spill-free rung) and everything else is probed."""
    if select is not None:
        if select not in _SELECT_MODES:
            raise ValueError(f"select must be one of {_SELECT_MODES}, got {select!r}")
        return select
    return "measure" if tiling.tier(num_loads) == "very_wide" else "probe"


def _choose_config(
    candidates, lower, *, kernel_name, ndim, num_inputs, halo_widths, dtype,
    max_registers=None, num_outputs=1, select="probe",
):
    """Probe (and in ``"measure"`` mode time) *candidates* (``regcheck.Config``)
    and return ``(tile_sizes, kernel_hints, resources)`` of the chosen one.

    *lower* turns a configuration into generated source; the probe compiles
    it the way cuTile will, reads register and stack usage from the cubin and,
    when measuring, times it on :func:`regcheck.timing_shape`.
    """
    from cutile.runtime import regcheck

    probed: dict = {}
    time_shape = regcheck.timing_shape(ndim) if select == "measure" else None

    def _probe(cfg):
        code = lower(cfg)
        res = regcheck.probe_resources(
            code=code, kernel_name=kernel_name, ndim=ndim, num_inputs=num_inputs,
            num_outputs=num_outputs, has_consts=("consts," in code),
            tile_sizes=cfg.tile_sizes, halo_widths=halo_widths, dtype=dtype,
            time_shape=time_shape,
        )
        probed[id(cfg)] = res
        return res

    if select == "measure":
        chosen = regcheck.select_fastest(candidates, _probe, max_registers=max_registers)
    else:
        chosen = regcheck.select_config(candidates, _probe, max_registers=max_registers)
    return chosen.tile_sizes, dict(chosen.kernel_hints), probed.get(id(chosen))


def compile(
    stencil_fn,
    domain: tuple[int, ...] | None = None,
    hw=None,
    pipeline: Pipeline | None = None,
    temporal_blocking: bool = True,
    autotune: bool = True,
    num_gpus: int = 1,
    split_axis: int = 0,
    overlap: bool = True,
    layout: str | None = None,
    brick_size: int = 32,
    align_halo: bool = True,
    occupancy: int | None = None,
    select: str | None = None,
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
    num_gpus
        Number of GPUs for multi-GPU code generation (default ``1``).
        When ``> 1``, generates domain decomposition + halo exchange.
    split_axis
        Axis along which to split the domain for multi-GPU (default ``0``).
    overlap
        Whether to overlap compute and communication in multi-GPU mode
        (default ``True``).
    layout
        Memory layout for the kernel.  ``None`` or ``"flat"`` for standard
        row-major layout; ``"bricked"`` for bricked memory layout with
        flat-to-brick conversion.
    brick_size
        Brick side length when ``layout="bricked"`` (default ``32``).
    align_halo
        Pad the innermost halo to a 128-byte multiple so interior rows are
        aligned (default ``True``; single-GPU flat layout only).  The
        unpadded footprint stays available as ``CompileResult.stencil_halo``.
    occupancy
        cuTile ``@ct.kernel(occupancy=...)`` hint: expected resident blocks per
        SM, which bounds the compiler's register budget.  ``None`` lets the
        tiling pass decide (it sets a hint for wide stencils).
    select
        How to choose among the tiling pass's candidate configurations:
        ``"probe"`` keeps the first whose compiled kernel fits its register
        budget, ``"measure"`` times every candidate on a slab and keeps the
        fastest.  ``None`` (default) measures very wide stencils and probes
        the rest.

    Returns
    -------
    CompileResult
    """
    ir = stencil_fn._ir
    name = stencil_fn._fn.__name__

    dtype_bytes = 4 if getattr(stencil_fn, "_dtype", "float64") in (
        "float32", "fp32"
    ) else 8
    if hw is not None:
        dtype_bytes = hw.dtype_bytes

    # -------------------------------------------------------------- #
    # 1. Run the analysis pipeline on a clone
    # -------------------------------------------------------------- #
    analysis_clone = ir.clone()

    if pipeline is None:
        shared_mem = 49152 if hw is None else hw.shared_mem_bytes
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
    kernel_hints: dict = {}
    pass_occupancy: int | None = None
    num_loads: int = 0
    num_inputs = _num_inputs(ir)

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
            if "occupancy" in op.attributes:
                kernel_hints["occupancy"] = op.attributes["occupancy"].data
                pass_occupancy = op.attributes["occupancy"].data
            from cutile.passes.tiling import unique_access_count
            num_loads = unique_access_count(op)

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

    # For multi-GPU, each GPU runs a single step per iteration;
    # temporal looping is handled by the multi-GPU launcher itself.
    if num_gpus > 1:
        temporal_steps = 1

    # -------------------------------------------------------------- #
    # 2b. Pad the innermost halo so interior rows start 128B-aligned
    # -------------------------------------------------------------- #
    stencil_halo = halo_widths
    if align_halo and num_gpus == 1 and layout != "bricked":
        halo_widths = pad_inner_halo(halo_widths, dtype_bytes)

    # -------------------------------------------------------------- #
    # 3. Optionally autotune
    # -------------------------------------------------------------- #
    if autotune and domain is not None:
        from cutile.runtime.autotune import autotune as _autotune
        result = _autotune(stencil_fn, domain, hw=hw)
        tile_sizes = result.tile_sizes
        temporal_steps = result.temporal_steps
        kernel_hints = dict(getattr(result, "kernel_hints", {}) or {})

    # An explicit occupancy request wins over both the pass and the autotuner.
    if occupancy is not None:
        kernel_hints["occupancy"] = occupancy

    # -------------------------------------------------------------- #
    # 3b. Wide and very wide stencils: the pass ranks (tile, hints)
    #     configurations; keep the first whose compiled kernel neither
    #     spills nor exceeds the register budget of the occupancy target.
    #     Decided from the cubin, not from the stencil's shape.
    # -------------------------------------------------------------- #
    resources = None
    # Policy decisions live here; the lowering only consumes them.
    from cutile.passes.tiling import TilingPass as _Tiling
    tiling = next((p for p in pipeline.passes if isinstance(p, _Tiling)), _Tiling())
    spatial_term_order = num_loads > tiling.very_wide_stencil_loads
    candidates = tiling.candidates(len(halo_widths), num_loads)
    probe_wanted = (
        len(candidates) > 1 and occupancy is None
        and not (autotune and domain is not None)
        and num_gpus == 1 and layout != "bricked"
    )
    if probe_wanted:
        from cutile.lowering.stencil_to_cutile import lower_stencil_to_python
        from cutile.runtime import regcheck

        def _lower(cfg):
            return lower_stencil_to_python(
                ir.clone(), tile_sizes=cfg.tile_sizes, halo_widths=halo_widths,
                kernel_hints=cfg.kernel_hints, spatial_term_order=spatial_term_order,
            )

        tile_sizes, kernel_hints, resources = _choose_config(
            _budgeted_configs(tiling, candidates), _lower,
            kernel_name=f"{name}_kernel", ndim=len(halo_widths), num_inputs=num_inputs,
            halo_widths=halo_widths, dtype=getattr(stencil_fn, "_dtype", "float64"),
            select=_selection_mode(select, tiling, num_loads),
        )

    # -------------------------------------------------------------- #
    # 4. Lower to Python source using the *original* Dialect 1 IR
    # -------------------------------------------------------------- #
    boundary_spec = None
    if hasattr(stencil_fn, "_boundary") and stencil_fn._boundary is not None:
        boundary_spec = stencil_fn._boundary

    if num_gpus > 1:
        # Multi-GPU lowering
        from cutile.lowering.multigpu_emitter import lower_stencil_to_multigpu_python

        code = lower_stencil_to_multigpu_python(
            ir.clone(),
            num_gpus=num_gpus,
            split_axis=split_axis,
            tile_sizes=tile_sizes,
            halo_widths=halo_widths,
            temporal_steps=temporal_steps,
            overlap=overlap,
        )
    elif layout == "bricked":
        # Bricked layout lowering
        from cutile.lowering.multigpu_emitter import lower_stencil_to_bricked_python

        code = lower_stencil_to_bricked_python(
            ir.clone(),
            tile_sizes=tile_sizes,
            halo_widths=halo_widths,
            temporal_steps=temporal_steps,
            brick_size=brick_size,
            boundary_spec=boundary_spec,
        )
    else:
        # Standard cuTile lowering
        from cutile.lowering.stencil_to_cutile import lower_stencil_to_python

        code = lower_stencil_to_python(
            ir.clone(),
            domain=domain,
            tile_sizes=tile_sizes,
            halo_widths=halo_widths,
            temporal_steps=temporal_steps,
            boundary_spec=boundary_spec,
            kernel_hints=kernel_hints,
            spatial_term_order=spatial_term_order,
        )

    return CompileResult(
        name=name,
        code=code,
        ndim=stencil_fn._ndim or len(halo_widths),
        halo_widths=halo_widths,
        tile_sizes=tile_sizes,
        temporal_steps=temporal_steps,
        analysis=analysis,
        stencil_halo=stencil_halo,
        kernel_hints=kernel_hints,
        resources=resources,
        inputs=_input_names(ir),
        _ref_fn=stencil_fn._fn,
        _ref_fns=[stencil_fn._fn],
    )


# ---------------------------------------------------------------------- #
# compile_fused()
# ---------------------------------------------------------------------- #


def compile_fused(
    stencil_fns: list,
    domain: tuple[int, ...] | None = None,
    hw=None,
    tile_sizes: tuple[int, ...] | None = None,
    halo_widths: tuple[int, ...] | None = None,
    temporal_steps: int = 1,
    occupancy: int | None = None,
    align_halo: bool = True,
    select: str | None = None,
) -> CompileResult:
    """Compile several stencils over one domain into a single fused kernel.

    The kernel loads each distinct (field, offset) once and computes every
    output; inputs are matched by parameter name, so the launcher takes the
    union of the stencils' arrays followed by one output per stencil
    (``launch_<name>(*result.inputs, *outs)``).  Tile shape and kernel hints
    are chosen exactly as for a single stencil, from the merged access count
    and the compiled kernel's register usage.

    Parameters
    ----------
    stencil_fns : list
        ``@stencil``-decorated functions with distinct names and equal ndim.
    domain : tuple[int, ...] | None
        Accepted for API compatibility; unused.
    hw
        Optional :class:`~cutile.config.HardwareSpec`.
    tile_sizes, halo_widths : tuple[int, ...] | None
        Override the tile shape / footprint (the footprint defaults to the
        widest member's, innermost padded for alignment).
    temporal_steps : int
        Must be 1; temporal blocking is not available for fused kernels.
    occupancy : int | None
        Explicit ``@ct.kernel(occupancy=...)`` hint; disables probing.
    align_halo : bool
        Pad the innermost halo to a 128-byte multiple (default ``True``).
    select : str | None
        ``"probe"`` / ``"measure"`` / ``None`` as for :func:`compile`.
    """
    if not stencil_fns:
        raise ValueError("At least one stencil function is required")
    if temporal_steps != 1:
        raise NotImplementedError(
            "temporal blocking of fused kernels is not supported (temporal_steps must be 1)"
        )

    from cutile.lowering.fusion_emitter import lower_fused_stencils_to_python
    from cutile.lowering.stencil_to_target import extract_fused_meta
    from cutile.passes.tiling import TilingPass
    from cutile.runtime import regcheck

    modules = [fn._ir for fn in stencil_fns]
    meta = extract_fused_meta(modules)
    ndim = meta.ndim
    dtype = getattr(stencil_fns[0], "_dtype", "float64")
    dtype_bytes = 4 if dtype in ("float32", "fp32") else 8
    shared_mem = 49152
    if hw is not None:
        shared_mem, dtype_bytes = hw.shared_mem_bytes, hw.dtype_bytes

    # Footprint: the widest halo any member needs, per dimension.
    if halo_widths is None:
        per_stencil = []
        for module in modules:
            clone = module.clone()
            Pipeline.single_gpu(
                shared_mem_bytes=shared_mem, dtype_bytes=dtype_bytes, max_temporal_steps=1
            ).run(clone)
            for op in clone.walk():
                if isinstance(op, ApplyOp) and "halo_widths" in op.attributes:
                    per_stencil.append(tuple(a.data for a in op.attributes["halo_widths"]))
                    break
        halo_widths = tuple(max(hs) for hs in zip(*per_stencil)) if per_stencil else (1,) * ndim
    stencil_halo = tuple(halo_widths)
    if align_halo:
        halo_widths = pad_inner_halo(tuple(halo_widths), dtype_bytes)

    num_loads = len(meta.accesses)
    tiling = TilingPass()
    spatial_term_order = num_loads > tiling.very_wide_stencil_loads
    candidates = [(tuple(tile_sizes), {})] if tile_sizes is not None else tiling.candidates(ndim, num_loads)
    if occupancy is not None:
        candidates = [(candidates[0][0], {"occupancy": occupancy})]

    def _lower(cfg):
        return lower_fused_stencils_to_python(
            modules, tile_sizes=cfg.tile_sizes, halo_widths=halo_widths,
            kernel_hints=cfg.kernel_hints, spatial_term_order=spatial_term_order,
        )

    resources = None
    if len(candidates) > 1:
        tile_sizes, kernel_hints, resources = _choose_config(
            _budgeted_configs(tiling, candidates), _lower,
            kernel_name=f"{meta.name}_kernel", ndim=ndim, num_inputs=len(meta.input_names),
            num_outputs=len(meta.output_names), halo_widths=halo_widths, dtype=dtype,
            select=_selection_mode(select, tiling, num_loads),
        )
    else:
        tile_sizes, kernel_hints = tuple(candidates[0][0]), dict(candidates[0][1])

    code = _lower(regcheck.Config(tuple(tile_sizes), dict(kernel_hints)))
    return CompileResult(
        name=meta.name,
        code=code,
        ndim=ndim,
        halo_widths=tuple(halo_widths),
        tile_sizes=tuple(tile_sizes),
        temporal_steps=1,
        analysis={"unique_loads": num_loads},
        stencil_halo=stencil_halo,
        kernel_hints=kernel_hints,
        resources=resources,
        inputs=tuple(meta.input_names),
        outputs=tuple(meta.output_names),
        _ref_fns=[fn._fn for fn in stencil_fns],
        _ref_inputs=meta.stencil_inputs,
    )
