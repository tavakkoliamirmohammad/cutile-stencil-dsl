"""Spill-aware selection between kernel configurations.

cuTile's ``occupancy`` hint bounds the register budget tileiras compiles to.
Whether a kernel fits that budget without spilling to local memory is only
known after compiling it: the 125-point 3D box fits ``occupancy=4`` in 62
registers, the 81-point 2D box spills under ``occupancy=8``.  Rather than
guess per stencil, :func:`probe_resources` compiles a generated kernel the
way cuTile itself does at launch (one launch on small representative
arrays, capturing the cubin it produces) and reads register and stack usage
with the toolkit's ``cuobjdump``; :func:`select_config` keeps the first
candidate configuration that does not spill and, when a register budget is
given, does not exceed it.  The budget comes from :func:`register_budget`:
the registers per thread that still let the occupancy target's blocks be
resident on one SM (128 for four 128-thread blocks and a 64K register
file).  A kernel that compiles to 210 registers without spilling runs two
blocks per SM and is latency bound; the same stencil under the occupancy
hint compiles to 64 registers and runs 1.5x faster.

Exporting the kernel through :func:`cuda.tile.compilation.export_kernel`
instead would be cheaper, but its generic array constraints do not match the
alignment and stride facts cuTile infers from real arrays, and the register
counts differ enough to give the wrong answer (128 registers and spills vs
62 and none for the same 125-point kernel).

The probe is best effort: without the toolkit, a GPU, or on any error it
returns ``None`` and the caller keeps its first (heuristic) choice.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class Resources:
    """Per-thread resource usage of a compiled kernel, and optionally its
    measured time per launch on the probe's timing slab (``ms``)."""

    registers: int
    stack_bytes: int
    ms: Optional[float] = None

    @property
    def spills(self) -> bool:
        return self.stack_bytes > 0


@dataclass(frozen=True)
class Config:
    """A candidate kernel configuration.

    ``max_registers`` is the register budget this configuration must meet
    (normally :func:`register_budget` of its occupancy hint); ``None`` means
    the budget :func:`select_config` was given.
    """

    tile_sizes: tuple[int, ...]
    kernel_hints: dict = field(default_factory=dict)
    max_registers: Optional[int] = None


# ---------------------------------------------------------------------- #
# Toolkit helpers
# ---------------------------------------------------------------------- #


def parse_res_usage(text: str) -> Optional[Resources]:
    """Parse ``cuobjdump -res-usage`` output."""
    m = re.search(r"REG:(\d+)\s+STACK:(\d+)", text)
    if m is None:
        return None
    return Resources(registers=int(m.group(1)), stack_bytes=int(m.group(2)))


def cuobjdump_path() -> Optional[str]:
    """Locate ``cuobjdump``: on PATH, next to tileiras, or under CUDA_HOME."""
    found = shutil.which("cuobjdump")
    if found:
        return found
    try:
        from cuda.tile._compile import _find_compiler_bin

        candidate = os.path.join(os.path.dirname(_find_compiler_bin().path), "cuobjdump")
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    for var in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(var)
        if home and os.path.isfile(os.path.join(home, "bin", "cuobjdump")):
            return os.path.join(home, "bin", "cuobjdump")
    return None


# ---------------------------------------------------------------------- #
# Probe
# ---------------------------------------------------------------------- #


def _capture_cubin(run: Callable[[], None]) -> Optional[bytes]:
    """Run *run* while intercepting the cubin cuTile compiles for a launch."""
    from cuda.tile import _execution

    captured: list[bytes] = []
    original = _execution.kernel._compile

    def spy(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        cubin = result[0]
        if isinstance(cubin, (bytes, bytearray, memoryview)):
            captured.append(bytes(cubin))
        return result

    _execution.kernel._compile = spy
    try:
        run()
    finally:
        _execution.kernel._compile = original
    return captured[-1] if captured else None


def timing_shape(ndim: int) -> tuple[int, ...]:
    """Interior extent of the slab kernels are timed on: large enough to
    fill the GPU for several waves, small enough to allocate and run in
    milliseconds."""
    return {1: (1 << 20,), 2: (256, 1024), 3: (32, 64, 256)}[ndim]


def probe_resources(
    *,
    code: str,
    kernel_name: str,
    ndim: int,
    num_inputs: int,
    num_outputs: int = 1,
    has_consts: bool = False,
    tile_sizes: Sequence[int],
    halo_widths: Sequence[int],
    dtype: str = "float64",
    time_shape: Optional[Sequence[int]] = None,
    iterations: int = 5,
) -> Optional[Resources]:
    """Compile the generated kernel in *code* as cuTile would and report its
    register and stack usage, and with *time_shape* also its time per launch.

    The generated launcher is run once on small arrays (two tiles per
    dimension plus halo), so the compiler sees the same alignment and stride
    facts as in real use.  With *time_shape* (an interior extent, see
    :func:`timing_shape`) the arrays have that size instead and the launcher
    is timed over *iterations* launches after two warm-ups.  Returns ``None``
    when the probe cannot run (no toolkit, no GPU, compile error).  Never
    raises.
    """
    dump = cuobjdump_path()
    if dump is None:
        return None
    path = cubin_path = None
    try:
        import cupy as cp

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        spec = importlib.util.spec_from_file_location(f"_probe_{kernel_name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        launcher = getattr(mod, "launch_" + kernel_name.removesuffix("_kernel"))

        cp_dtype = cp.float32 if dtype in ("float32", "fp32") else cp.float64
        if time_shape is not None:
            shape = tuple(int(n) + 2 * h for n, h in zip(time_shape, halo_widths))
        else:
            shape = tuple(2 * h + 2 * t for h, t in zip(halo_widths, tile_sizes))
        arrays = [(cp.random.rand(*shape) + 0.5).astype(cp_dtype) for _ in range(num_inputs)]
        outs = [cp.zeros(shape, dtype=cp_dtype) for _ in range(num_outputs)]

        def run():
            launcher(*arrays, *outs)
            cp.cuda.Device().synchronize()

        cubin = _capture_cubin(run)
        if cubin is None:
            return None
        ms = None
        if time_shape is not None:
            for _ in range(2):
                launcher(*arrays, *outs)
            cp.cuda.Device().synchronize()
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            for _ in range(iterations):
                launcher(*arrays, *outs)
            end.record()
            end.synchronize()
            ms = cp.cuda.get_elapsed_time(start, end) / iterations
        with tempfile.NamedTemporaryFile("wb", suffix=".cubin", delete=False) as f:
            f.write(cubin)
            cubin_path = f.name
        res = subprocess.run([dump, "-res-usage", cubin_path], capture_output=True, text=True, timeout=120)
        parsed = parse_res_usage(res.stdout)
        if parsed is None:
            return None
        return Resources(parsed.registers, parsed.stack_bytes, ms)
    except Exception:
        return None
    finally:
        for p in (path, cubin_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------- #
# Selection
# ---------------------------------------------------------------------- #


def register_budget(
    occupancy: int, threads_per_block: int = 128, registers_per_sm: int = 65536
) -> int:
    """Registers per thread that keep *occupancy* blocks resident on one SM."""
    return registers_per_sm // (occupancy * threads_per_block)


def select_config(
    configs: Sequence[Config],
    probe: Callable[[Config], Optional[Resources]],
    max_registers: Optional[int] = None,
) -> Config:
    """Return the first configuration whose kernel fits.

    A configuration fits when it does not spill and uses at most its
    register budget (``Config.max_registers``, else *max_registers*, else
    unlimited).  If the probe is unavailable (returns ``None``) the
    configuration under test is kept; if none fits, the first that does not
    spill is returned, and failing that the last (the most conservative) one.
    """
    probed: list[tuple[Config, Resources]] = []
    for cfg in configs:
        res = probe(cfg)
        if res is None:
            return cfg
        budget = cfg.max_registers if cfg.max_registers is not None else max_registers
        if not res.spills and (budget is None or res.registers <= budget):
            return cfg
        probed.append((cfg, res))
    for cfg, res in probed:
        if not res.spills:
            return cfg
    return configs[-1]


def _fits(cfg: Config, res: Resources, max_registers: Optional[int]) -> bool:
    budget = cfg.max_registers if cfg.max_registers is not None else max_registers
    return not res.spills and (budget is None or res.registers <= budget)


def select_fastest(
    configs: Sequence[Config],
    probe: Callable[[Config], Optional[Resources]],
    max_spill_bytes: int = 512,
    margin: float = 0.03,
    max_registers: Optional[int] = None,
) -> Config:
    """Probe every configuration and return the fastest measured one.

    Configurations whose kernel spills more than *max_spill_bytes* per thread
    are not trusted even when their slab time looks good; among the rest the
    earliest candidate within *margin* of the best time wins, so the pass's
    ranking breaks near-ties.  Without timings (probe returns resources
    without ``ms``) this degrades to :func:`select_config`'s budget rule; if
    the probe is unavailable the first configuration is kept.
    """
    probed: list[tuple[Config, Resources]] = []
    for k, cfg in enumerate(configs):
        res = probe(cfg)
        if res is None:
            if k == 0:
                return cfg
            continue
        probed.append((cfg, res))
    if not probed:
        return configs[0]
    timed = [(cfg, res) for cfg, res in probed if res.ms is not None]
    if timed:
        trusted = [(cfg, res) for cfg, res in timed if res.stack_bytes <= max_spill_bytes] or timed
        best = min(res.ms for _, res in trusted)
        for cfg, res in trusted:
            if res.ms <= best * (1.0 + margin):
                return cfg
    for cfg, res in probed:
        if _fits(cfg, res, max_registers):
            return cfg
    for cfg, res in probed:
        if not res.spills:
            return cfg
    return configs[-1]
