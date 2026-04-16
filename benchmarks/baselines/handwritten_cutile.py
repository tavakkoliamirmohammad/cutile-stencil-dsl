"""Hand-written cuTile kernels: expert-optimized reference implementations.

These kernels use manually selected tile sizes and explicit cuTile API
calls. They represent the performance ceiling for what the cuTile API
can achieve with expert tuning.
"""

from __future__ import annotations

import importlib
import importlib.util
import tempfile
import textwrap

import cupy as cp

from benchmarks.stencils import STENCIL_META, interior_size


_HEAT_2D_SRC = textwrap.dedent("""\
import cuda.tile as ct
from cuda.tile import ConstInt

TX = ConstInt(32)
TY = ConstInt(32)
HX = ConstInt(1)
HY = ConstInt(1)

@ct.kernel
def heat_2d_kernel(u, output):
    bx = ct.bid(0)
    by = ct.bid(1)
    nx = u.shape[0] - 2 * HX
    ny = u.shape[1] - 2 * HY
    u_w  = u.slice(axis=0, start=HX-1, stop=HX-1+nx).slice(axis=1, start=HY, stop=HY+ny)
    u_e  = u.slice(axis=0, start=HX+1, stop=HX+1+nx).slice(axis=1, start=HY, stop=HY+ny)
    u_s  = u.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY-1, stop=HY-1+ny)
    u_n  = u.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY+1, stop=HY+1+ny)
    tw = ct.load(u_w, index=(bx, by), shape=(TX, TY))
    te = ct.load(u_e, index=(bx, by), shape=(TX, TY))
    ts = ct.load(u_s, index=(bx, by), shape=(TX, TY))
    tn = ct.load(u_n, index=(bx, by), shape=(TX, TY))
    result = 0.25 * (tw + te + ts + tn)
    out_view = output.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY, stop=HY+ny)
    ct.store(out_view, index=(bx, by), tile=result)

def launch_heat_2d(u_in, u_out):
    nx = u_in.shape[0] - 2
    ny = u_in.shape[1] - 2
    grid = ((nx + 31) // 32, (ny + 31) // 32)
    stream = ct.Stream()
    ct.launch(stream, grid, heat_2d_kernel, (u_in, u_out))
    stream.sync()
""")


_LAPLACIAN_2D_5PT_SRC = textwrap.dedent("""\
import cuda.tile as ct
from cuda.tile import ConstInt

TX = ConstInt(32)
TY = ConstInt(32)
HX = ConstInt(1)
HY = ConstInt(1)

@ct.kernel
def lap_2d_kernel(u, output):
    bx = ct.bid(0)
    by = ct.bid(1)
    nx = u.shape[0] - 2 * HX
    ny = u.shape[1] - 2 * HY
    u_c  = u.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY, stop=HY+ny)
    u_w  = u.slice(axis=0, start=HX-1, stop=HX-1+nx).slice(axis=1, start=HY, stop=HY+ny)
    u_e  = u.slice(axis=0, start=HX+1, stop=HX+1+nx).slice(axis=1, start=HY, stop=HY+ny)
    u_s  = u.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY-1, stop=HY-1+ny)
    u_n  = u.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY+1, stop=HY+1+ny)
    tc = ct.load(u_c, index=(bx, by), shape=(TX, TY))
    tw = ct.load(u_w, index=(bx, by), shape=(TX, TY))
    te = ct.load(u_e, index=(bx, by), shape=(TX, TY))
    ts = ct.load(u_s, index=(bx, by), shape=(TX, TY))
    tn = ct.load(u_n, index=(bx, by), shape=(TX, TY))
    result = tw + te + ts + tn - 4.0 * tc
    out_view = output.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY, stop=HY+ny)
    ct.store(out_view, index=(bx, by), tile=result)

def launch_lap_2d(u_in, u_out):
    nx = u_in.shape[0] - 2
    ny = u_in.shape[1] - 2
    grid = ((nx + 31) // 32, (ny + 31) // 32)
    stream = ct.Stream()
    ct.launch(stream, grid, lap_2d_kernel, (u_in, u_out))
    stream.sync()
""")


_LAPLACIAN_3D_7PT_SRC = textwrap.dedent("""\
import cuda.tile as ct
from cuda.tile import ConstInt

TX = ConstInt(8)
TY = ConstInt(8)
TZ = ConstInt(8)
HX = ConstInt(1)
HY = ConstInt(1)
HZ = ConstInt(1)

@ct.kernel
def lap_3d_kernel(u, output):
    bx = ct.bid(0)
    by = ct.bid(1)
    bz = ct.bid(2)
    nx = u.shape[0] - 2 * HX
    ny = u.shape[1] - 2 * HY
    nz = u.shape[2] - 2 * HZ
    u_c   = u.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY, stop=HY+ny).slice(axis=2, start=HZ, stop=HZ+nz)
    u_xm  = u.slice(axis=0, start=HX-1, stop=HX-1+nx).slice(axis=1, start=HY, stop=HY+ny).slice(axis=2, start=HZ, stop=HZ+nz)
    u_xp  = u.slice(axis=0, start=HX+1, stop=HX+1+nx).slice(axis=1, start=HY, stop=HY+ny).slice(axis=2, start=HZ, stop=HZ+nz)
    u_ym  = u.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY-1, stop=HY-1+ny).slice(axis=2, start=HZ, stop=HZ+nz)
    u_yp  = u.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY+1, stop=HY+1+ny).slice(axis=2, start=HZ, stop=HZ+nz)
    u_zm  = u.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY, stop=HY+ny).slice(axis=2, start=HZ-1, stop=HZ-1+nz)
    u_zp  = u.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY, stop=HY+ny).slice(axis=2, start=HZ+1, stop=HZ+1+nz)
    tc  = ct.load(u_c,  index=(bx, by, bz), shape=(TX, TY, TZ))
    txm = ct.load(u_xm, index=(bx, by, bz), shape=(TX, TY, TZ))
    txp = ct.load(u_xp, index=(bx, by, bz), shape=(TX, TY, TZ))
    tym = ct.load(u_ym, index=(bx, by, bz), shape=(TX, TY, TZ))
    typ = ct.load(u_yp, index=(bx, by, bz), shape=(TX, TY, TZ))
    tzm = ct.load(u_zm, index=(bx, by, bz), shape=(TX, TY, TZ))
    tzp = ct.load(u_zp, index=(bx, by, bz), shape=(TX, TY, TZ))
    result = txm + txp + tym + typ + tzm + tzp - 6.0 * tc
    out_view = output.slice(axis=0, start=HX, stop=HX+nx).slice(axis=1, start=HY, stop=HY+ny).slice(axis=2, start=HZ, stop=HZ+nz)
    ct.store(out_view, index=(bx, by, bz), tile=result)

def launch_lap_3d(u_in, u_out):
    nx = u_in.shape[0] - 2
    ny = u_in.shape[1] - 2
    nz = u_in.shape[2] - 2
    grid = ((nx + 7) // 8, (ny + 7) // 8, (nz + 7) // 8)
    stream = ct.Stream()
    ct.launch(stream, grid, lap_3d_kernel, (u_in, u_out))
    stream.sync()
""")


_KERNEL_SRC = {
    "heat_2d": (_HEAT_2D_SRC, "launch_heat_2d"),
    "laplacian_2d_5pt": (_LAPLACIAN_2D_5PT_SRC, "launch_lap_2d"),
    "laplacian_3d_7pt": (_LAPLACIAN_3D_7PT_SRC, "launch_lap_3d"),
}


def _load_kernel(name: str):
    """Load a hand-written cuTile kernel module from source string."""
    src, launch_name = _KERNEL_SRC[name]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        f.flush()
        spec = importlib.util.spec_from_file_location(f"handwritten_{name}", f.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return getattr(mod, launch_name)


def bench_handwritten(
    name: str,
    shape: tuple[int, ...],
    warmup: int = 30,
    iters: int = 100,
) -> dict:
    """Benchmark a hand-written cuTile kernel."""
    if name not in _KERNEL_SRC:
        raise ValueError(f"No hand-written kernel for {name!r}. Available: {list(_KERNEL_SRC)}")

    meta = STENCIL_META[name]
    launch = _load_kernel(name)
    u = cp.random.randn(*shape).astype(cp.float64)
    out = cp.zeros_like(u)

    for _ in range(warmup):
        launch(u, out)
    cp.cuda.Device(0).synchronize()

    e1, e2 = cp.cuda.Event(), cp.cuda.Event()
    e1.record()
    for _ in range(iters):
        launch(u, out)
    e2.record()
    e2.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(e1, e2) / iters
    domain = tuple(s - 2 * h for s, h in zip(shape, meta["halo"]))
    npts = interior_size(domain)
    gps = npts / elapsed_ms * 1e-6
    gbytes = npts * (meta["loads_per_point"] + meta["stores_per_point"]) * meta["dtype_bytes"] / elapsed_ms * 1e-6

    return {
        "name": name,
        "framework": "Hand-cuTile",
        "time_ms": elapsed_ms,
        "gpoints_per_s": gps,
        "gbytes_per_s": gbytes,
    }
