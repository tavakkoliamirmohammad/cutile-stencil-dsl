# cuTile Stencil DSL

A Python stencil compiler built on [xDSL](https://github.com/xdslproject/xdsl) that generates optimized GPU kernels via NVIDIA [cuTile](https://github.com/NVIDIA/cutile).

## Architecture

```
                         Three-Dialect Compilation Stack
                         ==============================

  @stencil               cutile_stencil.      stencil.apply {       cutile.kernel {       @ct.kernel
  def heat(u,i,j):         access %u [-1,0]     stencil.access        cutile.slice(...)    def heat_kernel():
    return 0.25*(...)      arith.mulf ...        [-1, 0]               cutile.load(...)       ct.load(...)
                           cutile_stencil.       arith.mulf            cutile.store(...)      ct.store(...)
                             yield %res          stencil.return      cutile.host_program {  def launch_heat():
                                                                       cutile.launch(...)     ct.launch(...)
                                                                    }

  Python source       Dialect 1            Dialect 2              Dialect 3           Python source
  (user writes)    (cutile_stencil)     (xDSL stencil)         (cutile_target)       (generated GPU)
       |                 |                   |                       |                     |
       |  AST parser     | normalize pass    |  analysis passes      | emit_python         |
       +---------------->+----------------->+----+--+--+--+-------->+------------------->--+
                                                 |  |  |  |
                                             footprint |  |
                                               tiling -+  |
                                             temporal ----+
                                             boundary, fusion,
                                             multi-GPU, ...
```

### Module Structure

```
cutile/
|-- frontend/           @stencil decorator, Python AST parser
|-- dialects/           xDSL dialect definitions
|   |-- cutile_stencil/ Dialect 1: mirrors Python syntax
|   |-- (xdsl.stencil)  Dialect 2: standard MLIR stencil (from xDSL, not ours)
|   |-- cutile_target/  Dialect 3: cuTile device + host IR
|   |-- comm/           Communication ops (halo exchange)
|   |-- timestep/       RK time integration
|   +-- layout/         Data layout types
|-- passes/             IR transformation passes
|   |-- analysis/       Footprint, roofline (read-only)
|   |-- tiling.py       Tile size selection
|   |-- temporal.py     Temporal blocking
|   |-- boundary.py     Boundary conditions
|   |-- decompose.py    Multi-GPU domain split
|   +-- halo.py         Halo exchange insertion
|-- lowering/           IR to code
|   |-- normalize.py    Dialect 1 -> Dialect 2 (xDSL stencil)
|   |-- stencil_to_target.py  Dialect 1 -> Dialect 3
|   +-- target_to_python.py   Dialect 3 -> Python source
|-- runtime/            Execution
|   |-- launcher.py     compile() API
|   |-- pipeline.py     Composable PassManager
|   |-- autotune.py     Empirical GPU autotuning
|   +-- communicator.py P2P / NCCL backends
+-- reference/          CPU NumPy reference
```

## Quick Start

```python
from cutile import stencil, compile

@stencil
def heat(u, i, j):
    return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])

result = compile(heat)
result.emit_to_file("heat_kernel.py")
```

The `@stencil` decorator auto-infers `ndim=2` and `order=2` from the function body. The `compile()` function runs the full pass pipeline (analysis, tiling, temporal blocking) and generates a cuTile GPU kernel.

### Multi-GPU (one-line change)

```python
result = compile(heat, num_gpus=2)
```

### Several outputs in one kernel

```python
from cutile import compile_fused

fused = compile_fused([flux_rho, flux_mx, flux_ene])   # inputs matched by parameter name
fused.load_module().launch_flux_rho_flux_mx_flux_ene(*fields, rho_out, mx_out, ene_out)
```

### Compilation Pipeline Example

Here is the IR at every level for a 2D heat stencil:

**Level 1 -- Python source (user writes):**
```python
@stencil
def heat(u, i, j):
    return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
```

**Level 2 -- Dialect 1 (cuTile Stencil Dialect):**
```
cutile_stencil.func @heat(ndim=2, order=2, dtype="float64") {
  %1 = cutile_stencil.access %0 [-1, 0] {"i", "j"} : f64
  %2 = cutile_stencil.access %0 [1, 0] {"i", "j"} : f64
  %3 = cutile_stencil.access %0 [0, -1] {"i", "j"} : f64
  %4 = cutile_stencil.access %0 [0, 1] {"i", "j"} : f64
  %5 = arith.constant 0.25 : f64
  %6 = arith.addf %1, %2 : f64
  %7 = arith.addf %6, %3 : f64
  %8 = arith.addf %7, %4 : f64
  %9 = arith.mulf %5, %8 : f64
  cutile_stencil.yield %9 : f64
}
```

**Level 3 -- Dialect 2 (xDSL Stencil Dialect -- all passes run here):**
```
func.func @heat() -> !stencil.temp<?x?xf64> {
  stencil.apply() {
    %1 = stencil.access %arg [-1, 0] : !stencil.temp<?x?xf64>
    %2 = stencil.access %arg [1, 0]  : !stencil.temp<?x?xf64>
    %3 = stencil.access %arg [0, -1] : !stencil.temp<?x?xf64>
    %4 = stencil.access %arg [0, 1]  : !stencil.temp<?x?xf64>
    %5 = arith.constant 0.25 : f64
    %9 = arith.mulf %5, ... : f64
    stencil.return %9 : f64
  } attributes {halo_widths=[1,1], tile_sizes=[1,1024], bound="memory"}
}
```

**Level 4 -- Dialect 3 (cuTile Target IR):**
```
cutile.kernel @heat(tile=[1,1024], halo=[1,16]) {
  cutile.bid(0), cutile.bid(1)
  cutile.slice(axis=0, start="HX-1", stop="HX-1+nx")
  cutile.slice(axis=1, start="HY",   stop="HY+ny")    -> u_m1_0
  cutile.load(u_m1_0)
  ...
  cutile.store(out, result)
}
cutile.host_program @launch_heat {
  cutile.launch(heat_kernel, grid, args)
}
```

**Level 5 -- Generated cuTile Python:**
```python
@ct.kernel
def heat_kernel(u, output, TX: ConstInt, TY: ConstInt, HX: ConstInt, HY: ConstInt):
    bx, by = ct.bid(0), ct.bid(1)
    u_m1_0 = u.slice(axis=0, start=HX-1, stop=HX-1+nx).slice(axis=1, start=HY, stop=HY+ny)
    t_u_m1_0 = ct.load(u_m1_0, index=(bx, by), shape=(TX, TY))
    ...
    result = 0.25 * (t_u_m1_0 + t_u_p1_0 + t_u_0_m1 + t_u_0_p1)
    ct.store(out, index=(bx, by), tile=result)

def launch_heat(u_in, u_out):
    ct.launch(stream, grid, heat_kernel, (u_in, u_out, TX, TY, HX, HY))
```

## Performance notes

Measured against [bricklib](https://github.com/CtopCsUtahEdu/bricklib) on a
256^3 float64 grid on an RTX PRO 4000 Blackwell (CUDA 13.2 toolkit: tileiras
13.2.51 with its libnvvm, see Setup), one sweep, same session. "Arr" is bricklib's hand-written CUDA array kernel,
"Trans" its vector-scatter code generator on 8^3 bricks; a device-to-device
copy of the 7-point arrays takes 0.495 ms.

| stencil | DSL (tile, hint) | bricklib Arr | bricklib Trans | DSL / Arr | DSL / Trans |
|---|---|---|---|---|---|
| 7-point star | 0.506 ms (1,1,256) | 0.509 | 0.555 | 0.99 | 0.91 |
| 13-point star | 0.795 ms (1,1,256) | 0.720 | 0.890 | 1.10 | 0.89 |
| 19-point star | 1.104 ms (1,2,64) | 1.015 | 1.234 | 1.09 | 0.89 |
| 25-point star | 1.405 ms (1,2,64) | 1.336 | 1.575 | 1.05 | 0.89 |
| 27-point box | 1.431 ms (1,2,64) | 1.418 | 1.628 | 1.01 | 0.88 |
| 125-point box | 6.696 ms (1,2,64), occupancy=4 | 5.939 | 6.888 | 1.13 | 0.97 |
| iso3dfd, 3 grids | 1.685 ms (1,2,64) | 1.683 | 1.726 | 1.00 | 0.98 |
| cond (max/abs) | 0.896 ms (1,1,256) | 0.938 | 0.921 | 0.96 | 0.97 |
| CNS hypterm, 8 in / 5 out, one fused kernel | 17.10 ms (1,2,64), occupancy=4 | 17.24 | 17.21 (5 kernels) | 0.99 | 0.99 |

Run-to-run drift on this power-limited GPU is 3 to 5 percent for both
frameworks (the 125-point hand-written kernel measured 5.94 and 6.22 ms in
two runs of the same binary), so ratios within that band are ties. The CNS
row is the mean of three rounds alternating both frameworks (DSL 16.9 to
17.2 ms, bricklib 16.9 to 17.5 ms).

This is the benchmark set of the Bricks papers (P3HPC'18 Table I, SC'23
Table 2): star Laplacians of order 2 to 8, compact Laplacians of order 2 and
4, the 8th-order isotropic finite-difference wave kernel with three input
grids, the `cond` kernel (`max`/`abs` conditionals) and the compressible
Navier-Stokes `hypterm` kernel (CNS: 8 input fields, 5 outputs, 8th-order
central differences with products of two fields, 466 flops per point). Only
the 4D 9-point kernel is out of scope, since the DSL is limited to three
dimensions. For CNS, "Arr" is bricklib's hand-written kernel computing all
five outputs in one pass and "Trans" the sum of five generated single-output
kernels: bricklib's vector-scatter generator, given five output assignments
in one script, accumulates every term into a single buffer and stores that
to all five outputs (values verified identical across outputs), so it
cannot produce a valid fused kernel. The DSL row is `compile_fused()`: one
kernel with 144 distinct loads and five stores.

The generated kernels beat bricklib's code generator on every stencil except
CNS, where the fused kernel ties both bricklib variants within drift; they
match or beat its hand-written kernel on the 7-point, 27-point, iso and cond
kernels and are within 5-13% of it on the 13/19/25/125-point ones (a
power-limit effect: the DSL kernels execute ~1.6x more instructions per
element for bounds checks and address arithmetic, which pins the 145 W cap
and lowers the clock to 1770 MHz against bricklib's 2200-2430 MHz; under
locked clocks the gap is 2.6%). Nothing in the pipeline is specific to these
shapes: every decision keys on the dimensionality, the number of accesses,
whether the expression is a sum, or the measured resource usage of the
compiled kernel, so it applies to any stencil the DSL can express (linear or
not, one to three dimensions, one or more input fields, one or more
outputs). Checked on shapes bricklib does
not ship: a 2D 81-point box (default 4.29 ms, best of its alternatives), a
49-point in-plane box (2.65 ms, best of its alternatives) and a two-field
nonlinear 125-point stencil (6.74 ms vs 9.30 ms without the hint).

The compiler is layered so that code generation never depends on how the
parameters were chosen: `lower_stencil_to_python()` emits correct code for
any tile shape, halo, kernel hint and term-order flag; the tiling pass, the
spill probe and the autotuner are interchangeable policies that pick them,
and the front end can override each. The decisions, in order of impact:

* **Row tiles sized by elements per thread.** cuTile runs a tile on a
  128-thread block and loads one full tile per stencil access, and every
  loaded tile stays live in registers until consumed. The tiling pass
  therefore picks two elements per thread in one 256-wide row for narrow
  stencils (up to 16 accesses, memory-bound: the long row streams best), and
  one element per thread for wide ones, laid out as two rows of 64 because a
  y-shifted access re-fetches `rows + 2r` rows per tile and two rows halve
  that (27-point equal, 25-point 4% and a 49-point in-plane box 20% faster
  than one row of 128). The launcher clamps each tile dimension to the actual
  interior extent. Row tiles alone are a 3.3x speedup over the previous
  `(4, 8, 8)` default; the per-thread rule avoids a 1.6x loss on 27-point
  and a 5x loss on 125-point stencils.
* **The access count only ranks candidates; the compiled kernel decides.**
  The pass returns its tier's configurations in order of preference
  (`TilingPass.candidates`: wide stencils try one element per thread
  unhinted first, then the hinted ladder; very wide ones the ladder first
  and unhinted last). For wide stencils (17 to 64 accesses) `compile()`
  keeps the first whose compiled kernel neither spills nor exceeds the
  register budget of its occupancy target (128 registers for four
  128-thread blocks per SM, 170 for three). For very wide stencils the best
  configuration is not predictable from registers alone (a 112-load product
  kernel runs 15.0 ms as a single 128-row at occupancy 3, 15.8 ms as two
  rows at occupancy 4 with a 144-byte spill, and 20.3 ms at the first
  spill-free rung), so `compile()` compiles every candidate, times it on a
  slab of 32x64x256 points and keeps the fastest one without a large spill,
  preferring the earlier candidate within 3%. `compile(..., select="probe")`
  or `"measure"` forces either mode.
  The count alone is not enough: the CNS momentum stencils have 56 accesses
  made of two-field products and compile to 210 spill-free registers at one
  element per thread, two blocks per SM and 5.2 ms, where the hinted tile
  takes 64 registers and 3.3 ms; the 27-point box with the same tier fits in
  84 registers and is fastest unhinted. Probing costs one extra compile per
  wide stencil.
* **Inline loads.** Single-use tiles are loaded inside the expression in
  evaluation order rather than hoisted above it. tileiras keeps every loaded
  tile live, so this halves the time of a 125-point kernel at two elements
  per thread (15.0 ms to 9.4 ms) and is neutral otherwise.
* **Occupancy hint for wide stencils, verified against the cubin.**
  `@ct.kernel(occupancy=N)` is the one knob that bounds the compiler's
  register budget. Its effect depends on the tile: at two elements per thread
  it turns the 254-register 125-point kernel into a 62-register, spill-free
  one (9.4 ms to 6.4 ms); at one element per thread as two rows of 64 it
  works as well (56 registers, 6.3 ms) and is what product-heavy kernels need
  (the fused CNS kernel spills at two elements per thread); as a single
  128-wide row it spills badly (28.8 ms), and larger tiles are 10x to 20x
  slower. Whether a given kernel fits a budget is only known after compiling
  it (an 81-point 2D box spills under `occupancy=8` and runs 7x slower), so
  the pass ranks a ladder of candidates, the hinted two-element tile, the
  one-element two-row tile at occupancy 4 and 3, and the one-element single
  row at occupancy 3, and `compile()` compiles each exactly as cuTile will
  at launch and reads its register and stack usage with `cuobjdump`
  (`cutile/runtime/regcheck.py`; the result, including the measured slab
  time when the kernel was timed, is on `CompileResult.resources`).
  `compile(..., occupancy=N)` overrides, and the autotuner tries the hinted
  variant of every two-element tile.
* **Spatial term order for very wide sums.** tileiras issues loads in program
  order, and for the 125-point box that order decides the schedule: terms
  sorted by access offset run in 6.5 ms, the source order grouped by
  coefficient in 7.3 to 7.7 ms. Above 64 accesses the emitter re-associates a
  top-level sum into (x, y, z) order of the accesses each term reads;
  narrower stencils keep source order and stay bit-identical.
* **Rows-fastest block order.** `bid(0)` walks the second-innermost dimension,
  so consecutive thread blocks stream through contiguous rows of one plane
  instead of hopping between planes.
* **Aligned halo.** The innermost halo is padded to a 128-byte multiple
  (16 float64 elements), so every interior row starts on a cache-line boundary.
  `CompileResult.halo_widths` is what arrays must be allocated with;
  `CompileResult.stencil_halo` is the stencil's own footprint. Pass
  `compile(..., align_halo=False)` to opt out.
* **Exact float64 constants.** cuTile evaluates Python float scalars in
  float32, even inside float64 kernels. The emitter folds constant
  sub-expressions in Python and delivers any constant that is not exactly
  representable in float32 (`0.1`, `1/12`, ...) through a small device array
  loaded once per block, so kernels agree with the NumPy reference to ~1e-15
  instead of ~1e-8.

* **Fused multi-output kernels.** `compile_fused([f, g, ...])` lowers several
  stencils over one domain into a single kernel through the same builders
  and emitter. Inputs are matched by parameter name (the frontend records
  `arg_names`), so members may read different subsets of the fields; each
  distinct (field, offset) is loaded once and shared by every output that
  uses it, the members share one table of exact constants, and the tile and
  hint are chosen from the merged access count and register probe like any
  other kernel. The launcher takes the union of the inputs followed by one
  output per stencil (`launch_<name>(*result.inputs, *outs)`), and
  `CompileResult.validate()` checks every output against its member's NumPy
  reference. CNS as five separate kernels reads 264 tiles per point and
  takes 17.4 ms; the fused kernel reads 144 and takes 17.1 ms (the table's
  row; the measured selection picks two rows of 64 at `occupancy=4`, 128
  registers with a 192-byte spill, over the spill-free `occupancy=3` rung).
  Fusion is also a knob the front end can turn: grouping the members by hand
  into two kernels of 96 loads each (`compile_fused([rho, mx, my])` and
  `compile_fused([mz, ene])`) measured 16.4 to 17.8 ms across rounds, so
  it is within drift of the single kernel here.

Fusing several time steps into one kernel was measured and does not pay off
in cuTile: each extra tile load costs about as much as the DRAM traffic it
saves. `temporal_steps > 1` therefore stays a sequence of launches, through
cached scratch buffers so a call does not allocate.

### Who does what: bricklib, this compiler, cuTile, the user

| Optimization | bricklib | This DSL | Done by (DSL side) |
|---|---|---|---|
| Data layout | 8^3 bricks with an adjacency list; brick-contiguous storage | Dense row-major arrays with a padded halo (`layout="bricked"` exists but is not the tuned path) | User allocates with `result.halo_widths` |
| Footprint / halo analysis | Manual ghost zone (`GZ`) in the driver | Per-dimension halo from the access offsets | Our IR (`AnalysisPass`) |
| Vectorized, coalesced 128-bit loads and stores | From brick alignment, in generated CUDA | `LDG.E.128`/`STG.E.128` on every aligned view | cuTile compiler emits them; our lowering makes views aligned (row tiles, 128-byte inner halo) |
| Register reuse of neighbors (load each element once, shuffle) | Vector-scatter code generator | Not expressible: a shifted tile view is a new load. Replaced by load minimization: one load per distinct (field, offset), single-use loads inline, shared loads in fused kernels | Our IR/emitter; the per-thread lowering belongs to cuTile |
| Tile / thread-block shape | Fixed 8^3 brick, 32 threads (4x8 fold), chosen by the user | Row tiles sized by elements per thread, tier by access count, candidate ladder | Our pass (`TilingPass`) + runtime selection; cuTile fixes 128 threads per tile |
| Register pressure / occupancy | nvcc register allocation | `@ct.kernel(occupancy=N)` chosen by compiling each candidate and reading its cubin (very wide: timing it) | Knob is cuTile's; the choice is our runtime (`regcheck`) |
| Instruction scheduling, unrolling, FMA contraction | nvcc | Every term becomes one `DFMA` | cuTile compiler (tileiras + libnvvm + ptxas) |
| Bounds checks | None (fixed grid, full bricks) | Predicated loads/stores on every view (~1.6x instructions per element, the source of the star-stencil gap) | cuTile compiler; not removable from Tile IR on 13.2 |
| Block traversal order | Brick order | `bid(0)` walks rows of a plane (rows-fastest) | Our IR (`grid_order`) |
| Coefficients | C macros / `ConstRef` literals | Constant folding in float64; inexact constants through a device array because cuTile rounds Python scalars to float32 | Our emitter |
| Term order | Codegen DAG (by offset) | Very wide sums re-associated into spatial (x, y, z) order | Our emitter |
| Multi-output fusion | Hand-written kernel only (its generator folds all outputs into one accumulator) | `compile_fused()`: inputs merged by name, one load per distinct access, one constants table | Our IR; which stencils to fuse is the user's call |
| Temporal blocking | None (one sweep per launch) | Measured, does not pay in cuTile; `temporal_steps > 1` is a sequence of launches with cached buffers | Our runtime |
| Shared memory / TMA staging | None (registers + shuffles) | None; cuTile never emits smem or TMA for these kernels | cuTile compiler |
| Autotuning | None (user picks brick and fold dims) | Optional `compile(autotune=True, domain=...)`; fusion grouping; `select`, `occupancy`, `align_halo` overrides | User |
| Validation | Driver compares against a CPU loop | `result.validate()` against the NumPy reference for any number of inputs/outputs | Our runtime |
| Multi-node communication | MPI ghost exchange (papers) | Multi-GPU emitter (`num_gpus`), not benchmarked here | Our IR |

For a new stencil everything in the "Our IR / runtime" rows is automatic;
the user supplies the expression, allocates arrays with the padded halo,
and may choose fusion groups and autotuning.

### Robustness: any expression, not just the paper stencils

Code generation is exercised with random expressions, not only the
benchmark set. A generator (`tests/random_stencils.py`) draws stencils over
one to four fields in one to three dimensions with radius up to 4: linear
terms, products of two fields, divisions, `max`/`min`/`abs`, `where` on a
comparison, nested parentheses, negations, and a mix of float32-exact,
inexact and integer constants, in float64 and float32, with and without
temporal blocking. Two checks run on every case:

* `tests/test_random_expressions.py` reads the emitted kernel expression back
  with NumPy (each `ct.load` becomes the shifted interior of its input, each
  `_c<k>` its constant) and requires it to reproduce the Python source bit
  for bit on 40 seeds, and validates eight seeds on the GPU against the
  reference;
* a longer campaign in the same style (140 cases, 3 to 40 terms each, all
  tile tiers and occupancy rungs exercised) compiles, launches and validates
  every case end to end.

The campaign found one code generation bug, since fixed and pinned by
`tests/test_expr_precedence.py`: the parenthesization check treated any
operand string that starts with `(` as already parenthesized, so
`c * (-0.125 * abs(x) - y)` was emitted as `c * (-0.125) * abs(x) - y`. It
also found that `CompileResult.validate()` only handled single-input
stencils; it now validates any number of inputs and outputs, passing a
stencil's unread parameters as zero arrays.

Eight of the generated 3D expressions (12 to 60 terms, one to four fields,
radius 1 to 4) were also put through bricklib from the same AST: a
vector-scatter codegen script per case and a hand-written CUDA kernel with a
CPU reference, both validated by bricklib's own compare. Same session,
256^3 float64, ms per sweep, DSL defaults with no per-case tuning:

| case | fields | radius | terms | distinct loads | DSL (tile, hint) | bricklib Arr | bricklib Trans | DSL / Arr | DSL / Trans |
|---|---|---|---|---|---|---|---|---|---|
| rnd0 | 1 | 1 | 12 | 15 | 1.831 ms (1,1,256) | 1.841 | 2.040 | 0.99 | 0.90 |
| rnd1 | 2 | 2 | 24 | 38 | 3.350 ms (1,2,64) | 3.438 | 4.839 | 0.97 | 0.69 |
| rnd2 | 3 | 3 | 40 | 62 | 5.717 ms (1,2,64), occupancy=4 | 5.919 | 7.926 | 0.97 | 0.72 |
| rnd3 | 4 | 4 | 60 | 114 | 10.740 ms (1,1,128), occupancy=3 | 10.225 | 16.471 | 1.05 | 0.65 |
| rnd4 | 2 | 4 | 40 | 73 | 9.284 ms (1,1,128), occupancy=3 | 8.760 | 11.077 | 1.06 | 0.84 |
| rnd5 | 4 | 2 | 60 | 76 | 12.581 ms (1,2,64), occupancy=4 | 11.447 | 14.535 | 1.10 | 0.87 |
| rnd6 | 1 | 3 | 60 | 112 | 14.959 ms (1,1,128), occupancy=3 | 13.926 | 15.991 | 1.07 | 0.94 |
| rnd7 | 3 | 1 | 24 | 25 | 4.126 ms (1,2,64), occupancy=4 | 3.826 | 5.007 | 1.08 | 0.82 |

The DSL beats bricklib's generator on all eight (by 6 to 35 percent) and is
within 10 percent of the hand-written kernels, the same picture as on the
paper set. Nothing was tuned per case; the four very wide cases are where
the measured candidate selection earns its compile time (rnd6 was 20.3 ms
under the register-budget rule alone).

### What the generated machine code looks like

Inspected with `cuobjdump -sass` on cubins exported through
`cuda.tile.compilation.export_kernel`, and profiled with Nsight Compute
(RTX PRO 4000 Blackwell, CUDA 13.2 toolkit):

* cuTile lowers a tile load to plain predicated `LDG.E.64` per thread, or
  `LDG.E.128` when the view is provably 16-byte aligned (which the padded
  halo makes true for all but the two innermost-shifted views). There is no
  TMA and no shared memory; a 256-element tile is 128 threads times two
  elements.
* The old `(4, 8, 8)` and the new `(1, 1, 256)` 7-point kernels have the same
  instruction mix (14 loads, 2 stores, 12 fp64 ops, 46 vs 52 registers). The
  3.3x comes from access pattern alone: DRAM throughput 36% of peak with
  345 MB moved for 268 MB of compulsory traffic, versus 82% of peak with no
  overfetch.
* Every stencil term compiles to exactly one `DFMA`, the same as bricklib's
  hand-written kernels, so wide stencils are fp64-FMA-bound on this GPU
  (fp64 runs at 1/64 of fp32; roughly 0.048 ms per stencil point at 256^3).
  bricklib's 27- and 125-point array kernels sit at that roofline with ~38
  registers.
* tileiras hoists every load of an expression above the arithmetic, so a
  125-point kernel holds 125 tiles live: 254 registers at one element per
  thread (one block per SM, 8.6% warp occupancy, 63% of issue slots stalled
  on the FMA dependency chain, fp64 pipe 66% busy against bricklib's 99%).
  Interleaving loads, partial-sum trees, and a runtime loop over stencil
  planes were all measured and either do nothing or get unrolled and hoisted
  again. The effective controls are the `occupancy` hint at two elements per
  thread and the spatial term order, both described above; with them the
  kernel compiles to 62 registers with no spills.

## Setup

```bash
git clone https://github.com/tavakkoliamirmohammad/cutile-stencil-dsl && cd cutile-stencil-dsl
python -m venv venv && source venv/bin/activate

# CPU only (DSL + analysis + codegen)
pip install -e ".[test]"

# With GPU support (cuda-tile >= 1.6; the tileiras compiler comes from the CUDA toolkit)
pip install -e ".[gpu,test]"
```

cuTile compiles kernels with the `tileiras` executable from a CUDA 13.1+
toolkit (on PATH or under `$CUDA_HOME/bin`). `tileiras` is only the front
end: it lowers Tile IR to NVVM IR and then loads `$CUDA_HOME/nvvm/lib64/libnvvm.so`
for the PTX and runs `$CUDA_HOME/bin/ptxas` for the SASS (seen with `strace`),
so the machine code depends on the toolkit `CUDA_HOME` points at, not on the
`tileiras` build. The CUDA 13.3 toolkit's `libnvvm` currently generates
*worse* code for stencils. Measured on identical Tile IR for the 27-point
kernel, mixing the components through a fake `CUDA_HOME`:

| tileiras | libnvvm | ptxas | instructions | branches | 27-point |
|---|---|---|---|---|---|
| 13.2.51 | 13.2 | 13.2 | 344 | 3 | 1.40 ms |
| 13.2.51 | 13.2 | 13.3 | 344 | 3 | |
| 13.2.51 | 13.3 | 13.3 | 1128 | 56 | 1.62 ms |
| 13.3.36 | 13.3 | 13.3 | 896 | 56 | 1.55 ms |
| 13.3.36 | 13.2 | 13.2 | compile error (return code 5) | | |
| 13.2.51 | 13.3 | 13.2 | compile error (return code 5) | | |

`libnvvm` 13.3 wraps every tile load in a branchy bounds check; `ptxas` and
the `tileiras` build make no difference. The PyPI `nvidia-cuda-tileiras`
wheels bundle the matching newer `libnvvm`/`ptxas` and behave like the
toolkit of the same version (13.4 also duplicates loads). Same session, DSL
defaults, ms per sweep:

| compiler stack | 7-point | 13-point | 27-point | iso | 125-point |
|---|---|---|---|---|---|
| CUDA 13.2 toolkit | 0.499 | 0.777 | 1.416 | 1.664 | 6.578 |
| pip 13.3.36 (= CUDA 13.3 toolkit) | 0.500 | 0.783 | 1.574 | 1.791 | 6.464 |
| pip 13.4.92 | 0.502 | 0.777 | 1.940 | 1.814 | 6.462 |

The narrow (memory-bound) and very wide (hinted) tiers are unaffected; the
wide tier loses 10% under 13.3 and 37% under 13.4 on the 27-point box.
bricklib rebuilt with the 13.3 toolkit's `nvcc` is unchanged within 2%, so a
newer toolkit moves the comparison against the DSL on the wide tier only.
The wheel is therefore deliberately *not* a dependency, and CUDA 13.2 is the
toolkit to stay on until this is fixed upstream. A `tileiras` paired with an older
`libnvvm`/`ptxas` than its own fails to compile outright ("Return code 5");
if the wheel is installed anyway, `cutile` puts its bundled libraries first
on `LD_LIBRARY_PATH` for the compiler subprocess so the pairing stays
consistent.

## Tests

```bash
python -m pytest tests/ -v
```

258 tests across 6 test files:

| Test file | Tests | What it covers |
|-----------|-------|----------------|
| `test_dialects.py` | 118 | All 5 xDSL dialects: ops, attrs, printers |
| `test_cutile_new.py` | 82 | Frontend, passes, lowering, compile API, reference |
| `test_all_modes_convergence.py` | 24 | 4 modes x 6 stencils (GPU vs CPU) |
| `test_cutile_gpu_apps.py` | 13 | FDTD, Gray-Scott, shallow water (GPU) |
| `test_lowering.py` | 21 | Code generation unit tests |

## Examples

```bash
python examples/heat_1d.py          # 1D heat equation
python examples/wave_2d.py          # 2D wave (4th-order)
python examples/laplacian_3d.py     # 3D Laplacian
python examples/gray_scott.py       # Reaction-diffusion (2 fields)
python examples/fdtd_maxwell_1d.py  # FDTD Maxwell
python examples/shallow_water.py    # Shallow water (3 fields)
python examples/advection_upwind.py # Upwind advection
python examples/heat_2d_bricked.py  # Bricked memory layout
```

## Benchmarks

```bash
# cuTile only
python run_benchmarks.py

# With autotuning
python run_benchmarks.py --autotune

# Compare against JAX/XLA
python run_benchmarks.py --autotune --jax

# Full sweep (all stencils x all modes x all sizes)
python run_full_benchmarks.py
```

## Dependencies

- **xDSL** (>= 0.62) -- Pure Python MLIR framework
- **NumPy** -- CPU reference
- **cuda-tile** + **CuPy** -- GPU execution (optional)
- **JAX** -- Benchmark comparison (optional)
