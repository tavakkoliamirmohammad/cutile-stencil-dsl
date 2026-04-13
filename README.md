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
  } attributes {halo_widths=[1,1], tile_sizes=[32,32], bound="memory"}
}
```

**Level 4 -- Dialect 3 (cuTile Target IR):**
```
cutile.kernel @heat(tile=[32,32], halo=[1,1]) {
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

## Setup

```bash
git clone https://github.com/tavakkoliamirmohammad/cutile-stencil-dsl && cd cutile-stencil-dsl
python -m venv venv && source venv/bin/activate

# CPU only (DSL + analysis + codegen)
pip install -e ".[test]"

# With GPU support
pip install -e ".[gpu,test]"
```

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
