# cuTile Stencil DSL

High-order stencil compilation and tile-based iterative solvers via [cuTile](https://github.com/NVIDIA/cutile).

## Architecture

The framework has two layers:

1. **Pure Python** — DSL, analysis, reference implementations, and tests. Runs on any machine with Python 3.10+ and NumPy.
2. **Generated cuTile code** — Syntactically correct Python files (using `cuda.tile`) ready to run when cuTile is available on an NVIDIA GPU.

```
cutile_stencil/
├── dsl/            # @stencil decorator, types, pipeline, registry, boundary conditions
├── analysis/       # Footprint extraction, tiling, temporal blocking, roofline model
├── codegen/        # cuTile kernel code generation (1D/2D/3D stencils, bricked layouts)
├── solvers/        # CG, persistent CG, mixed-precision CG, matrix-free stencil CG
└── reference/      # NumPy reference: stencil executor, SpMV, CG solver
```

## Setup

```bash
# Clone and create virtual environment
git clone https://github.com/tavakkoliamirmohammad/cutile-stencil-dsl && cd cutile-stencil-dsl
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pytest
```

## Running Tests

```bash
python -m pytest tests/ -v
```

This runs 327 tests across 19 test files:

| Test file | What it tests |
|---|---|
| `test_dsl.py` | `@stencil` decorator, footprint extraction, auto-inference of ndim/order |
| `test_pipeline.py` | `compile()` and `analyze()` pipeline, `CompileResult` / `AnalysisResult` |
| `test_tiling.py` | Tile config selection, temporal blocking, roofline analysis |
| `test_codegen.py` | Generated code passes `ast.parse()`, contains expected cuTile API calls |
| `test_ast_transform.py` | AST inlining, subscript replacement transforms |
| `test_reference.py` | NumPy stencil correctness (heat equation, 2D/3D Laplacian) |
| `test_boundary.py` | Boundary conditions: Dirichlet, Neumann, periodic, reflecting |
| `test_bricks.py` | Bricked memory layout conversion and validation |
| `test_solvers.py` | DIA/BSR SpMV, CG convergence (1D/2D Poisson), mixed-precision CG |
| `test_solver_compile.py` | Solver code generation (CG, persistent CG, mixed-precision) |
| `test_solver_analysis.py` | Solver roofline and tile analysis |
| `test_stencil_cg.py` | Matrix-free stencil CG compilation and validation |
| `test_stencil_bridge.py` | Laplacian stencil spec bridge for solvers |
| `test_blas_kernels.py` | Generated BLAS kernel correctness |
| `test_config.py` | GPU presets, `SolverConfig` |
| `test_applications.py` | Multi-input stencils, coupled fields, staggered grids |
| `test_benchmark.py` | Benchmark suite runner |
| `test_bugfixes.py` | Regression tests for fixed bugs |
| `test_examples.py` | Smoke tests — runs every example end-to-end |

## Running Examples

Each example demonstrates the full pipeline: define stencil → run analysis → generate cuTile kernel → validate with NumPy.

```bash
# 1D heat equation (explicit Euler, Gaussian initial condition)
python examples/heat_1d.py

# 2D acoustic wave equation (4th-order stencil, leapfrog integration)
python examples/wave_2d.py

# 3D 7-point Laplacian stencil
python examples/laplacian_3d.py

# 2D heat equation with bricked memory layout
python examples/heat_2d_bricked.py

# Gray-Scott reaction-diffusion (multi-input stencil, 2 coupled fields)
python examples/gray_scott.py

# 1D advection with upwind scheme (asymmetric stencil)
python examples/advection_upwind.py

# Shallow water equations (3 coupled fields, flux stencils)
python examples/shallow_water.py

# 1D FDTD Maxwell (staggered E-H grids)
python examples/fdtd_maxwell_1d.py

# Matrix-free CG with stencil operator
python examples/stencil_cg.py

# Poisson equation solved with Conjugate Gradient (1D and 2D)
python examples/poisson_cg.py

# Mixed-precision CG: FP64 outer refinement + FP32 inner solve
python examples/mixed_precision_cg.py

# Benchmark suite with roofline analysis
python examples/benchmark_suite.py
```

Generated cuTile kernels are written to `examples/generated/`.

## Quick Start

Define a stencil, compile it, and generate a cuTile kernel — all in a few lines:

```python
from cutile_stencil import stencil, compile

@stencil(dtype="float64")
def heat_1d(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

result = compile(heat_1d, domain=(1024,))
result.print_summary()
result.emit_to_file("heat_1d_kernel.py")
```

The `@stencil` decorator auto-infers `ndim` and `order` from the function signature and array accesses. You can override them explicitly:

```python
@stencil(ndim=2, order=4, dtype="float64")
def wave_2d(u, i, j):
    ...
```

Use GPU presets for hardware-aware analysis:

```python
from cutile_stencil import compile, HardwareSpec

hw = HardwareSpec.from_preset("A100_80GB")
result = compile(heat_1d, domain=(1024,), hw=hw)
result.print_summary()   # Shows tiling, temporal blocking, roofline
```

Validate the generated kernel against the NumPy reference (no GPU needed):

```python
import numpy as np

u0 = np.exp(-((np.linspace(0, 1, 128) - 0.5) ** 2) / 0.01)
result.validate(u0)
```

To run just the analysis without code generation:

```python
from cutile_stencil import analyze

analysis = analyze(heat_1d, domain=(1024,), hw=hw)
print(f"Arithmetic intensity: {analysis.roofline.arithmetic_intensity:.3f}")
print(f"Bound: {analysis.roofline.bound}")
```

## Boundary Conditions

Specify boundary handling with `BoundarySpec`:

```python
from cutile_stencil import stencil, BoundarySpec

@stencil(boundary=BoundarySpec.periodic(2))
def wave_periodic(u, i, j):
    return u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - 4*u[i, j]

@stencil(boundary=BoundarySpec.dirichlet(1, value=0.0))
def heat_dirichlet(u, i):
    return 0.25 * u[i-1] + 0.5 * u[i] + 0.25 * u[i+1]
```

Available types: `dirichlet`, `neumann`, `periodic`, `reflecting`.

## Bricked Memory Layout

For memory-aware kernels with spatial locality:

```python
from cutile_stencil import compile, BrickLayout

layout = BrickLayout(brick_sizes=(32, 32))
result = compile(heat_2d, domain=(128, 128), layout=layout)
```

## Solver Examples

### Standard CG

Solve Poisson's equation with Conjugate Gradient:

```python
import numpy as np
from cutile_stencil.solvers.formats import laplacian_1d_dia
from cutile_stencil.reference.spmv_ref import dia_spmv
from cutile_stencil.reference.cg_ref import cg_solve

N = 100
A = laplacian_1d_dia(N)
b = np.ones(N)

x, iters, residuals = cg_solve(lambda v: dia_spmv(A, v), b, tol=1e-10)
print(f"Converged in {iters} iterations, final residual: {residuals[-1]:.2e}")
```

### Matrix-Free Stencil CG

Use a compiled stencil directly as the CG operator (no sparse matrix needed):

```python
from cutile_stencil import compile_stencil_cg, laplacian_stencil_spec

spec = laplacian_stencil_spec(ndim=2)
result = compile_stencil_cg(spec, domain=(64, 64))
result.print_summary()
result.emit_to_file("stencil_cg_kernel.py")
result.validate()
```

### Compiled Solvers

Generate GPU-ready solver kernels:

```python
from cutile_stencil.solvers.cg import compile_cg
from cutile_stencil.solvers.persistent_cg import compile_persistent_cg
from cutile_stencil.solvers.mixed_precision import compile_mixed_precision_cg

# Standard CG
cg = compile_cg(dtype="float64")
cg.emit_to_file("cg_kernel.py")

# Persistent CG (single-kernel with atomic barriers)
pcg = compile_persistent_cg(dtype="float64")
pcg.emit_to_file("persistent_cg_kernel.py")

# Mixed-precision: FP64 outer refinement + FP32 inner solve
mpcg = compile_mixed_precision_cg(outer_dtype="float64", inner_dtype="float32")
mpcg.emit_to_file("mixed_cg_kernel.py")
```
