"""Show generated kernel code alongside analysis summary.

Compiles several example stencils and prints both the analysis
and the generated cuTile kernel code for educational purposes.

Usage:
    python examples/show_generated.py [--show-code]
"""

import argparse
from cutile_stencil import stencil, compile


@stencil(ndim=1, order=2)
def heat_1d(u, i):
    return 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]


@stencil(ndim=2, order=2)
def laplacian_2d(u, i, j):
    return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]


@stencil(ndim=3, order=2)
def laplacian_3d(u, i, j, k):
    return (u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k]
            + u[i,j,k-1] + u[i,j,k+1] - 6 * u[i,j,k])


STENCILS = [
    (heat_1d, (1024,)),
    (laplacian_2d, (256, 256)),
    (laplacian_3d, (64, 64, 64)),
]


def print_roofline_table(results):
    """Print a markdown-formatted roofline comparison table."""
    print("\n## Roofline Analysis Summary\n")
    print("| Stencil | ndim | Order | Accesses | FLOPs/pt | AI | Bound | Peak Gpts/s | Tile |")
    print("|---------|------|-------|----------|----------|----|-------|-------------|------|")
    for name, result in results:
        spec = result.spec
        rf = result.analysis.roofline
        tc = result.analysis.tile_config
        print(
            f"| {name} | {spec.ndim} | {spec.order} | "
            f"{len(spec.accesses)} | {rf.flops_per_point} | "
            f"{rf.arithmetic_intensity:.3f} | {rf.bound} | "
            f"{rf.peak_gpoints_s:.2f} | {tc.tile_sizes} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Show generated stencil kernels")
    parser.add_argument("--show-code", action="store_true",
                        help="Print the generated kernel source code")
    args = parser.parse_args()

    results = []
    for stencil_fn, domain in STENCILS:
        name = stencil_fn._stencil_spec.name
        print("=" * 70)
        print(f"Stencil: {name}")
        print("=" * 70)

        result = compile(stencil_fn, domain)
        result.print_summary()
        results.append((name, result))

        if args.show_code:
            print(f"\n--- Generated kernel ({len(result.code)} chars) ---")
            print(result.code)
            print("--- End kernel ---\n")

    print_roofline_table(results)
    print(f"\nCompiled {len(results)} stencils successfully.")


if __name__ == "__main__":
    main()
