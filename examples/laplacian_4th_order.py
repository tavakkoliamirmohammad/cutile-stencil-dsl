"""4th-order Laplacian stencil examples.

Demonstrates wider halo (2 cells) from higher-order finite differences.
The 4th-order 1D Laplacian uses 5 points:
    (-u[i-2] + 16*u[i-1] - 30*u[i] + 16*u[i+1] - u[i+2]) / (12*dx^2)
"""

from cutile_stencil import stencil, compile, HardwareSpec


# ---- 1D 4th-order Laplacian ----

dx = 1.0

@stencil(ndim=1, order=4)
def laplacian_4th_1d(u, i):
    return (-u[i - 2] + 16 * u[i - 1] - 30 * u[i] + 16 * u[i + 1] - u[i + 2]) / (12 * dx ** 2)


# ---- 2D 4th-order Laplacian (compact cross stencil) ----

@stencil(ndim=2, order=4)
def laplacian_4th_2d(u, i, j):
    d2x = -u[i - 2, j] + 16 * u[i - 1, j] - 30 * u[i, j] + 16 * u[i + 1, j] - u[i + 2, j]
    d2y = -u[i, j - 2] + 16 * u[i, j - 1] - 30 * u[i, j] + 16 * u[i, j + 1] - u[i, j + 2]
    return (d2x + d2y) / (12 * dx ** 2)


def main():
    print("=" * 60)
    print("4th-Order Laplacian Stencil Examples")
    print("=" * 60)

    # 1D
    print("\n--- 1D 4th-order Laplacian ---")
    result_1d = compile(laplacian_4th_1d, domain=(1024,))
    result_1d.print_summary()
    print(f"\nGenerated code length: {len(result_1d.code)} chars")
    assert result_1d.spec.halo_widths == (2,), f"Expected halo=2, got {result_1d.spec.halo_widths}"
    print("  Halo width = 2 (correct for 4th-order)")

    # 2D
    print("\n--- 2D 4th-order Laplacian ---")
    result_2d = compile(laplacian_4th_2d, domain=(256, 256))
    result_2d.print_summary()
    print(f"\nGenerated code length: {len(result_2d.code)} chars")
    assert result_2d.spec.halo_widths == (2, 2), f"Expected halo=(2,2), got {result_2d.spec.halo_widths}"
    print("  Halo widths = (2, 2) (correct for 4th-order)")

    print("\n" + "=" * 60)
    print("Done! Both stencils compiled successfully.")


if __name__ == "__main__":
    main()
