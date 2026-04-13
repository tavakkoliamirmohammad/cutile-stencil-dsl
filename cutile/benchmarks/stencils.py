"""Standard benchmark stencils for comparison."""

from cutile import stencil


@stencil(ndim=1, order=2)
def heat_1d(u, i):
    return 0.25 * u[i-1] + 0.5 * u[i] + 0.25 * u[i+1]


@stencil(ndim=2, order=2)
def laplacian_2d_5pt(u, i, j):
    return u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]


@stencil(ndim=2, order=4)
def laplacian_2d_9pt(u, i, j):
    return (-u[i-2,j] + 16*u[i-1,j] - 30*u[i,j] + 16*u[i+1,j] - u[i+2,j]
            -u[i,j-2] + 16*u[i,j-1] - 30*u[i,j] + 16*u[i,j+1] - u[i,j+2]) / 12.0


@stencil(ndim=3, order=2)
def laplacian_3d_7pt(u, i, j, k):
    return u[i-1,j,k] + u[i+1,j,k] + u[i,j-1,k] + u[i,j+1,k] + u[i,j,k-1] + u[i,j,k+1] - 6*u[i,j,k]


@stencil(ndim=2, order=2)
def heat_2d(u, i, j):
    return 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])


# Standard benchmark domains per dimensionality
BENCHMARK_DOMAINS = {
    1: [(2**16,), (2**18,), (2**20,), (2**22,), (2**24,)],
    2: [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)],
    3: [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)],
}
