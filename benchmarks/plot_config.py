"""Plot styling constants for publication figures."""

COLORS = {
    "cuTile-DSL": "#0072B2",
    "CuPy": "#D55E00",
    "JAX": "#009E73",
    "Devito": "#CC79A7",
    "Hand-cuTile": "#E69F00",
    "CUDA-naive": "#56B4E9",
    "CUDA-smem": "#F0E442",
}

MARKERS = {
    "cuTile-DSL": "o",
    "CuPy": "s",
    "JAX": "^",
    "Devito": "D",
    "Hand-cuTile": "*",
    "CUDA-naive": "v",
    "CUDA-smem": "P",
}

STENCIL_LABELS = {
    "heat_1d": "1D Heat",
    "heat_2d": "2D Heat",
    "laplacian_2d_5pt": "2D Lap-5pt",
    "laplacian_2d_9pt": "2D Lap-9pt",
    "laplacian_3d_7pt": "3D Lap-7pt",
}

FIG_SINGLE_COL = (3.33, 2.5)
FIG_DOUBLE_COL = (7.0, 3.0)
FIG_ROOFLINE = (3.33, 3.0)

FONT_SIZES = {
    "title": 10,
    "label": 9,
    "tick": 8,
    "legend": 7,
}
