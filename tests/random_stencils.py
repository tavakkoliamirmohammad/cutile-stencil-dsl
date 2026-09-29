"""Random stencil expressions for property tests (also used by the fuzz campaign).

``generate(seed, ndim, nfields, radius, nterms)`` draws an expression over up
to four fields from linear terms, products, divisions (inputs are positive so
they are safe), ``max``/``min``/``abs``, ``where`` on a comparison, nested
parentheses and a mix of float32-exact, inexact and integer constants.
``make_stencil`` writes it to a file and imports it, because the frontend
reads the function's source with ``inspect``.
"""

import importlib.util
import os
import random
import re

FIELDS = ["a", "b", "c", "d"]
INDEX = ["i", "j", "k"]
OPS = ["lin", "lin", "lin", "prod", "div", "max", "min", "abs", "where", "nested", "neg"]


def _ix(v, o):
    return v if o == 0 else (f"{v} + {o}" if o > 0 else f"{v} - {-o}")


def generate(seed, ndim, nfields, radius, nterms, ops=OPS):
    """Return ``(fields, index_vars, body)`` for a random expression."""
    rng = random.Random(seed)
    fields, idx = FIELDS[:nfields], INDEX[:ndim]

    def coef():
        k = rng.random()
        if k < 0.3:
            return repr(rng.choice([0.25, 0.5, 2.0, -0.125, 1.0, 4.0]))
        if k < 0.6:
            return repr(round(rng.uniform(-1, 1), 6))
        if k < 0.75:
            return str(rng.choice([2, 3, -1, 7]))
        return repr(rng.choice([1 / 3, 0.1, -1 / 12, 4 / 105, 0.8]))

    def acc():
        f = rng.choice(fields)
        off = [rng.randint(-radius, radius) for _ in idx]
        return f"{f}[{', '.join(_ix(v, o) for v, o in zip(idx, off))}]"

    def term(depth=0):
        kind = rng.choice(ops)
        if kind == "lin":
            return f"{coef()} * {acc()}"
        if kind == "prod":
            return f"{coef()} * {acc()} * {acc()}"
        if kind == "div":
            return f"{coef()} * {acc()} / ({acc()} + 1.5)"
        if kind == "max":
            return f"{coef()} * max({acc()}, {acc()})"
        if kind == "min":
            return f"{coef()} * min({acc()}, {coef()})"
        if kind == "abs":
            return f"{coef()} * abs({acc()} - {acc()})"
        if kind == "where":
            return f"where({acc()} > {acc()}, {acc()}, {coef()} * {acc()})"
        if kind == "neg":
            return f"-{acc()}"
        if kind == "nested" and depth < 2:
            return f"{coef()} * ({term(depth + 1)} - ({term(depth + 1)}))"
        return f"{coef()} * {acc()}"

    parts = [term() for _ in range(nterms)]
    body = parts[0] + "".join(f" {rng.choice(['+', '-'])} {p}" for p in parts[1:])
    return fields, idx, body


def used_fields(body, fields):
    return [f for f in fields if re.search(rf"\b{f}\[", body)]


def make_stencil(directory, name, fields, idx, body, dtype="float64"):
    """Write ``@stencil`` source for *body* into *directory* and import it."""
    dt = f', dtype="{dtype}"' if dtype != "float64" else ""
    src = (
        "from cutile import stencil, where\n\n"
        f"@stencil(ndim={len(idx)}{dt})\n"
        f"def {name}({', '.join(fields)}, {', '.join(idx)}):\n"
        f"    return ({body})\n"
    )
    path = os.path.join(str(directory), f"{name}.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"random_stencil_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name), src
