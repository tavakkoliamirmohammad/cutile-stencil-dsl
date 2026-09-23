"""Keep the cuTile compiler (``tileiras``) paired with matching CUDA libraries.

``cuda.tile`` compiles kernels by running the ``tileiras`` executable in a
subprocess.  It prefers the one from the ``nvidia-cuda-tileiras`` wheel when
that is installed, otherwise the CUDA toolkit's.  The binary ``dlopen``s
``libnvvm``; when a system CUDA toolkit module is loaded, ``LD_LIBRARY_PATH``
makes the *pip* compiler pick up the toolkit's (older) ``libnvvm`` instead of
the wheel's, and larger kernels then fail with "failed to compile Tile IR
program".

:func:`prefer_pip_cuda_libs` prepends the wheel's ``nvidia/cu13/lib`` directory
to ``LD_LIBRARY_PATH`` *only when the pip compiler is present*, so every
compiler subprocess resolves the matching libraries first.  With the toolkit
compiler nothing is changed.  It alters only the environment inherited by
subprocesses; the running interpreter is unaffected.
"""

from __future__ import annotations

import os
import site
import sys
from typing import MutableMapping


def _site_dirs() -> list[str]:
    dirs: list[str] = []
    try:
        dirs.extend(site.getsitepackages())
    except Exception:  # pragma: no cover - very old/embedded interpreters
        pass
    try:
        dirs.append(site.getusersitepackages())
    except Exception:  # pragma: no cover
        pass
    dirs.extend(p for p in sys.path if p.endswith("site-packages"))
    return dirs


def _pip_cuda_root() -> str | None:
    seen: set[str] = set()
    for base in _site_dirs():
        if base in seen:
            continue
        seen.add(base)
        for cu in ("cu13", "cu12"):
            root = os.path.join(base, "nvidia", cu)
            if os.path.isdir(os.path.join(root, "lib")):
                return root
    return None


def pip_cuda_lib_dir() -> str | None:
    """Directory holding the pip CUDA wheels' shared libraries, if installed."""
    root = _pip_cuda_root()
    return os.path.join(root, "lib") if root else None


def pip_tileiras() -> str | None:
    """Path of the pip-installed ``tileiras`` compiler, if any."""
    root = _pip_cuda_root()
    if root is None:
        return None
    exe = os.path.join(root, "bin", "tileiras")
    return exe if os.path.isfile(exe) else None


def prefer_pip_cuda_libs(env: MutableMapping[str, str] = os.environ) -> bool:
    """Put the pip CUDA library directory first on ``LD_LIBRARY_PATH``.

    Only acts when the pip ``tileiras`` is installed (its libraries must win
    over a loaded toolkit's).  Returns ``True`` when *env* was modified.
    """
    if pip_tileiras() is None:
        return False
    lib = pip_cuda_lib_dir()
    if lib is None:
        return False
    current = env.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if parts and parts[0] == lib:
        return False
    parts = [lib] + [p for p in parts if p != lib]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return True
