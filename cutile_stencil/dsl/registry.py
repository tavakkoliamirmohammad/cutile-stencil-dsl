"""Global stencil registry."""

from __future__ import annotations

from typing import Dict, List, Optional

from cutile_stencil.dsl.types import StencilSpec

_REGISTRY: Dict[str, StencilSpec] = {}


def register(spec: StencilSpec) -> None:
    _REGISTRY[spec.name] = spec


def lookup(name: str) -> Optional[StencilSpec]:
    return _REGISTRY.get(name)


def all_stencils() -> List[StencilSpec]:
    return list(_REGISTRY.values())


def clear() -> None:
    _REGISTRY.clear()
