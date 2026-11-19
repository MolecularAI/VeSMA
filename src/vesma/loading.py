from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import joblib

from .parallel import parallel_map, resolve_n_jobs, validate_n_jobs

# Bump whenever the exact molecule model changes shape -- a new CMol or AtomMasks
# field, a new BondKind, a different packing. The joblib cache keys on function code
# and arguments, not on the layout of the objects it stores, so without this a model
# change silently unpickles stale graphs that are missing their new fields.
MODEL_SCHEMA = 2

__all__ = [
    "CachedFunction",
    "MODEL_SCHEMA",
    "cache_dir",
    "cached_function",
    "parallel_map",
    "resolve_cache",
    "resolve_n_jobs",
    "validate_n_jobs",
]


class CachedFunction[T](Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> T: ...

    def check_call_in_cache(self, *args: Any, **kwargs: Any) -> bool: ...


def cache_dir() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        base = Path(xdg_cache)
    else:
        base = Path.cwd() / ".cache"
    return base / "vesma" / f"model-{MODEL_SCHEMA}"


def cached_function[T](
    function: Callable[..., T],
    *,
    ignore: Sequence[str] = (),
) -> CachedFunction[T]:
    return cast(
        CachedFunction[T],
        joblib.Memory(cache_dir(), verbose=0).cache(
            function,
            ignore=list(ignore),
        ),
    )


def resolve_cache(cache: bool | None) -> bool:
    if cache is not None:
        return cache

    value = os.environ.get("VESMA_CACHE")
    if value is None:
        return True

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "VESMA_CACHE must be one of 1, true, yes, on, 0, false, no, or off."
    )
