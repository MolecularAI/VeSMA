from __future__ import annotations

import platform
from collections.abc import Iterable, Sequence
from functools import partial
from pathlib import Path

import pandas as pd

from vesma.keyer import Keyer
from vesma.loading import cache_dir, cached_function, parallel_map, resolve_cache

from .core import K, M

__all__ = [
    "get_cache_dir",
    "from_smiles",
    "keys_from_smiles",
    "load_stock",
    "load_stock_parallel",
    "parse_stock_smiles",
    "read_smiles_column",
]

STOCK_CACHE_VERSION = 1


def get_cache_dir() -> Path:
    """Return the shared Vesma cache directory."""
    return cache_dir()


def read_smiles_column(filepath: Path | str) -> tuple[str, ...]:
    """Read the first whitespace-delimited column from a .smi or .smi.gz file."""
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        usecols=[0],
        compression="infer",
        dtype=str,
    )
    return tuple(df[0].dropna().tolist())


def parse_stock_smiles(
    smiles: str,
    keyer: Keyer[M, K],
) -> K | None:
    """Parse one stock row and return its key."""
    try:
        return keyer.from_smiles(smiles)
    except ValueError:
        return None


def cached_keys_from_smiles(
    smiles: tuple[str, ...],
    keyer_name: str,
    keyer_version: str,
    python_version: str,
    cache_version: int,
    keyer: Keyer[M, K],
    n_jobs: int | None,
) -> set[K]:
    """Encode stock from normalized SMILES rows, cached by exact tuple content."""
    if cache_version != STOCK_CACHE_VERSION:
        raise ValueError("Unsupported stock cache version.")
    return keys_from_smiles(smiles, keyer, n_jobs=n_jobs)


def keys_from_smiles(
    smiles: Sequence[str],
    keyer: Keyer[M, K],
    *,
    n_jobs: int | None = None,
) -> set[K]:
    """Encode an iterable of SMILES strings to a set of keys using keyer.from_smiles."""
    results = parallel_map(
        partial(parse_stock_smiles, keyer=keyer),
        smiles,
        n_jobs=n_jobs,
    )
    return {token for token in results if token is not None}


def from_smiles(
    smiles: Iterable[str],
    keyer: Keyer[M, K],
    *,
    n_jobs: int | None = None,
    cache: bool | None = None,
) -> set[K]:
    """Encode stock from SMILES rows, cached by exact tuple content."""
    rows = tuple(smiles)
    jobs = n_jobs
    if not resolve_cache(cache):
        return keys_from_smiles(rows, keyer, n_jobs=jobs)

    cached = cached_function(cached_keys_from_smiles, ignore=("keyer", "n_jobs"))
    return cached(
        rows,
        keyer.name,
        keyer.version(),
        platform.python_version(),
        STOCK_CACHE_VERSION,
        keyer,
        jobs,
    )


def load_stock(
    filepath: str | Path,
    keyer: Keyer[M, K],
    *,
    n_jobs: int | None = None,
    cache: bool | None = None,
) -> set[K]:
    """
    Load a set of molecule keys from a .smi or .smi.gz file.

    Results are cached by exact first-column SMILES rows, keyer name/version,
    cache version, and Python version.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Stock file not found: {filepath}")

    return from_smiles(
        read_smiles_column(filepath),
        keyer,
        n_jobs=n_jobs,
        cache=cache,
    )


def load_stock_parallel(
    filepath: str | Path,
    keyer: Keyer[M, K],
    *,
    cache: bool | None = None,
) -> set[K]:
    """Load stock with all available joblib workers."""
    return load_stock(filepath, keyer, n_jobs=-1, cache=cache)
