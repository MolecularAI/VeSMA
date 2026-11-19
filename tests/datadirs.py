from __future__ import annotations

from pathlib import Path


def external_data_root() -> Path:
    """Return the committed, read-only test-data root (``tests/data``).

    This is the single source of truth for locating external datasets. Pytest tests
    receive it through the ``readonly_datadir`` fixture; non-pytest ``python -m`` runners
    call this function directly. Nothing copies the tree — callers read it in place and
    must not write into it (accidental writes show up in ``git status``).
    """
    return Path(__file__).resolve().parent / "data"
