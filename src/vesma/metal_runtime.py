"""One Metal runtime: the cache directory, the ctypes pointer casts, the dylib loader.

Four modules in this package drive a `.mm` translation unit of their own -- the matcher, the
retro constructor, the circular fingerprints and the folded path fingerprints. Each one needs
the same three things before it can call a kernel: somewhere to put the compiled dylib, a way to
hand numpy an array as a typed pointer, and a compile-and-load step. They each grew their own
copy, which is how the cache directory came to be defined three times and the pointer casts four,
under two spellings and two visibilities.

There is nothing per-module about any of it, so it lives here once, and the loading policy in
particular is one policy rather than four:

* The dylib's filename carries the source's mtime, so editing a kernel compiles to a *new* file
  instead of overwriting the one another process may have mapped. These sources are edited far
  more often than they are run.
* The compile writes a temp sibling and `os.replace`s it into place, which is atomic on the same
  filesystem, so a reader never maps a half-written dylib.
* Siblings of the same source under a different key are unlinked, so each source keeps exactly
  one dylib in the cache instead of accumulating one per edit.

The cache directory is disposable by construction: `rm -rf` it and the next call rebuilds.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "f32_ptr",
    "f64_ptr",
    "i32_ptr",
    "load_metal_dylib",
    "metal_cache_dir",
    "u8_ptr",
    "u32_ptr",
    "u64_ptr",
]


def metal_cache_dir() -> Path:
    """Where compiled dylibs live.

    Honours ``XDG_CACHE_HOME`` when set; otherwise a workdir-local ``.cache``, which keeps the
    cache inside the project and so inside whatever sandbox the project is running under.
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.cwd() / ".cache"
    return base / "vesma-metal"


def u8_ptr(array: np.ndarray) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def u32_ptr(array: np.ndarray) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def u64_ptr(array: np.ndarray) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def i32_ptr(array: np.ndarray) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def f32_ptr(array: np.ndarray) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def f64_ptr(array: np.ndarray) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


_COMPILE_FLAGS = (
    "-dynamiclib",
    "-std=c++17",
    "-fobjc-arc",
    "-O2",
    "-framework",
    "Foundation",
    "-framework",
    "Metal",
)


@cache
def load_metal_dylib(source: Path) -> ctypes.CDLL:
    """Compile `source` if the cache has no dylib for its current mtime, then load it.

    The caller sets argtypes afterwards; this returns a bare `CDLL` and knows nothing about what
    the translation unit exports.

    Memoised per source, and that is load-bearing rather than tidy. Two of the four loaders this
    replaced were uncached -- `metal_pathfp`'s runs on every fingerprint and every prefilter
    dispatch, three call sites deep in the search's hot path -- and consolidating made the per-call
    cost *worse*, adding a `mkdir`, a `glob` and an unlink loop to the `stat` they already did. One
    compilation and one `CDLL` per source per process is what every caller wanted; `metal_retro`
    had said so with a module global and `metal_circularfp` with this same decorator.

    An edited source therefore needs a fresh process, which is already true: `ctypes.CDLL` caches
    at the OS level and the old image stays mapped regardless.
    """
    cache = metal_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    key = source.stat().st_mtime_ns
    dylib = cache / f"{source.stem}_{key}.dylib"

    if not dylib.exists():
        # A temp sibling plus `os.replace` rather than a direct `-o dylib`: clang writes the
        # output incrementally, and two processes compiling the same source concurrently would
        # otherwise interleave into one file. Whichever replace lands last wins, and both
        # contents are the same compilation anyway.
        staging = cache / f"{source.stem}_{key}.{os.getpid()}.staging"
        command = ["clang++", *_COMPILE_FLAGS, str(source), "-o", str(staging)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            staging.unlink(missing_ok=True)
            raise RuntimeError(f"Metal compilation failed:\n{exc.stderr}") from exc
        os.replace(staging, dylib)

    for stale in cache.glob(f"{source.stem}_*.dylib"):
        if stale != dylib:
            # Already-mapped dylibs survive being unlinked on macOS, so this is safe to do
            # while another process in this tree is running against the previous build.
            stale.unlink(missing_ok=True)

    return ctypes.CDLL(str(dylib))
