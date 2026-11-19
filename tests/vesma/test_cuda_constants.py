"""Every constant the CUDA kernel and its host must agree about, compared rather than commented.

The counterpart of `test_shader_constants.py`, which does this for the Metal shaders, and it
exists for the same reason: the CUDA source is a raw string inside `cuda_retro.py`, so it cannot
import the host's constants and each number is declared twice. `cuda_retro` imports
`GPU_PRECURSOR_ATOM_CAP` from `vesma.limits` and uses it in a warning message, while the kernel
beside it says `#define CUDA_PRECURSOR_ATOM_CAP 64` as a literal -- change the limit and the
warning moves while the kernel does not.

These numbers size buffers and decide refusals, so a disagreement is not a wrong answer some
parity test would catch; it is a buffer indexed past its end, or an application refused on one
side and kept on the other, on whichever side was left behind.

Unlike the Metal test this needs no device and no probe kernel: the source is a Python string, so
the declarations can simply be read. That means it runs on any machine, which matters because the
drift it guards against is introduced on machines that have no GPU at all.
"""

from __future__ import annotations

import re
from pathlib import Path

from vesma.cuda_retro import _CUDA_RETRO_SOURCE, _MAX_REACTION_EDGES
from vesma.limits import (
    ATOM_MASK_WORD_COUNT,
    CUDA_EDITED_ATOM_CAP,
    GPU_MATCH_ATOM_CAP,
    GPU_PRECURSOR_ATOM_CAP,
)

_DEFINE = re.compile(r"^#define\s+(\w+)\s+(\d+)\w*\s*$", re.MULTILINE)


def cuda_defines() -> dict[str, int]:
    """The integer `#define`s the embedded CUDA source declares."""
    return {name: int(value) for name, value in _DEFINE.findall(_CUDA_RETRO_SOURCE)}


def test_the_cuda_kernel_compiled_the_limits_the_host_declares() -> None:
    defines = cuda_defines()
    expected = {
        # The packed molecule width, and the atom cap that width represents.
        "CUDA_BUCKET_SIZE": GPU_MATCH_ATOM_CAP,
        # The edited-graph width, wider than a molecule because an edit may grow one.
        "CUDA_EDITED_BUCKET_SIZE": CUDA_EDITED_ATOM_CAP,
        # The cap that refuses an application whose precursor is too large. This is the one
        # `cuda_retro` already imports for its warning text, so a drift here makes the engine
        # refuse at one size and say it refused at another.
        "CUDA_PRECURSOR_ATOM_CAP": GPU_PRECURSOR_ATOM_CAP,
        # Words per packed atom mask.
        "ATOM_WORD_COUNT": ATOM_MASK_WORD_COUNT,
        # Reaction edge capacity, used host-side to size the reaction arrays.
        "MAX_REACTION_EDGES": _MAX_REACTION_EDGES,
    }
    missing = sorted(name for name in expected if name not in defines)
    assert not missing, f"the CUDA source no longer defines {missing}"
    disagreements = {
        name: (defines[name], want)
        for name, want in expected.items()
        if defines[name] != want
    }
    # Message hoisted: black and ruff format a long `assert ..., "message"` to different shapes
    # and `make check` runs both, so the condition stays short enough for neither to wrap it.
    detail = f"CUDA kernel and host disagree (kernel, host): {disagreements}"
    assert not disagreements, detail


def test_both_gpu_backends_hash_wl_keys_at_the_same_version() -> None:
    """The WL hash version is declared twice, in two languages, and nowhere in Python.

    It is what makes a key comparable: Metal's keys, CUDA's keys and the CPU reference's keys are
    only the same keys while all three hash the same way. Bump one source and the other keeps
    producing the old key -- and nothing fails loudly, because each backend remains
    self-consistent. The parity gates would catch it only on a machine that has both devices, and
    no machine here has both.

    So this compares the two declarations directly. If the version is ever bumped deliberately,
    both sources change together and this keeps passing; that is the whole intent.
    """
    cuda_version = cuda_defines().get("GPU_WL_HASH_VERSION")
    undeclared = "the CUDA source no longer defines GPU_WL_HASH_VERSION"
    assert cuda_version is not None, undeclared

    metal_source = (
        Path(__file__).resolve().parents[2] / "src" / "vesma" / "metal_retro.mm"
    ).read_text()
    found = re.search(r"#define\s+GPU_WL_HASH_VERSION\s+(\d+)", metal_source)
    assert found is not None, "metal_retro.mm no longer defines GPU_WL_HASH_VERSION"
    metal_version = int(found.group(1))

    mismatch = (
        f"CUDA hashes WL keys at version {cuda_version} and Metal at {metal_version}; "
        "their keys are not comparable until these agree"
    )
    assert cuda_version == metal_version, mismatch
