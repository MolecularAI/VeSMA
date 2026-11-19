"""Every constant a shader and its host must agree about, compared rather than commented.

MSL lives in a raw C++ string inside each ``.mm``, so it cannot ``#include`` the host's header.
A number like the packed row width is therefore declared three separate times -- in the shader
source, in the host code beside it, and in :mod:`vesma.limits` -- and the only thing holding them
together used to be a comment saying "kept in sync with".

The failure mode is what makes that worth a test. These numbers size buffers. A disagreement is
not a wrong answer some parity test would notice; it is a buffer indexed past its end, or a row
truncated in silence, on whichever side was left behind. So each shader carries a
``constants_probe`` kernel that reports what it actually compiled, and this asks.

Two shaders have such a kernel and two do not, which is not an omission. ``metal_matcher.mm`` and
``metal_circularfp.mm`` take the bucket and the word count as *runtime parameters* -- the host
passes them in with every dispatch -- so there is no second declaration there to drift from.
"""

from __future__ import annotations

import pytest

from vesma.bonds import BOND_KIND_MASK
from vesma.limits import (
    ATOM_MASK_WORD_COUNT,
    GPU_MATCH_ATOM_CAP,
    MAX_COMPONENTS_PER_ASSIGNMENT,
    METAL_EDITED_ATOM_CAP,
)
from vesma.pathfp import FOLDED_DEFAULT_BITS

pytestmark = pytest.mark.metal


def test_the_retro_shader_compiled_the_limits_the_host_declares() -> None:
    from vesma.metal_retro import MAX_REACTION_EDGES, probe_retro_pipelines

    probed = probe_retro_pipelines()

    assert probed["shader_bucket"] == GPU_MATCH_ATOM_CAP
    assert probed["shader_match_atom_cap"] == GPU_MATCH_ATOM_CAP
    assert probed["shader_atom_word_count"] == ATOM_MASK_WORD_COUNT
    assert probed["shader_edited_bucket"] == METAL_EDITED_ATOM_CAP
    assert probed["shader_max_components"] == MAX_COMPONENTS_PER_ASSIGNMENT
    assert probed["shader_max_reaction_edges"] == MAX_REACTION_EDGES


def test_the_retro_shader_and_the_host_code_beside_it_agree() -> None:
    """The third declaration: the C++ host mirrors in ``metal_retro.mm`` itself.

    Separate from the test above because it is a different drift. One edits the shader string and
    forgets the host code ten lines up; the other edits both and forgets Python.
    """
    from vesma.metal_retro import probe_retro_pipelines

    probed = probe_retro_pipelines()

    assert probed["host_max_components"] == probed["shader_max_components"]
    assert probed["host_max_reaction_edges"] == probed["shader_max_reaction_edges"]
    assert probed["host_edited_bucket"] == probed["shader_edited_bucket"]
    assert probed["host_wl_threads"] == probed["shader_wl_threads"]


def test_the_wl_kernel_is_dispatched_as_wide_as_its_reduction_reads() -> None:
    """``WL_THREADS`` is not a tuning knob; it is a contract with the dispatch.

    The kernel's edge count sums ``partial_edges[0 .. WL_THREADS)``, every slot written by one
    lane, so a dispatch narrower than ``WL_THREADS`` sums slots nobody initialised. The host used
    to spell this as a bare ``32`` at the dispatch site.
    """
    from vesma.metal_retro import probe_retro_pipelines

    probed = probe_retro_pipelines()

    assert probed["host_wl_threads"] == probed["shader_wl_threads"]
    # And the threadgroup has to be able to hold them.
    assert probed["shader_wl_threads"] <= probed["wl_max_threads_per_threadgroup"]


def test_the_pathfp_shader_compiled_the_limits_the_host_declares() -> None:
    from vesma.metal_pathfp import _FP_TILE_SIZE, probe_pathfp_constants

    probed = probe_pathfp_constants()

    assert probed["shader_bucket"] == GPU_MATCH_ATOM_CAP
    assert probed["shader_atom_word_count"] == ATOM_MASK_WORD_COUNT
    assert probed["shader_fp_tile_size"] == _FP_TILE_SIZE
    assert probed["shader_fp_word_count"] == FOLDED_DEFAULT_BITS // 64
    assert probed["shader_bond_class_mask"] == BOND_KIND_MASK


def test_the_two_shaders_mask_bond_flags_the_same_way() -> None:
    """Both shaders read the same packed bond byte, so one mask is one mask.

    ``metal_retro.mm`` does not export its ``BOND_CLASS_MASK`` -- its ``bond_class`` is applied at
    every read, which the WL identity relies on -- so the claim tested here is the one that can be
    tested: the folded-path shader and :mod:`vesma.bonds` agree, and the retro shader's reads are
    checked against the CPU codec by the keyer parity tests instead.
    """
    from vesma.metal_pathfp import probe_pathfp_constants

    assert probe_pathfp_constants()["shader_bond_class_mask"] == BOND_KIND_MASK
