"""CPU/Metal parity over the corpus: same precursors, and the same refusals.

The CPU chemistry is frozen so the GPU port has a fixed target, and this is what measures
whether it hit it. Two things are compared, and both matter:

- the multiset of ``(molecule, rule, precursor key)`` triples, which says the two backends
  build the same molecules;
- the per-reason refusal histogram, which says they decline the same things for the same
  stated causes. Comparing only the first would let a backend agree by refusing whatever it
  found hard, and a refusal that names its cause is the model's product, not its shortfall.

Capacity is reported, never silently absorbed. The device caps a precursor at 64 atoms, an
edited graph at 256, and an assignment at 6 components. The reference below is given the same
precursor cap, and all three backends refuse an application whole when one of its precursors
exceeds a capacity, so a cap now costs both sides the same application rather than leaving the
CPU with a partial set the device never built. Only the 6-component cap has no CPU counterpart,
which is what the bound on missing outcomes still allows for.

    make bench-metal-parity PARITY_SIZES="3 25 100"
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pytest

from tests.vesma.bench_matcher import load_match_corpus, load_positive_pairs
from tests.vesma.parity_reference import (
    compare,
    keyed_counter,
    load_or_build,
    require_current,
)
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.identity import short_ring_wl_molecule_id
from vesma.limits import GPU_PRECURSOR_ATOM_CAP
from vesma.metal_retro import MAX_COMPONENTS_PER_ASSIGNMENT
from vesma.reaction import GPU_PARITY_REACTION_PROCESSING
from vesma.rejection import by_name

pytestmark = [pytest.mark.perf, pytest.mark.gpu, pytest.mark.metal]

DEFAULT_SIZES = (3, 25)


def _sizes() -> tuple[int, ...]:
    raw = os.environ.get("PARITY_SIZES", "")
    if not raw.strip():
        return DEFAULT_SIZES
    return tuple(int(token) for token in raw.split())


def _require_metal() -> None:
    if sys.platform != "darwin":
        pytest.skip("Metal backend is only available on macOS.")
    try:
        from vesma.metal_matcher import probe_metal

        probe_metal()
    except Exception as exc:  # pragma: no cover - depends on host GPU
        pytest.skip(f"Metal backend unavailable: {exc}")


def _key_chunk(precursors: Sequence[Any]) -> list[Any]:
    return [short_ring_wl_molecule_id(precursor) for precursor in precursors]


def _keys_in_parallel(precursors: Any, hits: Any) -> list[Any]:
    """WL keys for the precursor of every hit, computed across processes.

    Keying dominates this comparison -- at 100 molecules it is 24s against 7s of
    matching, because the reference keyer walks rings and mixes 64-bit words per molecule
    in Python. It is also perfectly independent per precursor, so the corpus sizes worth
    running are reachable by spreading it rather than by making the keyer cleverer. The
    keyer is the thing being trusted here; it should stay the simple one.

    Forked workers inherit the precursors, so nothing large is pickled on the way out --
    only the tokens come back.
    """
    ordered = [precursors[int(hit[3])] for hit in hits]
    workers = max(1, (os.cpu_count() or 2) - 1)
    if len(ordered) < 4096 or workers == 1:
        return _key_chunk(ordered)
    size = (len(ordered) + workers - 1) // workers
    chunks = [ordered[i : i + size] for i in range(0, len(ordered), size)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [key for chunk in pool.map(_key_chunk, chunks) for key in chunk]


def _capped_applications(diagnostics) -> int:
    """Applications where a device capacity limit fired, so the CPU had more room."""
    if diagnostics is None:
        return 0
    return (
        diagnostics.component_overflow_assignments
        + diagnostics.precursor_atom_overflow_assignments
        + diagnostics.edited_atom_overflow_assignments
    )


def _assert_same_triples(got: Counter, want: Counter, size: int, delivery: str) -> None:
    """Every delivery of one call must support the same (molecule, rule, key) multiset."""
    if got == want:
        return
    only_got = got - want
    only_want = want - got
    raise AssertionError(
        f"{delivery} disagrees with match at size {size}: "
        f"{sum(only_got.values())} triple(s) only in {delivery}, "
        f"{sum(only_want.values())} only in match; "
        f"first divergence {next(iter((only_got | only_want).items()))}"
    )


@pytest.mark.parametrize("size", _sizes())
def test_metal_matches_cpu_over_the_corpus(size: int) -> None:
    _require_metal()
    from vesma.matcher import keyed_match_folding
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_concat_batches,
        metal_short_ring_wl_key,
    )

    corpus = load_match_corpus(size)
    pairs = load_positive_pairs(corpus)
    rules = [template.pattern for template in corpus.templates]

    # The CPU leg comes from the shared harness rather than from a run of its own, so this bench
    # and `bench_cuda_parity` run literally the same comparison -- one `compare`, one definition of
    # agreement. The reference *file* is not shared, and should not be: this one asks a capped
    # question and CUDA's an uncapped one, which is why `default_path` carries the cap. No machine
    # has both devices, so each is only ever checked against CPU, and sharing the comparison is
    # what keeps the two commensurable where sharing a file cannot.
    #
    # GPU parity mode, and the 64-atom cap, are part of the question rather than the answer:
    # the devices refuse an application whose precursor exceeds the cap, so an uncapped
    # reference holds outcomes no device can build and each would read as a shortfall. Only
    # that one cap actually binds, on 0.0085% of the corpus. The uncapped model remains the
    # default everywhere else; this is a benchmark setting, not a change to the chemistry.
    reference = load_or_build(
        size,
        processing=GPU_PARITY_REACTION_PROCESSING,
        processing_name="gpu-parity",
        max_precursor_atoms=GPU_PRECURSOR_ATOM_CAP,
        # Forked keying, as this bench has always done. The CUDA bench must pass 1 instead.
        n_jobs=None,
    )
    require_current(
        reference,
        molecule_indices=corpus.source_molecule_indices,
        template_count=len(rules),
        processing="gpu-parity",
        max_precursor_atoms=GPU_PRECURSOR_ATOM_CAP,
    )

    metal = MetalReactionMatcher(rules)
    metal_batch, metal_hits = metal.match(corpus.cmols, pairs)
    metal_keys = metal_short_ring_wl_key(metal_batch)
    metal_keyed = keyed_counter(metal_hits, [metal_keys[hit[3]] for hit in metal_hits])

    diagnostics = metal.last_diagnostics

    # The same corpus delivered in pieces, which is what the search and the scaling measurements
    # use. The multiset `keyed_counter` builds is (molecule, rule, key) -- it names no precursor
    # row and no application index -- so it is comparable across any delivery without rebasing
    # anything, and agreement has to be exact. The unit tests pin this array-by-array on a handful
    # of rules; this is it over the real corpus and all 42,554, which is the standard `match` is
    # held to and so the one `match_batches` has to meet before anything relies on it.
    batched_keyed: Counter = Counter()
    for batch, hits in metal.match_batches(corpus.cmols, pairs):
        keys = metal_short_ring_wl_key(batch)
        batched_keyed.update(keyed_counter(hits, [keys[hit[3]] for hit in hits]))
    _assert_same_triples(batched_keyed, metal_keyed, size, "match_batches")

    # And once more through the generic fold, which collapses duplicate molecules as the pieces
    # arrive. Deduplication must not change which (molecule, rule, key) triples exist -- that is
    # exactly the invariant that lets the search hold distinct molecules instead of every
    # precursor -- so this runs against the same reference as the other two.
    folded_batch, folded_keys, folded_hits = keyed_match_folding(
        lambda mols: metal.match_batches(mols, pairs),
        metal_short_ring_wl_key,
        metal_concat_batches,
    )(corpus.cmols)
    folded_keyed = keyed_counter(
        folded_hits, [folded_keys[hit[3]] for hit in folded_hits]
    )
    _assert_same_triples(folded_keyed, metal_keyed, size, "keyed_match_folding")
    # Named rather than inlined: black and ruff format a long `assert ..., "message"` to
    # different shapes, and `make check` runs both.
    no_orphan_rows = set(folded_hits[:, 3].tolist()) == set(range(len(folded_batch)))
    assert no_orphan_rows, "folded batch has rows no hit names"
    print(
        f"[metal] delivery parity at size {size}: {len(metal_keyed)} distinct "
        f"(molecule, rule, key) triples agree exactly across match, match_batches and the fold; "
        f"the fold holds {len(folded_batch)} rows against {len(metal_batch)} "
        f"({len(metal_batch) / max(1, len(folded_batch)):.2f}x fewer)"
    )
    print(
        compare(
            reference,
            backend="metal",
            device_keyed=metal_keyed,
            device_refusals={
                name: count
                for name, count in by_name(tuple(diagnostics.rejections)).items()
                if count
            },
            capped=_capped_applications(diagnostics),
            components_per_capped_application=MAX_COMPONENTS_PER_ASSIGNMENT,
        )
    )


# The generated frontier is matched at every layer below the first in a real search, but
# it is much wider than the corpus, so the second layer runs on a bounded sample of the
# molecules both backends built. Wide enough to exercise the shapes, small enough that
# the all-rules-by-all-molecules match at layer two stays a benchmark and not a batch job.
SECOND_LAYER_MOLECULES = 64


@pytest.mark.parametrize("size", _sizes())
def test_metal_matches_cpu_on_a_generated_frontier(size: int) -> None:
    """Parity at the layer whose molecules the device built, not the host.

    The corpus test above starts from host-packed CMols, which carry CPU-perceived ring
    topology and atom invariants whichever backend then matches them. That leaves the
    whole of device perception untested against the CPU model: a precursor the device
    builds has to answer `@`, `!@` and `x` out of its own perception, and for a long
    time it did not -- it declared every bond "ring or chain", so `!@` matched ring
    bonds. Only a second layer can catch that, because only a second layer asks.
    """
    _require_metal()
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_short_ring_wl_key,
    )

    corpus = load_match_corpus(size)
    pairs = load_positive_pairs(corpus)
    rules = [template.pattern for template in corpus.templates]

    cpu = CpuReactionMatcher(
        rules,
        max_precursor_atoms=GPU_PRECURSOR_ATOM_CAP,
        processing=GPU_PARITY_REACTION_PROCESSING,
    )
    cpu_precursors, cpu_hits = cpu.match(corpus.cmols, pairs)
    cpu_keys = _keys_in_parallel(cpu_precursors, cpu_hits)

    metal = MetalReactionMatcher(rules)
    metal_batch, metal_hits = metal.match(corpus.cmols, pairs)
    metal_keys = metal_short_ring_wl_key(metal_batch)

    # Address the second layer's molecules by WL key rather than by position: the two
    # backends emit precursors in their own order, and a cap may drop one of them
    # entirely. Taking the keys both produced asks them the same question.
    cpu_by_key = {
        key: cpu_precursors[int(hit[3])]
        for hit, key in zip(cpu_hits, cpu_keys, strict=True)
    }
    metal_by_key = {metal_keys[int(hit[3])]: int(hit[3]) for hit in metal_hits}
    shared = sorted(set(cpu_by_key) & set(metal_by_key))[:SECOND_LAYER_MOLECULES]
    assert shared, "no precursor key was produced by both backends"

    frontier_cpu = [cpu_by_key[key] for key in shared]
    frontier_metal = metal_batch[[metal_by_key[key] for key in shared]]

    cpu_second_precursors, cpu_second_hits = cpu.match(frontier_cpu)
    cpu_second_keyed = Counter(
        (int(hit[0]), int(hit[1]), key)
        for hit, key in zip(
            cpu_second_hits,
            _keys_in_parallel(cpu_second_precursors, cpu_second_hits),
            strict=True,
        )
    )

    metal_second_batch, metal_second_hits = metal.match(frontier_metal)
    metal_second_keys = metal_short_ring_wl_key(metal_second_batch)
    metal_second_keyed = Counter(
        (int(hit[0]), int(hit[1]), metal_second_keys[int(hit[3])])
        for hit in metal_second_hits
    )

    capped = _capped_applications(metal.last_diagnostics)
    print(
        f"\nsize={size} frontier={len(shared)} "
        f"cpu_outcomes={sum(cpu_second_keyed.values())} "
        f"metal_outcomes={sum(metal_second_keyed.values())} "
        f"capped_applications={capped}"
    )

    invented = metal_second_keyed - cpu_second_keyed
    detail = f"{len(invented)} device second-layer outcomes the CPU did not build"
    assert not invented, detail

    assert metal.last_diagnostics is not None
    assert cpu.last_diagnostics is not None
    assert metal.last_diagnostics.edited_atom_overflow_assignments == 0
    assert metal.last_diagnostics.component_overflow_assignments == 0
    assert metal_second_keyed == cpu_second_keyed
    assert by_name(tuple(metal.last_diagnostics.rejections)) == by_name(
        cpu.last_diagnostics.rejections
    )
