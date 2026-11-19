"""CPU/CUDA parity over the corpus: same precursors and typed refusals.

The CPU chemistry is frozen so the device port has a fixed target. This compares both
the multiset of ``(molecule, rule, precursor key)`` outcomes and the per-reason refusal
histogram. Capacity events are reported separately; both backends refuse an application
whole when one of its precursors exceeds a capacity, so a cap costs them the same
application rather than leaving one side with a partial set the other never built.

    make bench-cuda-parity PARITY_SIZES="3 25 100"
"""

from __future__ import annotations

import os
from collections import Counter

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
from vesma.reaction import GPU_PARITY_REACTION_PROCESSING
from vesma.rejection import by_name

pytestmark = [pytest.mark.perf, pytest.mark.gpu, pytest.mark.cuda]

DEFAULT_SIZES = (3, 25)


def sizes() -> tuple[int, ...]:
    raw = os.environ.get("PARITY_SIZES", "")
    if not raw.strip():
        return DEFAULT_SIZES
    return tuple(int(token) for token in raw.split())


def require_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


def capped_applications(diagnostics) -> int:
    if diagnostics is None:
        return 0
    return (
        diagnostics.precursor_atom_overflow_assignments
        + diagnostics.edited_atom_overflow_assignments
    )


def assert_same_triples(got: Counter, want: Counter, size: int, delivery: str) -> None:
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


@pytest.mark.parametrize("size", sizes())
def test_cuda_matches_cpu_over_the_corpus(size: int) -> None:
    require_cuda()
    from vesma.cuda_matcher import (
        CudaReactionMatcher,
        cuda_concat_mols,
        cuda_short_ring_wl_key,
    )
    from vesma.matcher import keyed_match_folding

    corpus = load_match_corpus(size)
    pairs = load_positive_pairs(corpus)
    rules = [template.pattern for template in corpus.templates]

    # One reference, one question, both devices. This used to ask an *uncapped* one where
    # `bench_metal_parity` asked a capped one, which made the two gates incommensurable in the
    # place it mattered most: the device refuses an application whose precursor exceeds 64 atoms,
    # so an uncapped reference holds outcomes CUDA cannot build and the gate read them as a
    # shortfall. At 100 molecules that was 8 missing outcomes against 8 capped applications --
    # explainable, but only by hand, and `compare` asserted nothing about them because CUDA bounds
    # no component count. Capping the reference removes the discrepancy at its source rather than
    # bounding it afterwards.
    #
    # The cap is the same 64 on both devices (`GPU_PRECURSOR_ATOM_CAP`, and
    # `CUDA_PRECURSOR_ATOM_CAP` in the kernel), so this is not a CUDA-flavoured question: it is
    # the question both devices actually answer. `max_components` stays `None` because that one
    # really does differ -- Metal caps components per assignment and CUDA does not.
    #
    # Sharing the filename is now correct for the same reason it was wrong before. It was wrong
    # while the two asked different things, because `load_or_build` treats a cap mismatch as a
    # miss and saved over it, so running both on one machine rebuilt the CPU leg every time. Two
    # benches asking one question should share one file, and a machine with one device simply
    # never builds the other's.
    #
    # `n_jobs=1` is a correctness requirement, not a throughput choice: forking after CuPy
    # has initialised CUDA is unsafe on Linux. The keys are identical either way, so this
    # decides only who computes them -- which is exactly what lets the two benches share a
    # reference built by whichever ran first.
    reference = load_or_build(
        size,
        processing=GPU_PARITY_REACTION_PROCESSING,
        processing_name="gpu-parity",
        max_precursor_atoms=GPU_PRECURSOR_ATOM_CAP,
        n_jobs=1,
    )
    require_current(
        reference,
        molecule_indices=corpus.source_molecule_indices,
        template_count=len(rules),
        processing="gpu-parity",
        max_precursor_atoms=GPU_PRECURSOR_ATOM_CAP,
    )

    cuda = CudaReactionMatcher(rules, frontier_mode="host")
    cuda_precursors, cuda_hits = cuda.match(corpus.cmols, pairs)
    cuda_keys = cuda_short_ring_wl_key(cuda_precursors)
    cuda_keyed = keyed_counter(cuda_hits, [cuda_keys[hit[3]] for hit in cuda_hits])

    diagnostics = cuda.last_diagnostics
    assert diagnostics is not None
    detail = "corpus exceeds the CUDA edited-atom cap"
    assert diagnostics.edited_atom_overflow_assignments == 0, detail
    assert_same_triples(cuda_keyed, reference.keyed, size, "capped CPU reference")
    # Asserted rather than assumed: the refusal histogram below is half of what parity means, and
    # a matcher that reported no diagnostics at all would otherwise fail here as an
    # AttributeError on None rather than as the missing evidence it is.
    assert diagnostics is not None, "the matcher reported no diagnostics for this call"

    # The same corpus delivered in pieces, which is what `match` is now a fold over and what
    # `keyed_match_folding` consumes. The multiset `keyed_counter` builds is
    # (molecule, rule, key): it names no precursor row and no application index, so it is
    # comparable across deliveries without rebasing anything, and agreement has to be exact.
    # Unit tests pin this on a handful of rules; this is it over the real corpus and all 42,554.
    batched_keyed: Counter = Counter()
    for batch, batch_hits in cuda.match_batches(corpus.cmols, pairs):
        batch_keys = cuda_short_ring_wl_key(batch)
        batched_keyed.update(
            keyed_counter(batch_hits, [batch_keys[hit[3]] for hit in batch_hits])
        )
    assert_same_triples(batched_keyed, cuda_keyed, size, "match_batches")

    # And through the generic fold, which collapses duplicate molecules as the pieces arrive.
    # Deduplicating must not change which (molecule, rule, key) triples exist -- that is the
    # invariant letting a search hold distinct molecules instead of every precursor, and it is
    # what `match_keyed` has always relied on here with its own `key_to_physical` map. Holding
    # the generic fold to the same reference keeps the two deliveries comparable. The native
    # keyed path also places the frontier, so it remains the host/device search delivery.
    folded_batch, folded_keys, folded_hits = keyed_match_folding(
        lambda mols: cuda.match_batches(mols, pairs),
        cuda_short_ring_wl_key,
        cuda_concat_mols,
    )(corpus.cmols)
    folded_keyed = keyed_counter(
        folded_hits, [folded_keys[hit[3]] for hit in folded_hits]
    )
    assert_same_triples(folded_keyed, cuda_keyed, size, "keyed_match_folding")
    no_orphan_rows = set(folded_hits[:, 3].tolist()) == set(range(len(folded_batch)))
    assert no_orphan_rows, "folded batch has rows no hit names"
    print(
        f"[cuda] delivery parity at size {size}: {len(cuda_keyed)} distinct "
        f"(molecule, rule, key) triples agree exactly across match, match_batches and the fold; "
        f"the fold holds {len(folded_batch)} rows against {len(cuda_precursors)} "
        f"({len(cuda_precursors) / max(1, len(folded_batch)):.2f}x fewer)"
    )
    print(
        compare(
            reference,
            backend="cuda",
            device_keyed=cuda_keyed,
            device_refusals={
                name: count
                for name, count in by_name(tuple(diagnostics.rejections)).items()
                if count
            },
            capped=capped_applications(diagnostics),
            # CUDA has no component cap, so there is no equivalent bound; see `compare`.
            components_per_capped_application=None,
        )
        + f" undescribed_components={diagnostics.undescribed_components}"
    )


# The generated frontier is matched at every layer below the first in a real search, but
# it is much wider than the corpus, so the second layer runs on a bounded sample of the
# molecules both backends built. Kept in step with bench_metal_parity.py.
SECOND_LAYER_MOLECULES = 64


@pytest.mark.parametrize("size", sizes())
def test_cuda_matches_cpu_on_a_generated_frontier(size: int) -> None:
    """Parity at the layer whose molecules the device built, not the host.

    The corpus test above starts from host-packed CMols, which carry CPU-perceived ring
    topology and atom invariants whichever backend then matches them. That leaves the
    whole of device perception untested against the CPU model: a precursor the device
    builds has to answer `@`, `!@` and `x` out of its own perception, and for a long
    time it did not -- it declared every bond "ring or chain", so `!@` matched ring
    bonds. Only a second layer can catch that, because only a second layer asks.
    """
    require_cuda()
    from vesma.cuda_matcher import CudaReactionMatcher, cuda_short_ring_wl_key

    corpus = load_match_corpus(size)
    pairs = load_positive_pairs(corpus)
    rules = [template.pattern for template in corpus.templates]

    # The Metal gate and first-layer reference ask the same capped question. An uncapped
    # second-layer CPU run would count applications neither GPU can materialize.
    cpu = CpuReactionMatcher(
        rules,
        processing=GPU_PARITY_REACTION_PROCESSING,
        max_precursor_atoms=GPU_PRECURSOR_ATOM_CAP,
    )
    cpu_precursors, cpu_hits = cpu.match(corpus.cmols, pairs, n_jobs=1)
    cpu_keys = [short_ring_wl_molecule_id(cpu_precursors[hit[3]]) for hit in cpu_hits]

    cuda = CudaReactionMatcher(rules, frontier_mode="host")
    cuda_batch, cuda_hits = cuda.match(corpus.cmols, pairs)
    cuda_keys = cuda_short_ring_wl_key(cuda_batch)

    # Address the second layer's molecules by WL key rather than by position: the two
    # backends emit precursors in their own order, and a cap may drop one of them
    # entirely. Taking the keys both produced asks them the same question.
    cpu_by_key = {
        key: cpu_precursors[int(hit[3])]
        for hit, key in zip(cpu_hits, cpu_keys, strict=True)
    }
    cuda_by_key = {cuda_keys[int(hit[3])]: int(hit[3]) for hit in cuda_hits}
    shared = sorted(set(cpu_by_key) & set(cuda_by_key))[:SECOND_LAYER_MOLECULES]
    assert shared, "no precursor key was produced by both backends"

    frontier_cpu = [cpu_by_key[key] for key in shared]
    frontier_cuda = cuda_batch[[cuda_by_key[key] for key in shared]]

    cpu_second_precursors, cpu_second_hits = cpu.match(frontier_cpu, n_jobs=1)
    cpu_second_keyed = Counter(
        (
            int(hit[0]),
            int(hit[1]),
            short_ring_wl_molecule_id(cpu_second_precursors[hit[3]]),
        )
        for hit in cpu_second_hits
    )

    cuda_second_batch, cuda_second_hits = cuda.match(frontier_cuda)
    cuda_second_keys = cuda_short_ring_wl_key(cuda_second_batch)
    cuda_second_keyed = Counter(
        (int(hit[0]), int(hit[1]), cuda_second_keys[int(hit[3])])
        for hit in cuda_second_hits
    )

    capped = capped_applications(cuda.last_diagnostics)
    print(
        f"\nsize={size} frontier={len(shared)} "
        f"cpu_outcomes={sum(cpu_second_keyed.values())} "
        f"cuda_outcomes={sum(cuda_second_keyed.values())} "
        f"capped_applications={capped}"
    )

    invented = cuda_second_keyed - cpu_second_keyed
    detail = f"{sum(invented.values())} second-layer CUDA outcomes absent on CPU"
    assert not invented, detail

    assert cuda.last_diagnostics is not None
    assert cpu.last_diagnostics is not None
    detail = "generated frontier exceeds the CUDA edited-atom cap"
    assert cuda.last_diagnostics.edited_atom_overflow_assignments == 0, detail
    assert cpu.last_diagnostics.pairs_search_incomplete == 0
    assert_same_triples(
        cuda_second_keyed, cpu_second_keyed, size, "capped generated frontier"
    )
    assert by_name(tuple(cuda.last_diagnostics.rejections)) == by_name(
        cpu.last_diagnostics.rejections
    )
