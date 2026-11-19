"""Accumulating against streaming precursors, across workload size, on the Metal path.

`MetalReactionMatcher` can either join its precursors into one batch (`match`) or yield each
construction sub-dispatch as it lands (`match_batches`). The two do identical device work --
`_RETRO_ASSIGNMENTS_PER_DISPATCH` assignments per `retro_construct` dispatch however large the
call -- and differ only in host retention: joining keeps every sub-dispatch's gathered arrays
alive and then allocates the joined result beside them, so host footprint grows linearly in the
workload at ~7,684 bytes a precursor, twice over.

That growth is what used to bend the throughput curve down past ~100 molecules. Two legs here
hold it to account:

``accumulate``
    One `match` per size, the joined result dropped. The raw cost of delivering one array.
``stream``
    The same work through `match_batches`, each piece dropped as it arrives. The floor.
``keyed``
    `match`, then WL-key the joined batch -- what the search does today.
``folded``
    `match_batches` through `matcher.keyed_match_folding`, which keys each piece and collapses
    duplicate molecules as it goes -- what the search would do. The pair `keyed`/`folded` is the
    honest A/B, because both pay for identity; `accumulate`/`stream` isolate delivery alone.

All legs carry the matcher's own phase diagnostics and the machine's page-compression counters,
which is what separates "the host ran out of memory" from "the kernel got slower". `folded` also
records how many rows it retained against how many applications there were, since that ratio is
the whole reason to fold. Set ``STREAMING_LEGS`` to a comma-separated subset to run fewer. Records
land in ``output/benchmarks/metal-precursor-streaming/`` and are diagnostic, not paper-tracked.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
import pytest

from tests.vesma.bench_matcher import (
    _require_metal_reaction_matcher,
    load_match_corpus,
    load_positive_pairs,
)
from vesma import metal_reaction_matcher as mrm
from vesma.limits import ATOM_MASK_WORD_COUNT, GPU_PRECURSOR_ATOM_CAP
from vesma.metal_retro import probe_retro_pipelines

pytestmark = [
    pytest.mark.perf,
    pytest.mark.external_data,
    pytest.mark.gpu,
    pytest.mark.metal,
]

#: Host bytes one precursor row costs: the packed atom masks, the dense bond matrix, and the count.
#: This is the constant the accumulating path's footprint is proportional to, so it is derived from
#: the shared limits rather than written as 7684.
PRECURSOR_ROW_BYTES = (
    GPU_PRECURSOR_ATOM_CAP * ATOM_MASK_WORD_COUNT * 4  # atoms, uint32
    + GPU_PRECURSOR_ATOM_CAP * GPU_PRECURSOR_ATOM_CAP  # bonds, uint8
    + 4  # atom count, uint32
)

CORPUS_SIZE = int(os.environ.get("STREAMING_CORPUS", "2881"))
SIZES = tuple(
    int(piece)
    for piece in os.environ.get("STREAMING_SIZES", "1 10 100 1000 2881").split()
)
ROUNDS = int(os.environ.get("STREAMING_ROUNDS", "3"))
LEGS = tuple(
    leg.strip()
    for leg in os.environ.get("STREAMING_LEGS", "accumulate,stream,keyed,folded").split(
        ","
    )
    if leg.strip()
)
OUT_DIR = Path(
    os.environ.get("STREAMING_OUT", "output/benchmarks/metal-precursor-streaming")
)


# --------------------------------------------------------------------------- #
# Memory counters
# --------------------------------------------------------------------------- #
#: `vm_stat` lines worth differencing across a round. Compressions and swapouts are the ones that
#: say "capacity"; zero-filled pages are the ones that say "fresh allocation", and the two have to
#: be told apart because they imply different things about where the time went.
VM_STAT_KEYS = (
    "Pages free",
    "Pages active",
    "Pages inactive",
    "Pages wired down",
    "Pages zero filled",
    "Pages stored in compressor",
    "Pages occupied by compressor",
    "Decompressions",
    "Compressions",
    "Pageins",
    "Pageouts",
    "Swapins",
    "Swapouts",
)


def vm_stat() -> dict[str, int]:
    """Current `vm_stat` counters, in pages."""
    text = subprocess.run(
        ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True
    ).stdout
    out: dict[str, int] = {}
    for line in text.splitlines():
        label, _, value = line.partition(":")
        label = label.strip().strip('"')
        if label in VM_STAT_KEYS:
            out[label] = int(value.strip().rstrip("."))
    return out


def max_rss_bytes() -> int:
    """Peak resident size for this process. Darwin reports `ru_maxrss` in bytes already."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def rss_bytes() -> int:
    """*Current* resident size. `ru_maxrss` only ever rises, so it cannot show a live set
    shrinking back between rounds -- which is exactly what the streaming leg is meant to show.
    """
    return int(psutil.Process().memory_info().rss)


@dataclass(frozen=True)
class MemoryDelta:
    """What one timed round did to the machine's memory, not just to this process."""

    seconds: float
    max_rss_bytes_before: int
    max_rss_bytes_after: int
    rss_bytes_before: int
    rss_bytes_after: int
    page_size: int
    vm_stat_delta: dict[str, int]

    @property
    def compressor_pages(self) -> int:
        return self.vm_stat_delta.get("Compressions", 0)

    @property
    def swapout_pages(self) -> int:
        return self.vm_stat_delta.get("Swapouts", 0)


def measure(call: Callable[[], Any]) -> tuple[Any, MemoryDelta]:
    """Run `call` once, bracketed by the memory counters."""
    page_size = resource.getpagesize()
    rss_before = max_rss_bytes()
    live_before = rss_bytes()
    before = vm_stat()
    started = perf_counter()
    result = call()
    seconds = perf_counter() - started
    live_after = rss_bytes()
    after = vm_stat()
    return result, MemoryDelta(
        seconds=seconds,
        max_rss_bytes_before=rss_before,
        max_rss_bytes_after=max_rss_bytes(),
        rss_bytes_before=live_before,
        rss_bytes_after=live_after,
        page_size=page_size,
        vm_stat_delta={key: after[key] - before[key] for key in before},
    )


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StreamingSetup:
    corpus: Any
    pairs: list[tuple[int, int]]
    matcher: Any
    compile_seconds: float
    probe: dict[str, int]


@pytest.fixture(scope="module")
def setup() -> StreamingSetup:
    """One matcher for both legs, so the 1.1 GB reaction-table upload is paid once."""
    MetalReactionMatcher = _require_metal_reaction_matcher()
    corpus = load_match_corpus(CORPUS_SIZE)
    pairs = load_positive_pairs(corpus)
    started = perf_counter()
    matcher = MetalReactionMatcher([template.pattern for template in corpus.templates])
    compile_seconds = perf_counter() - started
    return StreamingSetup(
        corpus=corpus,
        pairs=pairs,
        matcher=matcher,
        compile_seconds=compile_seconds,
        probe=probe_retro_pipelines(),
    )


def slice_workload(
    setup: StreamingSetup, lo: int, hi: int
) -> tuple[tuple[Any, ...], list[tuple[int, int]]]:
    """The molecules in `[lo, hi)` and their positive pairs, renumbered to the slice.

    Pairs are `(mol_idx, rule_id)` as `MetalReactionMatcher.match` takes them, so only the
    molecule column needs shifting. Slicing by molecule rather than by pair keeps every slice a
    well-formed workload -- the same call a search would make with a smaller frontier.
    """
    cmols = setup.corpus.cmols[lo:hi]
    pairs = [(mol - lo, rule) for mol, rule in setup.pairs if lo <= mol < hi]
    return cmols, pairs


def record(name: str, payload: dict[str, Any], setup: StreamingSetup) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    host = os.environ.get("BENCH_HOST_LABEL", "unknown")
    path = OUT_DIR / f"{host}__{name}__{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "datetime": datetime.now(UTC).isoformat(),
                "leg": name,
                "host": host,
                "corpus_size": CORPUS_SIZE,
                "rounds": ROUNDS,
                "precursor_row_bytes": PRECURSOR_ROW_BYTES,
                "retro_assignments_per_dispatch": mrm._RETRO_ASSIGNMENTS_PER_DISPATCH,
                "assignment_buffer_budget": mrm._ASSIGNMENT_BUFFER_BUDGET,
                "compile_seconds": setup.compile_seconds,
                "retro_probe": setup.probe,
                **payload,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return path


def summarise(label: str, rows: Sequence[dict[str, Any]]) -> None:
    print(f"\n=== {label} ===")
    print(
        f"{'N':>6} {'prec':>9} {'best_s':>10} {'us/prec':>9} "
        f"{'prec/s':>10} {'live_GB':>8} {'peak_GB':>8} {'zerofill_GB':>11} "
        f"{'compr_pg':>10} {'swapout':>8}"
    )
    for row in rows:
        # The host's own page size, not a literal: vm_stat counts pages, and an Intel Metal host
        # pages at 4 KiB where Apple silicon pages at 16 KiB.
        zero_filled_bytes = row["zero_filled_pages"] * row["page_size"]
        print(
            f"{row['n']:>6} {row['precursors']:>9} "
            f"{row['best_seconds']:>10.4f} {row['us_per_precursor']:>9.2f} "
            f"{row['precursors_per_second']:>10.0f} "
            f"{row['live_rss_bytes'] / 2**30:>8.2f} "
            f"{row['peak_rss_bytes'] / 2**30:>8.2f} "
            f"{zero_filled_bytes / 2**30:>11.2f} "
            f"{row['compressor_pages']:>10} {row['swapout_pages']:>8}"
        )


def row_of(n: int, precursors: int, deltas: Sequence[MemoryDelta]) -> dict[str, Any]:
    best = min(delta.seconds for delta in deltas)
    return {
        "n": n,
        "precursors": precursors,
        "best_seconds": best,
        "mean_seconds": sum(delta.seconds for delta in deltas) / len(deltas),
        "all_seconds": [delta.seconds for delta in deltas],
        "us_per_precursor": 1e6 * best / precursors if precursors else 0.0,
        "precursors_per_second": precursors / best if best else 0.0,
        "live_result_bytes": precursors * PRECURSOR_ROW_BYTES,
        "page_size": deltas[0].page_size,
        "peak_rss_bytes": max(delta.max_rss_bytes_after for delta in deltas),
        "live_rss_bytes": max(delta.rss_bytes_after for delta in deltas),
        "zero_filled_pages": max(
            delta.vm_stat_delta.get("Pages zero filled", 0) for delta in deltas
        ),
        "compressor_pages": max(delta.compressor_pages for delta in deltas),
        "swapout_pages": max(delta.swapout_pages for delta in deltas),
        "memory": [asdict(delta) for delta in deltas],
    }


def _accumulate(setup: StreamingSetup, cmols: Any, pairs: Any) -> Callable[[], int]:
    """`match`, the joined array dropped. The raw cost of delivering one array."""

    def call() -> int:
        precursors, _hits = setup.matcher.match(cmols, pairs)
        produced = len(precursors)
        del precursors, _hits
        return produced

    return call


def _stream(setup: StreamingSetup, cmols: Any, pairs: Any) -> Callable[[], int]:
    """`match_batches`, each piece dropped as it arrives. The floor."""

    def call() -> int:
        produced = 0
        for batch, _hits in setup.matcher.match_batches(cmols, pairs):
            produced += len(batch)
        return produced

    return call


def _keyed(setup: StreamingSetup, cmols: Any, pairs: Any) -> Callable[[], int]:
    """`match`, then key the joined batch: what the search pays today."""
    from vesma.metal_reaction_matcher import metal_short_ring_wl_key

    def call() -> int:
        precursors, _hits = setup.matcher.match(cmols, pairs)
        keys = metal_short_ring_wl_key(precursors)
        produced = len(keys)
        del precursors, _hits, keys
        return produced

    return call


def _folded(setup: StreamingSetup, cmols: Any, pairs: Any) -> Callable[[], int]:
    """`keyed_match_folding` over `match_batches`: what the search would pay.

    Keys each piece as it arrives and keeps one row per distinct molecule, so the comparison with
    `keyed` is like for like -- both end holding a keyed batch -- and the difference is whether
    every application's precursor was held on the way there.
    """
    from vesma.matcher import keyed_match_folding
    from vesma.metal_reaction_matcher import (
        metal_concat_batches,
        metal_short_ring_wl_key,
    )

    keyed = keyed_match_folding(
        lambda mols: setup.matcher.match_batches(mols, pairs),
        metal_short_ring_wl_key,
        metal_concat_batches,
    )

    def call() -> int:
        batch, keys, hits = keyed(cmols)
        # What the fold bought, recorded on the call so the row can carry it.
        call.retained = (len(batch), len(hits))  # type: ignore[attr-defined]
        # Return the *precursors produced*, not the rows kept, so `us_per_precursor` counts the
        # same thing in every leg. The rows kept are the memory story, not the denominator.
        produced = len(hits)
        del batch, keys, hits
        return produced

    call.retained = None  # type: ignore[attr-defined]
    return call


#: Leg name -> builder. Order is the order they are interleaved in, which only affects which leg
#: runs first within a round.
LEG_BUILDERS: dict[str, Callable[[StreamingSetup, Any, Any], Callable[[], int]]] = {
    "accumulate": _accumulate,
    "stream": _stream,
    "keyed": _keyed,
    "folded": _folded,
}


def test_delivery_sweep(setup: StreamingSetup) -> None:
    """Every selected leg at every size, **interleaved round by round**.

    One test rather than one per leg, and interleaved rather than leg-by-leg, because leg order
    was biasing the answer: running each leg's whole sweep in turn left the later legs measured on
    a machine the earlier ones had warmed, and at 100 molecules that alone reversed the sign of
    the `keyed`/`folded` comparison -- the fold measured 54% slower run last and 12% faster when
    the two alternated. Within a round every leg now sees the same machine, which is the only way
    a ratio between them means anything on this host (the same kernel on the same workload drifts
    a quarter between sessions).
    """
    legs = [name for name in LEG_BUILDERS if name in LEGS]
    if not legs:
        pytest.skip(f"no known leg in STREAMING_LEGS={LEGS}")
    rows: dict[str, list[dict[str, Any]]] = {leg: [] for leg in legs}

    for size in SIZES:
        cmols, pairs = slice_workload(setup, 0, size)
        calls = {leg: LEG_BUILDERS[leg](setup, cmols, pairs) for leg in legs}
        deltas: dict[str, list[MemoryDelta]] = {leg: [] for leg in legs}
        produced: dict[str, int] = dict.fromkeys(legs, 0)
        diagnostics: dict[str, dict[str, Any]] = {leg: {} for leg in legs}

        for round_index in range(ROUNDS + 1):  # one warmup round, then ROUNDS timed
            for leg in legs:
                count, delta = measure(calls[leg])
                produced[leg] = count
                if round_index:
                    deltas[leg].append(delta)
                    diagnostics[leg] = asdict(setup.matcher.last_diagnostics)

        for leg in legs:
            row = row_of(size, produced[leg], deltas[leg])
            row["leg"] = leg
            row["positive_pairs"] = len(pairs)
            row["diagnostics"] = diagnostics[leg]
            row.update(_dedup_fields(getattr(calls[leg], "retained", None)))
            rows[leg].append(row)

    for leg in legs:
        summarise(
            f"{leg}: one call per size, interleaved with {len(legs) - 1} other leg(s)",
            rows[leg],
        )
        print(
            "wrote", record(leg, {"rows": rows[leg], "interleaved_with": legs}, setup)
        )

    _compare(rows, legs)


def _dedup_fields(counts: tuple[int, int] | None) -> dict[str, Any]:
    if counts is None:
        return {}
    rows, hits = counts
    return {
        "retained_rows": rows,
        "hit_rows": hits,
        "dedup_ratio": hits / rows if rows else 0.0,
        "retained_result_bytes": rows * PRECURSOR_ROW_BYTES,
    }


def _compare(rows: dict[str, list[dict[str, Any]]], legs: Sequence[str]) -> None:
    """The ratio the run exists to produce, printed only for legs that both ran."""
    for faster, baseline in (("stream", "accumulate"), ("folded", "keyed")):
        if faster not in legs or baseline not in legs:
            continue
        print(f"\n=== {faster} against {baseline} ===")
        print(f"{'N':>6} {baseline:>12} {faster:>12} {'ratio':>8} {'compr saved':>12}")
        for fast_row, base_row in zip(rows[faster], rows[baseline], strict=True):
            ratio = base_row["best_seconds"] / fast_row["best_seconds"]
            print(
                f"{fast_row['n']:>6} {base_row['best_seconds']:>12.4f} "
                f"{fast_row['best_seconds']:>12.4f} {ratio:>7.2f}x "
                f"{base_row['compressor_pages'] - fast_row['compressor_pages']:>12,}"
            )
