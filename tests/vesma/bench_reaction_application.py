"""Stream usable, unkeyed precursor batches to the host; release each after counting.

Input parsing, rule compilation and positive-pair selection are outside the clock.
Matching, editing, splitting, perception and host delivery are inside it. No identity
keys, canonical SMILES, deduplication or search. RDChiral is intentionally excluded.
Run one backend/size per process through the Makefile for attributable parent RSS.
"""

from __future__ import annotations

import hashlib
import math
import os
import resource
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from tests.datasets import chemistry_fingerprint
from tests.vesma.bench_match_scaling import rounds, warmup_rounds
from tests.vesma.bench_matcher import (
    _require_metal_reaction_matcher,
    load_match_corpus,
    load_positive_pairs,
    processing_names,
    template_smarts,
)
from vesma.cmol import CMol
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.limits import GPU_PRECURSOR_ATOM_CAP
from vesma.matcher import HitArray, Pair
from vesma.parallel import effective_n_jobs, resolve_n_jobs
from vesma.rdkit_matcher import RdkitReactionMatcher, rdkit_reactions_from_smarts
from vesma.reaction import DEFAULT_REACTION_PROCESSING

if TYPE_CHECKING:
    from vesma.cuda_matcher import CudaMoleculeHostBatch, CudaReactionMatcher

pytestmark = [pytest.mark.perf, pytest.mark.external_data]
CONTRACT = "streamed-unkeyed-host-v1"
BACKENDS = (
    "rdkit-serial",
    "rdkit-parallel",
    "cpu-serial",
    "cpu-parallel",
    "metal",
    "cuda-host",
)


def drain_batches(batches: Iterable[tuple[Any, HitArray]]) -> dict[str, int]:
    precursors = hits = chunks = maximum = 0
    for batch, rows in batches:
        count = len(batch)
        precursors += count
        hits += len(rows)
        chunks += 1
        maximum = max(maximum, count)
        # Release ownership before asking the generator to construct its next batch.
        del batch, rows
    return {
        "precursors": precursors,
        "hits": hits,
        "batches": chunks,
        "max_batch_precursors": maximum,
    }


def cuda_host_batches(
    matcher: CudaReactionMatcher, mols: Sequence[CMol], pairs: Sequence[Pair]
) -> Iterator[tuple[CudaMoleculeHostBatch, HitArray]]:
    """Copy each complete, unkeyed precursor batch to independently owned host arrays."""
    from vesma.cuda_matcher import CudaMoleculeHostBatch, CudaMoleculeHostPart

    for batch, hits in matcher.match_batches(mols, pairs):
        host = CudaMoleculeHostBatch((CudaMoleculeHostPart.from_cuda(batch, keys=()),))
        yield host, hits
        del host, batch, hits


@pytest.mark.parametrize(
    "backend", os.environ.get("APPLICATION_BACKENDS", "metal").split()
)
@pytest.mark.parametrize(
    "size",
    [
        int(item)
        for item in os.environ.get("APPLICATION_SIZES", "1 10 100 1000 2881").split()
    ],
)
def test_reaction_application(benchmark, backend: str, size: int) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported application backend: {backend}")
    corpus = load_match_corpus(size)
    pairs = load_positive_pairs(corpus)
    pair_batch_size = int(os.environ.get("APPLICATION_PAIRS_PER_BATCH", "4096"))
    gpu = backend in {"metal", "cuda-host"}
    jobs = 1 if backend.endswith("serial") else resolve_n_jobs(None)
    pair_batch_limit = pair_batch_size
    if backend.endswith("parallel"):
        # Fill the workers for small frontiers without relaxing the memory bound.
        pair_batch_size = min(
            pair_batch_limit,
            max(1, math.ceil(len(pairs) / (4 * effective_n_jobs(jobs)))),
        )
    started = perf_counter()
    matcher: Any
    if backend.startswith("rdkit"):
        matcher = RdkitReactionMatcher(
            rdkit_reactions_from_smarts(template_smarts(corpus)), n_jobs=jobs
        )
        mols = corpus.rd_mols
        kwargs = {"pairs_per_batch": pair_batch_size}
        worker_lifetime = "matcher"
    elif backend.startswith("cpu"):
        matcher = CpuReactionMatcher(
            [template.pattern for template in corpus.templates],
            max_precursor_atoms=GPU_PRECURSOR_ATOM_CAP,
        )
        mols = corpus.cmols
        kwargs = {"n_jobs": jobs, "pairs_per_batch": pair_batch_size}
        worker_lifetime = "application-call"
    elif backend == "cuda-host":
        cp = pytest.importorskip("cupy")
        if not cp.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        from vesma.cuda_matcher import CudaReactionMatcher

        matcher = CudaReactionMatcher(
            [template.pattern for template in corpus.templates], frontier_mode="host"
        )
        mols = corpus.cmols
        kwargs = {}
        worker_lifetime = "none"
    else:
        matcher = _require_metal_reaction_matcher()(
            [template.pattern for template in corpus.templates]
        )
        mols = corpus.cmols
        kwargs = {}
        worker_lifetime = "none"
    compile_seconds = perf_counter() - started

    def target() -> dict[str, int]:
        if backend == "cuda-host":
            result = drain_batches(cuda_host_batches(matcher, mols, pairs))
            cp.cuda.Stream.null.synchronize()
            return result
        return drain_batches(matcher.match_batches(mols, pairs, **kwargs))

    try:
        result = benchmark.pedantic(
            target, rounds=rounds(), warmup_rounds=warmup_rounds(), iterations=1
        )
        benchmark.extra_info.update(
            {
                "experiment": "reaction-application",
                "contract": CONTRACT,
                "chemistry_fingerprint": chemistry_fingerprint(),
                "backend": backend,
                "size": size,
                "molecules": len(mols),
                "rules": len(corpus.templates),
                "positive_pairs": len(pairs),
                "pair_sha256": hashlib.sha256(
                    np.asarray(pairs, dtype="<i8").tobytes()
                ).hexdigest(),
                "smiles_sha256": hashlib.sha256(
                    "\n".join(corpus.smiles).encode()
                ).hexdigest(),
                "templates_sha256": hashlib.sha256(
                    "\n".join(template_smarts(corpus)).encode()
                ).hexdigest(),
                "precursor_identity": "none",
                "delivery": "host-batches-released",
                "compile_seconds": compile_seconds,
                "pairs_per_batch": pair_batch_size if not gpu else None,
                "pairs_per_batch_limit": (pair_batch_limit if not gpu else None),
                "workers": effective_n_jobs(jobs) if not gpu else 0,
                "worker_lifetime": worker_lifetime,
                "precursor_atom_cap": (
                    None if backend.startswith("rdkit") else GPU_PRECURSOR_ATOM_CAP
                ),
                "reaction_processing": (
                    {
                        "before_match": processing_names(
                            DEFAULT_REACTION_PROCESSING.before_match
                        ),
                        "after_split": processing_names(
                            DEFAULT_REACTION_PROCESSING.after_split
                        ),
                    }
                    if not backend.startswith("rdkit")
                    else "rdkit-sanitize-fragments"
                ),
                "diagnostics": asdict(matcher.last_diagnostics),
                "peak_parent_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "rss_scope": "parent-process-high-water-including-setup; not-total-process-tree-or-device-memory",
                **result,
            }
        )
    finally:
        if backend.startswith("rdkit"):
            matcher.close()
