"""Compare retained keyed CUDA frontiers; separate from unkeyed host delivery."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pytest

from tests.datasets import chemistry_fingerprint
from tests.vesma.bench_match_scaling import rounds, warmup_rounds
from tests.vesma.bench_matcher import load_match_corpus, load_positive_pairs
from vesma.cmol import CMol
from vesma.cuda_matcher import CudaReactionMatcher, import_cupy
from vesma.matcher import Pair

pytestmark = [pytest.mark.perf, pytest.mark.cuda, pytest.mark.external_data]


def run_frontier(
    matcher: CudaReactionMatcher, mols: Sequence[CMol], pairs: Sequence[Pair]
) -> dict[str, int]:
    cp = import_cupy()

    batch, keys, hits = matcher.match_keyed(mols, pairs)
    result = {"unique_precursors": len(batch), "keys": len(keys), "hits": len(hits)}
    del batch, keys, hits
    cp.cuda.Stream.null.synchronize()
    return result


def test_cuda_frontier_residency(benchmark) -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mode = os.environ.get("FRONTIER_MODE", "device")
    if mode not in {"host", "device"}:
        raise ValueError(f"Unknown frontier mode: {mode}")
    size = int(os.environ.get("FRONTIER_SIZE", "2881"))
    cap = int(os.environ.get("FRONTIER_PAIR_CAP", "0"))
    corpus = load_match_corpus(size)
    pairs = load_positive_pairs(corpus)
    matcher = CudaReactionMatcher(
        [template.pattern for template in corpus.templates],
        frontier_mode="host" if mode == "host" else "device",
        max_pairs_per_launch=cap or None,
    )
    result = benchmark.pedantic(
        run_frontier,
        args=(matcher, corpus.cmols, pairs),
        rounds=rounds(),
        warmup_rounds=warmup_rounds(),
        iterations=1,
    )
    assert matcher.last_diagnostics is not None
    benchmark.extra_info.update(
        experiment="cuda-frontier-residency",
        contract="keyed-frontier-retained-v1",
        backend=f"cuda-{mode}",
        molecules=size,
        rules=len(corpus.templates),
        positive_pairs=len(pairs),
        pair_sha256=hashlib.sha256(
            np.asarray(pairs, dtype="<i8").tobytes()
        ).hexdigest(),
        chemistry_fingerprint=chemistry_fingerprint(),
        max_pairs_per_launch=matcher.max_pairs_per_launch,
        diagnostics=asdict(matcher.last_diagnostics),
        **result,
    )
