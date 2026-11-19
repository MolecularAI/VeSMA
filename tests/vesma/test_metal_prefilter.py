import numpy as np
import pytest

from tests.datasets import load_retro_benchmark_corpus
from vesma.metal_matcher import (
    GpuUnavailableError,
    MetalFoldedPrefilteredMatcher,
    MetalMatcher,
    probe_metal,
)

pytestmark = [pytest.mark.slow, pytest.mark.gpu, pytest.mark.metal]


def skip_if_no_metal_device() -> None:
    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")


def test_metal_prefilter_no_false_negatives() -> None:
    """Verify that the Metal folded prefilter produces exact bit-for-bit identical results
    to the raw MetalMatcher on the pinned generated retro-benchmark corpus.
    This guarantees no false negatives (or false positives) are introduced by the prefilter.
    """
    skip_if_no_metal_device()
    corpus = load_retro_benchmark_corpus()

    gmols = [molecule.molecule for molecule in corpus.molecules]
    patterns = [pattern.query for pattern in corpus.patterns]

    # 1. Run raw MetalMatcher
    raw_matcher = MetalMatcher()
    raw_matches = raw_matcher.has_match_many(patterns, gmols)

    # 2. Run prefiltered MetalMatcher
    prefiltered_matcher = MetalFoldedPrefilteredMatcher()
    prefiltered_matches = prefiltered_matcher.has_match_many(patterns, gmols)

    # 3. Assert shapes match
    assert raw_matches.shape == prefiltered_matches.shape

    # 4. Assert identically identical results
    differences = np.sum(raw_matches != prefiltered_matches)

    if differences > 0:
        false_negatives = np.sum(raw_matches & ~prefiltered_matches)
        false_positives = np.sum(~raw_matches & prefiltered_matches)
        pytest.fail(
            f"Found {differences} mismatched pairs! "
            f"False negatives (missed by prefilter): {false_negatives}, "
            f"False positives (added by prefilter): {false_positives}"
        )

    # Also assert at least some matches were found to ensure the test is actually meaningful
    assert raw_matches.sum() > 0
