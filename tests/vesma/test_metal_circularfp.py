"""Metal circular fingerprints and nearest-stock distances match packed host results.

:func:`vesma.packed_circularfp.nearest_stock_distance` is the scan specification and
:func:`vesma.metal_circularfp.nearest_stock_distance_metal` moves its popcounts to the GPU. The
claim these tests defend is stronger than "agrees to within tolerance": the two are **bit-identical**,
because the kernel picks its nearest neighbour by exact integer cross-multiplication rather than by
comparing quotients, and because the shader is compiled with safe math so its one division is
correctly rounded. Nothing downstream tolerates less. A search that scores its frontier on the GPU
and its stock through the host path would otherwise be measuring distance to a catalogue that is not
quite the one solvedness checks, and the discrepancy would show up as route quality rather than as
an error.

Tail cases carry most of the weight here. The kernel tiles both sides and pairs candidate rows, so
there are four independent ways to fall off an end — a query batch that does not fill its tile, a
catalogue shorter than one stock tile, an odd number of catalogue rows, and a fingerprint width that
is not a whole number of ``uint4`` — and each of them is silent rather than loud when wrong.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_cmol
from vesma.circularfp import CircularFingerprintConfig
from vesma.metal_circularfp import (
    METAL_QUERY_TILE_BYTES,
    compute_circular_fingerprints_metal,
    metal_circularfp_available,
    metal_tiling,
    nearest_stock_distance_metal,
    nearest_stock_similarity_metal,
    next_stock_token,
    packed_bit_counts,
)
from vesma.metal_matcher import _pack_atom_masks, _pack_bond_matrix
from vesma.packed_circularfp import (
    PACKED_BUCKET,
    circular_fingerprints_packed,
    nearest_stock_distance,
)

pytestmark = [pytest.mark.gpu, pytest.mark.metal]

MOLECULES = (
    "C",
    "CC",
    "CCCCCCCC",
    "c1ccccc1",
    "CC(=O)OC",
    "COC(=O)CC(=O)OC",
    "O=C(O)c1ccc(N)cc1",
    "C1CNCCN1",
    "Cn1nc(C2CCC(=O)NC2=O)c2ccc(N3CCNCC3)cc21",
    "CC(C)(C)OC(=O)N1CCOCC(O)(c2ccc(Br)cc2)C1",
    "c1ccc2[nH]ccc2c1",
    "FC(F)(F)c1ccncc1",
    "OS(=O)(=O)c1ccccc1",
    "CC.O",
)


def skip_if_no_metal_device() -> None:
    if not metal_circularfp_available():
        pytest.skip("Metal backend not available")


def molecule_fingerprints(n_bits: int = 2048) -> np.ndarray:
    """Real circular fingerprints, so the bit densities are a catalogue's and not a coin's."""
    gmols = [gmol_from_cmol(cmol_from_smiles(smi)[0]) for smi in MOLECULES]
    return circular_fingerprints_packed(
        np.stack([_pack_atom_masks(gmol, PACKED_BUCKET) for gmol in gmols]),
        np.stack([_pack_bond_matrix(gmol, PACKED_BUCKET) for gmol in gmols]),
        np.array([gmol.atom_count for gmol in gmols], dtype=np.uint32),
        CircularFingerprintConfig(n_bits=n_bits, radius=2),
    )


@pytest.mark.parametrize("n_bits", [64, 512, 2048])
@pytest.mark.parametrize("radius", [0, 1, 2, 3])
def test_metal_circular_fingerprints_match_packed_host(
    n_bits: int, radius: int
) -> None:
    skip_if_no_metal_device()
    gmols = [gmol_from_cmol(cmol_from_smiles(smi)[0]) for smi in MOLECULES]
    atoms = np.stack([_pack_atom_masks(gmol, PACKED_BUCKET) for gmol in gmols])
    bonds = np.stack([_pack_bond_matrix(gmol, PACKED_BUCKET) for gmol in gmols])
    counts = np.array([gmol.atom_count for gmol in gmols], dtype=np.uint32)
    config = CircularFingerprintConfig(n_bits=n_bits, radius=radius)
    assert np.array_equal(
        compute_circular_fingerprints_metal(atoms, bonds, counts, config),
        circular_fingerprints_packed(atoms, bonds, counts, config),
    )


def random_fingerprints(rows: int, words: int, seed: int) -> np.ndarray:
    """Sparse random bits: an AND of two draws, which lands near a fingerprint's own density."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**64, size=(rows, words), dtype=np.uint64) & rng.integers(
        0, 2**64, size=(rows, words), dtype=np.uint64
    )


def test_the_scan_is_bit_identical_to_the_host_on_real_fingerprints() -> None:
    skip_if_no_metal_device()
    fingerprints = molecule_fingerprints()
    stock, queries = fingerprints[:8], fingerprints[5:]
    assert np.array_equal(
        nearest_stock_distance_metal(
            queries, stock, weight=1.0, stock_token=next_stock_token()
        ),
        nearest_stock_distance(queries, stock, weight=1.0),
    )


@pytest.mark.parametrize("n_bits", [64, 128, 256, 512, 2048])
def test_the_scan_is_bit_identical_at_every_width(n_bits: int) -> None:
    """Widths that are not a whole number of ``uint4`` are padded, which must not move the answer.

    64 bits is one ``uint64`` word and so half a ``uint4``; the padding word is zero, which changes
    neither the intersection nor the union, and this is what says so.
    """
    skip_if_no_metal_device()
    stock = random_fingerprints(400, n_bits // 64, seed=n_bits)
    queries = random_fingerprints(97, n_bits // 64, seed=n_bits + 1)
    assert np.array_equal(
        nearest_stock_distance_metal(
            queries, stock, weight=1.0, stock_token=next_stock_token()
        ),
        nearest_stock_distance(queries, stock, weight=1.0),
    )


# 1 and 2 straddle the paired-row loop; 3 and 17 leave it an odd row to finish alone; 512 and 513
# straddle the smallest stock tile the tiling will cut; 4097 forces several tiles per threadgroup.
@pytest.mark.parametrize("stock_rows", [1, 2, 3, 17, 512, 513, 4097])
# 1 and 31 are a part-filled 64-query tile, 64 fills one exactly, 65 spills into a second.
@pytest.mark.parametrize("query_rows", [1, 31, 64, 65, 300])
def test_the_scan_is_bit_identical_at_every_tile_boundary(
    stock_rows: int, query_rows: int
) -> None:
    skip_if_no_metal_device()
    stock = random_fingerprints(stock_rows, 32, seed=stock_rows)
    queries = random_fingerprints(query_rows, 32, seed=query_rows + 9000)
    assert np.array_equal(
        nearest_stock_distance_metal(
            queries, stock, weight=2.5, stock_token=next_stock_token()
        ),
        nearest_stock_distance(queries, stock, weight=2.5),
    )


def test_a_stock_molecule_is_at_distance_zero_from_the_stock() -> None:
    """The property :class:`vertsy.heuristics.Heuristic` requires: the estimate vanishes there."""
    skip_if_no_metal_device()
    fingerprints = molecule_fingerprints()
    assert nearest_stock_distance_metal(
        fingerprints, fingerprints, weight=3.0, stock_token=next_stock_token()
    ) == pytest.approx(np.zeros(len(fingerprints)))


def test_an_all_zero_pair_shares_nothing_rather_than_being_a_nan() -> None:
    """An empty union divides zero by zero; both paths answer "shares nothing" instead."""
    skip_if_no_metal_device()
    empty = np.zeros((4, 32), dtype=np.uint64)
    assert nearest_stock_distance_metal(
        empty, empty, weight=1.0, stock_token=next_stock_token()
    ) == pytest.approx(np.ones(4))


def test_the_resident_stock_is_reused_and_not_confused() -> None:
    """The device-side cache must be a cache: same token, same catalogue, same answer.

    A token is an identity the caller mints, so the failure this guards against is not a wrong
    number from one call but a *stale* one from the second — the first upload answering for a
    catalogue it never saw.
    """
    skip_if_no_metal_device()
    first = random_fingerprints(500, 32, seed=1)
    second = random_fingerprints(500, 32, seed=2)
    queries = random_fingerprints(80, 32, seed=3)
    first_token, second_token = next_stock_token(), next_stock_token()

    expected_first = nearest_stock_distance(queries, first, weight=1.0)
    expected_second = nearest_stock_distance(queries, second, weight=1.0)
    for _ in range(2):
        assert np.array_equal(
            nearest_stock_distance_metal(
                queries, first, weight=1.0, stock_token=first_token
            ),
            expected_first,
        )
        assert np.array_equal(
            nearest_stock_distance_metal(
                queries, second, weight=1.0, stock_token=second_token
            ),
            expected_second,
        )


def test_concurrent_scans_do_not_exchange_resident_stocks() -> None:
    skip_if_no_metal_device()
    first = random_fingerprints(1_000, 32, seed=101)
    second = random_fingerprints(1_000, 32, seed=102)
    queries = random_fingerprints(32, 32, seed=103)
    stocks = (first, second)
    tokens = (next_stock_token(), next_stock_token())
    expected = tuple(
        nearest_stock_distance(queries, stock, weight=1.0) for stock in stocks
    )

    indices = (0, 1) * 8
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                nearest_stock_distance_metal,
                queries,
                stocks[index],
                weight=1.0,
                stock_token=tokens[index],
            )
            for index in indices
        ]
        actual = tuple(future.result() for future in futures)
    for index, distances in zip(indices, actual, strict=True):
        assert np.array_equal(distances, expected[index])


def test_a_token_reused_for_a_differently_shaped_catalogue_is_refused() -> None:
    """The one caller error the device cache can catch, so it does rather than answering."""
    skip_if_no_metal_device()
    token = next_stock_token()
    queries = random_fingerprints(8, 32, seed=5)
    nearest_stock_similarity_metal(
        queries, random_fingerprints(100, 32, seed=6), stock_token=token
    )
    with pytest.raises(RuntimeError, match="was uploaded as 100 rows"):
        nearest_stock_similarity_metal(
            queries, random_fingerprints(200, 32, seed=7), stock_token=token
        )


def test_a_zero_token_always_reuploads_and_stays_correct() -> None:
    """The escape hatch for a caller with no stable identity to offer, and it must still be exact."""
    skip_if_no_metal_device()
    first = random_fingerprints(300, 32, seed=11)
    second = random_fingerprints(300, 32, seed=12)
    queries = random_fingerprints(40, 32, seed=13)
    for stock in (first, second, first):
        assert np.array_equal(
            nearest_stock_distance_metal(queries, stock, weight=1.0, stock_token=0),
            nearest_stock_distance(queries, stock, weight=1.0),
        )


def test_an_empty_query_batch_scores_nothing() -> None:
    skip_if_no_metal_device()
    stock = random_fingerprints(4, 32, seed=17)
    assert nearest_stock_distance_metal(
        np.zeros((0, 32), dtype=np.uint64),
        stock,
        weight=1.0,
        stock_token=next_stock_token(),
    ).shape == (0,)


@pytest.mark.parametrize(
    ("queries", "stock", "match"),
    [
        (np.zeros((2, 16), np.uint64), np.zeros((4, 32), np.uint64), "widths differ"),
        (
            np.zeros((2, 32), np.uint64),
            np.zeros((0, 32), np.uint64),
            "must not be empty",
        ),
        (np.zeros((2, 32), np.float64), np.zeros((4, 32), np.uint64), "dtype uint64"),
        (np.zeros(32, np.uint64), np.zeros((4, 32), np.uint64), "two-dimensional"),
    ],
)
def test_the_same_inputs_the_host_refuses_are_refused_here(
    queries: np.ndarray, stock: np.ndarray, match: str
) -> None:
    """Signature-compatible means the errors match too, or a backend swap changes behaviour."""
    skip_if_no_metal_device()
    with pytest.raises(ValueError, match=match):
        nearest_stock_similarity_metal(queries, stock, stock_token=0)


def test_a_negative_weight_is_refused() -> None:
    skip_if_no_metal_device()
    stock = random_fingerprints(4, 32, seed=19)
    with pytest.raises(ValueError, match="weight must be non-negative"):
        nearest_stock_distance_metal(stock, stock, weight=-1.0, stock_token=0)


def test_bit_counts_are_the_union_identitys_other_half() -> None:
    """Passing them in must be the same as letting the scan derive them."""
    skip_if_no_metal_device()
    stock = random_fingerprints(600, 32, seed=23)
    queries = random_fingerprints(70, 32, seed=24)
    token = next_stock_token()
    assert np.array_equal(
        nearest_stock_similarity_metal(
            queries, stock, stock_token=token, stock_bit_counts=packed_bit_counts(stock)
        ),
        nearest_stock_similarity_metal(queries, stock, stock_token=token),
    )


def test_wrongly_sized_bit_counts_are_refused() -> None:
    skip_if_no_metal_device()
    stock = random_fingerprints(20, 32, seed=29)
    with pytest.raises(ValueError, match="one entry per stock row"):
        nearest_stock_similarity_metal(
            stock,
            stock,
            stock_token=0,
            stock_bit_counts=np.zeros(19, dtype=np.uint32),
        )


@pytest.mark.parametrize("word_count", [1, 2, 8, 32])
def test_the_tiling_fits_the_threadgroup_budget(word_count: int) -> None:
    """The query tile is a memory decision, and exceeding the budget is a launch failure."""
    tile, _ = metal_tiling(20_000, 988_086, word_count)
    assert tile % 32 == 0, "a tile that is not a multiple of the simd width idles lanes"
    assert tile * word_count * 8 <= METAL_QUERY_TILE_BYTES or tile == 32


def test_a_tiny_frontier_still_fills_the_device() -> None:
    """Why the stock is tiled at all: AO*'s first call scores one molecule, not twenty thousand.

    With query tiles alone that one molecule would run in a single threadgroup on an eight-core GPU.
    Splitting the catalogue is what turns it back into a parallel problem, and the split has to be
    *large* — a couple of tiles would not do it.
    """
    tile, rows_per_group = metal_tiling(1, 988_086, 32)
    groups = -(-988_086 // rows_per_group)
    assert groups >= 256, f"one query spread over only {groups} threadgroups"
    # And a full frontier already supplies the parallelism, so it should not be split much.
    _, full_rows = metal_tiling(19_296, 988_086, 32)
    assert -(-19_296 // tile) * -(-988_086 // full_rows) < 2048
