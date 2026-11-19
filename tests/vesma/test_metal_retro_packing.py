from __future__ import annotations

import numpy as np

from vesma.bonds import BondKind
from vesma.limits import METAL_EDITED_ATOM_CAP
from vesma.metal_retro import reaction_data_from_pattern
from vesma.reaction import parse_reaction_smarts


def test_reaction_data_from_pattern_uses_reaction_edit_fields() -> None:
    reaction = parse_reaction_smarts("[C:1][O][C:2]>>[C:1]-[C:2].[N:3]")
    data = reaction_data_from_pattern(reaction)

    assert data.deleted_lhs_count == 1
    assert data.deleted_lhs_indices[0] == 1
    assert data.common_count == 2
    assert data.common_maps[0] == 1
    assert data.common_lhs_indices[1] == 2
    assert data.common_rhs_indices[1] == 1
    assert data.lhs_edge_count == 2
    assert data.rhs_edge_count == 1
    assert data.rhs_edge_left[0] == 0
    assert data.rhs_edge_right[0] == 1
    assert data.rhs_edge_bond_types[0] == int(BondKind.SINGLE)


def test_reaction_data_packs_rhs_atoms_atom_major() -> None:
    reaction = parse_reaction_smarts("[C:1]>>[C:1]-[O:2]")
    data = reaction_data_from_pattern(reaction)

    rhs_atoms = np.ctypeslib.as_array(data.rhs_atoms).reshape(METAL_EDITED_ATOM_CAP, 14)

    assert int(rhs_atoms[0, 0]) == 1 << 6
    assert int(rhs_atoms[1, 0]) == 1 << 8


def test_reaction_data_packs_zero_common_map_reactions() -> None:
    reaction = parse_reaction_smarts("[C:1]>>[O:2]")
    data = reaction_data_from_pattern(reaction)

    assert data.common_count == 0
    assert data.deleted_lhs_count == 1
    assert data.deleted_lhs_indices[0] == 0
    assert data.rhs_atom_maps[0] == 2


def test_metal_retro_packs_aromatic_reactions() -> None:
    # Aromatic reactions are no longer rejected: the kernel re-perceives
    # aromaticity after bond edits, so packing must succeed.
    reaction = parse_reaction_smarts("[c:1]:[c:2]>>[c:1].[c:2]")

    data = reaction_data_from_pattern(reaction)

    assert data.common_count == 2
    assert data.lhs_edge_count == 1
    assert data.rhs_atom_count == 2


def test_adjacency_bitsets_set_the_bit_the_neighbour_index_names() -> None:
    """Bit ``j`` of atom ``i``'s word means atom ``j`` is bonded to atom ``i``, and nothing else.

    `_adjacency_rows_from_bonds` is a `packbits`, which is the fast spelling but also the one with a
    byte-order argument that is wrong by default: `bitorder="big"` reverses each group of eight, so
    a molecule's neighbours would be permuted within every byte and the matcher would prune against
    an adjacency that is not the molecule's. That failure is silent -- the bit *count* is right, so
    a popcount check passes -- which is why this asserts the positions.

    Checked against the weighted-sum form the function used to be, over a random sparse batch, since
    that form was obviously correct and merely slow. The batch carries every `BondKind` union rather
    than only `SINGLE`, because the fast form packs the bond bytes without an `!= 0` and so relies on
    `packbits` setting a bit for any nonzero byte rather than testing its low bit -- which `DOUBLE`
    (2), `TRIPLE` (4) and `DELOCALIZED` (8) would each silently fail.
    """
    from vesma.metal_reaction_matcher import _BUCKET, _adjacency_rows_from_bonds

    bonds = np.zeros((1, _BUCKET * _BUCKET), dtype=np.uint8)
    bonds[0, 0 * _BUCKET + 5] = np.uint8(BondKind.SINGLE)
    bonds[0, 5 * _BUCKET + 0] = np.uint8(BondKind.SINGLE)
    bonds[0, 3 * _BUCKET + 63] = np.uint8(BondKind.DOUBLE)

    rows = _adjacency_rows_from_bonds(bonds)
    assert rows.shape == (1, _BUCKET)
    assert rows[0, 0] == np.uint64(1) << np.uint64(5)
    assert rows[0, 5] == np.uint64(1) << np.uint64(0)
    assert rows[0, 3] == np.uint64(1) << np.uint64(63)
    assert rows[0, 1] == np.uint64(0)

    rng = np.random.default_rng(20261007)
    kinds = np.array(
        [BondKind.SINGLE, BondKind.DOUBLE, BondKind.TRIPLE, BondKind.DELOCALIZED],
        dtype=np.uint8,
    )
    batch = np.zeros((64, _BUCKET * _BUCKET), dtype=np.uint8)
    for row in range(batch.shape[0]):
        where = rng.integers(0, _BUCKET * _BUCKET, 40)
        batch[row, where] = kinds[rng.integers(0, kinds.size, where.size)]
    bits = np.left_shift(np.uint64(1), np.arange(_BUCKET, dtype=np.uint64))
    present = (batch.reshape(-1, _BUCKET, _BUCKET) != 0).astype(np.uint64)
    weighted = (present * bits[None, None, :]).sum(axis=2, dtype=np.uint64)
    assert np.array_equal(_adjacency_rows_from_bonds(batch), weighted)


def test_metal_concat_batches_joins_nothing_into_the_empty_batch() -> None:
    """`concat` of no parts is the empty batch, not an exception.

    `matcher.keyed_match_folding` takes the backend's own join as its `concat` and calls it with
    no parts when a layer produced no precursors at all -- a frontier the rules do not touch, or
    one whose pairs all screened out. `np.concatenate` refuses an empty sequence, so without this
    case an empty answer arrived as a crashed search. `cuda_concat_mols` has always preserved the
    empty batch; this is the Metal counterpart.
    """
    from vesma.matcher import keyed_match_folding
    from vesma.metal_reaction_matcher import (
        metal_concat_batches,
        metal_stack_mols,
    )

    assert len(metal_concat_batches([])) == 0

    # And through the fold, which is the caller that actually does this.
    folded = keyed_match_folding(
        lambda _mols: iter(()),
        lambda batch: [()] * len(batch),
        metal_concat_batches,
    )
    batch, keys, hits = folded(metal_stack_mols([]))
    assert len(batch) == 0
    assert list(keys) == []
    assert hits.shape == (0, 4)
