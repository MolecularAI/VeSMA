"""Two contracts about state that is shared rather than owned.

Both bugs here are silent: nothing raises, the wrong answer just comes back. That is what makes
them worth pinning — a crash would have found itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesma.metal_reaction_matcher import MetalMoleculeBatch, metal_concat_batches
from vesma.parallel import ResidentForkPool

# ------------------------------------------------------- serial owners share module globals

STATE: list[str] = []


def publish(name: str) -> None:
    """Stand in for an initializer that writes module globals, which is what they all do."""
    STATE.clear()
    STATE.append(name)


def observe(_item: int) -> str:
    return STATE[0]


def test_each_serial_owner_reinitialises_before_it_maps() -> None:
    """Two inline pools in one process must not read each other's state.

    ``n_jobs=1`` maps in the calling process, so an initializer's globals are shared with every
    other owner rather than private to a fork. Initialising once per owner would leave whichever
    ran last in place, and the other owner would then quietly compute against the wrong rules —
    for a matcher, the wrong chemistry, with no error to notice.
    """
    first = ResidentForkPool(initializer=publish, initargs=("first",), n_jobs=1)
    second = ResidentForkPool(initializer=publish, initargs=("second",), n_jobs=1)

    assert first.map(observe, [0]) == ("first",)
    assert second.map(observe, [0]) == ("second",)
    # The one that matters: going back to an owner that has already mapped once.
    assert first.map(observe, [0]) == ("first",)
    assert second.map(observe, [0]) == ("second",)


# ------------------------------------------------- Metal keys are per molecule, all or nothing


def batch(count: int) -> MetalMoleculeBatch:
    return MetalMoleculeBatch(
        atom_counts=np.ones(count, dtype=np.int64),
        atoms=np.zeros((count, 4), dtype=np.int64),
        bonds=np.zeros((count, 4), dtype=np.int64),
    )


def test_concatenating_batches_concatenates_their_molecules() -> None:
    """What concat promises, now that identity is not a field it has to reconcile.

    Three tests used to live here about reconciling `keys` across parts: a keyless frontier joined
    to keyed precursors had to drop the lot, because a tuple shorter than the batch hands a
    molecule its neighbour's key; an empty keyless part had to be exempted from that rule; and
    keyed parts had to concatenate in order. All three were consequences of identity riding on the
    batch, and all three are gone with it -- a batch is molecules, and `metal_short_ring_wl_key`
    keys whatever batch it is handed, so there is nothing ragged to guard against.
    """
    joined = metal_concat_batches([batch(1), batch(2)])

    assert len(joined) == 3


def test_an_empty_batch_contributes_no_rows() -> None:
    joined = metal_concat_batches([batch(0), batch(2)])

    assert len(joined) == 2


@pytest.mark.parametrize("count", [1, 3])
def test_a_single_batch_is_returned_unchanged(count: int) -> None:
    only = batch(count)
    assert metal_concat_batches([only]) is only
