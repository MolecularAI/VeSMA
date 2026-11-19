"""Checks on whether the heuristic is steering, and one repair for when it is not.

Every measured failure of this search so far has been a heuristic that was *silently* useless
rather than wrong: the estimate stayed admissible, the ranking stayed well-formed, and the
frontier filled with molecules the fingerprint could not tell apart. A Retro\\* target interned
540,038 molecules over sixteen passes without solving, and 127 of every 128 picks carried an
estimate of exactly zero -- against a catalogue that did not contain them. Nothing raised,
nothing warned, and the run reported a plausible mean estimate of 0.083.

So the checks here answer three different questions, and they sit in two different places
because that is where each question is free to ask.

**The fold sees leaf values and the stock mask**, so it can compare the estimate against
whether the molecule is buyable at all. :func:`auditing_fold` does, and it is the only check
that raises -- see below. :func:`floored` is the repair, and lives here because it is the same
wrapper shape over the same argument.

**Selection sees the ranking**, which the fold does not, so it is the only place a *plateau*
is visible. :func:`auditing_select` reports how much of a pass was decided by a tie: if more
leaves tie with the width-th best than there are slots, the picks are drawn arbitrarily out of
that tie and no reweighting of the terms can change which ones win. This generalises the
zero-estimate case -- those 127 leaves are a tie set -- and catches plateaus that sit
somewhere other than zero, which a test for exact zero cannot see.

Why one check raises and the others only warn
---------------------------------------------
It is tempting to raise on an inadmissible estimate, and wrong: **this search is run
inadmissibly on purpose.** A step cost of 0.001 against a ``[0, 1]`` heuristic is weighted A\\*
at an effective weight of 1000, which is the setting every measurement in the plan file was
taken at. Admissibility is not a live guarantee here, so breaking it cannot be an error.

An estimate above zero on a molecule that *is* in the stock is a different kind of fact. Equal
molecules produce equal fingerprints, so a stock molecule's distance to the nearest stock
molecule -- itself -- is zero by construction. A positive value there is not a tuning choice
and not a weighting artefact; it means the heuristic and the search disagree about what the
catalogue contains, and every solvedness decision downstream is being made against a different
set than the estimates. That is a representation invariant, so :func:`auditing_fold` raises.

The converse -- zero on a molecule the stock does *not* contain -- stays a warning, because it
is the fingerprint being coarser than molecular identity, which is what fingerprints are for.
Binary ECFP4 assigns ``C=CCCCCCCO`` and ``C=CCCCCCCCCCO`` the same sixteen bits, since every
interior methylene of a chain has the same radius-2 environment; past about five carbons chain
length is invisible. The estimate is still admissible. It is simply not information, and a
count fingerprint is the fix rather than an exception.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from .aostar import Fold, Rank, Select, by_total, reachable_leaves
from .core import K
from .costs import StepPrice
from .metrics import TROPICAL, GraphState, Valuation

__all__ = ["Report", "auditing_fold", "auditing_select", "floored", "warn"]


# Where a finding goes. A callable rather than the `warnings` module directly, so a benchmark
# can collect findings into a row and a test can assert on them without `pytest.warns`.
type Report = Callable[[str], None]


def warn(message: str) -> None:
    """Default :data:`Report` -- one ``UserWarning``, at the caller's stacklevel."""
    warnings.warn(message, UserWarning, stacklevel=3)


def _stock_mask(state: GraphState[K], count: int) -> np.ndarray:
    mask = np.zeros(count, dtype=np.bool_)
    if state.stock_sids:
        inside = [sid for sid in state.stock_sids if sid < count]
        if inside:
            mask[np.asarray(inside, dtype=np.int64)] = True
    return mask


def auditing_fold(fold: Fold[K], *, report: Report = warn) -> Fold[K]:
    """Wrap ``fold``, checking the estimates it is handed against what the stock contains.

    Raises :class:`ValueError` if any buyable molecule carries a positive estimate, and reports
    -- without changing anything -- how many unexpanded non-stock leaves carry exactly zero.

    Reads ``leaf_value`` before the fold consumes it, so the check sees what the search
    believes rather than what the valuation made of it. **This ordering is the whole reason the
    check and :func:`floored` are separate wrappers**: composed as ``auditing_fold(floored(f))``
    the audit runs first and sees the raw estimates, while ``floored(auditing_fold(f))`` would
    clamp the zeros away and then report none, which is a check that repairs its own evidence.
    """

    def folded(
        state: GraphState[K],
        *,
        leaf_value: np.ndarray,
        price: StepPrice,
        max_depth: int,
    ) -> Valuation[float]:
        values = np.asarray(leaf_value, dtype=np.float64)
        stock = _stock_mask(state, values.size)

        # Finite, because a stock molecule is interned but never handed to the heuristic, so its
        # slot holds the semiring's zero -- infinity -- rather than an estimate. That is the
        # normal case and not a disagreement: `inside_table` overrides stock to free regardless.
        buyable_but_priced = np.flatnonzero(
            stock & (values > 0.0) & np.isfinite(values)
        )
        if buyable_but_priced.size:
            worst = int(buyable_but_priced[int(np.argmax(values[buyable_but_priced]))])
            raise ValueError(
                f"{buyable_but_priced.size} stock molecules carry a positive estimate "
                f"(worst: sid {worst} at {values[worst]:.6f}). Equal molecules give equal "
                "fingerprints, so a buyable molecule's distance to the nearest buyable "
                "molecule is zero by construction. The heuristic and the search are reading "
                "different catalogues."
            )

        # Unscored sids sit at the semiring's zero (infinity), not at 0.0, so they cannot be
        # mistaken for a molecule the heuristic called free.
        priced_free = int(np.count_nonzero(~stock & (values == 0.0)))
        if priced_free:
            report(
                f"{priced_free} non-stock molecules are estimated at exactly 0: the heuristic "
                "calls them finished and the catalogue does not contain them. Admissible, but "
                "it cannot rank them -- suspect the fingerprint rather than the weights."
            )
        return fold(state, leaf_value=leaf_value, price=price, max_depth=max_depth)

    return folded


def floored(fold: Fold[K], *, floor: float) -> Fold[K]:
    """Wrap ``fold``, raising every estimate to at least ``floor``.

    A molecule that is not in the stock needs at least one reaction, so its true remaining cost
    is at least one step, and an estimate below that understates a known quantity.
    Pass the cheapest step as ``floor`` and the bound stays admissible while being strictly
    tighter than the estimate it replaces.

    **Stock needs no exemption**, which is what keeps this to one operation:
    :func:`vertsy.tropical.inside_table` overrides every stock entry to the semiring's one
    after the leaf values are read, at every depth, so a clamped stock estimate never survives
    to be used.

    What to expect of it, stated so it is not oversold: the effect is bounded by ``floor``, so
    against a heuristic whose useful spread is a few tenths it is negligible at a step cost of
    0.001 and material at 0.1. It makes ``h = 0`` mean "buyable" and nothing else -- which is
    what lets :func:`auditing_fold`'s zero report be read as a statement about the fingerprint.
    It does **not** break a tie: a hundred colliding molecules all clamp to the same floor and
    stay exactly as indistinguishable as they were.
    """
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor!r}.")

    def folded(
        state: GraphState[K],
        *,
        leaf_value: np.ndarray,
        price: StepPrice,
        max_depth: int,
    ) -> Valuation[float]:
        return fold(
            state,
            leaf_value=np.maximum(np.asarray(leaf_value, dtype=np.float64), floor),
            price=price,
            max_depth=max_depth,
        )

    return folded


def auditing_select(
    select: Select[K], *, report: Report = warn, rank: Rank = by_total
) -> Select[K]:
    """Wrap ``select``, reporting how much of each pass was decided by a tie rather than a rank.

    A pass takes the ``limit`` cheapest open leaves. If more leaves share the score of the last
    one taken than there are places, which of them go is settled by sid order — so the ranking
    chose the *score* and insertion order chose the *molecules*. That is invisible in every
    statistic a run already reports: mean estimate, mean context and the depth histogram all
    look ordinary while the selection is arbitrary.

    Reported per pass rather than summed, because the failure arrives at a particular pass and
    then persists. On the target that motivated this, the frontier is decided by rank for three
    passes and by tie for the remaining thirteen.

    **The threshold is "more tied than taken", and it needs no parameter.** An earlier version
    scaled it by a factor, on the theory that AND-siblings would otherwise make it fire on every
    healthy graph: siblings tie *exactly* under :func:`vertsy.aostar.by_total`, necessarily,
    since both carry the cost of the cheapest solution graph containing them and it is the same
    graph. That worry does not survive arithmetic. A sibling tie is the size of one
    application's arity — two or three — so it can only exceed a batch when ``limit`` is
    smaller than the arity, which no real width is. The factor was calibrated against toy
    widths in this module's own tests and protected nothing.

    And in the one case it does fire on siblings, firing is right: at ``limit`` below the arity
    the pass really does take part of a conjunction and leave the rest, which biases against
    finishing a group a route needs whole. That is a documented failure mode worth a line in the
    log, not noise to suppress.

    ``rank`` must match the one ``select`` ranks on, since the tie is only meaningful in the
    score that ordered the batch; it defaults to :func:`vertsy.aostar.by_total`, which is
    :func:`vertsy.aostar.global_leaves`'s own default and what every measurement here used.

    Costs a second outside sweep per pass. Selection is about 1% of a high-effort run, so this
    roughly doubles the cheapest stage rather than adding to the expensive one — which is why
    it is a wrapper a caller opts into and not a line inside the search.
    """

    def selecting(
        target: int,
        state: GraphState[K],
        value: Valuation[float],
        *,
        max_depth: int,
        limit: int,
        price: StepPrice,
    ) -> list[int]:
        chosen = select(
            target, state, value, max_depth=max_depth, limit=limit, price=price
        )
        if not chosen:
            return chosen

        leaves = reachable_leaves(
            target, state, value, max_depth=max_depth, price=price
        )
        scores = {
            sid: rank(context, total - context)
            for sid, (total, context, _) in leaves.items()
            if total != TROPICAL.zero
        }
        boundary = scores.get(int(chosen[-1]))
        if boundary is None:
            return chosen
        tied = sum(1 for score in scores.values() if score == boundary)
        if tied > len(chosen):
            report(
                f"{len(chosen)} leaves expanded out of {tied} tied at {boundary:.6f}: the "
                "ranking chose the score and sid order chose the molecules. Widening draws "
                "more from the same tie; only a finer estimate breaks it."
            )
        return chosen

    return selecting
