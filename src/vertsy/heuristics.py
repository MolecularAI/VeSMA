from __future__ import annotations

from typing import Protocol

import numpy as np

from .core import M
from .search import MoleculeBatch

__all__ = ["Heuristic", "strongest", "zero_heuristic"]


class Heuristic(Protocol[M]):
    def __call__(self, molecules: MoleculeBatch[M]) -> np.ndarray:
        """Estimate what it still costs to reach the stock, one float per molecule.

        Shape-wise this is a :data:`vertsy.search.MoleculeScore` and nothing more -- a batch in, a
        column out -- and it is named separately only because of the two properties below. Those are
        a *contract on one channel*, not on measurement in general: a forward-selectivity count is
        the same arrow and obeys neither.

        The unit is whatever ``rule_cost`` charges a reaction, because those two are added
        together and nothing else reconciles them. With the default unit cost that is literally
        reactions; under a log-probability cost (a step worth ``-log 0.9``, say) it is
        log-probability. **Commensurability with ``rule_cost`` is the requirement**; "reactions" is
        only the most common way to satisfy it.

        Two properties decide whether a function belongs here.

        It must **vanish at the stock**. A product that is already buyable needs no further
        reactions, so its estimate must be ``0`` — and, more importantly, the estimate must
        *fall* as a molecule gets closer to being buyable. AND aggregation sums over the
        precursors of an application, so an estimate with a non-zero floor is charged once
        per open precursor, which quietly penalises convergent routes: exactly the chemistry
        one wants. This is why absolute complexity scores (SA score, Bertz CT, heavy-atom
        count) are not heuristics — they measure how hard a molecule is in the abstract, not
        how far it is from *this* stock, and they do not go to zero when it is reached. A
        complexity penalty belongs on ``rule_cost`` in :func:`vertsy.metrics.valuation`,
        where an edge weight is charged once per reaction and need not vanish anywhere.

        It should be **admissible** if the caller wants the cheapest route: never estimate more
        than the true remaining cost. Since any product not in stock needs at least one reaction,
        **any estimate bounded by the cheapest ``rule_cost`` is admissible for free** — which is
        ``1`` only because the default cost is. Scaling the estimate up and scaling the step cost
        down are the same move, so the bound is a ratio and not a number.

        Molecules, not keys, because the caller already has them: the matcher built every one
        of these and :func:`vertsy.aostar.aostar` kept it. Taking keys would oblige every
        implementation to reconstruct a molecule that existed moments earlier — measured at
        ~430 ms per expansion for the ECFP4 heuristics — and would confine the search to key
        spaces that can be inverted at all.

        A whole batch, and the batch the matcher returned: the search never holds a molecule on
        its own, so there is nothing else it could pass. An implementation that needs to iterate
        its argument is therefore declaring which batches it works with — object arrays iterate,
        a device batch does not — and pairs with the ``stack_mols`` that produces them.
        """
        ...


def zero_heuristic(molecules: MoleculeBatch[object]) -> np.ndarray:
    """Estimate nothing. Trivially admissible, so AO* returns the cheapest route.

    Uninformed, which makes this the correctness baseline rather than a fast one: search is
    guided only by the reactions already committed to, so it explores in cost order like
    Dijkstra. The tests use it to pin down what "optimal" means before any real heuristic
    is allowed to perturb it. It never looks at a molecule, so it satisfies `Heuristic[M]`
    for every `M`.
    """
    return np.zeros(len(molecules), dtype=np.float32)


def strongest(*heuristics: Heuristic[M]) -> Heuristic[M]:
    """The pointwise maximum of several estimates: the tightest bound they can jointly justify.

    The only combination of admissible estimates that stays admissible. Each ``h_i`` never exceeds
    the true remaining cost, so neither does their maximum — while the maximum is at least as large
    as every input, and for a lower bound "larger" *is* "better informed". Averaging throws
    information away; **summing breaks admissibility** and re-introduces precisely the depth
    over-charging that a cheap ``rule_cost`` exists to undo, since two estimates of 0.5 would price
    a molecule at 1.0 without anything having got further from the stock.

    Read chemically it is a consensus AND. A molecule scores low only if *every* view agrees it is
    close to the catalogue, and one view calling it far keeps it far. Correlated fingerprints
    therefore still earn a place: they are not voting on a shared opinion, they are vetoing each
    other's false positives, and a false positive is exactly what a search wastes a pass on. What
    the combination cannot do is invent discrimination — if two views agree everywhere, their
    maximum is one of them, and the cost is the second featurisation for nothing.

    Both :class:`Heuristic` properties survive: a maximum of estimates that vanish at the stock
    vanishes there too, and commensurability with ``rule_cost`` is inherited from the parts, which
    is why the parts must already share a scale. Combining a ``[0, 1]`` distance with a raw
    heavy-atom count would let the count decide everything.
    """
    if not heuristics:
        raise ValueError("strongest needs at least one heuristic to combine.")

    def combined(molecules: MoleculeBatch[M]) -> np.ndarray:
        if not len(molecules):
            return np.zeros(0, dtype=np.float32)
        # `reduce` rather than a running maximum so the batch is walked once per heuristic and no
        # intermediate is kept beyond the accumulator.
        return np.maximum.reduce(
            [
                np.asarray(estimate(molecules), dtype=np.float32)
                for estimate in heuristics
            ]
        )

    return combined
