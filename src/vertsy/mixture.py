"""What a chemist should expect to find in the pot, and where on a spectrum to look for it.

The selectivity channels count. This names. A count steers a search -- a step with three products
costs more than a step with one -- but a chemist about to run the step wants the other thing: which
compounds will be in the flask, what they weigh, and which of them will *not* show up as a separate
peak because they weigh the same as the product.

That last part is why this is worth more than a price. Measured over 25 PaRoutes targets, **31.7%
of competitor products are isobaric with the intended product** -- 122 of 385 -- because a
competition is usually the same two partners joining differently, which rearranges atoms without
changing their number. Three `MC 90b` N-oxide regioisomers come out at `C18H21F2N5O2S` together. A
price throws that distinction away; a report can say "look for a peak at 309.07" for the two thirds
that resolve, and "this one hides under your product" for the third that does not.

**The by-product is the step's own atom balance**, not something a template has to declare. What
went into the pot minus what came out is the leaving group: an esterification sheds `H2O`, an acid
chloride `HCl`, a Suzuki `H2BBrO2`. No forward rule is consulted for it, which is why it is
available on every step rather than only the paired ones.

Where it is *wrong*, it is wrong informatively. A reductive amination reads as shedding bare `O`,
because the two hydrogens that would make it water came from a hydride the pot never held. A
balance that loses a heteroatom and no hydrogen is a step whose reagent is implicit -- the same
fault :func:`vertsy.selectivity.conditions_allow` gates competitors on -- so the formula is the
diagnostic as well as the datum.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

from vesma import (
    cmol_canonicalish_smiles,
    cmol_from_smiles,
    generate_precursors,
)
from vesma.cmol import CMol, cmol_union
from vesma.formula import (
    Coincidence,
    composition,
    difference,
    exact_mass,
    hill_formula,
    mass_coincidence,
    net_charge,
)
from vesma.reaction import ReactionPattern

__all__ = ["ExpectedPeak", "PeakKind", "expected_mixture"]

PeakKind = Literal[
    "product",
    "by-product",
    "atoms lost",
    "same reaction, another site",
    "another reaction, same site",
    "another reaction, another site",
    "another reaction, site unknown",
]

# The order a reader wants: what you are making, what leaves with it, then the three ways the pot
# can go somewhere else, worst first -- a competitor at the same atom cannot be protected against,
# while one at another atom can.
KIND_ORDER: tuple[PeakKind, ...] = (
    "product",
    "by-product",
    "atoms lost",
    "another reaction, same site",
    "same reaction, another site",
    "another reaction, another site",
    # Last because it is the least informative, not because it is the least dangerous: a competitor
    # whose site nobody can place is a peak to expect like any other, and saying "unknown" is the
    # only honest label for it. Lumping it with one of the two halves above is what this replaces.
    "another reaction, site unknown",
)


@dataclass(frozen=True)
class ExpectedPeak:
    """One compound to expect, and whether a mass spectrum will tell it from the product."""

    kind: PeakKind
    formula: str
    exact_mass: float | None
    """``None`` where no honest number can be given: an element with no monoisotopic mass on
    file, or a composition the balance could only partly see.

    Reported rather than omitted or guessed: a peak whose mass is unknown is still a compound to
    expect, and a mass that silently dropped a contribution looks exactly like a different one.
    """
    smiles: str
    """Empty for ``atoms lost``, which is a composition rather than a structure.

    The balance says `HCl` left; it does not say the hydrogen came from the amine rather than the
    solvent, and writing a structure would claim more than was derived. A *declared* by-product
    does have one, because a template named it.
    """
    source: str
    """The rule that would make it, or empty for the product and its by-product."""
    coincidence: Coincidence
    """How badly this collides with the product on a mass spectrum.

    Three tiers rather than a bool, because "no instrument can separate these" and "a better
    instrument could" are different advice -- see :data:`vesma.formula.Coincidence`. Always
    ``"resolved"`` on the product itself, which coincides with itself trivially and would otherwise
    make the column read as a warning about the thing being made."""

    @property
    def isobaric(self) -> bool:
        """Shares the product's formula, so **no** mass will separate the two.

        Derived rather than stored, so the tier is the single source of truth and the two cannot
        drift. This is the top tier only: a peak at the product's nominal mass but a different
        formula is a real warning and is *not* isobaric, because a high-resolution instrument does
        separate those.
        """
        return self.coincidence == "same formula"


def peak(
    kind: PeakKind,
    counts: Counter[int],
    *,
    charge: int = 0,
    smiles: str = "",
    source: str = "",
    incomplete: bool = False,
    product_counts: Counter[int] | None = None,
) -> ExpectedPeak:
    """One peak from a composition, with the mass left ``None`` where it cannot be summed.

    ``product_counts`` is the composition to compare against, not a formula string: the nominal-mass
    tier needs the atoms, and comparing compositions is also what makes ``same formula`` exact
    rather than a string coincidence. Omit it and the peak is ``"resolved"``, which is what the
    product's own entry wants.
    """
    try:
        mass: float | None = None if incomplete else exact_mass(counts, charge=charge)
    except ValueError:
        mass = None
    collides: Coincidence = "resolved"
    if kind != "product" and product_counts is not None:
        try:
            collides = mass_coincidence(counts, product_counts)
        except ValueError:
            # An element with no mass on file cannot be placed on a spectrum at all, which is
            # already said by `exact_mass` leaving the mass `None`. Claiming a tier on top of that
            # would be claiming the comparison succeeded.
            collides = "resolved"
    return ExpectedPeak(
        kind=kind,
        formula=hill_formula(counts),
        exact_mass=mass,
        smiles=smiles,
        source=source,
        coincidence=collides,
    )


def expected_mixture(
    product: str,
    precursors: Sequence[str],
    *,
    same_reaction: Sequence[tuple[str, ReactionPattern]] = (),
    same_site: Sequence[tuple[str, ReactionPattern]] = (),
    different_site: Sequence[tuple[str, ReactionPattern]] = (),
    unknown_site: Sequence[tuple[str, ReactionPattern]] = (),
    key: Callable[[CMol], Hashable],
    render: Callable[[CMol], str] = cmol_canonicalish_smiles,
) -> tuple[ExpectedPeak, ...]:
    """The product, its by-product, and what else this pot can give, as peaks.

    The four rule groups are **classified by the caller**, not here, and that seam is deliberate:
    which forward rules are this step's own, which compete at the same atom and which at another
    are questions about pairings, arities, conditions and reaction centres that
    :mod:`vertsy.selectivity` already answers and that this module has no business re-deciding.
    What it adds is the chemistry a count discards -- apply the rule, name the molecule, weigh it.

    ``unknown_site`` is the fourth group and exists because the site question is not always
    answerable: it compares the intended reaction's reaction centre against the candidate's, and
    the intended reaction has a centre only where some forward row describes it -- which is true of
    305 of 459 compiled rules. Those candidates used to be labelled same-site, by an out-of-range
    guard that returned "same" for anything it could not place. A peak is still a peak, so they are
    reported, under a label that claims nothing.

    Each group is ``(label, parsed forward rule)``. Re-running the rules here rather than carrying
    products through the search is what makes this affordable: a report covers a handful of steps,
    where the search covers millions of applications, and nothing in the graph has to grow a field.

    **Parsed by the caller, and that is the difference between affordable and not.** This used to
    take SMARTS and call :func:`vesma.reaction.parse_reaction_smarts` inside the loop below, once
    per rule per step -- while the caller had the whole table parsed and cached already and threw
    it away. Measured on two PaRoutes targets it cost ~3,200 parses apiece, and that was
    essentially the entire report phase: 2.42 s of a 2.43 s report, with extraction as a whole
    running at ~42% of the search it followed. The caller is the one place that can hold a parsed
    table for a sweep, so the caller holds it.

    ``key`` decides **identity** and ``render`` decides only how a molecule reads, and the two are
    separate arguments because conflating them is a bug this function had. Two peaks are the same
    peak when the search would call them the same molecule, so the dedup has to use whatever the
    arm keys by -- WL hashes on the Metal and CPU arms, canonical SMILES on the RDKit one. Keying
    by a *rendering* instead made the intended product reappear as a competitor of itself whenever
    the report's spelling of it differed from the renderer's: `O=C(C)NCC` against `CCNC(C)=O`, one
    molecule, two strings. A canonical SMILES is a perfectly good key -- one arm uses it -- but it
    is a key because the arm chose it, not because it is a string.

    ``render`` is injected for the reason :func:`route_report` injects a namer: the readable form
    is the caller's choice, and the rdkit-free default keeps this importable without the optional
    dependency.
    """
    target, _info = cmol_from_smiles(product)
    product_counts = composition(target)
    target_key = key(target)
    found = [
        peak(
            "product",
            product_counts,
            charge=net_charge(target),
            smiles=product,
            product_counts=product_counts,
        )
    ]

    parts = [cmol_from_smiles(one)[0] for one in precursors]
    pot = cmol_union(parts) if len(parts) > 1 else parts[0] if parts else None

    seen = {target_key}
    declared = False
    for kind, group in (
        ("same reaction, another site", same_reaction),
        ("another reaction, same site", same_site),
        ("another reaction, another site", different_site),
        ("another reaction, site unknown", unknown_site),
    ):
        for label, reaction in group:
            if pot is None:
                break
            try:
                outcomes = generate_precursors(pot, reaction)
            except Exception:
                # A rule the chemistry refused on this pot has nothing to report, and a report is
                # not the place to surface a refusal the probe already counts.
                continue
            for outcome in outcomes:
                components = list(outcome.precursors)
                identities = [key(one) for one in components]
                # What the reaction made is the *target*, wherever the target is there at all, and
                # everything else came off with it: a forward rule that declares its by-product
                # emits them together, so the water a reductive amination writes must not read as
                # another site of the same reaction -- a competing product, which it is the
                # opposite of.
                #
                # Where an outcome does *not* reach the target there is no intended product to
                # find, and nothing here knows which component the rule was aiming at. A
                # competitor rule is by construction aiming elsewhere, and the one piece of
                # evidence that would settle it -- which right-hand-side group the template calls
                # the product -- does not survive the matcher: component order is not group order,
                # and `RetroOutcome.mapping` is the left-hand-side match, so it cannot say which
                # component an atom ended up in. So this claims nothing, and every component
                # carries the group's own label.
                #
                # Two guesses stood here before and both were wrong in the same way. Taking the
                # heaviest component inverted the labels on `FWD H 50`, whose triphenylphosphine
                # oxide is twenty heavy atoms against the stilbene's seventeen; taking the one
                # nearest the target's composition got that case right and was still a guess at
                # something unknowable.
                makes_target = target_key in identities
                for component, identity in zip(components, identities, strict=True):
                    if identity == target_key:
                        # The intended reaction, reached by one of its own rules. Anything else in
                        # the same outcome is a *declared* by-product of it, and a declaration
                        # beats the balance below.
                        declared = declared or len(components) > 1
                        continue
                    if identity in seen:
                        continue
                    seen.add(identity)
                    found.append(
                        peak(
                            "by-product" if makes_target else kind,
                            composition(component),
                            charge=net_charge(component),
                            smiles=render(component),
                            source=label,
                            product_counts=product_counts,
                        )
                    )
    # The step's own balance, which needs no template: what the pot held minus what it made. Used
    # only where no rule declared one, because the two are accounts of the same leaving group and
    # the declaration is the better account -- a reductive amination's balance can only see the
    # oxygen, since the hydrogens that make it water came from a hydride the pot never held.
    if pot is not None and not declared:
        lost, gained = difference(composition(pot), product_counts)
        if lost:
            found.append(
                peak(
                    # Not a `by-product`, which is a compound a reader should look for. This is
                    # the arithmetic: these atoms left the pot, as *something*. A Negishi's
                    # balance says `Br2`, which is the right composition and the wrong molecule --
                    # the bromides leave with the zinc, and nobody should look for bromine at
                    # 157.84. Declaring a by-product is what turns the arithmetic into a peak.
                    "atoms lost",
                    lost,
                    # An open balance means the reaction took atoms from a reagent the pot never
                    # held, so what `lost` names is *part* of the by-product and a mass for it
                    # would be confidently wrong. A reductive amination that has not declared its
                    # water shows `O` at 15.995, which is not a compound; `incomplete` says so
                    # instead of printing a number a reader would look for.
                    incomplete=bool(gained),
                    product_counts=product_counts,
                )
            )

    return tuple(
        sorted(
            found, key=lambda one: (KIND_ORDER.index(one.kind), one.formula, one.smiles)
        )
    )
