"""Measure how selective a disconnection is by running its chemistry forwards.

A retro template says a product *can* be made from some precursors. It does not say the intended
reaction would dominate: chlorinating one primary hydroxyl of a free pentitol is a graph
transformation the rules permit and a selectivity a chemist would not bet on. The observation that
separates the two is cheap and entirely mechanical -- **run the forward reaction on the precursors
and count the distinct products.** One means the step is clean. Several mean competing sites,
competing chemoselectivity, or a competing reaction, and the step is then worth several steps,
which is what :func:`vertsy.costs.selective` charges it as.

Counting **products, not match sites**, and counting them by *key*. A pentitol has five hydroxyls
and three distinct monosubstitution products, because two pairs of them are equivalent by symmetry
-- and symmetry-equivalent sites are exactly what a chemist does not count twice. Keying through the
run's own keyer removes them for free, which is the whole reason the count is 3 and not 5.

Forward templates are **curated data, not a transformation of the retro ones.** Reversing a retro
SMARTS mechanically does not work and is not attempted: a left-hand side is a match pattern and may
contain alternation, a right-hand side is a rewrite program where alternation means nothing, and
:func:`vesma.reaction.parse_reaction_smarts` validates the right-hand side for concrete elements and
bond types -- which a query-laden product side is not. So a forward rule is written, once, by
someone who knows the reaction, and paired with the retro rules it probes.

Association is by rule id, and **an unpaired rule is unprobed rather than clean**. That distinction
is the one this module most needs to keep: a library nobody has written forward rules for must cost
exactly what it costs today, not be quietly declared perfectly selective. :class:`ProbeReport`
exists so a run can state its own coverage rather than leaving it to be assumed.

Two channels, one mechanism
---------------------------
**Competing sites** is the intended reaction happening in more than one place. Its templates are
the ones *paired* with the retro rule that made the disconnection -- a reagent's scope, since a
chlorinating agent does not know which hydroxyl was meant -- and its cost is a separation.
:func:`site_probe` builds it.

**Competing reactions** is a *different* reaction the same starting materials could undergo
instead, and its cost may be that the intended step does not work at all. Its templates are every
forward rule *not* paired with the intended one, filtered to those with **two or more components**.
:func:`competitor_probe` builds it.

That arity filter is the whole physical argument, and without it the channel is worthless. A
multi-component forward rule can only fire if all of its partners are present -- and they are,
because the probe is matching against exactly this application's precursors, which is what the pot
contains. So a hit means *these starting materials could also do this to each other*, which is a
real competition. A **unimolecular** rule matching a functional group elsewhere on the substrate
means nothing of the kind: it needs its own reagent, catalyst and conditions, none of which are in
the pot, so it would read every polyfunctional molecule as a mixture and measure molecular
complexity rather than a competition.

They are one mechanism rather than two because the measurement, the store relation, the price and
the report are identical in shape. Only the template set and the pairing differ, and those are
already parameters. They are separate *channels* rather than one number because their chemistry
differs, and :class:`vertsy.costs.StepPrice` gives each its own multiplier.

Where the future goes
---------------------
Counting every multi-component rule as a competitor is the crudest reading, and ``competes`` is the
one place that changes. It is a predicate on ``(intended rule, candidate rule)``, so the two things
that would refine it are the same plug point seen twice. **Thermodynamics**: given a favourability
per reaction, a candidate far less favourable than the intended one does not compete. **Conditions
and catalysts**: two reactions needing different ones cannot run together, so a candidate outside
the intended reaction's conditions does not compete -- an explicit allow-list of pairs is the
simplest form of that, and this signature is what it would plug into. Neither table exists yet, and
the default -- everything multi-component competes -- is the honest starting point rather than a
placeholder.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import numpy as np

from vesma.matcher import Pair

from .core import HitArray, M
from .costs import UNPROBED
from .interner import SidArray
from .search import (
    Column,
    MoleculeBatch,
    ReactionScore,
    object_mol_array,
)
from .template_lint import all_grouped, top_level_pieces

__all__ = [
    "Competes",
    "ForwardIndex",
    "competitor_probe",
    "forward_index",
    "ProbeReport",
    "Combine",
    "distinct_outcomes",
    "severity_outcomes",
    "Reduce",
    "forward_probe",
    "grouped_applications",
    "object_combine",
    "paired_with",
    "site_probe",
    "conditions_allow",
    "declared_molecularity",
    "template_arities",
    "unmeasured",
]

# Which forward templates probe a given retro template. A rule absent from this is unprobed, and an
# unprobed application costs exactly what it cost before any of this existed.
type ForwardIndex = Mapping[int, tuple[int, ...]]

# How one application's forward outcomes become one number: the group's products as keys, the hit
# rows, how many groups, per group the key of the molecule being disconnected, and per group the
# **retro rule** that made it. Parameterised for the reason `candidates` is -- it is the other thing
# that distinguishes one channel from another, and the two are independent. `distinct_outcomes`
# answers "how many products", `severity_outcomes` answers "how bad is the worst one", over the same
# hits.
#
# The retro rule is in the signature because a *relative* reduction needs it. How badly a competitor
# hurts depends on what it is competing against, so a severity is not a property of the competitor
# alone -- and this is the only place both reactions are known at once. A count ignores it.
type Reduce = Callable[
    [Sequence[Hashable], HitArray, int, Sequence[Hashable] | None, Sequence[int]],
    list[int],
]
# What this module builds is a `vertsy.search.ReactionScore`: one value per hit row, here the
# number of distinct products the forward chemistry makes from that application's precursors, with
# ``0`` meaning nobody measured it. Named by its axis rather than by its subject, because the axis is
# what composition cares about.
# One molecule per group, each the disjoint union of the group's members: what a multi-component
# forward pattern has to match across. Handed a batch and a CSR grouping -- the member rows and
# where each group starts in them, the same shape ``vertsy.tropical.Shape`` keeps its precursors in
# -- so nothing ever holds a molecule on its own and a packed device batch works as an object array
# does. ``None`` where a backend has no such operation, and then multi-precursor applications go
# unprobed rather than half-probed.
# Answers the combined batch *and* which groups it could actually build, because that is the one
# fact only the combiner has. A backend with a fixed atom budget cannot hold every union, and a
# group it could not build was never examined -- which is not the same as examining it and finding
# nothing. Conflating the two would charge a device limit as a broken pairing.
type Combined[M] = tuple[MoleculeBatch[M], np.ndarray]
type Combine[M] = Callable[[MoleculeBatch[M], SidArray, SidArray], Combined[M]]
# Whether a candidate forward rule actually competes with the intended one, given both rule ids.
#
# The single plug point for everything that would refine the competitor channel. Today's default
# says every unpaired multi-component rule competes, which is the crudest honest reading. A
# favourability table would say a candidate far less favourable than the intended reaction does not
# compete; a conditions table, or an explicit allow-list of pairs, would say two reactions needing
# different catalysts cannot run together and so do not compete. Both are this signature.
type Competes = Callable[[int, int], bool]


class ReactionPatternLike(Protocol):
    """What a reaction centre needs of a compiled template: its left side and its edit.

    Structural rather than imported, for the reason :class:`ForwardMatcher` is: this module
    describes what it uses and leaves :mod:`vesma` free to type its own objects.
    """

    @property
    def lhs(self) -> Any: ...

    @property
    def edit(self) -> Any: ...


@dataclass(frozen=True)
class CentreAtom:
    """One atom a template changes, named by *what the change does to it*.

    Not by how the author numbered it. A map number is bookkeeping -- templates routinely map
    spectator atoms to carry them through, and this repository's own star-mapped reagents are
    exactly that -- so "mapped by one rule and not the other" says nothing about chemistry. An
    atom is in the reaction centre when its **bonding changes**, which is a property of the edit.
    """

    elements: tuple[int, ...]
    """Atomic numbers the left-hand side admits here; several when the query is a disjunction."""
    aromaticity: int
    """The query's aromatic bitmask: 1 aliphatic, 2 aromatic, 3 either.

    Carried because element and role alone lump every electrophilic carbon together, and an acyl
    carbon is not an aryl carbon. Without it an amide coupling reads as competing with a Suzuki.
    """
    loses_heavy_bond: bool
    """A bond to another heavy atom breaks -- a leaving group departs, or a ring opens."""
    gains_bond: bool
    loses_hydrogen: bool
    """An explicitly written hydrogen leaves. Implicit hydrogen is invisible here; see below."""


def _elements(masks: Any, index: int) -> tuple[int, ...]:
    """The atomic numbers one left-hand-side atom admits, from its element bitmask."""
    lo, hi = int(masks.Z_lo[index]), int(masks.Z_hi[index])
    return tuple(
        [z for z in range(64) if lo >> z & 1]
        + [64 + z for z in range(64) if hi >> z & 1]
    )


def reaction_centre(pattern: ReactionPatternLike) -> tuple[CentreAtom, ...]:
    """The atoms a template changes, as roles rather than positions.

    **The edit rebuilds bonds rather than editing them**, so ``removed_lhs_edges`` holds bonds that
    are immediately put back: an amide coupling reports its own carbonyl among the removals. The
    bonds actually broken are the removals the additions do not restore, and the bonds actually
    made are the additions the removals did not already hold -- so both sets are differences, and
    reading either alone names the whole matched region as a reaction centre.

    Hydrogen is only half visible. A template writing ``[N:1]-[H]`` says so in its edges; one
    writing ``[NH2:1]`` changes the same nitrogen through an implicit count that the edit records
    as unspecified. So :attr:`CentreAtom.loses_hydrogen` is evidence when true and silence when
    false, and nothing here should treat it as a negative.
    """
    edit = pattern.edit
    # Right-hand index -> left-hand index. The two arrays are *positionally aligned and ordered by
    # map number*, so pairing them is the translation and indexing either one by an atom index is
    # not. Reading `mapped_lhs_indices[rhs_atom]` happens to work where the mapped right-hand atoms
    # sit at right-hand positions 0..k-1 in that order -- 142 of the 480 shipped rules -- and
    # silently mistranslates the other 338. On `Hartenfeller 33a`,
    # `[c:2]-[CH3:1]>>[c:2]-[Cl].[C:1]-[Br]`, it read the new c-Cl bond as an *added* aryl-methyl
    # bond, which cancelled the genuinely broken one and returned an empty reaction centre for a
    # rule that breaks a bond and halogenates both ends.
    lhs_of_rhs = dict(
        zip(
            np.asarray(edit.mapped_rhs_indices).tolist(),
            np.asarray(edit.mapped_lhs_indices).tolist(),
            strict=True,
        )
    )
    removed = {
        tuple(sorted(edge))
        for edge in np.asarray(edit.removed_lhs_edges).reshape(-1, 2).tolist()
    }
    added: set[tuple[int, int]] = set()
    gains_new_atom: set[int] = set()
    for left, right in np.asarray(edit.added_rhs_edges).reshape(-1, 2).tolist():
        # An atom the right-hand side creates has no left-hand counterpart, so the bond is to
        # something that was not there before: the mapped end gains a bond and nothing else is
        # known about the other.
        ends = [lhs_of_rhs.get(side) for side in (left, right)]
        if ends[0] is None or ends[1] is None:
            gains_new_atom.update(end for end in ends if end is not None)
            continue
        added.add((min(ends), max(ends)))
    broken, made = removed - added, added - removed
    deleted = set(np.asarray(edit.deleted_lhs_indices).tolist())

    masks = pattern.lhs.atom_masks
    hydrogen = {
        index
        for index in range(int(pattern.lhs.atom_count))
        if _elements(masks, index) == (1,)
    }
    centre = []
    for index in range(int(pattern.lhs.atom_count)):
        if index in deleted or index in hydrogen:
            continue
        partners_lost = {
            other
            for edge in broken
            for other in edge
            if index in edge and other != index
        }
        if (
            not partners_lost
            and index not in gains_new_atom
            and not any(index in edge for edge in made)
        ):
            continue
        centre.append(
            CentreAtom(
                elements=_elements(masks, index),
                aromaticity=int(masks.A[index]),
                loses_heavy_bond=bool(partners_lost - hydrogen),
                gains_bond=index in gains_new_atom
                or any(index in edge for edge in made),
                loses_hydrogen=bool(partners_lost & hydrogen),
            )
        )
    return tuple(centre)


def _same_atom(left: CentreAtom, right: CentreAtom) -> bool:
    """Whether two templates want the same *kind* of atom, in the same role."""
    return (
        bool(set(left.elements) & set(right.elements))
        and bool(left.aromaticity & right.aromaticity)
        and left.loses_heavy_bond == right.loses_heavy_bond
        and left.gains_bond == right.gains_bond
        and left.loses_hydrogen == right.loses_hydrogen
    )


def same_site(
    patterns: Sequence[ReactionPatternLike],
    intended_centres: Mapping[int, Sequence[CentreAtom]],
) -> Competes:
    """A :data:`Competes` that keeps the candidates wanting the **same atom** as the intended rule.

    ``intended_centres`` maps a **retro** rule id to that reaction's centre atoms as a *forward* rule
    describes them. It is what makes the comparison well-posed at all: ``patterns`` is indexed by
    forward rule id while ``intended`` is a retro rule id, so the intended reaction's centre has to
    be looked up rather than indexed.

    **It has to come from a forward row, and the broader alternative is wrong.** Reading the centre
    off the retro template itself would cover 452 of 459 compiled rules against 305 -- which was
    tried and reverted. A retro rule's left-hand side is the *product*, so its centre describes each
    atom as it is **after** the reaction, while every candidate here is a forward rule describing
    atoms as they are **in the pot**. Mirroring the bond roles is necessary and nowhere near
    sufficient: over the 271 rules with exactly one describing forward row, only 86 centres coincide
    after the mirror, 22 differ in the aromaticity mask alone and **163 differ in element or role**.
    An amine nitrogen that becomes an aromatic ring nitrogen is a different query on each side. It is
    the same fact that makes `templates-forward.csv` exist: a retro SMARTS cannot be reversed
    mechanically.

    **Every enrichment figure once recorded for this split was invalid, and the reason was here.**
    The predicate read ``centres[intended]`` directly, with an out-of-range guard returning
    ``True``: 204 forward patterns against 459 compiled rules meant 255 rules (55.6%) fell through
    it and were labelled same-site unconditionally. That is what produced the 39%-same-atom against
    4.8%-different-atom volumes and the reading that the predicate was "loose rather than wrong" --
    a majority of the same-atom half was an index error, not a signature.

    **Re-measured once it was fixed, and the halves swap.** Over 320,353 applications on 20 targets
    with the veto rows as ground truth, at the ``>= 2`` threshold: same-site 1.71% of applications
    and 4.68% of condemned ones (**2.7x**), different-site 0.17% and 0.31% (**1.8x**), against the
    unsplit channel's 2.02% / 5.59% / 2.8x. The record said the *different*-atom half was the
    enriched one at 1.1x against 2.2x; it is the other way round, which means the chemical
    expectation this function was written on -- that the same-atom half carries the signal -- was
    right, and the measurement that appeared to refute it was the bug.

    The *different*-atom half is the thinly measured one, at 3,790 firings against 22,791, and it is
    the half an automatic protection step would act on -- so broadening its coverage is worth
    wanting. The honest way is the match-level tier below, not the retro template.

    **It is still no basis for a price**, for a reason the correction does not change: the same-atom
    half is 2.7x where the whole channel is 2.8x and carries almost all of its volume, so splitting
    separates nothing a price could use, and the enriched rows are the ones the vetoes already
    refuse.

    **What it is for is deciding whether protection can help**, which is a question about what to
    *do* rather than what to charge. A competitor at a different atom is answered by protect, react,
    deprotect -- a bounded three-step detour -- and one at the same atom is not, because protecting
    the site stops both reactions. So an engine that inserts a protection sequence into a route
    needs exactly this bit to know whether inserting one would help, and that is a better reason to
    keep the split than any enrichment figure. Until then it is also what tells a reader of a route
    whether a competitor attacks the same atom or another one. What *did* separate is severity: `competitors:destructive` reads 6.0x
    on a 0.04% base in the same run.

    The reasoning below is chemistry and stands.
    The split is by what *fixes* each half. A competitor attacking a
    different site is answered by protection -- protect the other group, react, deprotect -- which
    is a bounded, buyable, three-step detour the selector can actually purchase, and which `sites`
    already prices at that threshold. A competitor attacking the **same** atom cannot be protected
    against, because protecting the site stops both reactions; it is answered only by a different
    reagent or a different disconnection, so its yield split is set by a rate ratio and nothing in
    the step repertoire removes it.

    Same *kind* of atom in the same role, which is all a rule pair can say: two templates are
    compared, not two matches, so this cannot know they would land on the same atom of this
    molecule. That is the approximation the rule-level tier makes, and the match-level tier -- the
    competitor's matched atoms overlapping the intended reaction's -- is what removes it.
    """
    return _site_half(patterns, intended_centres, same=True)


def different_site(
    patterns: Sequence[ReactionPatternLike],
    intended_centres: Mapping[int, Sequence[CentreAtom]],
) -> Competes:
    """A :data:`Competes` that keeps the candidates wanting a **different** atom.

    The complement of :func:`same_site` over the candidates the question can be *asked* about, and
    its own builder rather than `not same_site(...)` for exactly that reason: where the intended
    reaction has no forward row describing it there is no reaction centre to compare against, and
    both halves have to abstain. Composing one from the other with `not` would send every one of
    those to this half, which is 154 of 459 compiled rules.
    """
    return _site_half(patterns, intended_centres, same=False)


def _site_half(
    patterns: Sequence[ReactionPatternLike],
    intended_centres: Mapping[int, Sequence[CentreAtom]],
    *,
    same: bool,
) -> Competes:
    """One half of the site split, abstaining where the intended reaction has no description."""
    centres = [reaction_centre(pattern) for pattern in patterns]

    def competes(intended: int, candidate: int) -> bool:
        if not 0 <= candidate < len(centres):
            return False
        # `intended` is a **retro** rule id and `centres` is indexed by **forward** rule id, so the
        # intended reaction's centre has to be looked up and cannot be indexed. Indexing
        # `centres[intended]` directly is the bug this argument exists to fix: with 204 forward
        # patterns against 459 compiled rules, 255 of them were out of range and the guard returned
        # `True`, so a majority of applications read as same-site unconditionally.
        wanted = intended_centres.get(intended, ())
        if not wanted:
            # No forward row describes the intended reaction, so there is no site question to ask.
            # Neither half claims it, exactly as a candidate the medium cannot turn over is in
            # neither half.
            return False
        overlaps = any(
            _same_atom(left, right) for left in wanted for right in centres[candidate]
        )
        return overlaps is same

    return competes


class ForwardMatcher(Protocol[M]):
    """What the probe needs of a matcher: hand it a whole batch, get products and hit rows.

    Narrower than :class:`vesma.matcher.ReactionMatcher` in what it promises and wider in what it
    accepts, because that protocol's ``Sequence[M]`` understates its own implementations -- the Metal
    matcher's signature already reads ``Sequence[CMol] | MetalMoleculeBatch``, and an object array is
    both. Stating it here keeps the probe honest about the one thing it relies on without widening a
    protocol in :mod:`vesma` to mention a type that lives up here.
    """

    def match(
        self, mols: Any, pairs: Sequence[Pair] | None = None
    ) -> tuple[Any, HitArray]: ...


@dataclass(frozen=True)
class ProbeReport:
    """What one probe pass did, so coverage is a number a run states rather than one it assumes."""

    channel: str
    """Which measurement this pass took. Reports from different channels do not sum."""
    applications: int
    """Disconnections the probe was offered."""
    probed: int
    """Disconnections the probe actually ran on -- it had candidate rules and a molecule to match.

    "Examined", not "found something", so ``probed / applications`` means the same thing on every
    channel. What running found is :attr:`silent`'s business.
    """
    unpaired: int
    """Disconnections whose retro rule has no forward rule. Unprobed, and charged nothing."""
    silent: int
    """Of those, the ones whose candidate rules fired on nothing. **Read per channel.**

    On ``sites`` this is a curation defect: a forward rule that cannot reproduce its own partner's
    step from its own partner's precursors is wrong, and the round-trip test over the forward table
    is where that gets caught. Counted here so a run notices it.

    On ``competitors`` it is the *good* answer -- no other reaction these starting materials could
    undergo was found. A channel where finding nothing is success and one where it is a bug share a
    field because they share a mechanism, and only the channel says which reading applies.

    Charged nothing either way: a measurement of zero is "nobody measured this", and neither a
    defect nor an absence is evidence about a molecule.
    """
    unbuildable: int
    """Disconnections whose precursors could not be combined into one molecule to match against.

    Multi-component applications with no ``union``, or a combined molecule the backend cannot hold.
    Unprobed, for the same reason.
    """
    offered_by_rule: Mapping[int, int] = field(default_factory=dict)
    """Per forward rule, the disconnections it was *asked* about. Rules asked nothing are absent."""
    fired_by_rule: Mapping[int, int] = field(default_factory=dict)
    """Per forward rule, the disconnections it actually produced something on.

    The attribution the aggregate counts cannot give, and the same arithmetic answers a different
    question on each channel.

    On ``competitors`` it says *which* reaction competes. :attr:`silent` says only that something
    fired, and the count a disconnection carries says only how many distinct products came out, so
    neither can distinguish "this particular reaction would win here" from "some rule somewhere
    matched". A veto row is retired in favour of this channel only when the channel finds the
    *same* chemistry, and co-occurrence alone cannot show that.

    On ``sites`` the complement is what matters: a rule offered and never firing is precisely the
    curation defect :attr:`silent` counts in aggregate, and ``offered_by_rule`` minus this names it.

    Counted per *disconnection* and not per match, so a rule firing twice on one application counts
    once -- the same unit as every other field here.
    """

    @property
    def coverage(self) -> float:
        """Share of offered disconnections the probe actually answered."""
        return self.probed / self.applications if self.applications else 0.0


def forward_index(
    probes: Mapping[str, Sequence[str]],
    retro_ids: Mapping[str, int],
    forward_ids: Mapping[str, int],
) -> ForwardIndex:
    """Resolve ``forward code -> retro codes`` into ``retro rule id -> forward rule ids``.

    Codes are how a chemist writes a pairing and ids are how a graph stores one, and the two never
    meet: a rule id is a position in whichever rule set a run happened to compile, so the mapping
    belongs wherever the names live -- the bench, which already holds a ``template_code`` column per
    compiled rule -- and this module only ever sees ints.

    A forward code naming a retro code nothing compiled is dropped rather than raising. A run may
    legitimately load a subset of the rule sets, and a pairing for a rule that is simply not present
    is not an error; it is a pairing with nothing to do. What *is* worth raising over -- a code that
    exists nowhere at all -- is caught by the test over the table, before any run.
    """
    index: dict[int, list[int]] = {}
    for forward_code, retro_codes in probes.items():
        forward = forward_ids.get(forward_code)
        if forward is None:
            continue
        for retro_code in retro_codes:
            retro = retro_ids.get(retro_code)
            if retro is not None:
                index.setdefault(retro, []).append(forward)
    return {retro: tuple(sorted(set(found))) for retro, found in index.items()}


def conditions_allow(
    provides: Mapping[int, frozenset[str]],
    requires: Sequence[frozenset[str]],
) -> Competes:
    """A :data:`Competes` that keeps only candidates the intended reaction's medium could turn over.

    This is the ``requires`` column the tag vocabulary was written for, and it answers the question
    the channel's filter could not. A competitor is admitted on the argument "both partners are in
    the pot, so they could do this to each other" -- but a reaction needs more than its two
    partners. A Grignard addition needs the organometallic, a Friedel-Crafts needs the Lewis acid,
    a Buchwald needs the palladium, and none of those is a precursor. So a forward rule whose
    reagent is implicit fires wherever its two functional groups appear, which is how
    "Grignard addition to a carbonyl" came to compete in a pot whose halide was `OCCCCl` -- a
    molecule that would quench the organometallic it never became.

    **The test is a subset, not a formula.** Every condition class the candidate needs must be one
    the intended reaction supplies: ``requires[candidate] <= provides[intended]``. That is a
    conjunction over the candidate's own requirements and nothing more -- no disjunction, no
    grouping, no expression language -- because a reaction needs *all* of its conditions and the
    question is only whether this medium happens to have them. A rule requiring nothing competes
    everywhere, which is right: a base-mediated alkylation or a condensation runs in most pots.

    ``provides`` is keyed by **retro** rule id, because the intended reaction is the one that made
    the application, and ``requires`` is indexed by **forward** rule id, which is what a candidate
    is. Both are resolved from one curated table of ``tag -> template codes``, read by
    :func:`vertsy.filters.read_tags` and shared with the veto layer's ``tag:`` scope entries: what
    poisons a catalyst and what a reaction needs are the same per-rule attribute, so there is one
    list rather than two.
    """

    def allowed(intended: int, candidate: int) -> bool:
        return requires[candidate] <= provides.get(intended, frozenset())

    return allowed


def grouped_applications(hits: HitArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split hit rows into applications: the row permutation, each group's start, its key row.

    An application is a ``(mol_idx, rule, app_idx)`` group -- the same key
    :func:`vertsy.expansions.layer_tables` sorts by and :func:`vertsy.filters.refused` groups on,
    for the same reason: a raw ``app_idx`` is local to a ``(product, rule)`` pair, so grouping
    without the rule pools one template's precursors with another's.
    """
    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    if not rows.shape[0]:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    keys = rows[order][:, :3]
    breaks = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    starts = np.concatenate(([0], breaks)).astype(np.int64)
    return order, starts, keys[starts]


def distinct_outcomes(
    keys: Sequence[Hashable],
    hits: HitArray,
    groups: int,
    intended: Sequence[Hashable] | None = None,
    rules: Sequence[int] = (),
) -> list[int]:
    """Per forward application group, how many distinct products came out of it.

    An *outcome* is the whole multiset of components an application emitted, sorted so two
    spellings of one outcome are one outcome. Counting distinct outcomes rather than distinct
    components is what makes the answer "how many products can this reaction give" and not "how
    many molecules did it emit".

    **That quotient is now needed, and this function does not have it.** The measurement this
    docstring used to rest on -- zero of 178 forward rows with a multi-component right-hand side,
    all 68,974 applications over 1,500 reference pots emitting exactly one, so keying the whole
    multiset was identical to keying the main product -- described a table that no longer exists.
    By-products have since been declared across the forward table to feed `vertsy.mixture`:
    **130 of 191 rows emit more than one component, and 75.7% of applications do.**

    So the question that measurement dismissed is live in principle: two reactions giving one
    product and different salts are not competing, but their multisets differ, so this would count
    them as two outcomes.

    **Measured, it almost never happens.** Over 2,070 applications from 60 reference products, the
    mean distinct-outcome count per application against the whole multiset versus against the
    heaviest component alone: `sites` 1.106 and 1.106, an inflation of **1.000x**; `competitors`
    1.251 and 1.248, **1.003x**. The reason is that two candidate rules firing on one pot and
    agreeing on the main product while disagreeing on the salt is a rare coincidence -- rules that
    differ usually differ in the product too.

    A global count does look alarming and is the wrong unit: 1,423 distinct outcome keys against
    603 distinct product keys across a corpus, 2.36x. That is the *key space* widening, which is
    expected and harmless, not the per-application count that gets priced. The first version of
    this note quoted that number as if it were the inflation; it is not.

    So the quotient remains unimplemented on purpose, and nothing is being paid for. If it is ever
    wanted, "heaviest" is not the right discriminator -- a Stille sheds trimethyltin bromide and an
    Appel bromination triphenylphosphine oxide, either of which can outweigh the product -- and the
    sound one, which component carries the rule's mapped atoms, is not visible to a function that
    receives only keys.

    **Nothing is subtracted here, and that is a change.** A two-component pattern matched against
    a pot built by disjoint union used to match both of its components *inside one molecule*
    whenever that molecule carried both groups -- which, for drug-like precursors, is most of the
    time -- and the cyclisation that resulted was counted as a competition. Measured by hand on
    real steps: a Suzuki's "competitors" included a four-membered N-bridged ring, a benzoxirene,
    and a gem-diol cyclopropane, none of which is a molecule. `require_span` worked around it by
    probing each starting material alone and subtracting what it made by itself.

    The forward rules now say it instead: every multi-component row's left-hand side is written
    `(A).(B)`, which the matcher enforces, so such a match cannot happen and there is nothing left
    to subtract. Measured over 15,601 applications on 20 PaRoutes targets, the subtraction changed
    not one count -- and it cannot, because a grouped pattern matched against a single molecule
    produces no outcome for the solo pass to find.

    ``rules`` is accepted and ignored. A count of products does not depend on which reaction was
    intended -- it is the same number whatever was being made -- and the parameter exists only so
    that this and :func:`severity_outcomes` share one :data:`Reduce` shape. A reduction that *does*
    depend on it is relative rather than absolute, which is the whole distinction between the two.

    ``intended`` is, per group, the key of the molecule the application was disconnecting, and an
    outcome **containing** it is not counted. A product that *is* the intended product cannot compete
    with itself: whatever route reached it, it is the thing the step was for. Two shapes of this
    turn up and both are counted by mistake without it. One reaction under two codes --
    `MC 75` and `iSCB 70` are the same esterification, each appearing as the other's competitor --
    and one bond by two mechanisms, `Hartenfeller 51g` Buchwald against `FWD H 54r` SNAr on an
    azine that admits both. Measured over 25 PaRoutes targets: 39 of 590 competitor products,
    6.6%.
    """
    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    seen: list[set[tuple[Hashable, ...]]] = [set() for _ in range(groups)]
    parts: dict[tuple[int, int, int], list[Hashable]] = {}
    for product_idx, rule, app, precursor_idx in rows.tolist():
        parts.setdefault((product_idx, rule, app), []).append(keys[precursor_idx])
    for (product_idx, _rule, _app), components in parts.items():
        seen[product_idx].add(tuple(sorted(components, key=repr)))
    if intended is not None:
        for group, outcomes in enumerate(seen):
            # *Contains* the intended product, not *equals* it. An outcome that made the intended
            # product alongside something else still made the intended product, so it is not
            # competing with it -- and equality missed exactly that. `FWD MC 85a` emits two
            # components today, and an ester reduction reproducing the intended alcohol beside the
            # other alcohol read as a competition. Adding by-products to the forward table would
            # make every row that shape, so this is the test that has to hold first.
            seen[group] = {
                outcome for outcome in outcomes if intended[group] not in outcome
            }
    return [len(outcomes) for outcomes in seen]


def severity_outcomes(
    efficiency: Sequence[float],
    intended_yield: Callable[[int], float],
    edges: Sequence[float],
) -> Reduce:
    """Per application group, **how badly the worst competitor hurts**, as a band ordinal.

    The alternative reduction to :func:`distinct_outcomes`, over the same hit rows, and it exists
    because a count is the wrong question. A pot can make several weak side products and still
    deliver the one you wanted; one reaction that consumes the substrate delivers nothing.
    Multiplicity carries almost no information here and severity carries nearly all of it, which is
    what the measurements said: the count reading of `competitors` fires on 2.02% of applications at
    2.8x enrichment, while of the applications where something destructive fired, 1,354 made one
    such product and 122 made several -- and a substrate consumed twice is not worse off than one
    consumed once.

    **The band is relative, because selectivity is.** ``efficiency`` is per *forward* rule, how
    completely that reaction consumes its substrate -- its ``typical_yield``, read in exactly the
    sense every other row's is: what fraction of material this reaction turns into its own product.
    Nothing is reinterpreted; the column means one thing and a row nobody wants is still a row.
    ``intended_yield`` is the same number for the retro rule that was *wanted*. What survives for
    the intended reaction is then ``y / (y + e)``: two reactions competing for one substrate split
    it by their rates, and standalone yield is the only proxy for a rate this table has.

    That form matters more than its numbers. ``1 - e`` was the first thing here and it is a **mass
    balance** -- it assumes the degradation runs first and takes its share -- whereas these are
    *parallel* competitors. The relative form also makes the intended reaction's own strength count,
    so a robust step resists a weak degradation better than a hard step does, which is chemistry the
    absolute form could not express.

    ``edges`` are descending survival thresholds, and the band is how many of them the worst
    survival falls below, plus two: band ``1`` is "nothing destructive fired", so clean lands on the
    price's free reference, and higher bands keep less. **Bands rather than the ratio itself**, for
    the reason ``typical_yield`` has four of them: both inputs are class-level priors from a
    four-band vocabulary, so the thirteen distinct ratios they can produce would claim a resolution
    nobody has measured. :func:`vertsy.costs.banded` is the other half, and it decodes what this
    encodes.

    ``intended`` drops the same outcomes a count drops, and for the same reason: a candidate that
    remade the molecule the step was for destroyed nothing, so it contributes no severity. Contains
    rather than equals, as above.
    """

    def reduce(
        keys: Sequence[Hashable],
        hits: HitArray,
        groups: int,
        intended: Sequence[Hashable] | None = None,
        rules: Sequence[int] = (),
    ) -> list[int]:
        rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
        parts: dict[tuple[int, int, int], list[Hashable]] = {}
        for product_idx, rule, app, precursor_idx in rows.tolist():
            parts.setdefault((product_idx, rule, app), []).append(keys[precursor_idx])
        # The *least* that survives, over the candidates that actually made something else.
        survived = [1.0] * groups
        for (product_idx, rule, _app), components in parts.items():
            if intended is not None and intended[product_idx] in components:
                continue
            spent = efficiency[rule] if 0 <= rule < len(efficiency) else 0.0
            if spent <= 0.0:
                # Not a destructive row, so it takes nothing from the intended reaction here. An
                # unknown rule lands here too, which is the opposite of a guard that invents
                # severity for anything it cannot place.
                continue
            wanted = intended_yield(int(rules[product_idx])) if rules else 0.0
            if wanted <= 0.0:
                continue
            survived[product_idx] = min(
                survived[product_idx], wanted / (wanted + spent)
            )
        return [1 + sum(1 for edge in edges if share < edge) for share in survived]

    return reduce


def _tally_by_rule(hits: HitArray) -> dict[int, int]:
    """Per forward rule, how many distinct applications it produced something on.

    Per ``(application, rule)`` first and then per rule, because a pattern with several matches in
    one pot returns several hit rows and the unit everything else here counts is the application.
    """
    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    if not rows.shape[0]:
        return {}
    # `(group, rule)` and deliberately not `(group, rule, app)`: a forward application id is minted
    # per match, so a pattern with four matches in one pot carries four of them, and including the
    # column counts one application four times -- which showed up at once as `fired` exceeding
    # `offered`.
    rules, counts = np.unique(np.unique(rows[:, :2], axis=0)[:, 1], return_counts=True)
    return dict(zip(rules.tolist(), counts.tolist(), strict=True))


def unmeasured(values: np.ndarray) -> int:
    """A :data:`vertsy.expansions.Reduce` for a count channel: disagreement means nobody measured.

    The store collapses two matcher applications onto one disconnection when they share a product,
    a rule and their precursors. If this channel gave them different counts then it is not a
    function of the chemistry it claims to describe, and neither value can be trusted -- so the
    disconnection carries :data:`vertsy.costs.UNPROBED`, which already means exactly that and
    already costs exactly nothing.

    Picking one of the two would be worse than useless: it would price a step by whichever row the
    store happened to write, and a wrong price is paid on every route that step appears in.
    """
    del values
    return UNPROBED


def template_arities(smarts: Sequence[str]) -> tuple[int, ...]:
    """How many components each forward template's left-hand side has.

    The number that decides whether a rule can be a *competitor*: a pattern needing two starting
    materials can only fire if both are in the pot, and matching against an application's combined
    precursors is exactly asking whether they are.

    Dots are counted only at zero level, which is the whole correction. A dot inside a zero-level
    group is *within* one molecule -- `(A.B)` is two fragments of a single starting material, the
    intramolecular form -- so counting it would admit an intramolecular rule to competitor
    analysis, where it would then match both halves inside one molecule and report the result as
    something two starting materials did to each other. That was the measured defect, and this is
    where it dies: the candidate never enters the channel.

    A template that writes no parentheses is unchanged, which is why the unmigrated library keeps
    behaving as it did.
    """
    return tuple(len(top_level_pieces(text.split(">>")[0])) for text in smarts)


def declared_molecularity(smarts: Sequence[str]) -> tuple[int, ...]:
    """How many separate starting materials each forward rule *states* it needs, or ``0``.

    Three answers, not two, and the third is the point. One top-level piece is one molecule. Two
    or more pieces, each in its own zero-level group, is that many molecules. Two or more pieces
    written with a bare dot **states nothing**: a query dot is no constraint, so the pattern
    matches across two molecules and inside one indifferently, and reading it as "two" is the
    assumption that produced the measured defect -- a cyclisation counted as a competition between
    starting materials that never met.

    ``0`` is that third answer, and it keeps the arithmetic honest everywhere: a competitor filter
    asking ``>= 2`` excludes it without having to know it exists, and nothing can mistake it for a
    count. :func:`template_arities` still reports the literal piece count, which is what a row's
    own example precursors have to match.

    For the shipped forward table this is currently the same as the piece count, because all 129
    multi-component rows are grouped and `test_no_forward_rule_leaves_its_left_hand_side_unstated`
    keeps them so. It matters for a row added later, and for `AOSTAR_HOST_EXTRA_TEMPLATES`.
    """
    return tuple(
        len(pieces) if len(pieces) == 1 or all_grouped(pieces) else 0
        for pieces in (top_level_pieces(text.split(">>")[0]) for text in smarts)
    )


def object_combine(union: Callable[[Sequence[M]], M]) -> Combine[M]:
    """A :data:`Combine` for an object array, built from a per-molecule disjoint union.

    The host shape of the operation: pull each group's members out of the batch, hand them to
    ``union``, and pack the results back. A backend that holds molecules one per slot can only do it
    this way, and a backend that holds them packed should not -- see
    :func:`vesma.metal_reaction_matcher.metal_combine`, which writes the blocks directly and never
    materialises a molecule at all.
    """

    def combined(
        precursors: MoleculeBatch[M], members: SidArray, indptr: SidArray
    ) -> Combined[M]:
        rows = cast("Sequence[M]", precursors)
        built = [
            union([rows[int(row)] for row in members[start:stop]])
            for start, stop in zip(indptr[:-1].tolist(), indptr[1:].tolist())
        ]
        # A host union has no size budget, so every group is buildable.
        return object_mol_array(built), np.ones(len(built), dtype=np.bool_)

    return combined


def paired_with(index: ForwardIndex) -> Callable[[int, int], bool]:
    """Whether a forward rule is one of the ones paired with a retro rule."""

    def paired(intended: int, candidate: int) -> bool:
        return candidate in index.get(intended, ())

    return paired


def forward_probe(
    forward: ForwardMatcher[M],
    candidates: Callable[[int], Sequence[int]],
    *,
    channel: str,
    key: Callable[[MoleculeBatch[M]], Sequence[Hashable]],
    combine: Combine[M] | None = None,
    drop_intended: bool = False,
    reduce: Reduce = distinct_outcomes,
    report: Callable[[ProbeReport], None] | None = None,
) -> ReactionScore[M]:
    """Measure the forward outcomes of every application, application by application.

    ``forward`` is an ordinary reaction matcher over *forward* templates. The machinery is
    direction-agnostic -- it matches a pattern's left-hand side and applies its edit -- so what a
    retro matcher calls precursors, this one calls products, and no new chemistry code is needed.

    ``candidates`` says which forward rules to run against an application, given the retro rule that
    made it. **It is the only thing that separates the two channels**, and everything else here is
    shared: :func:`site_probe` passes the rules paired with the retro rule, and
    :func:`competitor_probe` passes the multi-component rules that are not. It is consulted per
    application, so the matcher is asked for a handful of ``(molecule, rule)`` pairs rather than an
    all-vs-all pass -- which is what makes the probe affordable at all, since a forward library is a
    few hundred rules against a retro library of tens of thousands.

    ``combine`` builds one molecule per multi-precursor application, because a bimolecular forward
    pattern has a two-component left-hand side and has to match across both starting materials at
    once. It is handed the batch and a **CSR grouping** -- the member rows and where each group
    starts in them -- rather than lists of molecules, which is the whole reason this probe runs
    anywhere: nothing here ever holds a molecule on its own, so a packed device batch works exactly
    as an object array does. Omit it and only *unimolecular* applications are probed, where the
    combination is a batch gather; that is the honest behaviour for a backend with no such
    operation, and the competitor channel needs one by construction since every rule it runs has two
    or more components.

    ``key`` names the products the same way the run names its molecules, and is handed the whole
    returned batch rather than its elements -- so a backend that already carries keys can answer
    with them and nothing crosses back to the host.

    ``drop_intended`` does not count an outcome equal to the molecule the application was
    disconnecting. **Only the competitor channel wants this**, and the asymmetry is the whole
    difference between the two: a competitor that remakes the intended product is not competing
    with anything, while for `sites` the intended product is exactly what is being counted -- it is
    the first of the places the intended reagent attacked, and dropping it would turn a perfectly
    selective step from one into zero.

    **What the two starting materials do together** is now the pattern's own statement rather
    than this function's. A multi-component forward rule is written `(A).(B)`, which requires its
    pieces to match in different components of the pot, so the cyclisation a bare dot used to admit
    -- "these starting materials could do this to each other", silently unenforced -- is simply not
    a match. The earlier workaround, a second pass over each member alone with its outcomes
    subtracted, is gone: it provably subtracts nothing once the rules are grouped, and
    `vertsy.template_lint.ungrouped_components` is what keeps them that way.
    """

    def measure(
        products: MoleculeBatch[M], precursors: MoleculeBatch[M], hits: HitArray
    ) -> Column:
        # `products` is read only to name what each application was disconnecting, and only when
        # `drop_intended` asks. A selectivity count is otherwise a question about one side: what the
        # forward chemistry does to the precursors, with the frontier they came from saying nothing
        # extra -- unlike mass balance, which is a comparison and needs both.
        rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
        # `UNPROBED` and not zero, because zero is an answer: an application whose rule has no
        # forward partner, or whose precursors no backend here can combine, was never examined, and
        # a price that cannot tell that from "examined, made nothing" charges a broken pairing the
        # same as a clean step.
        counts = np.full(rows.shape[0], UNPROBED, dtype=np.int64)
        order, starts, group_keys = grouped_applications(hits)
        if not len(starts):
            if report is not None:
                report(ProbeReport(channel, 0, 0, 0, 0, 0))
            return cast("Column", counts)

        stops = np.concatenate((starts[1:], [len(order)])).astype(np.int64)
        members: list[int] = []
        indptr: list[int] = [0]
        probe_rows: list[int] = []
        # The intended retro rule per probed group, which a relative reduction needs and a count
        # ignores. Parallel to `probe_rows`, so the reduction indexes it the way it indexes groups.
        probe_rules: list[int] = []
        pairs: list[Pair] = []
        unpaired = unbuildable = 0
        for group, (start, stop) in enumerate(zip(starts.tolist(), stops.tolist())):
            rule = int(group_keys[group][1])
            forward_rules = candidates(rule)
            if not forward_rules:
                unpaired += 1
                continue
            group_rows = [int(rows[row, 3]) for row in order[start:stop]]
            if len(group_rows) > 1 and combine is None:
                unbuildable += 1
                continue
            position = len(probe_rows)
            members.extend(group_rows)
            indptr.append(len(members))
            probe_rows.append(group)
            probe_rules.append(rule)
            pairs.extend((position, forward_rule) for forward_rule in forward_rules)

        # Gathered for the probed groups only. Keying the whole frontier to use a few of its rows
        # would cost a pass over every molecule in it; on a backend that already carries keys this
        # is free either way.
        intended: Sequence[Hashable] | None = None
        if drop_intended and probe_rows:
            disconnected = np.asarray(
                [int(group_keys[group][0]) for group in probe_rows], dtype=np.int64
            )
            intended = key(products[disconnected])
            if len(intended) != len(probe_rows):
                # Loud rather than an IndexError twenty lines later, and kept even though no
                # backend fails it today. It states the contract a `key` block owes this channel:
                # **total over the batch type it declares**, one key per row, including a frontier
                # gather. The Metal arm used to fail exactly here, because its keyer read tokens
                # stored on the batch and a frontier carried none; it computes them now and the
                # condition is unreachable. A short list would silently drop the same-product
                # filter and count every competitor that merely remade the intended product, so
                # the guard outlives the backend that motivated it.
                raise ValueError(
                    f"`key` answered {len(intended)} keys for {len(probe_rows)} probed "
                    "applications, so the intended product cannot be named. A backend's `key` "
                    "must be total over the batch type it declares, this one is not."
                )

        answered = np.zeros(len(probe_rows), dtype=np.int64)
        buildable = np.ones(len(probe_rows), dtype=np.bool_)
        fired_by_rule: dict[int, int] = {}
        if probe_rows:
            member_rows = np.asarray(members, dtype=np.int64)
            offsets = np.asarray(indptr, dtype=np.int64)
            # A gather when every group has one member -- which every batch supports -- and the
            # backend's own combination otherwise.
            if combine is None:
                probes = precursors[member_rows]
            else:
                probes, buildable = combine(precursors, member_rows, offsets)
            # A group the backend could not build is *unexamined*, so its pairs are dropped rather
            # than run against whatever placeholder the batch holds for it. Leaving them in would
            # return no hits and read as `SILENT`, charging a device atom budget as a broken
            # pairing -- the one confusion the sentinel exists to prevent.
            if not bool(buildable.all()):
                pairs = [pair for pair in pairs if buildable[pair[0]]]
            made, forward_hits = forward.match(probes, pairs)
            answered = np.array(
                reduce(key(made), forward_hits, len(probe_rows), intended, probe_rules),
                dtype=np.int64,
            )
            fired_by_rule = _tally_by_rule(forward_hits)
        for group, measured, built in zip(
            probe_rows, answered.tolist(), buildable.tolist()
        ):
            if built:
                counts[order[starts[group] : stops[group]]] = measured
        refused = int(np.count_nonzero(~buildable))
        if report is not None:
            report(
                ProbeReport(
                    channel=channel,
                    applications=len(starts),
                    probed=len(probe_rows) - refused,
                    unpaired=unpaired,
                    silent=int(np.count_nonzero(answered[buildable] == 0)),
                    unbuildable=unbuildable + refused,
                    # Counted from `pairs` after the unbuildable groups were dropped from it, so a
                    # rule is "offered" only where it actually got a molecule to match against.
                    offered_by_rule=dict(Counter(rule for _position, rule in pairs)),
                    fired_by_rule=fired_by_rule,
                )
            )
        return cast("Column", counts)

    return measure


def site_probe(
    forward: ForwardMatcher[M],
    index: ForwardIndex,
    *,
    key: Callable[[MoleculeBatch[M]], Sequence[Hashable]],
    combine: Combine[M] | None = None,
    report: Callable[[ProbeReport], None] | None = None,
) -> ReactionScore[M]:
    """How many distinct products the **intended** reaction makes: competing sites.

    The candidates are the forward rules *paired* with the retro rule that made the disconnection.
    Pairing is a statement about a reagent's scope rather than a template's mirror image -- a
    chlorinating agent does not know which hydroxyl was meant, so the primary and the secondary
    chlorination both probe the same disconnection, and together they say three where either alone
    says one.

    More than one product here means a mixture to separate, which is a cost and not a failure.
    """
    return forward_probe(
        forward,
        lambda intended: index.get(intended, ()),
        channel="sites",
        key=key,
        combine=combine,
        report=report,
    )


def competitor_probe(
    forward: ForwardMatcher[M],
    declared: Sequence[int],
    *,
    key: Callable[[MoleculeBatch[M]], Sequence[Hashable]],
    combine: Combine[M],
    index: ForwardIndex | None = None,
    competes: Competes | None = None,
    admit_unimolecular: bool = False,
    channel: str = "competitors",
    reduce: Reduce = distinct_outcomes,
    report: Callable[[ProbeReport], None] | None = None,
) -> ReactionScore[M]:
    """How many distinct products *other* reactions make from the same starting materials.

        **Only multi-component rules**, and that filter is the whole physical argument. A pattern
        needing two starting materials can fire only if both are present, and they are: the probe
        matches against exactly this application's precursors, which is what the pot contains. So a hit
        means *these starting materials could also do this to each other*, a real competition.

        A **unimolecular** rule matching a functional group elsewhere on the substrate means nothing of
        the kind. It needs its own reagent, catalyst and conditions, none of which are in the pot, so
        counting it would read every polyfunctional molecule as a mixture and measure molecular
        complexity rather than a competition. That is why this channel is not simply "the whole library
        on the whole substrate".

        ``index`` excludes the intended reaction's own rules, which belong to the sites channel and
        would otherwise be counted twice under two different penalties.

        ``competes`` narrows it further and is the one plug point everything else would enter through.
        Its default -- every unpaired multi-component rule competes -- is the crudest honest reading: a
        favourability table would drop candidates far less favourable than the intended reaction, and a
        conditions table or an explicit allow-list of pairs would drop candidates that cannot run under
        the intended reaction's conditions at all. Neither exists yet, and neither changes anything
        here but this predicate.

    ``declared`` is :func:`declared_molecularity`, not :func:`template_arities`: a rule that does
        not *state* that its pieces are separate molecules does not compete. A bare dot makes no such
        statement -- it matches inside one molecule as readily as across two -- so admitting it would
        put back exactly the reading this channel was found to be making wrongly. Unstated rules are
        dropped rather than guessed at, which is the same choice `(A).(B)` records in the template.

        ``admit_unimolecular`` drops that filter, and the reason it exists is that the filter is a
        **proxy**. "Only multi-component rules" stands in for "its reagent is not in the pot", which
        is what `competes` now says directly. Measured over 2665 applications once every forward row
        declared its own conditions, mixture peaks per application:

        =========================== ======== ==========
        channel                     ungated  gated
        =========================== ======== ==========
        multi-component only        0.7      0.5
        the whole forward table     12.5      1.0
        =========================== ======== ==========

        So the honest filter brings the whole table to the same order as the proxy did, and the 0.5
        extra peaks are unimolecular reactions whose reagent *is* supplied -- a reduction competing
        in a reducing pot, which is a real peak and one the proxy hid. Default still ``False``: this
        changes what a run reports, so flipping it is a measured decision rather than a side effect.
        Before 190 forward rows were tagged the same relaxation gave 12.4 gated peaks per
        application, which is why this argument did not exist until they were.

        ``combine`` is required rather than optional, because every rule this runs has two or more
        components and there is nothing to match without it.

        **Measured, the default reading is not yet discriminating.** Over four PaRoutes targets at 16x6
        against medchem + hartenfeller + protecting-groups, with 34 multi-component forward rules: every
        application is examined, and **57.3% of them have at least one competing reaction**. A signal
        that fires on more than half of everything separates very little, and multiplying the step cost
        of half a graph is a large intervention to make on it. That number is the argument for
        ``competes`` rather than an argument against the channel: what it says is that "another rule
        could also join these two molecules" is true far more often than "another rule *would win*", and
        the difference between those is favourability and conditions. Until one of those tables exists,
        prefer measuring this channel without pricing it.
    """
    multi = (
        tuple(range(len(declared)))
        if admit_unimolecular
        else tuple(rule for rule, stated in enumerate(declared) if stated >= 2)
    )
    paired = paired_with(index or {})
    allowed = competes if competes is not None else lambda intended, candidate: True

    # Resolved once per intended rule rather than once per application. The answer is a pure
    # function of `intended` -- `multi`, `paired` and `allowed` are all fixed when this closure is
    # built -- while the probe asks it for every disconnection in every layer, which measured
    # 29,018 calls and 0.27 s on a single two-target run. A dict rather than `lru_cache` because
    # the key space is the rule table and so is small and known, and because the cache then lives
    # and dies with the probe instead of outliving it on the function object.
    resolved: dict[int, Sequence[int]] = {}

    def candidates(intended: int) -> Sequence[int]:
        found = resolved.get(intended)
        if found is None:
            found = tuple(
                candidate
                for candidate in multi
                if not paired(intended, candidate) and allowed(intended, candidate)
            )
            resolved[intended] = found
        return found

    # `channel` names the report, so a sub-channel's measurements are attributable to it. This was
    # hardcoded to "competitors", which meant every sub-channel's `ProbeReport` claimed the base
    # name: `rule_attribution` found nothing under `competitors:same-site` and summed all three
    # under `competitors`. The selectivity columns were always keyed correctly, so the enrichment
    # tables were right and only the per-rule attribution was wrong.
    return forward_probe(
        forward,
        candidates,
        channel=channel,
        key=key,
        combine=combine,
        drop_intended=True,
        reduce=reduce,
        report=report,
    )
