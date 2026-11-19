"""Vetoes: measure a verdict about every disconnection, then drop the ones it condemns.

Two arrows, not one, and the split is the whole design:

    measured(match, reactions={"vetoed": vetoed(...)})   # judge
    refused(match, applications=("vetoed",))             # enforce

A verdict is an ordinary :data:`vertsy.search.ReactionScore` -- one value per hit row, exactly like a
selectivity count -- and :func:`refused` is one function that reads named columns and knows nothing
about what produced them. So a new veto is a new column and never a new wrapper, several arrive
together in one mapping rather than by nesting, and everything arm-specific lives inside the score
where a packed device batch is as good as an object array.

This is deliberately upstream of everything else: a disconnection removed here never enters the
store, never gets valued, never competes for a width slot, and never appears in a route. Every other
lever this package offers only *reorders* candidates; a veto changes the set.

That distinction is what makes it worth more than ranking, and it is also why the first one had to be
written. A small number of PaRoutes templates lose mapped atoms -- **71 of 67,557 in raw68k, 4 of
13,978 in used14k** -- and rarity is no protection, because a search seeks them out. The clearest
example is an ester hydrolysis::

    [C:4]-[O;H0;D2;+0:3]-[C:2]=[O;D1;H0:1]  >>  [O;D1;H0:1]=[C:2]-[OH;D1;+0:3]

``[C:4]`` is matched on the product side and absent from the precursors, so RDKit deletes it and
everything hanging off it. Applied to a 45-heavy-atom ester it returns 4-nitrobenzoic acid alone, and
the alcohol half -- 33 atoms of it -- simply vanishes.

A search does not merely tolerate such a rule, it **hunts for it**: a disconnection that discards most
of a molecule yields a precursor that is small and already in the catalogue, which is the most
attractive thing a stock-distance heuristic can see. One such application in 1,493 was enough to
"solve" four complex natural products in two or three steps each -- so a defect present in a tenth of
a percent of the library set every solve rate measured against it.

:mod:`vertsy.template_lint` finds the same fault in the template text, before any molecule is matched,
and is the cheaper of the two by the number of applications a run makes. It is opt-in and wired
nowhere, because removing templates changes which library a benchmark ran against.

Why the two arrows, and why in that order
-----------------------------------------
They do not commute, and the asymmetry is the point. Measuring and then refusing costs a verdict on
every application and can say what it refused; refusing and then measuring is cheaper and cannot.
**Composition order is therefore the choice between cheap and reportable**, which is worth saying out
loud because the cheap order looks like an optimisation.

The first use of a new veto is the reportable order with no refusal at all: measure how often it
*would* fire, over a real sweep, and change not one route. That is how a hand-written rule earns the
right to remove things, and it is the order of operations that showed a selectivity multiplier was
not worth shipping.

The two axes, and the two kinds of veto
---------------------------------------
:class:`vertsy.search.Expansion` carries *"exactly two id spaces and no third"*, and those are exactly
the two claims a veto can make. "These reagents will not react as intended" is per application, a
``ReactionScore``. "This molecule cannot exist" is per precursor, a ``MoleculeScore``. :func:`refused`
reads both, and a molecule-axis refusal *induces* a reaction-axis one through a hit row's precursor
column, so there is no third concept.

Rejection is per *application*, never per edge: a disconnection is a conjunction, so keeping half of
one would claim a molecule can be made from some of the pieces it needs.

The table of structural vetoes
------------------------------
A veto row is a query pattern plus a scope, and when it fires the disconnection is condemned. The
design is ``docs/note-vetoes.md``; what follows is the contract.

**What the pattern is matched against.** ``on=product`` matches the molecule being disconnected;
``on=precursors`` matches **the disjoint union of everything it disconnects to**, not each precursor
in turn. That single choice is why one mechanism covers both kinds of real failure: a boronic acid
carried through an NBS bromination is boron on the brominated precursor, which a one-component
pattern finds, while a phenoxide and a benzyl bromide in one pot is a *two-component* pattern matched
across the union -- which is how "these reagents react with each other instead" is said. The union is
built by a :data:`vertsy.selectivity.Combine`, the same injected operation the forward probe uses, so
nothing here ever holds a molecule on its own and a packed device batch works as an object array does.

**Which side to read, and why it matters.** A veto on the precursor union has no atom mapping, so it
cannot tell the template's own substrate from an interferent: forbidding a benzylic halide on the
precursors of an N-alkylation would veto the reaction that uses one. So a group the step *consumes*
is read on the product instead, where a leaving group still present is one the step did not use. And
where the incompatibility is between the template's substrate and a foreign group, name only the
foreign group -- the substrate is present by construction, since the template would not have fired
without it.

Pre- versus post-filter is therefore not a property of a veto but of (direction x side): on a retro
template ``on=product`` is the matcher's *input* and ``on=precursors`` its output, and on a forward
template it is the other way round. That is why these are called vetoes and not filters of either
half.

**Scope, and why it is affordable.** A row names the templates it applies to, looked up per
application on the rule column -- the access pattern :func:`vertsy.selectivity.forward_probe` relies
on for the same reason, so the matcher is asked for a handful of ``(pattern, molecule)`` pairs rather
than an all-vs-all pass over tens of thousands of rules. An empty scope means every template, which
is right for a row about a molecule rather than about a reaction: "no free acyl chloride anywhere"
needs no list.

**What a row may express**, tighter than SMARTS, and the limits bind every arm equally: **no
recursive** ``$()``, refused when the table is built -- vesma's CPU matcher evaluates it, but the
device backends do not yet, and a row only the host arm could answer would make the arms disagree
-- which for now rules out lifting PAINS, Brenk or the Daylight alert sets wholesale; **no chirality**; and **no ring membership** (``R``,
``r``), which depends on one spanning forest's fundamental cycles and which the GPU backends leave
unknown -- the same prohibition ``tests/data/templates/README.md`` imposes on hand-written templates.
Available: element, aromaticity, charge, explicit degree, total connectivity, valence, total and
implicit hydrogen counts, isotope, ring *connectivity* (``x``), bond order, and bond-level ``@`` /
``!@``. Enough for the reagent incompatibilities that motivated this, which is why the rows are
hand-written and flat.

What belongs here and what does not
-----------------------------------
Vetoes answer yes or no about a disconnection, and they are curated chemistry. There used to be one
that was not -- a mass-balance veto, refusing an application whose precursors weighed less than its
product -- and it is gone, because **31% of the hand-written templates lose heavy atoms legitimately**:
the atoms come from a reagent nobody writes as a precursor, so a Miyaura borylation sheds two, an
Fmoc installation seventeen. It was the wrong signal even on the PaRoutes corpus it was measured
against, where 33 templates lose mass and one is actually broken.

The defect it was reaching for is detected exactly, and at the root: an application that leaves a
piece of the molecule in no precursor the template named is refused by
:func:`vesma.reaction.precursors_from_mapping` itself, with the dropped fragment's formula on the
diagnostics, and :mod:`vertsy.template_lint` finds the same fault in the template text before
anything is matched. Neither needs a veto.

*Scores* do not belong here. "This disconnection is one of five regiochemical options" or "this
reaction runs at 40% yield" are not vetoes, and collapsing them to one would throw away what makes
them useful. They belong on the cost of an application, which is :class:`vertsy.costs.StepPrice` and
can read the precursors a template fired on -- :func:`vertsy.selectivity.probed` is the wrapper that
measures one, and it is a score rather than a veto for exactly this reason. A veto changes the set of
routes; a score changes only their order, and a wrong score therefore costs a ranking rather than an
answer.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .core import HitArray, K, M
from .interner import SidArray
from .search import (
    Column,
    Expansion,
    Match,
    MoleculeBatch,
)
from .selectivity import Combine, grouped_applications

if TYPE_CHECKING:
    from vesma.cmol import CMol
    from vesma.gmol import GMol

__all__ = [
    "CLEAR",
    "SENSES",
    "FORCES",
    "SIDES",
    "COSTS",
    "UNEXAMINED",
    "Screen",
    "Force",
    "Sense",
    "Side",
    "Cost",
    "DeficitReport",
    "Veto",
    "VetoReport",
    "gathered",
    "host_screen",
    "read_tags",
    "read_vetoes",
    "refusal_wins",
    "unaccounted",
    "refused",
    "validated",
    "verdict_rows",
    "vetoed",
    "vetoes_in_scope",
]


UNEXAMINED = -1
"""Nobody could judge this row.

Out of band rather than ``0`` for the reason :data:`vertsy.costs.UNPROBED` is: zero is a *result* --
the veto ran and had no objection. An arm that cannot evaluate a pattern, or a precursor past an atom
cap, is a row nothing examined, and a layer that conflates the two reads a silently skipped veto as
chemistry that passed it.

An unexamined row is **not** refused, because claiming it would be inventing a verdict. Counting them
is the measuring arrow's job, not :func:`refused`'s -- a ``Match -> Match`` has nowhere to report to.
"""

CLEAR = 0
"""The veto ran and had no objection. Anything greater is a refusal."""


def refusal_wins(values: np.ndarray) -> int:
    """A :data:`vertsy.expansions.Reduce` for a verdict column: the strictest value survives.

    The opposite policy from :func:`vertsy.selectivity.unmeasured`, on the same mechanism, which is
    why neither belongs in the store. A count that contradicts itself has measured nothing. A
    *verdict* that contradicts itself has still found something: one of the two applications the
    store collapsed was condemned, and a disconnection with a refusal on file is refused. The same
    asymmetry :func:`verdict_rows` relies on -- a refusal is definite, a gap is not.

    ``UNEXAMINED`` therefore loses to any real verdict, which `max` gives for free: it is negative,
    `CLEAR` is zero and a fired count is positive.
    """
    return int(np.asarray(values, dtype=np.int64).max())


def gathered(column: Column, rows: np.ndarray) -> Column:
    """Narrow one column to the hit rows that survived, by the same index."""
    return column[rows]


def verdict_rows(verdict: Column) -> np.ndarray:
    """Which rows a verdict column condemns.

    One place, because the convention is a convention and not an invariant anyone can check: a count
    of how many vetoes fired, ``CLEAR`` for none and ``UNEXAMINED`` for an unjudged row. A *count*
    rather than the identity of the offender, which is what makes the molecule and application axes
    combine by plain disjunction with no winner to pick -- a refusal is a refusal. Which rule fired
    is a question for an aggregate report, or for re-running the vetoes on the one application.
    """
    return np.asarray(verdict, dtype=np.int64) > CLEAR


def refused(
    match: Match[M, K],
    *,
    applications: Sequence[str] = (),
    molecules: Sequence[str] = (),
) -> Match[M, K]:
    """Wrap ``match``, dropping every disconnection a named verdict column condemns.

    The two arguments are the two axes, mirroring :func:`vertsy.search.measured`, which is what put
    the columns there. ``applications`` names columns parallel to ``hits``; ``molecules`` names
    columns parallel to ``keys``, and a condemned precursor condemns every application that names it.

    **A missing name raises.** Asking to enforce a verdict nobody measured is a wiring error, and the
    alternative -- treating absence as consent -- is precisely the confusion the sentinels exist to
    prevent: a run with its vetoes unwired would look exactly like a run whose chemistry passed them.

    Grouped by ``(product, rule, application)`` and **not** by ``(product, application)``: a raw
    application id is minted per rule, so the same id recurs under different rules and grouping
    without the rule silently merges disconnections that have nothing to do with each other. That is
    the key :func:`vertsy.expansions.layer_tables` sorts by, for the same reason. Getting it wrong
    does not raise -- it pools one rule's precursors with another's, which is precisely enough to hide
    a mass-losing rule behind a well-behaved one.

    Survivors keep their rows unchanged so ids downstream mean what they meant, and the batch and keys
    are left whole rather than compacted: a precursor no surviving application names is simply never
    interned -- :func:`vertsy.expansions.add_layer` reads the hit rows, not the batch -- so compacting
    would cost a rebuild to save nothing.

    **This wrapper requires nothing of a batch.** It reads columns and hit rows only, so it works on
    an object array and on a device batch alike. That was not true of its predecessor, which asked a
    Python predicate about one disconnection at a time and so had to iterate the molecules; moving the
    judgement into a column is what removed the restriction rather than working around it.
    """

    def matched(mols: MoleculeBatch[M]) -> Expansion[M, K]:
        found = match(mols)
        rows = np.asarray(found.hits, dtype=np.int64).reshape(-1, 4)
        if not rows.shape[0]:
            return found

        condemned = np.zeros(rows.shape[0], dtype=bool)
        for name in applications:
            condemned |= verdict_rows(_column(found.reaction_scores, name, "reaction"))
        for name in molecules:
            per_precursor = verdict_rows(
                _column(found.molecule_scores, name, "molecule")
            )
            condemned |= per_precursor[rows[:, 3]]

        if not bool(condemned.any()):
            return found

        groups, inverse = _application_groups(rows)
        doomed = np.bincount(inverse, weights=condemned, minlength=groups.shape[0]) > 0
        kept = np.flatnonzero(~doomed[inverse])
        # Reaction scores are per hit row, so they follow the same gather -- which is the whole reason
        # that axis was chosen. Molecule scores are untouched, like the batch they parallel.
        return replace(
            found,
            hits=rows[kept],
            reaction_scores={
                name: gathered(column, kept)
                for name, column in found.reaction_scores.items()
            },
        )

    return matched


@dataclass(frozen=True)
class DeficitReport:
    """What one no-mass-loss pass found, per rule, so the table can be repaired rather than guessed at."""

    applications: int
    """Disconnections the pass judged."""
    unaccounted: int
    """Disconnections whose precursors do not account for every atom of the product."""
    atoms: int
    """Heavy atoms unaccounted for, summed. A rule short by one reads very differently from one
    short by seventeen: the first is an implicit halogen or hydroxyl, the second a whole reagent."""
    by_rule: Mapping[int, int]
    """Per rule, the disconnections it came up short on. The runtime list of templates still
    leaving a reagent implicit, which is the thing a reader can act on."""


def unaccounted(
    sizes: Callable[[MoleculeBatch[M]], np.ndarray],
    *,
    slack: int = 0,
    report: Callable[[DeficitReport], None] | None = None,
) -> Callable[[MoleculeBatch[M], MoleculeBatch[M], HitArray], Column]:
    """A verdict on **retro** applications whose precursors do not account for the product's atoms.

    *No mass loss*, and deliberately not mass conservation. A retro template may legitimately name
    **more** atoms than the product has -- a reagent contributes atoms that leave again as a
    by-product nobody writes down -- so the constraint is one-sided: the precursors may weigh more
    than the product and must not weigh less.

    One-sided and **retro only**. Forward, the same comparison means nothing: an alkylation adds a
    whole group and a condensation expels water, so neither direction of inequality is a defect.

    What it enforces is that **every atom of the product can be bought**. A retro step whose
    precursors are short has produced those atoms from nowhere, and a search does not merely
    tolerate such a step -- it seeks it out, because a disconnection that quietly invents atoms
    yields precursors that are smaller and closer to the catalogue than honest ones.

    An earlier version of this lived here, was measured against the *hand-written* library, and was
    deleted: 31% of those templates came up short, every one of them legitimately, because the atoms
    came from a reagent nobody wrote as a precursor. That measurement was right and the conclusion
    drawn from it was too narrow. The deficit is real either way -- those routes cannot be costed,
    because the reagent they need is not a molecule the search ever has to find -- so the repair is
    to **name the reagent**, and this is the runtime check for whether that has been done. Measured
    per rule for exactly that reason: the report is a worklist.

    Opt-in, because it is a claim about a library rather than about chemistry. A vendored table
    nobody intends to repair should be searched as it is, with the deficit reported and nothing
    refused.

    ``sizes`` counts one batch's heavy atoms and is injected because every backend spells it
    differently -- a packed Metal batch carries ``atom_counts``, a ``cmol`` array has
    ``atom_count`` per molecule, RDKit has ``GetNumHeavyAtoms``. Guessing is how such a thing
    silently becomes metal-only.

    ``slack`` tolerates a deficit of that many atoms. The honest default is ``0``: a template short
    by one has left a halogen or a hydroxyl implicit, which is the same defect as leaving a whole
    reagent implicit and is more common.

    **It knows about no other veto and no other veto knows about it.** The verdict is a named column
    like any other, and :func:`refused` takes several names and combines them by disjunction, so
    composing this with the pattern table is naming both and nothing else.
    """

    def measure(
        products: MoleculeBatch[M], precursors: MoleculeBatch[M], hits: HitArray
    ) -> Column:
        rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
        counts = np.full(rows.shape[0], CLEAR, dtype=np.int64)
        if not rows.shape[0]:
            return cast("Column", counts)
        wanted = np.asarray(sizes(products), dtype=np.int64)
        supplied_by = np.asarray(sizes(precursors), dtype=np.int64)
        groups, inverse = _application_groups(rows)
        # Summed over the application's rows, because an application is a conjunction: it needs
        # every precursor it names, so what it supplies is their total.
        supplied = np.bincount(
            inverse, weights=supplied_by[rows[:, 3]], minlength=groups.shape[0]
        ).astype(np.int64)
        deficit = wanted[groups[:, 0]] - supplied
        short = deficit > slack
        verdict = np.where(short, 1, CLEAR)
        if report is not None:
            rules, tallies = np.unique(groups[short][:, 1], return_counts=True)
            report(
                DeficitReport(
                    applications=int(groups.shape[0]),
                    unaccounted=int(np.count_nonzero(short)),
                    atoms=int(deficit[short].sum()),
                    by_rule=dict(zip(rules.tolist(), tallies.tolist(), strict=True)),
                )
            )
        return cast("Column", verdict[inverse].astype(np.int64))

    return measure


def validated(
    match: Callable[..., tuple[Any, Any, Any]],
    *,
    verdict: Callable[[Any, Any, HitArray], Column],
    report: Callable[[Mapping[str, int]], None] | None = None,
) -> Callable[..., tuple[Any, Any, Any]]:
    """Wrap a **replay** match so a condemned application cannot be reproduced.

    The same predicate as :func:`refused` and the same disjunction, at the other end of the
    pipeline. Replay is where an expensive veto belongs, because it is the only place molecules
    exist again: :func:`vertsy.replay.template_replay_routes` rehydrates them to re-run the
    templates, where the store holds interned ids and nothing to screen.

    The shapes differ and that is all that stops ``refused`` being reused here. A layer match
    answers a :class:`vertsy.search.Expansion`; a replay match answers
    ``(precursors, keys, hits)``. The verdict is the same :data:`vertsy.search.ReactionScore`
    either way, which is what makes this a wrapper rather than a second mechanism.

    **Dropping a row is how a route is refused.** Replay reproduces a route step by step and reports
    a mismatch when a step cannot be reproduced; a condemned application simply is not there to
    reproduce, so the route fails to replay and the caller moves to the next one. That is the
    cheapest possible implementation of "extract the best route, validate it, take the next if it
    fails", and it needs no route-exclusion machinery.

    What it does *not* yet do is cache the verdict per application so the next extraction avoids the
    disconnection rather than rediscovering it. That is the step that turns this from a validator
    into a learned constraint, and it is the one piece of the design still missing.

    ``report`` receives per-veto refusal counts for the call, so a run can say how many candidate
    routes validation refused and by which rows. That number is the point: large, while routes are
    missing, tells a reader to turn validation off and read the veto table.
    """

    def inner(molecules: Any, pairs: Any) -> tuple[Any, Any, Any]:
        precursors, keys, hits = match(molecules, pairs)
        rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
        if not rows.shape[0]:
            return precursors, keys, hits
        column = np.asarray(verdict(molecules, precursors, rows), dtype=np.int64)
        condemned = verdict_rows(cast("Column", column))
        if not bool(condemned.any()):
            return precursors, keys, hits
        groups, inverse = _application_groups(rows)
        doomed = np.bincount(inverse, weights=condemned, minlength=groups.shape[0]) > 0
        kept = np.flatnonzero(~doomed[inverse])
        if report is not None:
            report({"refused": int(doomed.sum())})
        return precursors, keys, rows[kept]

    return inner


def _application_groups(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct ``(product, rule, application)`` triples, and each row's triple."""
    groups, inverse = np.unique(rows[:, :3], axis=0, return_inverse=True)
    return groups, inverse.reshape(-1)


def _column(columns: Mapping[str, Column], name: str, axis: str) -> Column:
    if name not in columns:
        known = ", ".join(sorted(columns)) or "none"
        raise KeyError(
            f"no {axis} column named {name!r} to refuse by; this layer carries {known}. "
            "A verdict has to be measured before it can be enforced -- wrap the match in "
            "`measured` with that name first. Absence is not consent."
        )
    return columns[name]


type Cost = Literal["pattern", "geometry"]
"""What evaluating a veto costs, which is what decides *where* it may be evaluated.

This replaced a ``stage`` field whose values were ``expansion`` / ``extraction`` / ``replay``, and
the replacement is a simplification rather than a rename. That field named a mechanism, so only
someone who knew the pipeline could fill it in -- and one of its three values could not work at
all: ``extraction`` holds interned ids and no molecules, so a veto that screens a molecule, which
is every veto there is, had nothing to screen there. The question a chemist can answer instead is
**is this check cheap or expensive?**

``pattern`` is a subgraph match, cheap enough for the inner loop, and may run anywhere.

``geometry`` embeds a structure -- *no 3D arrangement satisfies these distances*, the predicate
that catches every absurd bridged system without a rule being written for any of them. It is far
too slow per application per layer and **may never run at expansion**. That prohibition is the
point of declaring a cost at all: it is a safety property, enforced by what schedules the veto
rather than left to care.

Where an expensive veto *does* run is replay, because that is the only place molecules exist again
-- :func:`vertsy.replay.template_replay_routes` rehydrates them to re-run the templates. A verdict
reached there is cached per *application* and not per route, so one expensive evaluation excludes
every route that uses that disconnection, and the anytime schedule's next pass sees what this one
learned.
"""

type Side = Literal["product", "precursors"]
type Sense = Literal["forbid", "require"]

type Force = Literal["refuse", "price"]
"""What a fired row does: remove the disconnection, or make it more expensive.

A **third axis**, and deliberately not folded into :data:`Cost`. Cost says what evaluating the
predicate costs, and so where it may be evaluated; force says what a fired row *does*. The same
predicate, at the same price, can be a wall or a surcharge.

The split exists because a chemist does not grade incompatibilities one way. A boronic acid carried
through NBS is destroyed and the step cannot be run -- a wall. A single protic hydrogen in a
Grignard costs an equivalent of reagent, and excess overcomes it -- a cost. A table that could only
say "wall" would either forbid chemistry that works or stay silent about chemistry that is worse
than the alternative, and both are wrong in the same way a selectivity count would be if it could
only say "clean" or "impossible".

**No new machinery is needed to price one.** A verdict is a named reaction score;
:func:`vertsy.search.measured` persists every score under its own name and
:func:`vertsy.expansions.layer_tables` stores it as a channel, so
:func:`vertsy.costs.selective` -- whose ``penalties`` is a mapping from *channel name* to
multiplier -- can charge for it exactly as it charges a selectivity count. What a row declares here
is which of the two consumers reads it, and the two never mix in one column.

Infinity is the wrong encoding of a wall and is not offered. A refused disconnection never enters
the store, never gets valued and never competes for a width slot; an infinitely-priced one is
expanded first and rejected after, and pollutes every sum it reaches.
"""

type Screen[M] = Callable[[MoleculeBatch[M], Sequence[tuple[int, int]]], np.ndarray]
"""Which ``(veto, molecule row)`` pairs match: one bool per pair, in order.

The whole of what an arm has to supply, and it is deliberately pair-shaped rather than all-vs-all
because that is both what scoping produces and what every backend already takes --
``MetalMatcher.dispatch_match_packed`` wants exactly ``pair_indices`` over packed arrays, so a device
veto is a match-only dispatch and no kernel has to change. A screen closes over its compiled
patterns so a device one can upload them once under a stable token.
"""


@dataclass(frozen=True)
class Veto:
    """One row of the veto table.

    ``scope`` holds template codes, empty meaning every template. ``pattern`` is query SMARTS within
    the limits in the module docstring. ``rationale`` is not read by anything and is the only defence
    against a row nobody can later justify.
    """

    code: str
    cost: Cost
    scope: frozenset[str]
    on: Side
    force: Force
    sense: Sense
    pattern: str
    rationale: str


COSTS: frozenset[str] = frozenset(("pattern", "geometry"))
SIDES: frozenset[str] = frozenset(("product", "precursors"))
SENSES: frozenset[str] = frozenset(("forbid", "require"))
FORCES: frozenset[str] = frozenset(("refuse", "price"))


@dataclass(frozen=True)
class VetoReport:
    """What one veto pass judged, so a table earns its place before it removes anything.

    The reportable half of the composition order this module argues for: measure first, refuse
    later, and report per *row* rather than per pass, because the decision a table faces is always
    about one row -- whether it fires often enough to be worth the chemistry it removes, and whether
    anything else already catches what it catches.

    Deliberately shaped like :class:`vertsy.selectivity.ProbeReport`, which is the other thing that
    measures every application and has to say what it could not judge. The fields mean the same, and
    :attr:`fired` is the one addition: a verdict column carries a *count* and not the identity of the
    offender, by design, so the only place a row's own tally can be kept is where the rows are.
    """

    applications: int
    """Disconnections the pass was offered."""
    examined: int
    """Disconnections every veto in scope could be evaluated on.

    An application with no veto in scope at all is examined: "no objection on file" is an answer,
    and the same answer a row in scope that did not fire gives.
    """
    unexamined: int
    """Disconnections something went unjudged on -- a pot no backend here could build."""
    condemned: int
    """Disconnections at least one veto fired on. What a refusing pass would remove."""
    fired: Mapping[str, int]
    """Per veto code, how many disconnections it condemned. Rows that fired on nothing are absent.

    These **do not sum to** :attr:`condemned`: two rows can condemn one application, which is
    exactly the overlap a table wants to know about before a row is deleted as redundant.
    """


def read_vetoes(path: Path) -> tuple[Veto, ...]:
    """Read a veto table, refusing anything it cannot read exactly.

    Every field is validated against a closed vocabulary and a duplicate code is an error, because a
    veto is a *veto*: a row silently misread does not degrade a score, it removes chemistry nobody
    asked it to remove, or fails to remove chemistry somebody did. A typo in ``sense`` inverts the
    row's meaning entirely.

    Scope is ``;``-separated. A ``tag:<name>`` entry parses and is kept verbatim so a reagent-class
    vocabulary can arrive without a format change; nothing resolves tags yet, and
    :func:`vetoes_in_scope` says so rather than matching them silently.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "veto_code",
        "cost",
        "scope",
        "on",
        "force",
        "sense",
        "pattern",
        "rationale",
    }
    vetoes: list[Veto] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=2):
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"{path.name}:{number} is missing {sorted(missing)}; a veto table needs "
                f"every column, since a default for `sense` or `on` would be a guess at chemistry."
            )
        code = row["veto_code"].strip()
        if not code:
            raise ValueError(f"{path.name}:{number} has no veto_code.")
        if code in seen:
            raise ValueError(
                f"{path.name}:{number} repeats veto_code {code!r}; a report naming a code "
                f"has to name one row."
            )
        seen.add(code)
        vetoes.append(
            Veto(
                code=code,
                cost=cast("Cost", _one_of(row["cost"], COSTS, "cost", path, number)),
                scope=frozenset(
                    entry.strip() for entry in row["scope"].split(";") if entry.strip()
                ),
                on=cast("Side", _one_of(row["on"], SIDES, "on", path, number)),
                force=cast(
                    "Force", _one_of(row["force"], FORCES, "force", path, number)
                ),
                sense=cast(
                    "Sense", _one_of(row["sense"], SENSES, "sense", path, number)
                ),
                pattern=row["pattern"].strip(),
                rationale=row["rationale"].strip(),
            )
        )
        if not vetoes[-1].pattern:
            raise ValueError(f"{path.name}:{number} ({code}) has no pattern.")
        if not vetoes[-1].rationale:
            raise ValueError(
                f"{path.name}:{number} ({code}) has no rationale. A veto removes chemistry, so a "
                f"row that cannot say why is a row nobody can later audit or delete."
            )
    return tuple(vetoes)


def _one_of(
    value: str, allowed: frozenset[str], field: str, path: Path, number: int
) -> str:
    cleaned = value.strip()
    if cleaned not in allowed:
        raise ValueError(
            f"{path.name}:{number} has {field}={value!r}; expected one of "
            f"{sorted(allowed)}."
        )
    return cleaned


def read_tags(path: Path) -> Mapping[str, frozenset[str]]:
    """Read ``tag -> template codes`` from the tag table.

    A tag names a **class of reaction conditions** -- "turned over by a palladium catalyst" -- and
    exists because what poisons a catalyst is a property of the catalyst rather than of any one
    coupling. Without it a veto about palladium would carry eighty-nine template codes in its scope
    field and would silently stop covering the ninetieth.

    Deliberately not a vocabulary of functional *groups*: those are what a veto's ``pattern`` is
    for, and SMARTS-RX and the ZBH collections already name them. This is the other axis, and it is
    the same per-rule attribute the ``competes`` predicate wants as its ``requires`` column -- what
    a reaction *needs* -- so the two share one curated list rather than minting two.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_tag: dict[str, set[str]] = {}
    for number, row in enumerate(rows, start=2):
        tag = (row.get("tag") or "").strip()
        code = (row.get("template_code") or "").strip()
        if not tag or not code:
            raise ValueError(
                f"{path.name}:{number} needs both a tag and a template_code; "
                "a half-written row would narrow a veto's scope without saying so."
            )
        by_tag.setdefault(tag, set()).add(code)
    return {tag: frozenset(codes) for tag, codes in by_tag.items()}


def vetoes_in_scope(
    vetoes: Sequence[Veto],
    codes: Sequence[str],
    tags: Mapping[str, frozenset[str]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Which vetoes apply to each rule, indexed by rule.

    Resolved once against the rule table rather than per application, so the per-application lookup is
    a tuple index. ``codes`` is the template code of each rule, parallel to the compiled rule table;
    a rule whose code is empty -- a library row with no identity of its own -- is in scope only for
    vetoes that name no templates at all.

    A ``tag:`` scope entry expands through ``tags``, which :func:`read_tags` supplies. A tag nothing
    defines **raises**: a scope that quietly matched nothing would be a veto that looks curated and
    removes no chemistry, which is the one failure mode a veto table cannot afford. That is the one
    thing this function used to get wrong -- it dropped unknown tags silently.
    """
    known = tags or {}
    universal = tuple(index for index, veto in enumerate(vetoes) if not veto.scope)
    by_code: dict[str, list[int]] = {}
    for index, veto in enumerate(vetoes):
        for entry in veto.scope:
            if entry.startswith("tag:"):
                name = entry[len("tag:") :]
                if name not in known:
                    raise ValueError(
                        f"veto {veto.code} is scoped to {entry!r} and no tag table defines "
                        f"{name!r}; known tags are {sorted(known) or 'none'}. A scope that "
                        "resolves to nothing is a veto that removes no chemistry while looking "
                        "curated."
                    )
                for code in known[name]:
                    by_code.setdefault(code, []).append(index)
                continue
            by_code.setdefault(entry, []).append(index)
    return tuple(tuple(sorted((*universal, *by_code.get(code, ())))) for code in codes)


def vetoed(
    vetoes: Sequence[Veto],
    in_scope: Sequence[tuple[int, ...]],
    *,
    screen: Screen[M],
    cost: Cost = "pattern",
    force: Force = "refuse",
    combine: Combine[M] | None = None,
    report: Callable[[VetoReport], None] | None = None,
) -> Callable[[MoleculeBatch[M], MoleculeBatch[M], HitArray], Column]:
    """A :data:`vertsy.search.ReactionScore` counting how many vetoes condemn each application.

    ``CLEAR`` where every veto in scope ran and none fired, and where none was in scope at all --
    those are the same claim, "no objection on file", and a veto table is a list of known
    incompatibilities rather than a certificate. ``UNEXAMINED`` is narrower and means something went
    unjudged: a group the backend could not combine, or a veto needing a union where no
    :data:`Combine` was supplied. A fired count wins over an unexamined one, because a refusal is
    definite and a gap is not.

    ``cost`` selects which rows run, so one table serves every placement and an expensive row cannot
    reach the inner loop by accident -- a safety property, not a convenience. ``force`` selects them the same way and for a
    different reason: a column mixing rows meant to refuse with rows meant to be priced could be
    given to neither consumer honestly, since one fired count would mean "remove this" for some
    rows and "charge for this" for others. One table, one call per force, two columns.

    ``report`` receives a :class:`VetoReport` per pass that actually ran, which is how a table is
    measured before it refuses anything. A pass with no rows to judge, or no row live at this cost and force,
    reports nothing rather than a row of zeroes: the counts would describe a pass that did not
    happen, and a zero in a tally of fire counts is a claim about chemistry.
    """
    live = tuple(
        index
        for index, veto in enumerate(vetoes)
        if veto.cost == cost and veto.force == force
    )
    live_set = frozenset(live)

    def measure(
        products: MoleculeBatch[M], precursors: MoleculeBatch[M], hits: HitArray
    ) -> Column:
        rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
        counts = np.full(rows.shape[0], CLEAR, dtype=np.int64)
        if not live_set or not rows.shape[0]:
            return cast("Column", counts)

        order, starts, keys = grouped_applications(rows)
        stops = np.append(starts[1:], order.size)
        groups = keys.shape[0]

        # Per group, which vetoes apply: scope first, then cost and force. All set lookups,
        # so the expensive part stays proportional to the pairs actually dispatched.
        applicable = [
            tuple(
                index
                for index in (in_scope[rule] if rule < len(in_scope) else ())
                if index in live_set
            )
            for rule in keys[:, 1].tolist()
        ]

        product_pairs: list[tuple[int, int]] = []
        product_owner: list[tuple[int, int]] = []
        for group, indices in enumerate(applicable):
            for index in indices:
                if vetoes[index].on == "product":
                    product_pairs.append((index, int(keys[group, 0])))
                    product_owner.append((group, index))

        union_groups = [
            group
            for group, indices in enumerate(applicable)
            if any(vetoes[index].on == "precursors" for index in indices)
        ]
        unexamined = np.zeros(groups, dtype=bool)
        union_pairs: list[tuple[int, int]] = []
        union_owner: list[tuple[int, int]] = []
        probes: MoleculeBatch[M] | None = None
        if union_groups:
            probes, buildable, placed = _unions(
                precursors, rows, order, starts, stops, union_groups, combine
            )
            for group in union_groups:
                slot = placed.get(group)
                if slot is None or not bool(buildable[slot]):
                    unexamined[group] = True
                    continue
                for index in applicable[group]:
                    if vetoes[index].on == "precursors":
                        union_pairs.append((index, slot))
                        union_owner.append((group, index))

        fired = np.zeros(groups, dtype=np.int64)
        by_code: dict[str, int] = {}
        for batch, pairs, owners in (
            (products, product_pairs, product_owner),
            (probes, union_pairs, union_owner),
        ):
            if not pairs or batch is None:
                continue
            matched = np.asarray(screen(batch, pairs), dtype=bool)
            for (group, index), hit in zip(owners, matched.tolist(), strict=True):
                # `forbid` fires on a match, `require` on its absence. The two senses are one
                # comparison, which is the point of keeping them one field rather than two tables.
                if hit == (vetoes[index].sense == "forbid"):
                    fired[group] += 1
                    code = vetoes[index].code
                    by_code[code] = by_code.get(code, 0) + 1

        verdict = np.where(fired > 0, fired, np.where(unexamined, UNEXAMINED, CLEAR))
        for group in range(groups):
            counts[order[starts[group] : stops[group]]] = verdict[group]
        if report is not None:
            # Read off `verdict` rather than its inputs, so the report says what the column says: a
            # fired count wins over an unexamined one, so an application whose pot could not be built
            # but whose product a veto condemned is condemned and not unexamined.
            unjudged = int(np.count_nonzero(verdict == UNEXAMINED))
            report(
                VetoReport(
                    applications=groups,
                    examined=groups - unjudged,
                    unexamined=unjudged,
                    condemned=int(np.count_nonzero(verdict > CLEAR)),
                    fired=dict(sorted(by_code.items())),
                )
            )
        # Cast for the same reason `selectivity.forward_probe` does: an ndarray satisfies `Column`,
        # and the checker cannot see that through `np.full`'s overloads.
        return cast("Column", counts)

    return measure


def _unions(
    precursors: MoleculeBatch[M],
    rows: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    wanted: Sequence[int],
    combine: Combine[M] | None,
) -> tuple[MoleculeBatch[M], np.ndarray, Mapping[int, int]]:
    """One probe molecule per wanted group, and which groups got one.

    With a ``combine`` every group gets its union, and the combiner answers which it could actually
    build -- a device batch has an atom budget and a group past it was never examined.

    Without one, only *single-precursor* groups are examinable, because a one-member union is a batch
    gather and needs no such operation. A multi-precursor group is then left out and reported
    unexamined, which is the honest answer for an arm that cannot build a pot rather than quietly
    testing one half of it.
    """
    members = rows[order, 3]
    if combine is not None:
        sizes = [int(stops[group] - starts[group]) for group in wanted]
        gathered = np.concatenate(
            [members[starts[group] : stops[group]] for group in wanted]
        ).astype(np.int64)
        indptr = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
        built, buildable = combine(
            precursors, cast("SidArray", gathered), cast("SidArray", indptr)
        )
        return built, buildable, {group: slot for slot, group in enumerate(wanted)}

    alone = [group for group in wanted if int(stops[group] - starts[group]) == 1]
    picked = np.array([int(members[starts[group]]) for group in alone], dtype=np.int64)
    return (
        precursors[picked],
        np.ones(len(alone), dtype=np.bool_),
        {group: slot for slot, group in enumerate(alone)},
    )


def host_screen(vetoes: Sequence[Veto]) -> Screen[M]:
    """A :data:`Screen` over molecules the host holds, one subgraph match per pair.

    Compiles each pattern once. The arm-specific half of a veto is this function and nothing else:
    a device screen packs the same patterns once under a stable upload token and answers the same
    pairs with ``dispatch_match_packed``, which is why no kernel has to change for vetoes to run
    there.

    Patterns are compiled eagerly so a table with an unparsable row fails when it is built rather
    than on whichever layer first brings that rule into scope.
    """
    from vesma import gmol_from_cmol
    from vesma.cpu_matcher import CpuMatcher
    from vesma.gmol_builder import gmol_from_smarts
    from vesma.predicates import require_no_predicates

    patterns = tuple(gmol_from_smarts(veto.pattern)[0] for veto in vetoes)
    for pattern in patterns:
        # The arms must agree, and the device arm cannot evaluate a recursive predicate yet.
        require_no_predicates(pattern, backend="veto table")
    matcher = CpuMatcher()

    def screen(batch: MoleculeBatch[M], pairs: Sequence[tuple[int, int]]) -> np.ndarray:
        molecules = cast("Sequence[CMol]", batch)
        # Memoised per call, because scoping hands one molecule to several vetoes and converting it
        # is the expensive half of a single-pattern test.
        queries: dict[int, GMol] = {}
        found = np.zeros(len(pairs), dtype=bool)
        for position, (veto, row) in enumerate(pairs):
            if row not in queries:
                queries[row] = gmol_from_cmol(molecules[row])
            found[position] = matcher.has_match(patterns[veto], queries[row])
        return found

    return screen
