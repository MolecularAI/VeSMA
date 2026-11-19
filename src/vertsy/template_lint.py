"""Check reaction templates for atoms that vanish, without applying them to anything.

**Applying this changes which templates a run uses**, and several benchmarks are pinned to a full,
unfiltered library — ``uspto42k-compat`` exists precisely to reproduce numbers against every
template, defect included. So callers opt in explicitly and say what they dropped:
``bench_host_aostar.load_library`` removes what :func:`lint` and :func:`duplicates` report and
counts each separately. Nothing else — no matcher, no search — consults this module on its own.

The defect it finds is a template whose product side maps an atom its precursor side never
mentions::

    [C:4]-[O;H0;D2;+0:3]-[C:2]=[O;D1;H0:1]  >>  [O;D1;H0:1]=[C:2]-[OH;D1;+0:3]

Read as retrosynthesis this says an ester becomes a carboxylic acid — and says nothing about
``[C:4]``, the alcohol half. RDKit does not treat the omission as "leave it alone"; it deletes the
matched atom and everything hanging off it. Applied to a 45-heavy-atom ester the template returns
4-nitrobenzoic acid and discards 33 atoms.

It is rare and it does not need to be common: **71 of 67,557 raw68k templates and 4 of 13,978
used14k templates** map a product atom no precursor mentions — 0.1% and 0.03%. Seventy-one bad
templates were enough to "solve" four complex natural products in two or three steps each, because
a search hunts for them. (An earlier count here said a tenth of each library; that came from
counting every ``:`` in the SMARTS, which swallows the aromatic bond colons in ``[c:1]:[n:2]`` and
inflates the product side of every aromatic template. The parse below counts map numbers.) This
is a property of the extraction, not of the underlying reactions — Daylight SMIRKS forbids exactly
this, requiring every mapped atom on one side to appear on the other, and RDKit's reaction SMARTS
relaxes the rule on purpose so that fragment-deleting transforms can be written at all.

Two places can catch it and they are not redundant. This one is a pure function of the template
text, so it costs one pass over a library and never runs again;
:func:`vesma.reaction.precursors_from_mapping` watches the molecules, refuses any application that
orphans a component, and catches anything a parse cannot see. Linting is the cheaper of the two by
the number of applications a run makes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported for typing only: every vesma import in this module is deferred into the function
    # that needs it, so a caller using the text-only lints never pays for the chemistry core.
    from vesma.reaction import ReactionPattern

__all__ = [
    "duplicates",
    "half_migrated_duplicates",
    "identical_patterns",
    "lint",
    "lost_atom_maps",
    "subsumed_patterns",
    "unbalanced",
    "ungrouped_components",
]


def _atom_maps(fragment: str) -> set[int]:
    """The atom-map numbers written in one side of a reaction SMARTS.

    Read off the text rather than a parse. RDKit would answer the same question and would also
    reject a template the matcher happily compiles, which would conflate "this template is
    unbalanced" with "this template is unparseable" — two findings a linter should keep apart.
    """
    found: set[int] = set()
    index = 0
    while (index := fragment.find(":", index)) != -1:
        index += 1
        digits = index
        while digits < len(fragment) and fragment[digits].isdigit():
            digits += 1
        # A map number is digits terminated by `]`; `[n;H0;D2:3]` qualifies and `[c:1]:[c:2]`'s
        # aromatic bond colon does not, since no digits follow it.
        if digits > index and digits < len(fragment) and fragment[digits] == "]":
            found.add(int(fragment[index:digits]))
        index = digits
    return found


def lost_atom_maps(smarts: str) -> frozenset[int]:
    """Map numbers the product side claims that no precursor accounts for.

    Empty for a well-formed template. Non-empty means RDKit will delete those atoms, and every
    substituent attached to them, from any molecule the template matches.

    Retro convention: the text is ``product >> precursors``, so the *left* is what gets matched and
    the right is what comes back. A template written with agents (``a>b>c``) has its middle section
    folded into the precursor side, since an agent's atoms are equally not-deleted.
    """
    left, _, right = smarts.partition(">>")
    if not right:
        pieces = smarts.split(">")
        if len(pieces) != 3:
            return frozenset()
        left, middle, right = pieces
        right = f"{middle}.{right}"
    return frozenset(_atom_maps(left) - _atom_maps(right))


def unbalanced(smarts: str, *, tolerance: int = 0) -> bool:
    """Whether this template loses more than ``tolerance`` mapped atoms.

    ``tolerance`` defaults to zero because the honest threshold is zero — a real retro template
    accounts for every atom it matches. It exists because a caller comparing libraries may want to
    count only the egregious cases, and the measured defects drop tens of atoms rather than one.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance!r}.")
    return len(lost_atom_maps(smarts)) > tolerance


def duplicates(templates: Iterable[str]) -> Iterator[int]:
    """Indices of templates whose SMARTS text has appeared earlier in the sequence.

    Indices rather than a deduplicated list, for the same reason :func:`lint` yields them: the
    caller keeps its own numbering, and a renumbered library would be a fourth rule-id scheme.
    The first occurrence is kept and every later one reported, so dropping these preserves
    source order.

    **Exact text, not chemistry.** Two SMARTS that differ only in atom-map numbering or in
    whitespace describe the same transformation and are not reported here. That is deliberate:
    deciding two spellings are the same reaction is what `template_hash` is for, and a linter
    that silently merged them would be making a chemical judgement from a string comparison.
    What this catches is the cheap case -- the same line twice -- which a hand-curated table
    accumulates easily and which costs a search real work, since a duplicate rule is matched,
    applied and expanded twice for identical precursors.
    """

    seen: set[str] = set()

    for index, smarts in enumerate(templates):
        if smarts in seen:
            yield index
        else:
            seen.add(smarts)


def ungrouped_components(templates: Iterable[str]) -> Iterator[tuple[int, int, int]]:
    """``(index, left-hand pieces, right-hand pieces)`` for every rule that leaves a side unstated.

    A multi-component side written with bare dots says nothing about molecularity. A query dot is
    *no constraint*: `CCC.NNN` matches `NNNOCCC`, one molecule, so a forward rule spelled that way
    admits an intramolecular reaction as though two starting materials had met, and a retro rule
    spelled that way can hand back one molecule while claiming two precursors. Zero-level
    parentheses are what state it -- `(A).(B)` separate, `(A.B)` one molecule in fragments -- and
    this is the worklist of rules that have not been migrated.

    Reported rather than refused. Both readings are chemically real, so which one a rule meant is a
    judgement about the reaction and not something a linter can make: `MC 91a` meant one molecule
    and `Hartenfeller 47` meant two, and the same text was used for both. A side whose every piece
    is parenthesised is migrated and is not reported; a side with one piece has nothing to state.
    """
    for index, smarts in enumerate(templates):
        pieces = tuple(top_level_pieces(side) for side in smarts.split(">>", 1)[:2])
        if len(pieces) < 2:
            continue
        unstated = tuple(
            len(side) if len(side) > 1 and not all_grouped(side) else 0
            for side in pieces
        )
        if any(unstated):
            yield index, unstated[0], unstated[1]


def identical_patterns(templates: Iterable[str]) -> Iterator[tuple[int, int]]:
    """Index pairs that are the same rule, however they were numbered or spelled.

    What :func:`duplicates` cannot see. It compares text, so `iSCB 63a` and `Hartenfeller 29a` --
    one rule under two atom-map numberings -- were duplicates only because somebody had already
    hand-rewritten one to match the other. This compares
    :func:`vesma.pattern_identity.pattern_certificate`, a canonical form of the compiled pattern, so
    the numbering cannot reach the answer and the match needs nobody's help.

    It is also *semantic* where `duplicates` is syntactic: the atom predicates in the certificate
    are the compiled mask words, and `[#6]` and `[C,c]` compile to the same masks. Two spellings of
    one predicate are one predicate, because that is what the matcher sees.

    Pairs rather than classes, and all pairs within a class, matching
    :func:`half_migrated_duplicates`; the caller keeps its own numbering, as everything here does.

    A template that does not compile is skipped. It cannot be a duplicate of anything, and the row
    that will not parse is a different lint's finding -- see
    `tests/vertsy/test_hand_written_templates.py::test_every_row_parses_and_is_balanced`.
    """
    from vesma import parse_reaction_smarts
    from vesma.pattern_identity import pattern_certificate

    by_certificate: dict[object, list[int]] = {}
    for index, smarts in enumerate(templates):
        try:
            certificate = pattern_certificate(parse_reaction_smarts(smarts))
        except Exception:
            continue
        by_certificate.setdefault(certificate, []).append(index)
    for found in by_certificate.values():
        for position, left in enumerate(found):
            for right in found[position + 1 :]:
                yield left, right


def subsumed_patterns(templates: Iterable[str]) -> Iterator[tuple[int, int]]:
    """Index pairs ``(broad, narrow)`` where the broad rule already covers the narrow one.

    The asymmetric case, and where the defects turn out to be.
    :func:`vesma.pattern_subsumption.subsumes` decides it: every molecule `narrow` matches, `broad`
    matches too, and both do the same thing to it -- so `narrow` offers a search nothing `broad`
    does not.

    Reported, never acted on, and the direction is the reason. A rule broader than its near-twin is
    broader *somewhere*, and that somewhere is either the generalisation the author intended or a
    bug: `iSCB 70` was `MC 75` without the `-!@`, and the breadth it bought was opening lactones and
    reporting a two-precursor disconnection that had produced one molecule. Only a chemist reading
    the pair can say which it is, so this yields a worklist.

    Mutual pairs are identity and :func:`identical_patterns` says so more cheaply; they appear here
    too, in both directions, because suppressing them would hide a disagreement between the two
    relations. On the shipped table they agree exactly -- 11 mutual pairs, all certificate-equal.

    Rules that state their molecularity are skipped: zero-level grouping is an equality between two
    relations rather than a constraint that widens, so `subsumes` declines it rather than guess.
    """
    from vesma import parse_reaction_smarts
    from vesma.pattern_subsumption import subsumes

    compiled: list[tuple[int, ReactionPattern]] = []
    for index, smarts in enumerate(templates):
        try:
            compiled.append((index, parse_reaction_smarts(smarts)))
        except Exception:
            continue  # a row that will not parse is a different lint's finding
    for broad_index, broad in compiled:
        for narrow_index, narrow in compiled:
            if broad_index == narrow_index:
                continue
            # The screen that keeps this quadratic pass cheap, and it costs nothing: a pattern
            # cannot embed into a smaller one, so the full search never starts on those pairs.
            if broad.lhs.atom_count > narrow.lhs.atom_count:
                continue
            if subsumes(broad, narrow):
                yield broad_index, narrow_index


def half_migrated_duplicates(templates: Iterable[str]) -> Iterator[tuple[int, int]]:
    """Index pairs that are the same rule, one of them stating its molecularity and the other not.

    The fault this catches was made here and measured: `iSCB 63a` was character-for-character
    `Hartenfeller 29a`, so the loader deduplicated it first-wins and only one compiled. Grouping
    the Hartenfeller row made the texts differ, the dedup stopped firing, and the library quietly
    gained four rules doing the same chemistry under two different claims about their own
    molecularity -- the worse of both states.

    Decided on :func:`vesma.pattern_identity.pattern_certificate` with ``grouping=False``, which is
    "the same rule up to what it claims about its own molecularity" -- precisely the thing a
    half-finished migration varies. It replaces the text comparison this used to be, and strictly:
    stripping parentheses and comparing strings could not see a renumbering, so a pair that had been
    both renumbered *and* half migrated went unreported. Atom-map numbers never reach a certificate.

    A parent and its intramolecular sibling are *not* this: `Hartenfeller 47` and `MC 102` also
    differ only in parentheses, and both state a molecularity -- different ones, on purpose. What is
    reported is a pair where one side states one and the other states nothing, which can only be a
    migration left half done.
    """
    from vesma import parse_reaction_smarts
    from vesma.pattern_identity import pattern_certificate

    blind: dict[object, list[tuple[int, bool]]] = {}
    for index, smarts in enumerate(templates):
        try:
            pattern = parse_reaction_smarts(smarts)
        except Exception:
            continue  # a row that will not parse is a different lint's finding
        # "Unstated" is exactly what `ungrouped_components` reports, so the two lints cannot
        # disagree about which rules are migrated. A lone `(A.B)` *is* a statement -- one molecule
        # in two fragments -- which is why this cannot be a count of pieces.
        stated = not any(any(side) for _i, *side in ungrouped_components([smarts]))
        certificate = pattern_certificate(pattern, grouping=False)
        blind.setdefault(certificate, []).append((index, stated))
    for found in blind.values():
        grouped = [index for index, stated in found if stated]
        unstated = [index for index, stated in found if not stated]
        for left in grouped:
            for right in unstated:
                yield (left, right) if left < right else (right, left)


def top_level_pieces(side: str) -> tuple[str, ...]:
    """One SMARTS side split on its zero-level dots, groups and branches left intact."""
    pieces: list[str] = []
    start = 0
    depth = 0
    for position, character in enumerate(side):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "." and depth == 0:
            pieces.append(side[start:position])
            start = position + 1
    pieces.append(side[start:])
    return tuple(pieces)


def all_grouped(pieces: Sequence[str]) -> bool:
    """Whether every piece of a side is wrapped in its own zero-level group."""
    return all(
        piece.startswith("(") and piece.endswith(")") and balanced_once(piece)
        for piece in pieces
    )


def balanced_once(piece: str) -> bool:
    """Whether a piece's outer parentheses are its own, rather than two adjacent branches."""
    depth = 0
    for position, character in enumerate(piece):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return position == len(piece) - 1
    return False


def lint(
    templates: Iterable[str], *, tolerance: int = 0
) -> Iterator[tuple[int, frozenset[int], str]]:
    """``(index, lost map numbers, template)`` for every template that loses atoms.

    A generator over indices rather than a filtered list, so a caller keeps the correspondence
    between its own numbering and the library's — three separate rule-id schemes already exist in
    this repository, and handing back a renumbered library would quietly make it four.
    """
    for index, smarts in enumerate(templates):
        lost = lost_atom_maps(smarts)
        if len(lost) > tolerance:
            yield index, lost, smarts
