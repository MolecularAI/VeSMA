"""Prepare search targets: find duplicates, and run optional probes that change the molecule.

**This module no longer exists to make targets parseable.** It was written because campaign
target 16 -- a protonated ring nitrogen writing its hydrogen as its own atom node -- was refused
by the matcher, and the standing fix was a hand-made normalised copy of the targets outside the
repository. That case now belongs to ``vesma.parsing.reading.fold_explicit_hydrogens``, which
folds a terminal ``[H]`` into its neighbour's hydrogen count during the SMILES reading. The raw
the raw targets file parses in full, with no preprocessing and no RDKit.

What is left here is the work that genuinely does not belong in a matcher, because it is a
judgement about a campaign rather than about a molecule:

* **Duplicates.** Two targets can be written differently and be the same molecule. Worth saying
  before spending a search on each.
* **The free-base probe.** Neutralising a charged target is a different question, not a fix: a
  route to the free base is not a route to the salt. Off by default, and reported when used.

**Targets are not rewritten by default.** ``searched`` is the string the caller supplied unless a
probe was asked for. Canonicalisation is used to *compare* targets, never to replace them --
rewriting a target for cosmetic reasons would put a molecule in the report that nobody asked
about, and every campaign target so far differs from its canonical form.

**Importing this module requires the ``rdkit`` extra**, like ``vertsy.similarity`` and unlike
``vertsy`` itself. Callers that may run without it should import it inside the branch that uses
it.
"""

from __future__ import annotations

__all__ = [
    "UNCHANGED",
    "PreparedTarget",
    "canonical_smiles",
    "neutralized",
    "prepare_target",
    "prepare_targets",
]

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

#: The target was searched exactly as supplied.
UNCHANGED = "none"


@dataclass(frozen=True)
class PreparedTarget:
    """One target as supplied, as searched, and what happened in between.

    :param original: exactly the string the caller provided, never rewritten.
    :param searched: what the search runs on.
    :param note: provenance for the report.
    :param duplicate_of: 1-based index of the earlier target this one became identical to after
        preparation, or ``None``. Two inputs can be written differently and prepare to the same
        molecule, which is worth saying rather than searching twice.
    """

    original: str
    searched: str
    note: str
    duplicate_of: int | None = None

    @property
    def changed(self) -> bool:
        return self.searched != self.original


def canonical_smiles(smiles: str) -> str:
    """``smiles`` in RDKit's canonical isomeric form.

    Isomeric deliberately: ``vesma.keyer.rdkit_token_from_smiles`` writes
    ``isomericSmiles=False`` because it builds identity keys, and dropping stereochemistry from a
    *target* would search for a different molecule.

    Raises on unparseable input rather than passing it through, so a bad target is named here
    instead of surfacing as an empty result after a long search.
    """

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"target is not parseable as SMILES: {smiles!r}")

    return Chem.MolToSmiles(molecule)


def neutralized(smiles: str) -> str:
    """``smiles`` with chargeable atoms returned to their neutral form.

    Separate from :func:`prepare_target`'s default path and never applied automatically, because
    unlike canonicalisation this *changes the molecule*: a route to the free base is not a route
    to the salt. It is offered as a probe worth running deliberately, and its result has to be
    reported as the different question it is.

    **It is not needed as often as it looks.** An earlier note here claimed a charged target
    matches nothing because every template writes an explicit ``+0``. That was measured wrong:
    ``+0`` appears somewhere in 100% of raw68k templates, but **6.1% of them (4,106) carry a
    non-zero charge**, and Hartenfeller writes no ``+0`` at all, so its unconstrained atoms match
    charged and neutral alike. A protonated ring nitrogen -- one real target -- draws 2
    applications from each library once it parses. Charge narrows what matches; it does not
    silence it.

    It also cannot rescue every charged target: a quaternary ammonium and a dissociated salt have
    no neutral form and come back unchanged.
    """

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"target is not parseable as SMILES: {smiles!r}")

    return Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(molecule))


def prepare_target(smiles: str, *, neutralize: bool = False) -> PreparedTarget:
    """One target prepared for search, carrying a note about what was done.

    Returns the target untouched unless a probe was asked for. ``canonical_smiles`` is still
    called on the unchanged path, so an unparseable target is named here rather than after a
    long search -- but its result is discarded.

    :param neutralize: run the free-base probe. Off by default; see :func:`neutralized`.
    """

    canonical_smiles(smiles)

    if not neutralize:
        return PreparedTarget(original=smiles, searched=smiles, note=UNCHANGED)

    searched = neutralized(smiles)

    if searched == canonical_smiles(smiles):
        return PreparedTarget(original=smiles, searched=smiles, note=UNCHANGED)

    return PreparedTarget(
        original=smiles,
        searched=searched,
        note="RDKit Uncharger free-base probe; routes are to the neutral form",
    )


def prepare_targets(
    smiles: tuple[str, ...], *, neutralize: bool = False
) -> tuple[PreparedTarget, ...]:
    """Prepare a target list, flagging any that collapse onto an earlier one.

    Duplicates are reported rather than removed. The caller decides whether to reuse the earlier
    result, and a silently shortened list would make every downstream index -- report filenames,
    ``targets.jsonl`` line numbers -- disagree with the input file.
    """

    prepared: list[PreparedTarget] = []
    first_seen: dict[str, int] = {}

    for index, text in enumerate(smiles, start=1):
        target = prepare_target(text, neutralize=neutralize)
        # Compared canonically and kept verbatim: two targets are duplicates when they are the
        # same molecule, not when they are the same string.
        earlier = first_seen.get(canonical_smiles(target.searched))

        if earlier is None:
            first_seen[canonical_smiles(target.searched)] = index
            prepared.append(target)
            continue

        duplicate = f"duplicate of target {earlier}"
        prepared.append(
            PreparedTarget(
                original=target.original,
                searched=target.searched,
                note=(
                    duplicate
                    if target.note == UNCHANGED
                    else f"{target.note}; {duplicate}"
                ),
                duplicate_of=earlier,
            )
        )

    return tuple(prepared)
