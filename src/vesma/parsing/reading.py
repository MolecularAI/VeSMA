"""The SMILES reading of a parsed atom expression.

One grammar parses both languages, and the IR it produces records only what was
*written*. What SMILES and SMARTS disagree about is what an omission *means*, so the
disagreement belongs here -- in a reading applied to the IR -- and nowhere below.

Daylight's SMARTS theory states the difference::

    In SMILES, default values are used; in SMARTS, unspecified properties are not
    defined to be part of the pattern.

So the SMILES text ``O`` is water, while the SMARTS ``O`` is any aliphatic oxygen. For a
bracket atom OpenSMILES 1.0 section 3.1 is explicit::

    If no ``Hn`` is specified, it is identical to ``H0`` ... For example, ``[C]`` and
    ``[CH0]`` are identical, and ``[CH]`` and ``[CH1]`` are identical.

That equivalence is what keeps this small: the reading rewrites ``[C]`` into the ``[CH0]``
the rest of the pipeline already handles, so no hydrogen rule is duplicated here.

The reading runs on the raw IR, before normalization, for two reasons. Asking "did the
author write an ``H``?" of a spelling is a different and simpler question than asking
whether an arbitrary CNF constrains ``H``; and ``identity_seed_from_smiles`` consumes the
IR without ever normalizing, so a reading applied to the CNF could not serve it.
"""

from __future__ import annotations

import numpy as np

from ..gmol_atoms import (
    BUCKET_MAX,
    CHARGE_MAX,
    CHARGE_MIN,
    MAX_ISOTOPE,
    MIN_ISOTOPE,
)
from .builder import QBOND_DEFAULT
from .errors import NotMolecularSmilesError
from .ir import AtomFormula, PropVal, SmartsAtom, SmartsMol

# OpenSMILES 1.0 section 2.2 gives the whole of a bracket atom::
#
#     bracket_atom ::= '[' isotope? symbol chiral? hcount? charge? class? ']'
#
# Chirality is stripped before parsing and the atom class becomes ``atom_map``, so the
# properties that can survive into a formula are the isotope (``M``), the element
# (``symbol``), the hydrogen count (``H``) and the charge (``C``). Every other primitive
# the parser can produce -- ``D X v R r x h`` and ``#n`` -- is SMARTS-only.
SMILES_ATOM_PROPS = frozenset({"symbol", "M", "H", "C"})

# ``*`` is legal SMILES syntax but denotes an atom of unknown element, which is not a
# molecule this model can build. ``A``/``a`` are SMARTS primitives that reach the same
# ``symbol`` slot.
NON_ELEMENT_SYMBOLS = frozenset({"*", "A", "a"})

__all__ = [
    "NON_ELEMENT_SYMBOLS",
    "SMILES_ATOM_PROPS",
    "fold_explicit_hydrogens",
    "folds_into",
    "written_hydrogens",
    "smiles_reading",
    "validate_smiles_ir",
]


def conjunction(formula: AtomFormula) -> list[PropVal] | None:
    """The atom's literals when it is a single conjunction, else None.

    A SMILES atom expression has no ``,`` or ``;``, so its formula is one or-clause
    holding one and-clause. Anything else is a query. ``&`` is not detectable here -- the
    parser folds it into the same and-clause as juxtaposition -- but it is only a spelling
    of a conjunction the reading already allows, so it changes no meaning.
    """
    if len(formula) != 1 or len(formula[0]) != 1:
        return None
    return formula[0][0]


def validate_smiles_atom(atom: SmartsAtom, index: int) -> None:
    literals = conjunction(atom.formula)
    if literals is None:
        raise NotMolecularSmilesError(
            f"Atom {index} uses SMARTS alternation or low-precedence conjunction "
            "(',' or ';'), which no molecule has."
        )

    negated = [literal.prop for literal in literals if literal.negated]
    if negated:
        raise NotMolecularSmilesError(
            f"Atom {index} negates {', '.join(sorted(set(negated)))}; "
            "a molecule states what it is, not what it is not."
        )

    unsupported = {literal.prop for literal in literals} - SMILES_ATOM_PROPS
    if unsupported:
        raise NotMolecularSmilesError(
            f"Atom {index} carries the SMARTS-only "
            f"{'primitives' if len(unsupported) > 1 else 'primitive'} "
            f"{', '.join(sorted(unsupported))}."
        )

    for prop in sorted(SMILES_ATOM_PROPS):
        if sum(literal.prop == prop for literal in literals) > 1:
            raise NotMolecularSmilesError(f"Atom {index} states {prop} more than once.")

    symbols = [literal for literal in literals if literal.prop == "symbol"]
    if len(symbols) != 1:
        raise NotMolecularSmilesError(f"Atom {index} names no element.")
    if symbols[0].value in NON_ELEMENT_SYMBOLS:
        raise NotMolecularSmilesError(
            f"Atom {index} is written '{symbols[0].value}', which names no element."
        )

    validate_representable(literals, index)


def written_value(literals: list[PropVal], prop: str, index: int) -> int | None:
    """The integer an atom wrote for ``prop``, or None if it wrote nothing.

    A bare ``H`` carries no value and means one (OpenSMILES 3.1), which is inside every
    range checked here, so it needs no separate case.
    """
    literal = next((item for item in literals if item.prop == prop), None)
    if literal is None or literal.value is None:
        return None
    if not isinstance(literal.value, int):
        raise NotMolecularSmilesError(f"Atom {index} states a non-integer {prop}.")
    return literal.value


def validate_representable(literals: list[PropVal], index: int) -> None:
    """Refuse a property the mask encoding could only hold by rounding it.

    A molecule states its properties exactly, and the per-atom masks are buckets: charge
    saturates at ``CHARGE_MIN``/``CHARGE_MAX`` and counts at ``BUCKET_MAX``. Left
    unchecked, `[C+5]` becomes a carbon of charge +4 and `[S-4]` one of -3 -- built,
    returned, and wrong, which is the failure this reading exists to prevent. The isotope
    is a plain range rather than a bucket, but belongs to the same question.
    """
    charge = written_value(literals, "C", index)
    if charge is not None and not CHARGE_MIN <= charge <= CHARGE_MAX:
        raise NotMolecularSmilesError(
            f"Atom {index} has charge {charge:+d}, outside the representable "
            f"{CHARGE_MIN:+d} to {CHARGE_MAX:+d}."
        )

    hydrogens = written_value(literals, "H", index)
    if hydrogens is not None and not 0 <= hydrogens <= BUCKET_MAX:
        raise NotMolecularSmilesError(
            f"Atom {index} states {hydrogens} hydrogens, outside the representable "
            f"0 to {BUCKET_MAX}."
        )

    validate_isotope(literals, index)


def validate_isotope(literals: list[PropVal], index: int) -> None:
    mass = next((literal for literal in literals if literal.prop == "M"), None)
    if mass is None:
        return
    if not isinstance(mass.value, int):
        raise NotMolecularSmilesError(f"Atom {index} has a non-integer isotope.")
    if mass.value == 0:
        # OpenSMILES distinguishes "isotope written as 0" from "no isotope written" so a
        # writer can round-trip its own output. There is no nuclide of mass zero, so this
        # model refuses the spelling rather than keying it as natural abundance and
        # silently agreeing with ``[S]``. A documented divergence, and a loud one.
        raise NotMolecularSmilesError(
            f"Atom {index} is written with isotope 0, which is not a nuclide. "
            "Write the element alone for natural abundance."
        )
    if not MIN_ISOTOPE <= mass.value <= MAX_ISOTOPE:
        raise NotMolecularSmilesError(
            f"Atom {index} has isotope {mass.value}, outside the supported "
            f"{MIN_ISOTOPE}-{MAX_ISOTOPE}."
        )


def validate_smiles_ir(mol: SmartsMol) -> None:
    """Refuse a parsed expression that is a query rather than a molecule.

    This cannot be left to concretization. ``AtomMasks.exact_h_total`` answers -1 for a
    mask that is not a single bit, and ``exact_c`` answers 0 for any mask that merely
    includes neutral, so ``[C;!H0]``, ``[C;H1,H2]``, ``[C;R]`` and ``[C;D2]`` all reach
    the molecule builder and become methane. Only the element and the bond kinds are
    checked for exactness downstream.
    """
    for index, atom in enumerate(mol.atoms):
        validate_smiles_atom(atom, index)


def smiles_reading(mol: SmartsMol) -> SmartsMol:
    """Read a parsed expression as a molecule: an omitted bracket hydrogen count is zero.

    Validates first, so every atom is a single conjunction of literals and the rewrite
    has one well-defined place to land. Hydrogens written as their own atoms are folded
    afterwards, because folding increments a count that ``with_written_hydrogens`` may
    have just supplied.
    """
    validate_smiles_ir(mol)
    return fold_explicit_hydrogens(
        SmartsMol(
            atoms=[with_written_hydrogens(atom) for atom in mol.atoms],
            bonds=mol.bonds,
            groups=mol.groups,
        )
    )


def folds_into(mol: SmartsMol, index: int) -> int | None:
    """The neighbour this atom should become a hydrogen *count* on, or ``None``.

    ``None`` for every atom that has to stay a node of its own, and the exclusions are the
    whole content of this function:

    * not hydrogen, or a query rather than a plain conjunction -- nothing to fold;
    * an isotope. ``[2H]`` is deuterium, a different nuclide, and merging it into a count
      would silently relabel it as ordinary hydrogen;
    * a charge. ``[H+]`` is a proton, a species in its own right, not a substituent;
    * an atom map, which folding would discard;
    * anything but exactly one default-order bond. A bare ``[H]`` has no neighbour to join,
      a bridging hydride has two, and ``C=[H]`` is not a hydrogen substituent;
    * a hydrogen neighbour. ``[H][H]`` is dihydrogen, which has no heavy atom to carry the
      count, so it is left alone to fail downstream with the error it deserves rather than
      being quietly turned into a lone hydrogen atom.
    """
    literals = conjunction(mol.atoms[index].formula)

    if literals is None or mol.atoms[index].atom_map is not None:
        return None

    symbols = [literal for literal in literals if literal.prop == "symbol"]

    if len(symbols) != 1 or symbols[0].value != "H" or symbols[0].negated:
        return None

    if any(literal.prop == "M" for literal in literals):
        return None

    if any(
        literal.prop == "C" and literal.value not in (0, None) for literal in literals
    ):
        return None

    neighbours = [
        other
        for other in range(mol.atom_count)
        if other != index and mol.bonds[index][other]
    ]

    if len(neighbours) != 1 or mol.bonds[index][neighbours[0]] != QBOND_DEFAULT:
        return None

    partner = conjunction(mol.atoms[neighbours[0]].formula)

    if partner is None or any(
        literal.prop == "symbol" and literal.value == "H" for literal in partner
    ):
        return None

    return neighbours[0]


def fold_explicit_hydrogens(mol: SmartsMol) -> SmartsMol:
    """Rewrite ``C[H]`` into ``C``, and ``[N+]([H])`` into ``[NH+]``.

    Both spellings are valid SMILES for the same molecule, and RDKit reads either, but the
    matcher models hydrogen as a *property of an atom* rather than a node in the graph --
    which is exactly what makes ``C`` and ``C[H]`` one molecule to every keyer, fingerprint
    and template it has. Without this the second spelling reaches ``cmol_builder`` as a bare
    hydrogen atom and is refused: *occupies no declared configuration*. It is a real input,
    not a curiosity: a real campaign target writes a protonated ring nitrogen this way.

    **Only the molecule reading does this**, because :func:`smiles_reading` is the only
    caller and ``parse_smarts`` does not run it. So a pattern that names ``[H]`` still means
    an explicit hydrogen atom to match, and nothing about query semantics moves. Neither
    shipped rule set writes one -- 0 of raw68k's 67,557 templates, 0 of Hartenfeller's 146 --
    so nothing in use is affected either way.

    **No stereochemistry is involved.** Written as SMILES, dropping the explicit neighbour
    reorders the rest and the chirality tag has to flip from ``@`` to ``@@`` to keep its
    meaning. Here it does not arise: chirality is stripped before parsing and the matcher
    does not model it, so two enantiomers already produce one key.

    An unbracketed neighbour is left untouched on purpose. ``C[H]`` becomes ``C``, and the
    hydrogen law then fills it to methane -- the same molecule, reached by the rule that
    already governs unbracketed atoms rather than by a count written on top of it.
    """
    folded = {
        index: partner
        for index in range(mol.atom_count)
        if (partner := folds_into(mol, index)) is not None
    }

    if not folded:
        return mol

    gained: dict[int, int] = {}

    for partner in folded.values():
        gained[partner] = gained.get(partner, 0) + 1

    kept = [index for index in range(mol.atom_count) if index not in folded]

    return SmartsMol(
        atoms=[
            with_extra_hydrogens(mol.atoms[index], gained.get(index, 0))
            for index in kept
        ],
        bonds=mol.bonds[np.ix_(kept, kept)],
        # Folded away with their heavy neighbour, so the group array is reindexed exactly as the
        # atoms and the bond matrix are. A hydrogen never carried a group of its own.
        groups=mol.groups[kept],
    )


def written_hydrogens(literal: PropVal) -> int:
    """The count a written ``H`` literal carries. A bare ``H`` is one, not zero.

    OpenSMILES 1.0 section 3.1: "``[CH]`` and ``[CH1]`` are identical." The parser records a
    bare ``H`` as ``value=None``, so reading it as zero would turn ``C[C@@H]([H])O`` into a
    carbon short of two hydrogens -- which is how this was found.
    """
    return 1 if literal.value is None else int(literal.value)


def with_extra_hydrogens(atom: SmartsAtom, extra: int) -> SmartsAtom:
    """``atom`` carrying ``extra`` more hydrogens, when it is the kind that counts them.

    An unbracketed atom is returned unchanged even when it gained one: its hydrogens come
    from the valence law, which refills the bond the folded node vacated.
    """
    if not extra or not atom.bracketed:
        return atom

    literals = conjunction(atom.formula)

    if literals is None:
        return atom

    return SmartsAtom(
        formula=[
            [
                [
                    (
                        PropVal(
                            "H", written_hydrogens(literal) + extra, literal.negated
                        )
                        if literal.prop == "H"
                        else literal
                    )
                    for literal in literals
                ]
            ]
        ],
        is_aromatic=atom.is_aromatic,
        atom_map=atom.atom_map,
        bracketed=True,
    )


def with_written_hydrogens(atom: SmartsAtom) -> SmartsAtom:
    """``[C]`` becomes ``[CH0]``; an unbracketed ``C`` is left to the hydrogen law.

    Only brackets carry the rule. An organic-subset atom written without brackets means
    "fill to the lowest normal valence", which is what omitting the count already does.
    """
    if not atom.bracketed:
        return atom
    literals = conjunction(atom.formula)
    if literals is None or any(literal.prop == "H" for literal in literals):
        return atom
    # A new formula rather than an append: ``SmartsAtom`` is frozen but its formula is a
    # plain nested list, so mutating in place would reach every atom sharing it.
    return SmartsAtom(
        formula=[[[*literals, PropVal("H", 0)]]],
        is_aromatic=atom.is_aromatic,
        atom_map=atom.atom_map,
        bracketed=True,
    )
