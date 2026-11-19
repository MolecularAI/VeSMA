"""Counting only what two starting materials do *together* -- now stated by the pattern.

This file used to test `require_span`, a subtraction: probe each starting material alone and
remove what it makes by itself, because a two-component forward pattern matched against a
`cmol_union` pot matches both of its components *inside one* molecule whenever that molecule
carries both groups. The forward rules say it themselves now -- every multi-component left-hand
side is written `(A).(B)`, which the matcher enforces -- so the same three scenarios are checked
against the pattern rather than against the workaround, and the workaround is gone.

One answer changed with it, and deliberately. A single-precursor application used to keep its
intramolecular count, because the subtraction skipped groups of one member ("where the pot is one
molecule it *is* that molecule, and subtracting would leave nothing"). A bimolecular rule cannot
fire when there is one molecule in the pot, so the count is now zero, and that guard turns out to
have been preserving the defect it was written beside.
"""

from __future__ import annotations

import numpy as np

from vertsy.search import object_mol_array
from vertsy.selectivity import competitor_probe, object_combine
from vesma import cmol_from_smiles, reaction_patterns_from_smarts
from vesma.cmol import cmol_union
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.keyer import cmol_rdkit_smiles_keyer as keyer
from vesma.matcher import hit_array

# The shipped spelling: `FWD H 47`, an acyl chloride and a primary amine as two molecules.
AMIDE = "([C:1](=[O:2])-[Cl,Br]).([NH2;D1:3]-[#6:4])>>[C:1](=[O:2])-[N:3]-[#6:4]"
# The same chemistry written as one molecule carrying both halves, which is `FWD MC 102`.
LACTAM = "([C:1](=[O:2])-[Cl,Br].[NH2;D1:3]-[#6:4])>>[C:1](=[O:2])-[N:3]-[#6:4]"
# And the way the table was written before either could be said.
DOTTED = "[C:1](=[O:2])-[Cl,Br].[NH2;D1:3]-[#6:4]>>[C:1](=[O:2])-[N:3]-[#6:4]"


def keys(batch):
    return [keyer.encode(molecule) for molecule in batch]


def probe_counts(smarts: str, precursor_smiles: list[str]) -> list[int]:
    forward = CpuReactionMatcher(reaction_patterns_from_smarts((smarts,)))
    precursors = object_mol_array(
        [cmol_from_smiles(smiles)[0] for smiles in precursor_smiles]
    )
    products = object_mol_array([cmol_from_smiles("CCOCC")[0]])
    hits = hit_array([(0, 0, 0, row) for row in range(len(precursor_smiles))])
    probe = competitor_probe(
        forward,
        (2,),
        key=keys,
        combine=object_combine(cmol_union),
        index={0: ()},
    )
    return np.asarray(probe(products, precursors, hits)).tolist()


def test_a_cyclisation_of_one_precursor_is_not_a_competition_between_two() -> None:
    """The channel's own argument, enforced by the pattern that makes it.

    The amine and the acyl chloride are on the same precursor and the partner is an innocent
    alcohol, so there is nothing for the two to do to each other. The cyclisation itself is real
    chemistry -- this one closes to a six-membered lactam -- but it is a *different fate*:
    unimolecular, concentration-dependent, needing no second reagent. Written `(A).(B)`, the rule
    cannot claim it; written `(A.B)`, a different rule claims exactly it.
    """
    assert probe_counts(DOTTED, ["NCCCCC(=O)Cl", "CCO"]) == [1, 1], "the old reading"
    assert probe_counts(AMIDE, ["NCCCCC(=O)Cl", "CCO"]) == [0, 0]


def test_a_real_bimolecular_competition_survives() -> None:
    """The grouping must only remove what one molecule does by itself.

    An amine and an acyl chloride on *separate* precursors genuinely react with each other, and
    neither makes that amide alone.
    """
    assert probe_counts(AMIDE, ["CCN", "CC(=O)Cl"]) == [1, 1]


def test_a_single_precursor_is_no_pot_for_a_bimolecular_rule() -> None:
    """Zero, where the subtraction this replaced said one.

    `require_span` skipped groups of one member on the grounds that subtracting would leave
    nothing -- true, and it meant a unimolecular application kept whatever the bimolecular rule
    found inside its single precursor. There is no second molecule for that rule to have met, so
    zero is the answer, and the guard was holding the defect open on exactly the applications the
    subtraction could not reach.
    """
    assert probe_counts(AMIDE, ["NCCCCC(=O)Cl"]) == [0]
    # And the intramolecular sibling does claim it, which is where that chemistry now lives.
    assert probe_counts(LACTAM, ["NCCCCC(=O)Cl"]) == [1]
