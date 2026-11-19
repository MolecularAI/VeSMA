"""Composition and exact mass: what a report needs and matching never did.

The numbers are checked against RDKit rather than asserted from the table they came from. A
monoisotopic mass table is exactly the kind of thing that is wrong in one entry and never noticed,
and `Descriptors.ExactMolWt` is an independent implementation of the same sum.
"""

from __future__ import annotations

from collections import Counter

import pytest

from vesma import cmol_from_smiles
from vesma.formula import (
    ELECTRON_MASS,
    MONOISOTOPIC_MASS,
    composition,
    difference,
    exact_mass,
    hill_formula,
    mass_coincidence,
    net_charge,
    nominal_mass,
)

# Spanning the elements the engine declares, including the ones a mass table gets wrong: boron
# (where 11 is the abundant isotope, not 10), silicon, sulfur, and the heavy halogens.
MOLECULES = (
    "O",
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(=O)Cl",
    "Br",
    "CCCBr",
    "OB(O)c1ccccc1",
    "CS(C)(=O)=O",
    "FC(F)(F)c1ccncc1",
    "CC(=O)OCC",
    "O=S(=O)(O)O",
    "C[Si](C)(C)C",
    "ClCCI",
)


# Ions, where the electron mass shows: an anion is heavier than its neutral formula by one
# electron per negative charge. `[AlH4-]` is 31.01339, not 31.01284.
IONS = ("[AlH4-]", "[BH4-]", "CC(=O)[O-]", "[NH4+]", "[N-]=[N+]=[N-]")


@pytest.mark.parametrize("smiles", MOLECULES + IONS)
def test_exact_mass_agrees_with_rdkit(smiles: str) -> None:
    descriptors = pytest.importorskip("rdkit.Chem.Descriptors")
    chem = pytest.importorskip("rdkit.Chem")

    molecule, _info = cmol_from_smiles(smiles)
    ours = exact_mass(composition(molecule), charge=net_charge(molecule))
    theirs = descriptors.ExactMolWt(chem.MolFromSmiles(smiles))

    # The table is rounded to five decimals, so the tolerance is the rounding and not a fudge.
    assert abs(ours - theirs) < 1e-4, f"{smiles}: {ours} vs {theirs}"


def test_an_ion_is_heavier_than_its_neutral_formula() -> None:
    """Half a milli-mass-unit, below any instrument -- and modelled anyway.

    Agreeing with an outside implementation is worth more than an argument about whether the
    difference matters, and the comparison above would otherwise carry a tolerance wide enough to
    hide a real table error.
    """
    borohydride, _info = cmol_from_smiles("[BH4-]")
    counts = composition(borohydride)

    assert net_charge(borohydride) == -1
    gained = exact_mass(counts, charge=-1) - exact_mass(counts)
    assert gained == pytest.approx(ELECTRON_MASS, abs=1e-9)


def test_hydrogen_is_counted_and_that_is_the_point() -> None:
    """Without it a by-product formula is wrong where it matters: water is 18, not 16."""
    water, _info = cmol_from_smiles("O")

    assert composition(water) == Counter({1: 2, 8: 1})
    assert hill_formula(composition(water)) == "H2O"
    assert exact_mass(composition(water)) == pytest.approx(18.0106, abs=1e-4)


def test_hill_order_is_carbon_then_hydrogen_then_alphabetical() -> None:
    ethyl_acetate, _info = cmol_from_smiles("CC(=O)OCC")
    sulfuric = Counter({1: 2, 8: 4, 16: 1})

    assert hill_formula(composition(ethyl_acetate)) == "C4H8O2"
    assert hill_formula(sulfuric) == "H2O4S", "no carbon, so hydrogen leads"


def test_an_element_with_no_mass_on_file_raises() -> None:
    """A missing contribution looks exactly like a different compound, so it cannot be silent."""
    with pytest.raises(ValueError, match="no monoisotopic mass on file"):
        exact_mass(Counter({95: 1}))
    assert 95 not in MONOISOTOPIC_MASS


def test_difference_separates_what_left_from_what_was_invented() -> None:
    """Two values, because they say different things about the template that moved the atoms.

    What is lost going forward is the by-product -- the leaving group, the water, the hydrogen
    halide, the extra peak. What is gained can only have come from a reagent the rule did not
    name, so a non-empty ``gained`` says the balance is open and the by-product read off ``lost``
    alone is incomplete.
    """
    acid_and_alcohol = Counter({6: 4, 1: 10, 8: 3})
    ester = Counter({6: 4, 1: 8, 8: 2})

    lost, gained = difference(acid_and_alcohol, ester)

    assert hill_formula(lost) == "H2O" and not gained
    # And the other direction, which is a chlorination inventing its chlorine.
    lost, gained = difference(Counter({8: 1, 1: 1}), Counter({17: 1}))
    assert hill_formula(lost) == "HO" and hill_formula(gained) == "Cl"


def test_a_by_product_is_derivable_from_an_application_without_authoring_it() -> None:
    """The reason the forward table does not need rewriting to SMIRKS level.

    A forward rule writes only its main product, and the by-product is the difference between what
    went into the pot and what came out. Three of these are exact: an esterification sheds `H2O`,
    an acid chloride `HCl`, a Williamson `HBr`.

    The fourth is the honest limit and is a *diagnostic*. A reductive amination reads as losing a
    bare `O`, because the two hydrogens that make it water came from a hydride the rule never
    named. A balance that loses a heteroatom and no hydrogen is a rule that borrowed atoms, which
    is the same implicit-reagent fault `conditions_allow` gates on -- and no amount of by-product
    authoring fixes it, because the atoms are genuinely not in the pot.
    """
    from vesma import generate_precursors, parse_reaction_smarts
    from vesma.cmol import cmol_union

    cases = (
        (
            "([C:1](=[O:2])-[OX2H1]).([OX2H1:3]-[#6:4])>>[C:1](=[O:2])-[O:3]-[#6:4]",
            ("CC(=O)O", "CCO"),
            "H2O",
        ),
        (
            "([C:1](=[O:2])-[Cl,Br]).([NH2;D1:3]-[#6:4])>>[C:1](=[O:2])-[N:3]-[#6:4]",
            ("CC(=O)Cl", "CCN"),
            "HCl",
        ),
        (
            "([CH2:1](-[#6,#8:2])-[Cl,Br,I]).([OX2H1:3]-[CX4:4])"
            ">>[CH2:1](-[*:2])-[O:3]-[C:4]",
            ("CCCBr", "CCO"),
            "HBr",
        ),
        (
            "([#6:1]-[CH1:2]=[O]).([NH1;D2:3](-[#6:4])-[#6:5])"
            ">>[#6:1]-[CH2:2]-[N:3](-[#6:4])-[#6:5]",
            ("CCC=O", "CNCC"),
            "O",  # the hydride's two hydrogens are missing, and that is the signal
        ),
    )
    for smarts, parts, expected in cases:
        pot = cmol_union([cmol_from_smiles(part)[0] for part in parts])
        outcomes = generate_precursors(pot, parse_reaction_smarts(smarts))
        assert outcomes, smarts
        emitted = Counter()
        for component in outcomes[0].precursors:
            emitted += composition(component)
        lost, gained = difference(composition(pot), emitted)

        assert hill_formula(lost) == expected, smarts
        assert not gained, "none of these invents an atom"


def test_the_three_tiers_of_mass_coincidence_are_three_instruments() -> None:
    """Not three points on one scale, which is why this is not a ppm window.

    Same formula means the exact masses are *identical*: no spectrometer at any resolution
    separates them, so the answer is chromatography or a different measurement entirely. Same
    nominal mass means a unit-resolution instrument sees one peak and a high-resolution one sees
    two, which is a question about the kit rather than about the chemistry. A single tolerance
    would read those two alike and lose exactly that distinction.
    """
    propane = composition(cmol_from_smiles("CCC")[0])
    acetaldehyde = composition(cmol_from_smiles("CC=O")[0])
    benzene = composition(cmol_from_smiles("c1ccccc1")[0])

    assert mass_coincidence(propane, propane) == "same formula"
    assert mass_coincidence(propane, acetaldehyde) == "same nominal mass"
    assert mass_coincidence(propane, benzene) == "resolved"
    # The middle tier is 36 mDa apart -- separable, but not on every instrument.
    assert exact_mass(propane) != exact_mass(acetaldehyde)
    assert abs(exact_mass(propane) - exact_mass(acetaldehyde)) == pytest.approx(
        0.0364, abs=1e-4
    )


def test_two_spellings_of_one_molecule_are_the_same_formula() -> None:
    """Decided on the composition, not on a float equality that happens to hold."""
    kekule = composition(cmol_from_smiles("CC(=O)OC1=CC=CC=C1")[0])
    aromatic = composition(cmol_from_smiles("CC(=O)Oc1ccccc1")[0])

    assert mass_coincidence(kekule, aromatic) == "same formula"


def test_nominal_mass_is_the_rounded_monoisotopic_table() -> None:
    """One table, so an integer mass cannot drift from the exact one it is derived from.

    Chlorine at 34.96885 and bromine at 78.91834 are the two that would catch a wrong rounding
    rule, since both sit below the integer they belong to.
    """
    assert nominal_mass(composition(cmol_from_smiles("CCCl")[0])) == 64
    assert nominal_mass(composition(cmol_from_smiles("CCBr")[0])) == 108
    assert nominal_mass(composition(cmol_from_smiles("O")[0])) == 18
    # The charge is ignored: an electron is 0.00055 u and no integer notices it.
    assert nominal_mass(composition(cmol_from_smiles("[OH-]")[0])) == 17


def test_an_element_with_no_mass_on_file_is_refused_not_guessed() -> None:
    from collections import Counter

    with pytest.raises(ValueError, match="no monoisotopic mass"):
        nominal_mass(Counter({92: 1}))
