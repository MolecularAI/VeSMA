"""The peaks to expect from a step, and which of them a mass spectrum will not separate."""

from __future__ import annotations

import pytest

from vertsy.mixture import expected_mixture, peak
from vesma import cmol_from_smiles, parse_reaction_smarts
from vesma.formula import composition
from vesma.keyer import cmol_rdkit_smiles_keyer as rdkit_key
from vesma.keyer import short_ring_wl_keyer as wl_key
from vesma.reaction import ReactionPattern


def rule(code: str, smarts: str) -> tuple[str, ReactionPattern]:
    """One classified group entry, parsed as `expected_mixture` now takes them.

    The caller parses because the caller is the one that can: in the bench, `step_mixture` holds
    the whole forward table parsed and cached for the sweep, and handing SMARTS down made it
    re-parse every rule for every reported step. These tests are still about the SMARTS named
    beside them, so this is a wrapping rather than a change of subject.
    """
    return code, parse_reaction_smarts(smarts)


AMIDE = "([C:1](=[O:2])-[Cl,Br]).([NH2;D1:3]-[#6:4])>>[C:1](=[O:2])-[N:3]-[#6:4]"
ESTER = "([C:1](=[O:2])-[Cl,Br]).([OX2H1:3]-[#6:4])>>[C:1](=[O:2])-[O:3]-[#6:4]"
SNAR = "([c:1](:[n:2])-[Cl,F]).([NH2;D1:3]-[#6:4])>>[c:1](:[n:2])-[N:3]-[#6:4]"

# One of the measured isobaric pairs: an amine and a dichloropyrimidine, where substituting at
# either chloride gives `C11H17ClFN3O2S` and the two products weigh exactly the same.
AZINE_PRODUCT = "CC(C)(C)C(CS(C)(=O)=O)Nc1nc(Cl)ncc1F"
AZINE_POT = ("CC(C)(C)C(N)CS(C)(=O)=O", "Fc1cnc(Cl)nc1Cl")


def render(molecule) -> str:
    return rdkit_key.encode(molecule)


def identity(molecule):
    """Dedup by what the *search* keys by, which here is the WL hash the CPU and Metal arms use."""
    return wl_key.encode(molecule)


def test_the_by_product_is_the_steps_own_atom_balance() -> None:
    """No template is consulted for it, which is why every step can report one.

    What the pot held minus what it made: an acid chloride and an amine give an amide and `HCl`.
    The composition is derived, so there is no structure to show -- the balance says a hydrogen
    left with the chlorine, not which hydrogen.
    """
    peaks = expected_mixture(
        "CCNC(C)=O", ["CC(=O)Cl", "CCN"], key=identity, render=render
    )

    kinds = {peak.kind: peak for peak in peaks}
    assert set(kinds) == {"product", "atoms lost"}
    assert kinds["product"].formula == "C4H9NO"
    assert kinds["atoms lost"].formula == "HCl"
    assert kinds["atoms lost"].exact_mass == pytest.approx(35.9767, abs=1e-4)
    assert kinds["atoms lost"].smiles == "", "a composition, not a structure"


def test_an_isobaric_competitor_is_flagged_and_the_product_is_not() -> None:
    """The finding that makes this worth more than a count: 31.7% of competitor products share
    the intended product's formula, because a competition is usually the same two partners
    joining differently -- which rearranges atoms without changing their number.

    Here substituting the other chloride gives a different compound at the *same* exact mass. A
    single product peak is therefore not evidence of a clean step, and that is exactly what a
    price cannot say. The product itself is never flagged: it shares its own formula trivially.
    """
    peaks = expected_mixture(
        AZINE_PRODUCT,
        list(AZINE_POT),
        same_site=[rule("FWD H 54", SNAR)],
        key=identity,
        render=render,
    )

    product = next(peak for peak in peaks if peak.kind == "product")
    competitor = next(
        peak for peak in peaks if peak.kind == "another reaction, same site"
    )

    assert not product.isobaric
    assert competitor.isobaric
    assert competitor.formula == product.formula == "C11H17ClFN3O2S"
    assert competitor.exact_mass == product.exact_mass
    # And they are genuinely different compounds, not one molecule written twice.
    assert competitor.smiles != product.smiles
    assert competitor.source == "FWD H 54"


def test_a_resolvable_competitor_is_not_flagged() -> None:
    """Two thirds of them are, and for those the report says where to look."""
    peaks = expected_mixture(
        "CCNC(C)=O",
        ["CC(=O)Cl", "CCN", "CCO"],
        different_site=[rule("esterification", ESTER)],
        key=identity,
        render=render,
    )

    other = [peak for peak in peaks if peak.kind == "another reaction, another site"]

    assert other, "the alcohol in the pot gives an ester"
    assert not any(peak.isobaric for peak in other)
    assert {peak.formula for peak in other} == {"C4H8O2"}


def test_the_intended_product_is_not_reported_as_its_own_competitor() -> None:
    """Whichever way the report spelled it, because identity is the search's key and not a string.

    Two peaks are the same peak when the search would call them one molecule. Deduplicating on a
    *rendering* instead re-reported the product as a competitor of itself whenever the report's
    spelling differed from the renderer's -- `O=C(C)NCC` against `CCNC(C)=O`, one molecule, two
    strings -- which is the same confusion the probe's own same-product filter exists to prevent.
    A canonical SMILES is a fine key, and one arm uses it; it is a key because the arm chose it,
    not because it is a string.
    """
    for written in ("CCNC(C)=O", "O=C(C)NCC", "CC(=O)NCC"):
        peaks = expected_mixture(
            written,
            ["CC(=O)Cl", "CCN"],
            same_reaction=[rule("the step's own rule", AMIDE)],
            key=identity,
            render=render,
        )

        assert [peak.kind for peak in peaks] == ["product", "atoms lost"], written


def test_the_order_puts_the_unprotectable_competitor_first() -> None:
    """Product, by-product, then the three classes worst first.

    A competitor at the same atom cannot be protected against -- protecting the site stops both
    reactions -- while one at another atom is answered by a protect/react/deprotect detour. So the
    one a reader can do least about is listed first among the competitors.
    """
    peaks = expected_mixture(
        AZINE_PRODUCT,
        list(AZINE_POT),
        same_site=[rule("FWD H 54", SNAR)],
        different_site=[rule("esterification", ESTER)],
        key=identity,
        render=render,
    )

    assert [peak.kind for peak in peaks][:3] == [
        "product",
        "atoms lost",
        "another reaction, same site",
    ]


def test_a_pot_of_one_molecule_still_reports_a_by_product() -> None:
    """A unimolecular step has a balance too, and no bimolecular rule can fire in its pot."""
    peaks = expected_mixture("C1CCOC1", ["OCCCCCl"], key=identity, render=render)

    kinds = {peak.kind: peak for peak in peaks}
    assert kinds["atoms lost"].formula == "HCl"


def test_identity_follows_the_arm_and_not_the_rendering() -> None:
    """Each arm keys its search its own way, so each arm's report means its own "same molecule".

    Run with the WL keyer the CPU and Metal arms use and with the canonical SMILES the RDKit arm
    uses, and the mixture comes out the same here -- which is the point: the answer does not
    depend on the *renderer*, and swapping the keyer is how a caller says which search's notion of
    identity the report should speak.
    """
    for keyer in (wl_key, rdkit_key):
        peaks = expected_mixture(
            "O=C(C)NCC",
            ["CC(=O)Cl", "CCN"],
            same_reaction=[rule("the step's own rule", AMIDE)],
            key=keyer.encode,
            render=render,
        )

        assert [peak.kind for peak in peaks] == ["product", "atoms lost"], keyer


# The reductive amination as it is now written: it declares the water the condensation expels.
REDUCTIVE_AMINATION = (
    "([#6:1]-[CH1:2]=[O]).([NH1;D2:3](-[#6:4])-[#6:5])"
    ">>([#6:1]-[CH2:2]-[N:3](-[#6:4])-[#6:5]).([OH2])"
)


def test_a_declared_by_product_is_reported_as_one_and_not_as_another_site() -> None:
    """The bug declaring by-products exposed, and the reason the heaviest component is the product.

    A rule that declares its by-product emits both together, so every component of its outcome
    arrived from the same reaction. Treating them all as products made the water read as "same
    reaction, another site" -- a competing product, which is the opposite of what it is.
    """
    peaks = expected_mixture(
        "CCCN(C)CC",
        ["CCC=O", "CNCC"],
        same_reaction=[rule("FWD H 30c", REDUCTIVE_AMINATION)],
        key=identity,
        render=render,
    )

    kinds = {peak.kind: peak for peak in peaks}
    assert set(kinds) == {"product", "by-product"}
    water = kinds["by-product"]
    assert water.formula == "H2O"
    assert water.exact_mass == pytest.approx(18.0106, abs=1e-4)
    assert water.source == "FWD H 30c", "attributed to the rule that declared it"
    assert water.smiles, "a declared by-product has a structure, unlike a derived one"


def test_a_declaration_beats_the_balance_for_the_same_leaving_group() -> None:
    """Both are accounts of one thing, and the declaration is the better account.

    A reductive amination's atom balance can only see the oxygen: the two hydrogens that make it
    water came from a hydride the pot never held, so the derived by-product is `O` at 15.995 --
    not a compound. With the water declared, the report says `H2O` at 18.011 and does not also
    say `O`.
    """
    peaks = expected_mixture(
        "CCCN(C)CC",
        ["CCC=O", "CNCC"],
        same_reaction=[rule("FWD H 30c", REDUCTIVE_AMINATION)],
        key=identity,
        render=render,
    )

    assert [peak.formula for peak in peaks if peak.kind == "by-product"] == ["H2O"]
    # Without the declaration, the balance is all there is, and it is the half-seen version.
    bare = expected_mixture("CCCN(C)CC", ["CCC=O", "CNCC"], key=identity, render=render)
    assert [peak.formula for peak in bare if peak.kind == "atoms lost"] == ["O"]


def test_the_balance_is_not_presented_as_a_peak() -> None:
    """`atoms lost` is arithmetic; `by-product` is a compound. The distinction is the honest part.

    A Negishi's balance reads `Br2`, which is the right composition and the wrong molecule: the
    bromides leave with the zinc, and nobody should go looking for bromine at 157.84. Only a
    template that declares its by-product turns the arithmetic into a peak -- and then it has a
    structure and the rule that named it.

    This is also the answer to whether exact mass is robust. For a molecule the engine built it is
    exact, hydrogens and electrons included, checked against RDKit. For a composition nobody built
    it is a sum over atoms that left, and labelling it as such is what keeps the number from being
    read as advice.
    """
    negishi = expected_mixture(
        "Cc1ccccc1", ["Brc1ccccc1", "CBr"], key=identity, render=render
    )
    balance = next(peak for peak in negishi if peak.kind == "atoms lost")

    assert balance.formula == "Br2"
    assert not balance.smiles, "a composition, so no structure is claimed"
    assert not balance.source, "and no rule to attribute it to"

    # Against a row that does declare: a compound, with a structure and an attribution.
    declared = expected_mixture(
        "CCCN(C)CC",
        ["CCC=O", "CNCC"],
        same_reaction=[rule("FWD H 30c", REDUCTIVE_AMINATION)],
        key=identity,
        render=render,
    )
    water = next(peak for peak in declared if peak.kind == "by-product")

    assert water.smiles and water.source
    assert not any(peak.kind == "atoms lost" for peak in declared)


def test_a_peak_carries_which_tier_of_mass_coincidence_it_is() -> None:
    """Three tiers, and the middle one is the whole reason the bool was not enough.

    An ester and an amide at `C4H9NO` share a formula, so no instrument separates them. A peak at
    the product's *integer* mass with a different formula is a real warning and a different one: a
    high-resolution instrument does separate those, so it must not read as "isobaric".
    """
    product = composition(cmol_from_smiles("CCNC(C)=O")[0])

    # Butanamide, the same C4H9NO at the same 87.0684 -- an isomer, so nothing separates them.
    same_formula = peak(
        "another reaction, same site",
        composition(cmol_from_smiles("CCCC(N)=O")[0]),
        product_counts=product,
    )
    # Pentylamine is C5H13N at 87.1049: the same integer mass, 36 mDa away.
    near = peak(
        "another reaction, same site",
        composition(cmol_from_smiles("CCCCCN")[0]),
        product_counts=product,
    )
    resolved = peak(
        "another reaction, same site",
        composition(cmol_from_smiles("c1ccccc1")[0]),
        product_counts=product,
    )

    assert same_formula.coincidence == "same formula" and same_formula.isobaric
    assert near.coincidence == "same nominal mass" and not near.isobaric
    assert resolved.coincidence == "resolved" and not resolved.isobaric


def test_the_middle_tier_is_not_isobaric() -> None:
    """Derived from the tier, so the two cannot drift, and only the top tier is isobaric."""
    propane = composition(cmol_from_smiles("CCC")[0])
    acetaldehyde = composition(cmol_from_smiles("CC=O")[0])

    near = peak("by-product", acetaldehyde, product_counts=propane)

    assert near.coincidence == "same nominal mass"
    # A better instrument separates these, so it is not isobaric.
    assert not near.isobaric


def test_the_product_never_coincides_with_itself() -> None:
    """It shares its own formula trivially, and would read as a warning about what you are making."""
    product = composition(cmol_from_smiles("CCNC(C)=O")[0])

    assert peak("product", product, product_counts=product).coincidence == "resolved"


# The shape that exposed the bug: a Wittig whose triphenylphosphine oxide is twenty heavy atoms
# against the stilbene's seventeen, so the by-product outweighs the thing the step was for.
WITTIG = (
    "([#6:1]-[CH1:2]=[O:7]).([CH2:3](-[#6:4])-[Cl,Br,I:5])"
    ".(c1ccccc1P(c1ccccc1)c1ccccc1)>>([#6:1]-[C:2]=[C:3]-[#6:4])"
    ".([O:7]=P(c1ccccc1)(c1ccccc1)c1ccccc1).([*H1:5])"
)
WITTIG_PRODUCT = "Brc1ccccc1C=Cc1ccccc1I"
WITTIG_POT = (
    "O=Cc1ccccc1I",
    "ClCc1ccccc1Br",
    "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
)


def test_a_by_product_heavier_than_the_product_is_still_a_by_product() -> None:
    """Which component the reaction made is the target, not the biggest one.

    The heaviest-component rule is only a guess at the product, and it is needed only where an
    outcome does not reach the target at all. Applied unconditionally it inverted both labels on
    this step: the phosphine oxide read as "same reaction, another site" -- a competing product,
    which it is the opposite of -- and the HCl beside it as the lone by-product.
    """
    peaks = expected_mixture(
        WITTIG_PRODUCT,
        WITTIG_POT,
        same_reaction=(rule("FWD H 50", WITTIG),),
        same_site=(),
        different_site=(),
        key=identity,
        render=render,
    )
    kinds = {one.formula: one.kind for one in peaks}
    assert kinds["C14H10BrI"] == "product"
    assert kinds["C18H15OP"] == "by-product"  # triphenylphosphine oxide
    assert kinds["HCl"] == "by-product"
    # Nothing is a competing product, because the rule reached the target in the same outcome.
    assert not [one for one in peaks if "another site" in one.kind]


def test_a_rule_that_misses_the_target_claims_no_product_among_its_components() -> None:
    """With no target in the outcome there is no intended product, so none is named.

    Asked for an alkene this pot cannot give, the Wittig still fires and still makes the other
    stilbene. Which of the three components it was *aiming* at is not knowable here -- component
    order is not right-hand-side group order, and the outcome's mapping is the left-hand-side
    match -- so every one of them carries the group's label and nothing is called a by-product.
    """
    peaks = expected_mixture(
        "Brc1ccccc1C=Cc1ccccc1F",
        WITTIG_POT,
        same_reaction=(rule("FWD H 50", WITTIG),),
        same_site=(),
        different_site=(),
        key=identity,
        render=render,
    )
    kinds = {one.formula: one.kind for one in peaks}
    assert kinds["C14H10BrI"] == "same reaction, another site"
    assert kinds["C18H15OP"] == "same reaction, another site"
    assert kinds["HCl"] == "same reaction, another site"
    # Nothing is a by-product, because a by-product is a by-product *of the intended reaction*.
    assert not [one for one in peaks if one.kind == "by-product"]
