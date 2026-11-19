"""Atom configurations: one lookup, never a search."""

from __future__ import annotations

import pytest

from vesma.configurations import (
    Configuration,
    ConfigurationFailure,
    PiContribution,
    configurations_for,
    select_circuit_configuration,
    select_configuration,
    unwritten_hydrogen_configuration,
    unwritten_hydrogens,
)
from vesma.symbols import ELEMENT_TO_Z


def selected(symbol: str, charge: int, sigma_degree: int, total_h: int, **kwargs):
    return select_configuration(
        z=ELEMENT_TO_Z[symbol],
        charge=charge,
        sigma_degree=sigma_degree,
        total_h=total_h,
        bond_order_sum=kwargs.get("bond_order_sum", 0),
        delocalized=kwargs.get("delocalized", True),
    )


@pytest.mark.parametrize(
    "symbol, charge, sigma_degree, total_h, expected_pi",
    [
        ("C", 0, 2, 1, PiContribution.ONE),  # benzene
        ("N", 0, 2, 0, PiContribution.ONE),  # pyridine-like
        ("N", 0, 2, 1, PiContribution.TWO),  # pyrrole-like
        ("O", 0, 2, 0, PiContribution.TWO),  # furan
        ("S", 0, 2, 0, PiContribution.TWO),  # thiophene
        ("C", -1, 2, 1, PiContribution.TWO),  # cyclopentadienyl
        ("C", 1, 2, 1, PiContribution.NONE),  # tropylium
    ],
)
def test_circuit_members_select_one_configuration(
    symbol: str,
    charge: int,
    sigma_degree: int,
    total_h: int,
    expected_pi: PiContribution,
) -> None:
    configuration = selected(symbol, charge, sigma_degree, total_h)

    assert isinstance(configuration, Configuration)
    assert configuration.pi is expected_pi


def test_every_declared_circuit_sigma_frame_selects_one_configuration() -> None:
    """The circuit selector is a lookup for the whole table, not only named examples.

    Localized bond orders cannot disambiguate a delocalized atom because the orderless
    spelling does not contain them. The model therefore relies on every eligible
    element/charge/sigma frame naming exactly one circuit configuration. Keep that as a
    checked table invariant as new elements and charge states are added.
    """
    frames: dict[tuple[int, int, int], Configuration] = {}
    for z in range(128):
        for charge in range(-3, 5):
            for configuration in configurations_for(z, charge):
                if configuration.pi is PiContribution.INVALID:
                    continue
                frame = (z, charge, configuration.sigma_capacity)
                assert frame not in frames, frame
                frames[frame] = configuration

    assert frames
    for (z, charge, sigma), expected in frames.items():
        assert (
            select_circuit_configuration(
                z=z,
                charge=charge,
                sigma_degree=sigma,
                total_h=0,
            )
            == expected
        )


def test_pyridine_and_pyrrole_nitrogen_differ_only_by_hydrogen() -> None:
    """The same element and charge; the sigma frame alone separates the two roles."""
    pyridine = selected("N", 0, 2, 0)
    pyrrole = selected("N", 0, 2, 1)

    assert (pyridine.pi, pyrrole.pi) == (PiContribution.ONE, PiContribution.TWO)
    assert pyridine.graph_valence == pyrrole.graph_valence


def test_a_saturated_centre_has_no_orbital_left_for_a_circuit() -> None:
    sp3 = select_configuration(
        z=ELEMENT_TO_Z["C"],
        charge=0,
        sigma_degree=0,
        bond_order_sum=0,
        total_h=4,
        delocalized=False,
    )
    assert sp3.pi is PiContribution.INVALID


def test_bond_orders_separate_sp3_from_sp2_carbon() -> None:
    """A delocalized atom has no orders to read, so the concrete branch needs them."""
    sp3 = select_configuration(
        z=ELEMENT_TO_Z["C"],
        charge=0,
        sigma_degree=4,
        bond_order_sum=4,
        total_h=0,
        delocalized=False,
    )
    sp2 = select_configuration(
        z=ELEMENT_TO_Z["C"],
        charge=0,
        sigma_degree=1,
        bond_order_sum=2,
        total_h=2,
        delocalized=False,
    )

    assert sp3.sigma_capacity == 4
    assert sp2.sigma_capacity == 3
    assert sp2.pi is PiContribution.ONE


def test_octet_expansion_never_joins_a_circuit() -> None:
    """A sulfoxide sulfur has a pi bond and a lone pair, but no ring p-orbital."""
    sulfoxide = select_configuration(
        z=ELEMENT_TO_Z["S"],
        charge=0,
        sigma_degree=3,
        bond_order_sum=4,
        total_h=0,
        delocalized=False,
    )
    assert sulfoxide.graph_valence == 4
    assert sulfoxide.pi is PiContribution.INVALID


def test_an_impossible_local_state_is_incompatible() -> None:
    assert (
        selected("C", 0, 5, 0, delocalized=False, bond_order_sum=5)
        is ConfigurationFailure.INCOMPATIBLE
    )


@pytest.mark.parametrize(
    "symbol, charge, sigma_degree, valence",
    [
        ("Mg", 0, 2, 2),  # Grignard
        ("Zn", 1, 1, 1),  # organozinc
        ("Sn", 0, 4, 4),  # stannane
        ("Li", 0, 1, 1),  # organolithium
        ("K", 1, 0, 0),  # isolated counter-ion, template 21730
        ("K", 0, 1, 1),  # bonded potassium, same template
        ("Mn", 0, 4, 7),  # permanganate
        ("Os", 0, 4, 8),  # osmium tetroxide
    ],
)
def test_declared_metal_configurations(
    symbol: str, charge: int, sigma_degree: int, valence: int
) -> None:
    configuration = select_configuration(
        z=ELEMENT_TO_Z[symbol],
        charge=charge,
        sigma_degree=sigma_degree,
        bond_order_sum=valence,
        total_h=0,
        delocalized=False,
    )

    assert isinstance(configuration, Configuration)
    assert configuration.graph_valence == valence
    assert configuration.pi is PiContribution.INVALID


def test_metals_carry_no_hydrogen_capacity_beyond_their_bonds() -> None:
    for symbol in ("Mg", "Zn", "Sn", "Li", "Na", "K", "Cu", "Mn", "Os"):
        for configuration in configurations_for(ELEMENT_TO_Z[symbol], 0):
            assert configuration.sigma_capacity <= configuration.graph_valence
            assert configuration.pi is PiContribution.INVALID


def unwritten_outcome(
    symbol: str,
    *,
    charge: int = 0,
    sigma_degree: int,
    bond_order_sum: int = 0,
    delocalized: bool = False,
) -> int | ConfigurationFailure:
    configuration = unwritten_hydrogen_configuration(
        z=ELEMENT_TO_Z[symbol],
        charge=charge,
        sigma_degree=sigma_degree,
        bond_order_sum=bond_order_sum,
        delocalized=delocalized,
    )
    if isinstance(configuration, ConfigurationFailure):
        return configuration
    return unwritten_hydrogens(
        configuration,
        sigma_degree=sigma_degree,
        bond_order_sum=bond_order_sum,
        delocalized=delocalized,
    )


def unwritten(symbol: str, **kwargs) -> int:
    """Hydrogen the law gives an atom whose count was never written.

    Refuses to return a failure. ``ConfigurationFailure`` is an ``IntEnum`` whose
    INCOMPATIBLE member is 0, so a bare ``== 0`` assertion would pass just as happily for
    "rejected" as for "no hydrogens" -- which is how a removed configuration went on
    satisfying a test that should have caught it.
    """
    outcome = unwritten_outcome(symbol, **kwargs)
    if isinstance(outcome, ConfigurationFailure):
        raise AssertionError(f"expected a hydrogen count, got {outcome!r}")
    return outcome


@pytest.mark.parametrize(
    "symbol, sigma_degree, bond_order_sum, expected, role",
    [
        ("C", 1, 1, 3, "methyl"),
        ("C", 2, 2, 2, "methylene"),
        ("C", 1, 3, 1, "terminal alkyne carbon"),
        ("N", 1, 1, 2, "amine, not a pentavalent ylide"),
        ("N", 3, 3, 0, "tertiary amine"),
        ("O", 1, 1, 1, "hydroxyl"),
        ("O", 2, 2, 0, "ether"),
        ("S", 1, 1, 1, "thiol, not a sulfurane"),
        ("S", 2, 2, 0, "thioether"),
        ("P", 1, 1, 2, "phosphine, not a phosphorane"),
        ("Cl", 1, 1, 0, "chloride"),
        ("B", 3, 3, 0, "borane"),
    ],
)
def test_omitted_hydrogen_is_the_fewest_the_configurations_allow(
    symbol: str, sigma_degree: int, bond_order_sum: int, expected: int, role: str
) -> None:
    assert (
        unwritten(symbol, sigma_degree=sigma_degree, bond_order_sum=bond_order_sum)
        == expected
    ), role


@pytest.mark.parametrize(
    "symbol, sigma_degree, bond_order_sum, expected, role",
    [
        ("S", 3, 4, 0, "sulfoxide: four written orders reach valence 4"),
        ("S", 4, 6, 0, "sulfone: six written orders reach valence 6"),
        ("P", 4, 4, 1, "phosphonic acid keeps its P-H at valence 5"),
        ("P", 4, 5, 0, "phosphate ester"),
    ],
)
def test_octet_expansion_is_evidenced_by_bonding_never_by_hydrogen(
    symbol: str, sigma_degree: int, bond_order_sum: int, expected: int, role: str
) -> None:
    """Hydrogen never buys an expanded valence on its own; written bond orders do.

    The pair to read together is phosphorus at bond-order sum 1 and at 4: the first is a
    phosphine with two hydrogens because valence 3 is the least it reaches, and the second
    keeps exactly one hydrogen because four written orders leave no declared valence below
    5. Neither is a search for a valence that fits.
    """
    assert (
        unwritten(symbol, sigma_degree=sigma_degree, bond_order_sum=bond_order_sum)
        == expected
    ), role


@pytest.mark.parametrize(
    "symbol, sigma_degree, expected, role",
    [
        ("C", 2, 1, "aromatic CH"),
        ("C", 3, 0, "substituted aromatic carbon"),
        ("N", 2, 0, "lowercase n is pyridine-like, not pyrrole-like"),
        ("O", 2, 0, "furan oxygen"),
        ("S", 2, 0, "thiophene sulfur"),
    ],
)
def test_omitted_hydrogen_on_a_delocalized_atom_reads_its_sigma_frame(
    symbol: str, sigma_degree: int, expected: int, role: str
) -> None:
    hydrogens = unwritten(symbol, sigma_degree=sigma_degree, delocalized=True)
    assert hydrogens == expected, role


@pytest.mark.parametrize(
    "symbol, charge, sigma_degree, bond_order_sum, valence, reagent",
    [
        ("I", 0, 3, 3, 3, "PhI(OAc)2, a lambda-3 iodane"),
        ("I", 0, 3, 5, 5, "iodyl chloride O=I(=O)Cl, iodine(V)"),
        ("I", 0, 5, 5, 5, "Dess-Martin periodinane, a lambda-5 iodane"),
        ("S", 1, 4, 5, 5, "trimethylsulfoxonium, the Corey-Chaykovsky reagent"),
    ],
)
def test_declared_hypervalent_configurations(
    symbol: str,
    charge: int,
    sigma_degree: int,
    bond_order_sum: int,
    valence: int,
    reagent: str,
) -> None:
    """Hypervalent main-group centres that real reagents occupy.

    The sulfoxonium sulfur is reached by the isoelectronic shift rather than a declared
    row: S+ shares phosphorus's electron count, so it reaches phosphorus's expanded
    valence. An O+ inherits nothing of the sort, because expansion is the atom's own
    property and oxygen has no shell to expand into.
    """
    configuration = select_configuration(
        z=ELEMENT_TO_Z[symbol],
        charge=charge,
        sigma_degree=sigma_degree,
        bond_order_sum=bond_order_sum,
        total_h=0,
        delocalized=False,
    )

    assert isinstance(configuration, Configuration), reagent
    assert configuration.graph_valence == valence


def test_oxygen_does_not_inherit_an_expanded_valence_from_nitrogen() -> None:
    assert all(
        configuration.graph_valence == 3
        for configuration in configurations_for(ELEMENT_TO_Z["O"], 1)
    )


def test_an_undeclared_state_is_reported_rather_than_completed() -> None:
    """A boron dication is the one added atom in uspto42k the table will not place.

    It is *undeclared* rather than incompatible: the table says nothing at all about
    boron at charge +2, which is a gap in the model. Incompatible means the declared
    states exist and none of them fits, which is a claim about the molecule. An audit
    that counts causes needs those two apart.
    """
    assert (
        unwritten_outcome("B", charge=2, sigma_degree=1, bond_order_sum=1)
        is ConfigurationFailure.UNDECLARED
    )
    assert (
        unwritten_outcome("C", sigma_degree=5, bond_order_sum=5)
        is ConfigurationFailure.INCOMPATIBLE
    )


def test_neutral_nitrogen_never_reaches_graph_valence_five() -> None:
    """Nitrogen is second period: four valence orbitals, so four bonds at most.

    The octet is a hard limit there and only a soft one from period 3 down, which is why
    phosphorus reaches valence 5 and nitrogen does not, though they share a group. What is
    drawn as pentavalent nitrogen is charge separated -- nitro is [N+](=O)[O-], azide is
    [N-]=[N+]=N -- and those are already covered, because N+ is isoelectronic with carbon
    and reaches valence 4 with no expansion at all.
    """
    assert all(
        configuration.graph_valence == 3
        for configuration in configurations_for(ELEMENT_TO_Z["N"], 0)
    )
    assert any(
        configuration.graph_valence == 5
        for configuration in configurations_for(ELEMENT_TO_Z["P"], 0)
    )
    assert any(
        configuration.graph_valence == 4
        for configuration in configurations_for(ELEMENT_TO_Z["N"], 1)
    )
    assert (
        unwritten_outcome("N", sigma_degree=3, bond_order_sum=5)
        is ConfigurationFailure.INCOMPATIBLE
    )
